from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from .model_profiles import ModelProfile, load_model_profile
from .models import build_fast_model
from .preprocessing import preprocess_image, stack_prepared_images
from .tiling import build_tile_grid
from .types import ModelInput, ModelOutput, StationConfig


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for Hugging Face runtime adapters.") from exc
    return torch


def _require_transformers():
    try:
        import transformers
    except ImportError as exc:
        raise ImportError(
            "Hugging Face backend support requires the `transformers` package. Install `greymodel[framework]` or add `transformers`."
        ) from exc
    return transformers


def _normalized_text(value: Any) -> str:
    return str(value).strip().lower()


def _softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    if values.size == 0:
        return values
    shifted = values.astype(np.float32) - np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.maximum(exp_values.sum(axis=axis, keepdims=True), 1e-6)


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(value))))


def _repeat_grayscale_to_rgb(image_uint8: np.ndarray) -> np.ndarray:
    if image_uint8.ndim != 2:
        raise ValueError("Expected a 2D grayscale image.")
    return np.repeat(image_uint8[:, :, None], 3, axis=2)


def _normalize_map(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    values = values - float(values.min())
    maximum = float(values.max())
    if maximum > 0:
        values = values / maximum
    return np.clip(values, 0.0, 1.0)


def _map_label(profile: ModelProfile, label: str) -> str:
    normalized = _normalized_text(label)
    mapped = profile.label_mapping.get(normalized)
    if mapped:
        return _normalized_text(mapped)
    for candidate in profile.good_labels:
        if normalized == _normalized_text(candidate):
            return "good"
    for candidate in profile.bad_labels:
        if normalized == _normalized_text(candidate):
            return "bad"
    mapped_family = profile.defect_family_mapping.get(normalized)
    if mapped_family:
        return _normalized_text(mapped_family)
    return normalized


def _stable_logit(probability: float) -> float:
    probability = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    return float(np.log(probability / (1.0 - probability)))


def _heatmap_from_image(prepared_image: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    image = np.asarray(prepared_image, dtype=np.float32)
    image = np.abs(image - float(np.mean(image[valid_mask > 0])) if np.any(valid_mask) else image)
    heatmap = _normalize_map(image)
    return heatmap * np.asarray(valid_mask, dtype=np.float32)


def _tile_scores_from_heatmap(heatmap: np.ndarray, station_config: StationConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = build_tile_grid(heatmap.shape, station_config.tile_size_2d, station_config.tile_stride_2d)
    tile_scores = []
    for y1, x1, y2, x2 in grid.boxes:
        tile_scores.append(float(np.mean(heatmap[y1:y2, x1:x2])) if y2 > y1 and x2 > x1 else 0.0)
    tile_scores_array = np.asarray(tile_scores, dtype=np.float32)
    top_k = min(5, tile_scores_array.shape[0])
    top_indices = np.argsort(-tile_scores_array)[:top_k] if top_k else np.zeros((0,), dtype=np.int64)
    top_boxes = np.asarray([grid.boxes[index] for index in top_indices], dtype=np.int64) if top_k else np.zeros((0, 4), dtype=np.int64)
    top_tiles = (
        np.concatenate([top_boxes.astype(np.float32), tile_scores_array[top_indices][:, None]], axis=1)
        if top_k
        else np.zeros((0, 5), dtype=np.float32)
    )
    return top_tiles, top_indices, top_boxes


def _heatmap_from_boxes(boxes: np.ndarray, scores: np.ndarray, shape: tuple[int, int], valid_mask: np.ndarray) -> np.ndarray:
    heatmap = np.zeros(shape, dtype=np.float32)
    counts = np.zeros(shape, dtype=np.float32)
    for score, box in zip(scores, boxes):
        x1, y1, x2, y2 = [int(value) for value in box]
        x1 = max(0, min(x1, shape[1]))
        y1 = max(0, min(y1, shape[0]))
        x2 = max(x1 + 1, min(x2, shape[1]))
        y2 = max(y1 + 1, min(y2, shape[0]))
        heatmap[y1:y2, x1:x2] += float(score)
        counts[y1:y2, x1:x2] += 1.0
    heatmap = np.divide(heatmap, np.maximum(counts, 1.0))
    return _normalize_map(heatmap) * np.asarray(valid_mask, dtype=np.float32)


def _prepare_inputs(model_input: ModelInput, station_config: StationConfig):
    prepared = preprocess_image(model_input, station_config)
    rgb_image = _repeat_grayscale_to_rgb(prepared.image_uint8)
    return prepared, rgb_image


def _canonical_family_scores(
    profile: ModelProfile,
    labels: Sequence[str],
    probabilities: np.ndarray,
    defect_families: Sequence[str],
) -> tuple[dict[str, float], float]:
    score_by_label = {
        _map_label(profile, label): float(probability)
        for label, probability in zip(labels, probabilities)
    }
    good_score = max((score for label, score in score_by_label.items() if label == "good"), default=0.0)
    bad_score = max((score for label, score in score_by_label.items() if label == "bad"), default=0.0)
    canonical_scores: dict[str, float] = {}
    if defect_families:
        for family in defect_families:
            canonical_scores[str(family)] = max(
                score
                for label, score in score_by_label.items()
                if label == str(family) or profile.defect_family_mapping.get(label) == str(family)
            ) if any(
                label == str(family) or profile.defect_family_mapping.get(label) == str(family)
                for label in score_by_label
            ) else 0.0
    else:
        for label, score in score_by_label.items():
            if label not in {"good", "bad"}:
                canonical_scores[label] = float(score)
    reject_score = bad_score if bad_score > 0 else (max(canonical_scores.values()) if canonical_scores else 1.0 - good_score)
    reject_score = float(min(max(reject_score, 0.0), 1.0))
    return canonical_scores, reject_score


def _defect_logits_from_probs(defect_probs: np.ndarray) -> np.ndarray:
    if defect_probs.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return np.asarray([_stable_logit(float(probability)) for probability in defect_probs], dtype=np.float32)


def _build_model_output(
    *,
    reject_score: float,
    defect_family_probs: np.ndarray,
    heatmap: np.ndarray,
    top_tiles: np.ndarray,
    top_tile_indices: np.ndarray,
    top_tile_boxes: np.ndarray,
    metadata: Mapping[str, Any],
    local_heatmap: Optional[np.ndarray] = None,
    global_heatmap: Optional[np.ndarray] = None,
    tile_logits: Optional[np.ndarray] = None,
    tile_validity: Optional[np.ndarray] = None,
) -> ModelOutput:
    reject_logit = _stable_logit(reject_score)
    return ModelOutput(
        reject_score=np.float32(reject_score),
        accept_reject_logit=np.float32(reject_logit),
        defect_family_probs=np.asarray(defect_family_probs, dtype=np.float32),
        defect_logits=_defect_logits_from_probs(np.asarray(defect_family_probs, dtype=np.float32)),
        defect_heatmap=np.asarray(heatmap, dtype=np.float32),
        top_tiles=np.asarray(top_tiles, dtype=np.float32),
        top_tile_indices=np.asarray(top_tile_indices, dtype=np.int64),
        top_tile_boxes=np.asarray(top_tile_boxes, dtype=np.int64),
        tile_logits=np.asarray(tile_logits, dtype=np.float32) if tile_logits is not None else None,
        tile_validity=np.asarray(tile_validity, dtype=np.bool_) if tile_validity is not None else None,
        local_heatmap=np.asarray(local_heatmap, dtype=np.float32) if local_heatmap is not None else None,
        global_heatmap=np.asarray(global_heatmap, dtype=np.float32) if global_heatmap is not None else None,
        metadata=dict(metadata),
    )


class NativeFastProfileRuntime:
    def __init__(
        self,
        profile: ModelProfile,
        *,
        num_defect_families: int,
        defect_families: Sequence[str] = (),
        num_stations: int = 32,
    ) -> None:
        del num_stations
        self.profile = profile
        self.torch = _require_torch()
        self.defect_families = tuple(defect_families) if defect_families else tuple("defect_%d" % index for index in range(max(num_defect_families, 1)))
        self.model = build_fast_model(
            num_defect_families=max(num_defect_families, 1),
            defect_families=self.defect_families,
        )
        checkpoint_path = _normalized_text(profile.local_path) if profile.local_path else ""
        if checkpoint_path:
            state_path = Path(checkpoint_path)
            if state_path.exists():
                payload = self.torch.load(state_path, map_location="cpu", weights_only=False)
                if isinstance(payload, Mapping) and "backbone" in payload:
                    payload = payload["backbone"]
                if isinstance(payload, Mapping) and "state_dict" in payload:
                    payload = payload["state_dict"]
                if isinstance(payload, Mapping):
                    self.model.load_state_dict(payload, strict=False)
        self.device = self.torch.device("cuda" if self.torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def _screen_stage(self, model_input: ModelInput, station_config: StationConfig):
        prepared = preprocess_image(model_input, station_config)
        batch = stack_prepared_images([prepared], as_torch=True)
        batch.image = batch.image.to(self.device)
        batch.valid_mask = batch.valid_mask.to(self.device)
        with self.torch.no_grad():
            screen_logit, coarse_heatmap = self.model.screen_forward(batch.image.float(), batch.valid_mask.float())
        screen_heatmap = self.torch.nn.functional.interpolate(
            coarse_heatmap,
            size=batch.image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        screen_score = float(self.torch.sigmoid(screen_logit)[0].detach().cpu().item())
        return prepared, batch, screen_score, screen_heatmap[0, 0].detach().cpu().numpy()

    def forward(self, model_input: ModelInput, station_config: StationConfig) -> ModelOutput:
        prepared, batch, screen_score, screen_heatmap = self._screen_stage(model_input, station_config)
        if screen_score <= float(self.profile.uncertainty_low):
            return _build_model_output(
                reject_score=screen_score,
                defect_family_probs=np.zeros((len(self.defect_families),), dtype=np.float32),
                heatmap=screen_heatmap * np.asarray(prepared.valid_mask, dtype=np.float32),
                top_tiles=np.zeros((0, 5), dtype=np.float32),
                top_tile_indices=np.zeros((0,), dtype=np.int64),
                top_tile_boxes=np.zeros((0, 4), dtype=np.int64),
                metadata={
                    "backend": "native_fast",
                    "runtime_engine": self.profile.runtime_engine,
                    "mode": "screen_only",
                    "profile_id": self.profile.profile_id,
                    "screen_score": screen_score,
                },
                global_heatmap=screen_heatmap,
            )

        batch.station_id = batch.station_id.to(self.device)
        batch.geometry_id = batch.geometry_id.to(self.device)
        with self.torch.no_grad():
            output = self.model(batch)
        refined_score = float(np.asarray(output.reject_score[0].detach().cpu().numpy()).reshape(()))
        if screen_score < float(self.profile.uncertainty_high):
            refined_score = 0.5 * refined_score + 0.5 * screen_score
        return ModelOutput(
            reject_score=np.float32(refined_score),
            accept_reject_logit=np.float32(_stable_logit(refined_score)),
            defect_family_probs=output.defect_family_probs[0].detach().cpu().numpy(),
            defect_logits=output.defect_logits[0].detach().cpu().numpy(),
            defect_heatmap=output.defect_heatmap[0, 0].detach().cpu().numpy(),
            top_tiles=output.top_tiles[0].detach().cpu().numpy(),
            top_tile_indices=output.top_tile_indices[0].detach().cpu().numpy(),
            top_tile_boxes=output.top_tile_boxes[0].detach().cpu().numpy(),
            global_heatmap=output.global_heatmap[0, 0].detach().cpu().numpy() if output.global_heatmap is not None else screen_heatmap,
            metadata={
                "backend": "native_fast",
                "runtime_engine": self.profile.runtime_engine,
                "mode": "cascade_refine",
                "profile_id": self.profile.profile_id,
                "screen_score": screen_score,
            },
        )


class _BaseHuggingFaceAdapter:
    task_type: str = "classification"
    processor_class_name: str = "AutoImageProcessor"
    model_class_name: str = "AutoModelForImageClassification"

    def __init__(
        self,
        profile: ModelProfile,
        *,
        num_defect_families: int,
        defect_families: Sequence[str] = (),
        num_stations: int = 32,
    ) -> None:
        self.profile = profile
        self.num_defect_families = max(int(num_defect_families), 1)
        defect_family_list = list(defect_families) if defect_families else list(profile.canonical_defect_families())
        self.defect_families = tuple(defect_family_list[: self.num_defect_families])
        self.num_stations = max(int(num_stations), 1)
        self._processor = None
        self._model = None
        self._mode = "heuristic"

    def _load_transformers_backend(self):
        source = self.profile.source
        if not source:
            return None, None
        transformers = _require_transformers()
        processor_cls = getattr(transformers, self.processor_class_name, None)
        model_cls = getattr(transformers, self.model_class_name, None)
        if processor_cls is None or model_cls is None:
            return None, None
        from_pretrained_kwargs = {}
        if self.profile.cache_dir:
            from_pretrained_kwargs["cache_dir"] = self.profile.cache_dir
        if self.profile.revision:
            from_pretrained_kwargs["revision"] = self.profile.revision
        processor = processor_cls.from_pretrained(source, **from_pretrained_kwargs)
        model = model_cls.from_pretrained(source, **from_pretrained_kwargs)
        try:
            model.eval()
        except Exception:
            pass
        return processor, model

    def _ensure_backend(self):
        if self._processor is not None or self._model is not None:
            return
        processor, model = self._load_transformers_backend()
        if processor is None or model is None:
            self._mode = "heuristic"
            self._processor = None
            self._model = None
            return
        self._mode = "transformers"
        self._processor = processor
        self._model = model

    def _processor_call(self, rgb_image: np.ndarray):
        if self._processor is None:
            return None
        try:
            return self._processor(images=rgb_image, return_tensors="pt")
        except TypeError:
            return self._processor(rgb_image, return_tensors="pt")

    def _get_class_labels(self, logits: np.ndarray) -> list[str]:
        model = self._model
        if model is None:
            return [self.profile.good_labels[0], self.profile.bad_labels[0]] + list(self.defect_families)
        config = getattr(model, "config", None)
        id2label = getattr(config, "id2label", None) if config is not None else None
        if isinstance(id2label, Mapping):
            labels = [str(id2label.get(index, "class_%d" % index)) for index in range(int(logits.shape[-1]))]
            return labels
        return ["class_%d" % index for index in range(int(logits.shape[-1]))]

    def _prepare_class_scores(self, labels: Sequence[str], probabilities: np.ndarray) -> tuple[np.ndarray, float]:
        canonical_scores, reject_score = _canonical_family_scores(self.profile, labels, probabilities, self.defect_families)
        defect_probs = np.asarray([canonical_scores.get(family, 0.0) for family in self.defect_families], dtype=np.float32)
        return defect_probs, reject_score

    def _heuristic_heatmap(self, prepared, station_config: StationConfig) -> ModelOutput:
        heatmap = _heatmap_from_image(prepared.image_uint8, prepared.valid_mask)
        top_tiles, top_indices, top_boxes = _tile_scores_from_heatmap(heatmap, station_config)
        defect_probs = np.asarray([float(np.mean(heatmap))] * len(self.defect_families), dtype=np.float32) if self.defect_families else np.zeros((0,), dtype=np.float32)
        return _build_model_output(
            reject_score=float(np.mean(heatmap)),
            defect_family_probs=defect_probs,
            heatmap=heatmap,
            top_tiles=top_tiles,
            top_tile_indices=top_indices,
            top_tile_boxes=top_boxes,
            metadata={
                "backend": "huggingface",
                "runtime_engine": self.profile.runtime_engine,
                "mode": "heuristic",
                "profile_id": self.profile.profile_id,
                "task_type": self.task_type,
                "model_id": self.profile.model_id,
            },
        )

    def forward(self, model_input: ModelInput, station_config: StationConfig) -> ModelOutput:
        prepared, rgb_image = _prepare_inputs(model_input, station_config)
        self._ensure_backend()
        if self._mode != "transformers" or self._processor is None or self._model is None:
            return self._heuristic_heatmap(prepared, station_config)
        return self._forward_transformers(prepared, rgb_image, station_config)

    def _forward_transformers(self, prepared, rgb_image: np.ndarray, station_config: StationConfig) -> ModelOutput:
        raise NotImplementedError


class HuggingFaceClassificationAdapter(_BaseHuggingFaceAdapter):
    task_type = "classification"
    processor_class_name = "AutoImageProcessor"
    model_class_name = "AutoModelForImageClassification"

    def _forward_transformers(self, prepared, rgb_image: np.ndarray, station_config: StationConfig) -> ModelOutput:
        import torch

        encoded = self._processor_call(rgb_image)
        with torch.no_grad():
            outputs = self._model(**encoded) if encoded is not None else self._model(pixel_values=torch.from_numpy(rgb_image.transpose(2, 0, 1)[None].astype(np.float32)))
        logits = outputs.logits.detach().cpu().numpy()
        logits = logits[0] if logits.ndim > 1 else logits
        probabilities = _softmax(np.asarray(logits, dtype=np.float32), axis=-1)
        labels = self._get_class_labels(np.asarray(logits))
        defect_probs, reject_score = self._prepare_class_scores(labels, probabilities)
        heatmap = _heatmap_from_image(prepared.image_uint8, prepared.valid_mask)
        top_tiles, top_indices, top_boxes = _tile_scores_from_heatmap(heatmap, station_config)
        return _build_model_output(
            reject_score=reject_score,
            defect_family_probs=defect_probs,
            heatmap=heatmap,
            top_tiles=top_tiles,
            top_tile_indices=top_indices,
            top_tile_boxes=top_boxes,
            metadata={
                "backend": "huggingface",
                "runtime_engine": self.profile.runtime_engine,
                "mode": "transformers",
                "profile_id": self.profile.profile_id,
                "task_type": self.task_type,
                "model_id": self.profile.model_id,
                "labels": labels,
            },
        )


class HuggingFaceDetectionAdapter(_BaseHuggingFaceAdapter):
    task_type = "detection"
    processor_class_name = "AutoImageProcessor"
    model_class_name = "AutoModelForObjectDetection"

    def _forward_transformers(self, prepared, rgb_image: np.ndarray, station_config: StationConfig) -> ModelOutput:
        import torch

        encoded = self._processor_call(rgb_image)
        with torch.no_grad():
            outputs = self._model(**encoded) if encoded is not None else self._model(pixel_values=torch.from_numpy(rgb_image.transpose(2, 0, 1)[None].astype(np.float32)))
        if hasattr(self._processor, "post_process_object_detection"):
            target_sizes = torch.tensor([prepared.canvas_shape], dtype=torch.long)
            detections = self._processor.post_process_object_detection(outputs, threshold=0.0, target_sizes=target_sizes)[0]
            scores = detections.get("scores", torch.zeros((0,), dtype=torch.float32)).detach().cpu().numpy()
            labels = detections.get("labels", torch.zeros((0,), dtype=torch.long)).detach().cpu().numpy()
            boxes = detections.get("boxes", torch.zeros((0, 4), dtype=torch.float32)).detach().cpu().numpy()
        else:
            logits = outputs.logits.detach().cpu().numpy()
            if logits.ndim == 3:
                probabilities = _softmax(logits, axis=-1)
                scores = probabilities.max(axis=-1).reshape(-1)
                labels = probabilities.argmax(axis=-1).reshape(-1)
            elif logits.ndim == 2:
                probabilities = _softmax(logits, axis=-1)
                scores = probabilities.max(axis=-1).reshape(-1)
                labels = probabilities.argmax(axis=-1).reshape(-1)
            else:
                probabilities = _softmax(logits.reshape(1, -1), axis=-1)
                scores = probabilities.reshape(-1)
                labels = np.zeros_like(scores, dtype=np.int64)
            boxes = outputs.pred_boxes.detach().cpu().numpy().reshape(-1, 4)
        label_names = self._get_class_labels(np.asarray(scores))
        canonical_scores: dict[str, float] = {}
        for score, label_index in zip(np.asarray(scores, dtype=np.float32).reshape(-1), np.asarray(labels).reshape(-1)):
            label = label_names[int(label_index) % len(label_names)] if label_names else "class_%d" % int(label_index)
            canonical = _map_label(self.profile, label)
            canonical_scores[canonical] = max(canonical_scores.get(canonical, 0.0), float(score))
        defect_probs = np.asarray([canonical_scores.get(family, 0.0) for family in self.defect_families], dtype=np.float32)
        reject_score = float(np.max(defect_probs)) if defect_probs.size else max(canonical_scores.values(), default=0.0)
        heatmap = _heatmap_from_boxes(np.asarray(boxes, dtype=np.float32), np.asarray(scores, dtype=np.float32), prepared.canvas_shape, prepared.valid_mask)
        top_tiles, top_indices, top_boxes = _tile_scores_from_heatmap(heatmap, station_config)
        return _build_model_output(
            reject_score=reject_score,
            defect_family_probs=defect_probs,
            heatmap=heatmap,
            top_tiles=top_tiles,
            top_tile_indices=top_indices,
            top_tile_boxes=top_boxes,
            metadata={
                "backend": "huggingface",
                "runtime_engine": self.profile.runtime_engine,
                "mode": "transformers",
                "profile_id": self.profile.profile_id,
                "task_type": self.task_type,
                "model_id": self.profile.model_id,
                "labels": label_names,
            },
        )


class HuggingFaceSegmentationAdapter(_BaseHuggingFaceAdapter):
    task_type = "segmentation"
    processor_class_name = "AutoImageProcessor"
    model_class_name = "AutoModelForSemanticSegmentation"

    def _forward_transformers(self, prepared, rgb_image: np.ndarray, station_config: StationConfig) -> ModelOutput:
        import torch

        encoded = self._processor_call(rgb_image)
        with torch.no_grad():
            outputs = self._model(**encoded) if encoded is not None else self._model(pixel_values=torch.from_numpy(rgb_image.transpose(2, 0, 1)[None].astype(np.float32)))
        if hasattr(self._processor, "post_process_semantic_segmentation"):
            target_sizes = torch.tensor([prepared.canvas_shape], dtype=torch.long)
            mask = self._processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)[0]
            mask_np = mask.detach().cpu().numpy() if hasattr(mask, "detach") else np.asarray(mask)
            if mask_np.ndim == 2:
                heatmap = _normalize_map(mask_np)
            else:
                heatmap = _normalize_map(np.max(mask_np, axis=0))
        else:
            logits = outputs.logits.detach().cpu().numpy()
            probabilities = _softmax(logits[0], axis=0) if logits.ndim == 4 else _softmax(logits, axis=0)
            if probabilities.ndim == 3:
                heatmap = _normalize_map(np.max(probabilities, axis=0))
            elif probabilities.ndim == 2:
                heatmap = _normalize_map(np.max(probabilities, axis=0))
            else:
                heatmap = _heatmap_from_image(prepared.image_uint8, prepared.valid_mask)
        heatmap = np.asarray(heatmap, dtype=np.float32) * np.asarray(prepared.valid_mask, dtype=np.float32)
        top_tiles, top_indices, top_boxes = _tile_scores_from_heatmap(heatmap, station_config)
        defect_probs = np.asarray([float(np.mean(heatmap))] * len(self.defect_families), dtype=np.float32) if self.defect_families else np.zeros((0,), dtype=np.float32)
        return _build_model_output(
            reject_score=float(np.max(heatmap)) if heatmap.size else 0.0,
            defect_family_probs=defect_probs,
            heatmap=heatmap,
            top_tiles=top_tiles,
            top_tile_indices=top_indices,
            top_tile_boxes=top_boxes,
            metadata={
                "backend": "huggingface",
                "runtime_engine": self.profile.runtime_engine,
                "mode": "transformers",
                "profile_id": self.profile.profile_id,
                "task_type": self.task_type,
                "model_id": self.profile.model_id,
            },
        )


def build_huggingface_model_backend(
    profile: ModelProfile | Mapping[str, Any] | str | Path,
    *,
    num_defect_families: int,
    defect_families: Sequence[str] = (),
    num_stations: int = 32,
    registry_root: Path | str | None = None,
):
    loaded_profile = load_model_profile(profile, registry_root=registry_root)
    task_type = loaded_profile.task_type
    if loaded_profile.is_native:
        from .api import BaseModel, LiteModel

        variant = loaded_profile.native_variant
        if variant == "fast":
            return NativeFastProfileRuntime(
                loaded_profile,
                num_defect_families=num_defect_families,
                defect_families=defect_families,
                num_stations=num_stations,
            )
        if variant == "lite":
            return LiteModel(num_defect_families=num_defect_families, defect_families=defect_families, num_stations=num_stations)
        return BaseModel(num_defect_families=num_defect_families, defect_families=defect_families, num_stations=num_stations)
    if task_type == "classification":
        return HuggingFaceClassificationAdapter(
            loaded_profile,
            num_defect_families=num_defect_families,
            defect_families=defect_families,
            num_stations=num_stations,
        )
    if task_type == "detection":
        return HuggingFaceDetectionAdapter(
            loaded_profile,
            num_defect_families=num_defect_families,
            defect_families=defect_families,
            num_stations=num_stations,
        )
    if task_type == "segmentation":
        return HuggingFaceSegmentationAdapter(
            loaded_profile,
            num_defect_families=num_defect_families,
            defect_families=defect_families,
            num_stations=num_stations,
        )
    raise ValueError("Unsupported model profile task_type %r for profile %s." % (loaded_profile.task_type, loaded_profile.profile_id))
