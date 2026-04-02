from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import time
from typing import Mapping, Optional, Sequence

import numpy as np

from .api import BaseModel, LiteModel
from .preprocessing import preprocess_image, stack_prepared_images
from .profiles import resolve_model_profile
from .types import LatencyReport, ModelInput, ModelOutput, ModelProfile, StationConfig
from .utils import ensure_dir, stable_int_hash, write_json


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for model backend execution.") from exc
    return torch


def _require_transformers():
    try:
        from transformers import (
            AutoImageProcessor,
            AutoModelForImageClassification,
            AutoModelForObjectDetection,
            AutoModelForSemanticSegmentation,
        )
    except ImportError as exc:
        raise ImportError("Transformers is required for Hugging Face model backends.") from exc
    return AutoImageProcessor, AutoModelForImageClassification, AutoModelForObjectDetection, AutoModelForSemanticSegmentation


def _sigmoid(values):
    return 1.0 / (1.0 + np.exp(-values))


def _normalize_gray_to_rgb(image_uint8: np.ndarray) -> np.ndarray:
    return np.repeat(image_uint8[:, :, None], 3, axis=2)


def _profile_threshold(profile: ModelProfile, key: str, default: float) -> float:
    try:
        return float(profile.thresholds.get(key, default))
    except Exception:
        return float(default)


def _load_state_dict_if_available(torch_model, checkpoint_path: Optional[str]) -> None:
    if checkpoint_path is None:
        return
    path = Path(checkpoint_path)
    if not path.exists():
        return
    torch = _require_torch()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, Mapping) and "backbone" in payload:
        payload = payload["backbone"]
    if isinstance(payload, Mapping) and "state_dict" in payload:
        payload = payload["state_dict"]
    if isinstance(payload, Mapping):
        torch_model.load_state_dict(payload, strict=False)


class NativeReviewBackend:
    def __init__(self, profile: ModelProfile, defect_families: Sequence[str], num_stations: int = 32) -> None:
        variant = str(profile.variant or "base").lower()
        if variant == "lite":
            self.model = LiteModel(num_defect_families=max(len(defect_families), 1), defect_families=defect_families, num_stations=num_stations)
        else:
            self.model = BaseModel(num_defect_families=max(len(defect_families), 1), defect_families=defect_families, num_stations=num_stations)
        backend_model = getattr(self.model.backend, "model", None)
        if backend_model is not None:
            _load_state_dict_if_available(backend_model, profile.checkpoint_path)
            backend_model.eval()

    def forward(self, model_input: ModelInput, station_config: StationConfig) -> ModelOutput:
        return self.model.forward(model_input, station_config)


class NativeFastBackend:
    def __init__(self, profile: ModelProfile, defect_families: Sequence[str]) -> None:
        from .models import build_fast_model

        self.profile = profile
        self.defect_families = tuple(defect_families) if defect_families else ("unknown",)
        self.torch = _require_torch()
        self.device = self.torch.device("cuda" if self.torch.cuda.is_available() else "cpu")
        self.model = build_fast_model(
            num_defect_families=max(len(self.defect_families), 1),
            defect_families=self.defect_families,
        ).to(self.device)
        _load_state_dict_if_available(self.model, profile.checkpoint_path)
        self.model.eval()

    def _screen_only(self, model_input: ModelInput, station_config: StationConfig):
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
        return prepared, batch, float(self.torch.sigmoid(screen_logit)[0].detach().cpu().item()), screen_heatmap[0, 0].detach().cpu().numpy()

    def forward(self, model_input: ModelInput, station_config: StationConfig) -> ModelOutput:
        prepared, batch, screen_score, screen_heatmap = self._screen_only(model_input, station_config)
        reject_threshold = float(station_config.reject_threshold)
        good_max = _profile_threshold(self.profile, "good_max", 0.35)
        bad_min = _profile_threshold(self.profile, "bad_min", 0.65)
        screen_logit = float(np.log(screen_score / max(1.0 - screen_score, 1e-6)))
        if screen_score <= good_max:
            return ModelOutput(
                reject_score=np.float32(screen_score),
                accept_reject_logit=np.float32(screen_logit),
                defect_family_probs=np.zeros((len(self.defect_families),), dtype=np.float32),
                defect_logits=np.zeros((len(self.defect_families),), dtype=np.float32),
                defect_heatmap=screen_heatmap.astype(np.float32) * prepared.valid_mask.astype(np.float32),
                top_tiles=np.zeros((0, 5), dtype=np.float32),
                top_tile_indices=np.zeros((0,), dtype=np.int64),
                top_tile_boxes=np.zeros((0, 4), dtype=np.int64),
                global_heatmap=screen_heatmap.astype(np.float32),
                metadata={"backend": "native_fast", "cascade_stage": "screen_only", "screen_score": float(screen_score), "reject_threshold": reject_threshold},
            )

        batch.station_id = batch.station_id.to(self.device)
        batch.geometry_id = batch.geometry_id.to(self.device)
        with self.torch.no_grad():
            output = self.model(batch)
        reject_score = float(output.reject_score[0].detach().cpu().item())
        reject_score = 0.5 * reject_score + 0.5 * screen_score if screen_score < bad_min else reject_score
        reject_logit = float(np.log(reject_score / max(1.0 - reject_score, 1e-6)))
        return ModelOutput(
            reject_score=np.float32(reject_score),
            accept_reject_logit=np.float32(reject_logit),
            defect_family_probs=output.defect_family_probs[0].detach().cpu().numpy(),
            defect_logits=output.defect_logits[0].detach().cpu().numpy(),
            defect_heatmap=output.defect_heatmap[0, 0].detach().cpu().numpy(),
            top_tiles=output.top_tiles[0].detach().cpu().numpy(),
            top_tile_indices=output.top_tile_indices[0].detach().cpu().numpy(),
            top_tile_boxes=output.top_tile_boxes[0].detach().cpu().numpy(),
            global_heatmap=output.global_heatmap[0, 0].detach().cpu().numpy() if output.global_heatmap is not None else screen_heatmap.astype(np.float32),
            metadata={
                "backend": "native_fast",
                "cascade_stage": "refined",
                "screen_score": float(screen_score),
                "reject_threshold": reject_threshold,
            },
        )

    def export_onnx(self, export_dir: Path | str, image_shape: tuple[int, int]) -> Path:
        export_dir = ensure_dir(Path(export_dir))
        output_path = export_dir / "prod_fast_native_screen.onnx"
        torch = self.torch
        dummy_image = torch.randn(1, 1, image_shape[0], image_shape[1], device=self.device)
        dummy_mask = torch.ones_like(dummy_image)

        class _ScreenWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, image, mask):
                logit, heatmap = self.model.screen_forward(image, mask)
                return logit, heatmap

        wrapper = _ScreenWrapper(self.model).to(self.device).eval()
        torch.onnx.export(
            wrapper,
            (dummy_image, dummy_mask),
            output_path,
            input_names=["image", "valid_mask"],
            output_names=["screen_logit", "screen_heatmap"],
            dynamic_axes={"image": {2: "height", 3: "width"}, "valid_mask": {2: "height", 3: "width"}, "screen_heatmap": {2: "heat_h", 3: "heat_w"}},
            opset_version=17,
        )
        return output_path


class HuggingFaceBackend:
    def __init__(self, profile: ModelProfile, defect_families: Sequence[str]) -> None:
        self.profile = profile
        self.defect_families = tuple(defect_families)
        self.torch = _require_torch()
        auto_processor, model_cls, detection_cls, segmentation_cls = _require_transformers()
        model_id = profile.hf_local_path or profile.hf_model_id
        if not model_id:
            raise ValueError("Hugging Face profiles require hf_model_id or hf_local_path.")
        cache_dir = profile.cache_dir
        self.processor = auto_processor.from_pretrained(profile.hf_processor_id or model_id, cache_dir=cache_dir)
        backend_family = str(profile.backend_family)
        if backend_family == "hf_classification":
            self.model = model_cls.from_pretrained(model_id, cache_dir=cache_dir)
        elif backend_family == "hf_detection":
            self.model = detection_cls.from_pretrained(model_id, cache_dir=cache_dir)
        elif backend_family == "hf_segmentation":
            self.model = segmentation_cls.from_pretrained(model_id, cache_dir=cache_dir)
        else:
            raise ValueError("Unsupported Hugging Face backend family %r." % backend_family)
        self.model.eval()
        self.id2label = getattr(self.model.config, "id2label", {}) or {}

    def _mapping_payload(self):
        mapping = dict(self.profile.label_mapping or {})
        return {
            "good_labels": {str(value) for value in mapping.get("good_labels", ())},
            "bad_labels": {str(value) for value in mapping.get("bad_labels", ())},
            "defect_family_map": {str(key): str(value) for key, value in mapping.get("defect_family_map", {}).items()},
        }

    def _classification_output(self, model_input: ModelInput, station_config: StationConfig) -> ModelOutput:
        mapping = self._mapping_payload()
        image_rgb = _normalize_gray_to_rgb(model_input.image_uint8)
        inputs = self.processor(images=image_rgb, return_tensors="pt")
        with self.torch.no_grad():
            logits = self.model(**inputs).logits[0]
            probs = self.torch.softmax(logits, dim=-1).detach().cpu().numpy()
        labels = {str(index): str(self.id2label.get(index, index)) for index in range(len(probs))}
        good_score = 0.0
        bad_score = 0.0
        defect_probs = {name: 0.0 for name in self.defect_families}
        for index, prob in enumerate(probs.tolist()):
            label = labels[str(index)]
            if label in mapping["good_labels"]:
                good_score += float(prob)
            if label in mapping["bad_labels"]:
                bad_score += float(prob)
            family = mapping["defect_family_map"].get(label)
            if family in defect_probs:
                defect_probs[family] = max(defect_probs[family], float(prob))
        if bad_score == 0.0:
            bad_score = max(0.0, 1.0 - good_score)
        heatmap = np.zeros(model_input.image_uint8.shape, dtype=np.float32)
        return ModelOutput(
            reject_score=np.float32(bad_score),
            accept_reject_logit=np.float32(np.log(max(bad_score, 1e-6) / max(1.0 - bad_score, 1e-6))),
            defect_family_probs=np.asarray([defect_probs.get(name, 0.0) for name in self.defect_families], dtype=np.float32),
            defect_logits=np.asarray([defect_probs.get(name, 0.0) for name in self.defect_families], dtype=np.float32),
            defect_heatmap=heatmap,
            top_tiles=np.zeros((0, 5), dtype=np.float32),
            top_tile_indices=np.zeros((0,), dtype=np.int64),
            top_tile_boxes=np.zeros((0, 4), dtype=np.int64),
            metadata={
                "backend": "huggingface",
                "backend_family": self.profile.backend_family,
                "profile_id": self.profile.profile_id,
                "hf_labels": labels,
            },
        )

    def _detection_output(self, model_input: ModelInput, station_config: StationConfig) -> ModelOutput:
        mapping = self._mapping_payload()
        image_rgb = _normalize_gray_to_rgb(model_input.image_uint8)
        inputs = self.processor(images=image_rgb, return_tensors="pt")
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = self.torch.tensor([[image_rgb.shape[0], image_rgb.shape[1]]])
        if hasattr(self.processor, "post_process_object_detection"):
            processed = self.processor.post_process_object_detection(outputs, threshold=0.05, target_sizes=target_sizes)[0]
            boxes = processed["boxes"].detach().cpu().numpy()
            scores = processed["scores"].detach().cpu().numpy()
            labels = processed["labels"].detach().cpu().numpy()
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
        defect_probs = {name: 0.0 for name in self.defect_families}
        heatmap = np.zeros(model_input.image_uint8.shape, dtype=np.float32)
        top_tiles = []
        reject_score = 0.0
        for box, score, label_idx in zip(boxes, scores, labels):
            label = str(self.id2label.get(int(label_idx), int(label_idx)))
            family = mapping["defect_family_map"].get(label, label if label in defect_probs else None)
            if family in defect_probs:
                defect_probs[family] = max(defect_probs[family], float(score))
            reject_score = max(reject_score, float(score))
            x1, y1, x2, y2 = [int(max(value, 0)) for value in box.tolist()]
            heatmap[y1:y2, x1:x2] = np.maximum(heatmap[y1:y2, x1:x2], float(score))
            top_tiles.append([float(y1), float(x1), float(y2), float(x2), float(score)])
        top_tiles_array = np.asarray(top_tiles, dtype=np.float32) if top_tiles else np.zeros((0, 5), dtype=np.float32)
        return ModelOutput(
            reject_score=np.float32(reject_score),
            accept_reject_logit=np.float32(np.log(max(reject_score, 1e-6) / max(1.0 - reject_score, 1e-6))),
            defect_family_probs=np.asarray([defect_probs.get(name, 0.0) for name in self.defect_families], dtype=np.float32),
            defect_logits=np.asarray([defect_probs.get(name, 0.0) for name in self.defect_families], dtype=np.float32),
            defect_heatmap=heatmap,
            top_tiles=top_tiles_array,
            top_tile_indices=np.arange(len(top_tiles_array), dtype=np.int64),
            top_tile_boxes=top_tiles_array[:, :4].astype(np.int64) if len(top_tiles_array) else np.zeros((0, 4), dtype=np.int64),
            metadata={
                "backend": "huggingface",
                "backend_family": self.profile.backend_family,
                "profile_id": self.profile.profile_id,
                "num_detections": int(len(top_tiles_array)),
            },
        )

    def _segmentation_output(self, model_input: ModelInput, station_config: StationConfig) -> ModelOutput:
        mapping = self._mapping_payload()
        image_rgb = _normalize_gray_to_rgb(model_input.image_uint8)
        inputs = self.processor(images=image_rgb, return_tensors="pt")
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        if hasattr(self.processor, "post_process_semantic_segmentation"):
            masks = self.processor.post_process_semantic_segmentation(outputs, target_sizes=[model_input.image_uint8.shape])[0]
            if isinstance(masks, self.torch.Tensor):
                mask = masks.detach().cpu().numpy()
            else:
                mask = np.asarray(masks)
        else:
            resized = self.torch.nn.functional.interpolate(
                logits,
                size=model_input.image_uint8.shape,
                mode="bilinear",
                align_corners=False,
            )
            mask = resized.argmax(dim=1)[0].detach().cpu().numpy()
        defect_probs = {name: 0.0 for name in self.defect_families}
        heatmap = np.zeros(model_input.image_uint8.shape, dtype=np.float32)
        for label_idx in np.unique(mask):
            label = str(self.id2label.get(int(label_idx), int(label_idx)))
            family = mapping["defect_family_map"].get(label, label if label in defect_probs else None)
            region = mask == int(label_idx)
            if family in defect_probs:
                defect_probs[family] = max(defect_probs[family], float(region.mean()))
            if label not in mapping["good_labels"]:
                heatmap[region] = 1.0
        reject_score = float(heatmap.max(initial=0.0))
        return ModelOutput(
            reject_score=np.float32(reject_score),
            accept_reject_logit=np.float32(np.log(max(reject_score, 1e-6) / max(1.0 - reject_score, 1e-6))),
            defect_family_probs=np.asarray([defect_probs.get(name, 0.0) for name in self.defect_families], dtype=np.float32),
            defect_logits=np.asarray([defect_probs.get(name, 0.0) for name in self.defect_families], dtype=np.float32),
            defect_heatmap=heatmap.astype(np.float32),
            top_tiles=np.zeros((0, 5), dtype=np.float32),
            top_tile_indices=np.zeros((0,), dtype=np.int64),
            top_tile_boxes=np.zeros((0, 4), dtype=np.int64),
            metadata={
                "backend": "huggingface",
                "backend_family": self.profile.backend_family,
                "profile_id": self.profile.profile_id,
                "labels_present": [int(value) for value in np.unique(mask).tolist()],
            },
        )

    def forward(self, model_input: ModelInput, station_config: StationConfig) -> ModelOutput:
        if self.profile.backend_family == "hf_classification":
            return self._classification_output(model_input, station_config)
        if self.profile.backend_family == "hf_detection":
            return self._detection_output(model_input, station_config)
        return self._segmentation_output(model_input, station_config)


def create_inference_backend(
    *,
    profile: ModelProfile | str | Path | None,
    defect_families: Sequence[str] = (),
    profiles_dir: Path | str | None = None,
    num_stations: int = 32,
):
    if profile is None:
        return None
    resolved = profile if isinstance(profile, ModelProfile) else resolve_model_profile(profile, profiles_dir=profiles_dir)
    backend_family = str(resolved.backend_family)
    if backend_family == "native_fast":
        return NativeFastBackend(resolved, defect_families)
    if backend_family == "native_review":
        return NativeReviewBackend(resolved, defect_families, num_stations=num_stations)
    if backend_family in {"hf_classification", "hf_detection", "hf_segmentation"}:
        return HuggingFaceBackend(resolved, defect_families)
    raise ValueError("Unsupported backend family %r." % backend_family)


def benchmark_backend_latency(
    backend,
    model_input: ModelInput,
    station_config: StationConfig,
    *,
    profile: ModelProfile,
    iterations: int = 30,
    warmup_iterations: int = 5,
) -> LatencyReport:
    torch = None
    if hasattr(backend, "torch"):
        torch = backend.torch
    for _ in range(max(int(warmup_iterations), 0)):
        backend.forward(model_input, station_config)
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
    durations_ms = []
    peak_memory_mb = 0.0
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for _ in range(max(int(iterations), 1)):
        started = time.perf_counter()
        backend.forward(model_input, station_config)
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_memory_mb = max(peak_memory_mb, float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)))
        durations_ms.append((time.perf_counter() - started) * 1000.0)
    durations_np = np.asarray(durations_ms, dtype=np.float64)
    target_ms = profile.thresholds.get("target_ms")
    return LatencyReport(
        profile_id=profile.profile_id,
        backend_family=profile.backend_family,
        runtime_engine=profile.runtime_engine,
        accelerator="cuda" if torch is not None and torch.cuda.is_available() else "cpu",
        batch_size=1,
        image_shape=tuple(int(value) for value in model_input.image_uint8.shape),
        iterations=int(iterations),
        warmup_iterations=int(warmup_iterations),
        mean_ms=float(durations_np.mean()),
        p50_ms=float(np.percentile(durations_np, 50)),
        p95_ms=float(np.percentile(durations_np, 95)),
        throughput_per_second=float(1000.0 / max(durations_np.mean(), 1e-6)),
        peak_memory_mb=float(peak_memory_mb),
        target_ms=float(target_ms) if target_ms is not None else None,
        meets_target=(float(np.percentile(durations_np, 95)) <= float(target_ms)) if target_ms is not None else None,
        metadata={"station_id": str(model_input.station_id), "runtime_hash": stable_int_hash(profile.profile_id + profile.runtime_engine)},
    )


def save_latency_report(report: LatencyReport, output_path: Path | str) -> Path:
    return write_json(Path(output_path), asdict(report))
