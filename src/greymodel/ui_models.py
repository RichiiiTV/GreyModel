from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import statistics
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .api import BaseModel, LiteModel
from .data import DatasetRecord, load_dataset_index, load_station_configs_from_index, station_config_for_record
from .evaluation import evaluate_predictions, save_predictions
from .hf_backends import build_huggingface_model_backend
from .model_profiles import ModelProfile as RegistryModelProfile
from .preprocessing import preprocess_image
from .types import ModelInput, ModelOutput, PredictionEvidence, PredictionRecord, StationConfig
from .ui_workspace import ModelProfile
from .utils import load_uint8_grayscale, read_json, utc_timestamp, write_json


def _has_transformers() -> bool:
    return importlib.util.find_spec("transformers") is not None


def _variant_for_profile(profile: ModelProfile) -> str:
    variant = str(profile.native_variant).lower()
    if variant == "fast":
        return "fast"
    if variant == "lite":
        return "lite"
    return "base"


def _as_rgb(image_uint8: np.ndarray) -> np.ndarray:
    if image_uint8.ndim != 2:
        raise ValueError("Expected a grayscale image.")
    return np.repeat(image_uint8[:, :, None], 3, axis=2)


def _score_from_label_mapping(raw_labels: Mapping[str, float], mapping: Mapping[str, str]) -> tuple[float, dict[str, float]]:
    defect_scores: dict[str, float] = {}
    reject_score = 0.0
    for label, score in raw_labels.items():
        mapped = mapping.get(label, "")
        mapped_normalized = str(mapped).strip().lower()
        if not mapped_normalized:
            mapped_normalized = "bad" if any(token in label.lower() for token in ("defect", "anomaly", "bad", "error")) else "good"
        if mapped_normalized == "good":
            reject_score = max(reject_score, max(0.0, 1.0 - float(score)))
            continue
        if mapped_normalized == "bad":
            reject_score = max(reject_score, float(score))
            continue
        defect_scores[mapped_normalized] = max(defect_scores.get(mapped_normalized, 0.0), float(score))
        reject_score = max(reject_score, float(score))
    return float(reject_score), defect_scores


@dataclass
class BenchmarkResult:
    profile_id: str
    backend_family: str
    runtime_engine: str
    accelerator: str
    num_samples: int
    p50_ms: float
    p95_ms: float
    mean_ms: float
    throughput_samples_per_s: float
    memory_peak_mb: float | None
    latency_target_ms: float
    meets_target: bool
    report_path: str | None = None


class NativeProfileRuntime:
    def __init__(self, profile: ModelProfile, defect_families: Sequence[str]) -> None:
        num_defect_families = max(len(defect_families), 1)
        variant = _variant_for_profile(profile)
        if variant == "fast":
            registry_profile = RegistryModelProfile(
                profile_id=profile.profile_id,
                backend_family="native",
                task_type="native",
                local_path=profile.local_path,
                runtime_engine=profile.runtime_engine,
                metadata={"variant": "fast", "latency_target_ms": float(profile.latency_target_ms)},
            )
            self.model = build_huggingface_model_backend(
                registry_profile,
                num_defect_families=num_defect_families,
                defect_families=tuple(defect_families),
                num_stations=32,
            )
        elif variant == "lite":
            self.model = LiteModel(num_defect_families=num_defect_families, defect_families=tuple(defect_families))
        else:
            self.model = BaseModel(num_defect_families=num_defect_families, defect_families=tuple(defect_families))
        self.profile = profile
        self.defect_families = tuple(defect_families)

    def predict(self, image_uint8: np.ndarray, station_config: StationConfig) -> ModelOutput:
        model_input = ModelInput(
            image_uint8=image_uint8,
            station_id=station_config.station_id,
            geometry_mode=station_config.geometry_mode or "rect",
            metadata=station_config.metadata,
        )
        return self.model.forward(model_input, station_config)


class HuggingFaceProfileRuntime:
    def __init__(self, profile: ModelProfile, *, cache_root: Path | str | None = None, local_files_only: bool = False) -> None:
        if not _has_transformers():
            raise ImportError("Hugging Face model support requires the `transformers` package.")
        from transformers import AutoImageProcessor

        self.profile = profile
        self.cache_root = Path(cache_root) if cache_root is not None else None
        self.local_files_only = bool(local_files_only or profile.cache_policy == "local_only")
        self.processor = AutoImageProcessor.from_pretrained(
            profile.model_id,
            cache_dir=str(self.cache_root) if self.cache_root is not None else None,
            revision=profile.model_revision or None,
            local_files_only=self.local_files_only,
        )
        if profile.backend_family == "hf_detection":
            from transformers import AutoModelForObjectDetection

            self.model = AutoModelForObjectDetection.from_pretrained(
                profile.model_id,
                cache_dir=str(self.cache_root) if self.cache_root is not None else None,
                revision=profile.model_revision or None,
                local_files_only=self.local_files_only,
            )
        elif profile.backend_family == "hf_segmentation":
            try:
                from transformers import AutoModelForSemanticSegmentation

                self.model = AutoModelForSemanticSegmentation.from_pretrained(
                    profile.model_id,
                    cache_dir=str(self.cache_root) if self.cache_root is not None else None,
                    revision=profile.model_revision or None,
                    local_files_only=self.local_files_only,
                )
            except Exception:
                from transformers import AutoModelForUniversalSegmentation

                self.model = AutoModelForUniversalSegmentation.from_pretrained(
                    profile.model_id,
                    cache_dir=str(self.cache_root) if self.cache_root is not None else None,
                    revision=profile.model_revision or None,
                    local_files_only=self.local_files_only,
                )
        else:
            from transformers import AutoModelForImageClassification

            self.model = AutoModelForImageClassification.from_pretrained(
                profile.model_id,
                cache_dir=str(self.cache_root) if self.cache_root is not None else None,
                revision=profile.model_revision or None,
                local_files_only=self.local_files_only,
            )
        self.model.eval()

    def _device(self):
        try:
            import torch

            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except Exception:
            return "cpu"

    def predict(self, image_uint8: np.ndarray, station_config: StationConfig) -> tuple[dict[str, float], dict[str, Any]]:
        import torch

        image_rgb = _as_rgb(image_uint8)
        inputs = self.processor(images=image_rgb, return_tensors="pt")
        device = self._device()
        self.model.to(device)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)

        if self.profile.backend_family == "hf_detection":
            processed = self.processor.post_process_object_detection(outputs, target_sizes=[image_rgb.shape[:2]], threshold=0.0)[0]
            scores: dict[str, float] = {}
            boxes = []
            for label_id, score, box in zip(processed.get("labels", []), processed.get("scores", []), processed.get("boxes", [])):
                label = str(self.model.config.id2label.get(int(label_id), label_id))
                mapped = self.profile.label_mapping.get(label, label)
                scores[str(mapped)] = max(scores.get(str(mapped), 0.0), float(score))
                boxes.append(
                    {
                        "label": label,
                        "mapped_label": mapped,
                        "score": float(score),
                        "box": [float(v) for v in box.tolist()],
                    }
                )
            return scores, {"boxes": boxes, "output_kind": "detection"}

        if self.profile.backend_family == "hf_segmentation":
            if hasattr(self.processor, "post_process_semantic_segmentation"):
                processed = self.processor.post_process_semantic_segmentation(outputs, target_sizes=[image_rgb.shape[:2]])[0]
            else:
                processed = outputs.logits[0].argmax(dim=0).detach().cpu().numpy()
            scores = {}
            regions = []
            if isinstance(processed, np.ndarray):
                unique, counts = np.unique(processed, return_counts=True)
                for label_id, count in zip(unique.tolist(), counts.tolist()):
                    label = str(self.model.config.id2label.get(int(label_id), label_id))
                    mapped = self.profile.label_mapping.get(label, label)
                    scores[str(mapped)] = max(scores.get(str(mapped), 0.0), float(count) / float(processed.size))
                    regions.append({"label": label, "mapped_label": mapped, "fraction": float(count) / float(processed.size)})
            else:
                scores = {}
            return scores, {"regions": regions, "output_kind": "segmentation"}

        logits = outputs.logits[0].detach().softmax(dim=-1).cpu().numpy()
        labels = [str(self.model.config.id2label.get(index, index)) for index in range(len(logits))]
        scores = {label: float(score) for label, score in zip(labels, logits.tolist())}
        return scores, {"labels": labels, "output_kind": "classification"}


@lru_cache(maxsize=16)
def _native_runtime_cache(profile_cache_key: str, native_variant: str, defect_families: tuple[str, ...]) -> NativeProfileRuntime:
    profile = ModelProfile(
        profile_id=profile_cache_key.split("|", 1)[0],
        backend_family="native",
        native_variant=native_variant,
    )
    return NativeProfileRuntime(profile, defect_families)


def build_runtime_for_profile(
    profile: ModelProfile,
    *,
    defect_families: Sequence[str] = (),
    cache_root: Path | str | None = None,
    local_files_only: bool = False,
):
    if profile.is_native:
        return _native_runtime_cache(profile.cache_key, profile.native_variant, tuple(defect_families))
    return HuggingFaceProfileRuntime(profile, cache_root=cache_root, local_files_only=local_files_only)


def predict_record_with_profile(
    record: DatasetRecord,
    profile: ModelProfile,
    station_config: StationConfig,
    *,
    cache_root: Path | str | None = None,
    local_files_only: bool = False,
    defect_families: Sequence[str] = (),
) -> PredictionRecord:
    image = load_uint8_grayscale(Path(record.image_path))
    runtime = build_runtime_for_profile(
        profile,
        defect_families=defect_families,
        cache_root=cache_root,
        local_files_only=local_files_only,
    )
    if profile.is_native:
        output = runtime.predict(image, station_config)
        raw_scores = {
            family: float(score)
            for family, score in zip(defect_families, np.asarray(output.defect_family_probs).reshape(-1).tolist())
        }
        reject_score = float(np.asarray(output.reject_score).reshape(()))
        defect_scores = raw_scores
        evidence_metadata = {
            "backend_family": profile.backend_family,
            "runtime_engine": profile.runtime_engine,
            "native_variant": profile.native_variant,
            **dict(getattr(output, "metadata", {}) or {}),
            "top_tile_indices": np.asarray(output.top_tile_indices).reshape(-1).tolist()
            if output.top_tile_indices is not None
            else [],
            "top_tile_boxes": np.asarray(output.top_tile_boxes).tolist() if output.top_tile_boxes is not None else [],
        }
    else:
        defect_scores, evidence_metadata = runtime.predict(image, station_config)
        reject_score = max(defect_scores.values(), default=0.0)
        if "good" in defect_scores:
            reject_score = max(reject_score, float(1.0 - defect_scores.get("good", 0.0)))

    primary_label = "bad" if reject_score >= float(station_config.reject_threshold) else "good"
    predicted_label = int(primary_label == "bad")
    evidence = PredictionEvidence(
        station_decision={
            "station_id": str(record.station_id),
            "threshold": float(station_config.reject_threshold),
            "reject": bool(predicted_label),
            "backend_family": profile.backend_family,
            "runtime_engine": profile.runtime_engine,
        },
        metadata={"evidence": evidence_metadata},
    )
    return PredictionRecord(
        sample_id=record.sample_id,
        station_id=record.station_id,
        accept_reject=record.accept_reject,
        reject_score=reject_score,
        predicted_label=predicted_label,
        primary_label=primary_label,
        primary_score=reject_score,
        top_defect_family=max(defect_scores.items(), key=lambda item: item[1])[0] if defect_scores else None,
        defect_family_probs=defect_scores,
        evidence=evidence,
        split=record.split,
        defect_scale="unknown",
        metadata={
            "profile_id": profile.profile_id,
            "profile_backend_family": profile.backend_family,
            "profile_task_type": profile.task_type,
            "profile_model_id": profile.model_id,
            **dict(record.capture_metadata or {}),
        },
    )


def benchmark_profile_runtime(
    profile: ModelProfile,
    records: Sequence[DatasetRecord],
    station_config: StationConfig,
    *,
    cache_root: Path | str | None = None,
    local_files_only: bool = False,
    defect_families: Sequence[str] = (),
    max_samples: int = 8,
) -> BenchmarkResult:
    if not records:
        raise ValueError("benchmark_profile_runtime requires at least one record.")
    selected = list(records[: max(1, int(max_samples))])
    runtime = build_runtime_for_profile(
        profile,
        defect_families=defect_families,
        cache_root=cache_root,
        local_files_only=local_files_only,
    )
    latencies: list[float] = []
    memory_peak_mb: float | None = None
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for record in selected:
        image = load_uint8_grayscale(Path(record.image_path))
        start = time.perf_counter()
        if profile.is_native:
            runtime.predict(image, station_config)
        else:
            runtime.predict(image, station_config)
        latencies.append((time.perf_counter() - start) * 1000.0)
    if torch is not None and torch.cuda.is_available():
        memory_peak_mb = float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
    p50 = float(statistics.median(latencies))
    p95 = float(np.percentile(np.asarray(latencies, dtype=np.float32), 95))
    mean = float(np.mean(latencies))
    throughput = float(len(latencies) / max(sum(latencies) / 1000.0, 1e-6))
    return BenchmarkResult(
        profile_id=profile.profile_id,
        backend_family=profile.backend_family,
        runtime_engine=profile.runtime_engine,
        accelerator="cuda" if torch is not None and torch.cuda.is_available() else "cpu",
        num_samples=len(latencies),
        p50_ms=p50,
        p95_ms=p95,
        mean_ms=mean,
        throughput_samples_per_s=throughput,
        memory_peak_mb=memory_peak_mb,
        latency_target_ms=float(profile.latency_target_ms),
        meets_target=p95 <= float(profile.latency_target_ms),
    )


def save_benchmark_result(output_dir: Path | str, result: BenchmarkResult) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "profile_id": result.profile_id,
        "backend_family": result.backend_family,
        "runtime_engine": result.runtime_engine,
        "accelerator": result.accelerator,
        "num_samples": result.num_samples,
        "p50_ms": result.p50_ms,
        "p95_ms": result.p95_ms,
        "mean_ms": result.mean_ms,
        "throughput_samples_per_s": result.throughput_samples_per_s,
        "memory_peak_mb": result.memory_peak_mb,
        "latency_target_ms": result.latency_target_ms,
        "meets_target": result.meets_target,
        "written_at": utc_timestamp(),
    }
    return write_json(output_dir / "latency_report.json", payload)
