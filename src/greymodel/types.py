from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence, Tuple, Union
import zlib

import numpy as np


class GeometryMode(str, Enum):
    RECT = "rect"
    SQUARE = "square"

    @classmethod
    def from_value(cls, value: str) -> "GeometryMode":
        normalized = value.lower().strip()
        if normalized == cls.RECT.value:
            return cls.RECT
        if normalized == cls.SQUARE.value:
            return cls.SQUARE
        raise ValueError("Unsupported geometry mode: %s" % value)

    def to_id(self) -> int:
        return 0 if self is GeometryMode.RECT else 1


SizeLike = Union[int, Tuple[int, int]]


@dataclass
class StationConfig:
    canvas_shape: Tuple[int, int]
    station_id: Any = 0
    geometry_mode: Optional[GeometryMode] = None
    pad_value: int = 0
    normalization_mean: float = 127.5
    normalization_std: float = 50.0
    tile_size: SizeLike = (64, 64)
    tile_stride: SizeLike = (32, 32)
    adapter_id: Any = 0
    reject_threshold: float = 0.5
    defect_thresholds: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.canvas_shape = tuple(int(v) for v in self.canvas_shape)
        if isinstance(self.geometry_mode, str):
            self.geometry_mode = GeometryMode.from_value(self.geometry_mode)

    @property
    def canvas_size(self) -> Tuple[int, int]:
        return self.canvas_shape

    @property
    def tile_size_2d(self) -> Tuple[int, int]:
        if isinstance(self.tile_size, int):
            return (self.tile_size, self.tile_size)
        return tuple(int(v) for v in self.tile_size)

    @property
    def tile_stride_2d(self) -> Tuple[int, int]:
        if isinstance(self.tile_stride, int):
            return (self.tile_stride, self.tile_stride)
        return tuple(int(v) for v in self.tile_stride)


@dataclass
class Sample:
    image_uint8: np.ndarray
    station_id: Any
    product_family: str
    geometry_mode: GeometryMode
    accept_reject: int
    defect_tags: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        image = self.image_uint8
        if not isinstance(image, np.ndarray):
            raise TypeError("Sample.image_uint8 must be a numpy.ndarray.")
        if image.dtype != np.uint8:
            raise TypeError("Sample.image_uint8 must use uint8 grayscale pixels.")
        if image.ndim != 2:
            raise ValueError("Sample.image_uint8 must be a 2D grayscale array.")
        if self.accept_reject not in (0, 1):
            raise ValueError("Sample.accept_reject must be 0 or 1.")
        if isinstance(self.geometry_mode, str):
            self.geometry_mode = GeometryMode.from_value(self.geometry_mode)


@dataclass
class ModelInput:
    image_uint8: np.ndarray
    station_id: Any
    geometry_mode: Any
    metadata: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.image_uint8, np.ndarray):
            raise TypeError("ModelInput.image_uint8 must be a numpy.ndarray.")
        if self.image_uint8.dtype != np.uint8:
            raise TypeError("ModelInput.image_uint8 must use uint8 grayscale pixels.")
        if self.image_uint8.ndim != 2:
            raise ValueError("ModelInput.image_uint8 must be a 2D grayscale array.")
        if isinstance(self.geometry_mode, str):
            self.geometry_mode = GeometryMode.from_value(self.geometry_mode)


@dataclass
class TensorBatch:
    image: Any
    valid_mask: Any
    station_id: Any
    geometry_id: Any
    metadata: Optional[Mapping[str, Any]] = None


@dataclass
class TopTilePrediction:
    box: Any
    score: Any
    index: Any


@dataclass(frozen=True)
class BoxAnnotation:
    xyxy: Tuple[int, int, int, int]
    defect_tag: str = "unknown"
    confidence: float = 1.0
    annotator: str = "unknown"
    is_hard_case: bool = False

    def __post_init__(self) -> None:
        if len(self.xyxy) != 4:
            raise ValueError("BoxAnnotation.xyxy must contain four integers.")
        x1, y1, x2, y2 = [int(value) for value in self.xyxy]
        if x2 <= x1 or y2 <= y1:
            raise ValueError("BoxAnnotation.xyxy must define a positive-area box.")
        object.__setattr__(self, "xyxy", (x1, y1, x2, y2))

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.xyxy
        return int(max(x2 - x1, 0) * max(y2 - y1, 0))


@dataclass
class DatasetRecord:
    sample_id: str
    image_path: str
    station_id: Any
    product_family: str
    geometry_mode: GeometryMode
    accept_reject: int
    defect_tags: Tuple[str, ...] = ()
    boxes: Tuple[BoxAnnotation, ...] = ()
    mask_path: Optional[str] = None
    split: str = "unspecified"
    capture_metadata: Mapping[str, Any] = field(default_factory=dict)
    source_dataset: str = "unknown"
    review_state: str = "unreviewed"

    def __post_init__(self) -> None:
        if isinstance(self.geometry_mode, str):
            self.geometry_mode = GeometryMode.from_value(self.geometry_mode)
        if self.accept_reject not in (0, 1):
            raise ValueError("DatasetRecord.accept_reject must be 0 or 1.")
        if isinstance(self.defect_tags, list):
            self.defect_tags = tuple(self.defect_tags)
        if isinstance(self.boxes, list):
            self.boxes = tuple(self.boxes)

    def to_sample(self, image_uint8: np.ndarray) -> Sample:
        metadata = dict(self.capture_metadata)
        metadata.update(
            {
                "sample_id": self.sample_id,
                "source_dataset": self.source_dataset,
                "review_state": self.review_state,
            }
        )
        return Sample(
            image_uint8=image_uint8,
            station_id=self.station_id,
            product_family=self.product_family,
            geometry_mode=self.geometry_mode,
            accept_reject=self.accept_reject,
            defect_tags=self.defect_tags,
            metadata=metadata,
        )


@dataclass
class DatasetIndex:
    manifest_version: str
    ontology_version: str
    root_dir: str
    manifest_path: str
    splits_path: str
    ontology_path: str
    hard_negatives_path: str
    index_path: Optional[str] = None
    split_seed: int = 17
    grouping_keys: Tuple[str, ...] = ("station_id", "capture_day", "batch_id", "camera_id")
    split_assignments: Mapping[str, str] = field(default_factory=dict)
    hard_negative_ids: Tuple[str, ...] = ()
    review_subset_ids: Tuple[str, ...] = ()
    station_configs: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelProfile:
    profile_id: str
    display_name: str
    backend_family: str
    task_type: str
    variant: Optional[str] = None
    checkpoint_path: Optional[str] = None
    hf_model_id: Optional[str] = None
    hf_revision: Optional[str] = None
    hf_local_path: Optional[str] = None
    hf_processor_id: Optional[str] = None
    runtime_engine: str = "pytorch"
    cache_policy: str = "online_cache"
    cache_dir: Optional[str] = None
    label_mapping: Mapping[str, Any] = field(default_factory=dict)
    thresholds: Mapping[str, float] = field(default_factory=dict)
    export_path: Optional[str] = None
    supports_training: bool = False
    supports_prediction: bool = True
    supports_explain: bool = True
    supports_benchmark: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkspaceConfig:
    workspace_root: str
    run_root: str = "artifacts"
    data_root: str = "data"
    profiles_dir: str = "profiles"
    cache_root: Optional[str] = None
    active_dataset_manifest: Optional[str] = None
    active_dataset_index: Optional[str] = None
    active_profile_id: Optional[str] = None
    slurm_defaults: Mapping[str, Any] = field(default_factory=dict)
    ui_preferences: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PredictionEvidence:
    heatmap_path: Optional[str] = None
    top_tiles_path: Optional[str] = None
    sample_dir: Optional[str] = None
    explanation_bundle_path: Optional[str] = None
    station_decision: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PredictionRecord:
    sample_id: str
    station_id: Any
    accept_reject: int
    reject_score: float
    predicted_label: int
    defect_probs: Mapping[str, float] = field(default_factory=dict)
    primary_label: str = ""
    primary_score: Optional[float] = None
    top_defect_family: Optional[str] = None
    defect_family_probs: Mapping[str, float] = field(default_factory=dict)
    evidence: PredictionEvidence = field(default_factory=PredictionEvidence)
    split: str = "unspecified"
    defect_scale: str = "unknown"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.accept_reject not in (0, 1):
            raise ValueError("PredictionRecord.accept_reject must be 0 or 1.")
        if self.predicted_label not in (0, 1):
            raise ValueError("PredictionRecord.predicted_label must be 0 or 1.")
        probs = dict(self.defect_family_probs or self.defect_probs or {})
        probs = {str(key): float(value) for key, value in probs.items()}
        object.__setattr__(self, "defect_probs", probs)
        object.__setattr__(self, "defect_family_probs", probs)
        if self.primary_score is None:
            object.__setattr__(self, "primary_score", float(self.reject_score))
        if not self.primary_label:
            object.__setattr__(self, "primary_label", "bad" if int(self.predicted_label) == 1 else "good")
        if self.primary_label not in {"good", "bad"}:
            raise ValueError("PredictionRecord.primary_label must be 'good' or 'bad'.")
        if self.top_defect_family is None and probs:
            object.__setattr__(self, "top_defect_family", max(probs.items(), key=lambda item: item[1])[0])
        if isinstance(self.evidence, Mapping):
            object.__setattr__(self, "evidence", PredictionEvidence(**self.evidence))


@dataclass(frozen=True)
class HierarchicalPredictionRecord:
    sample_id: str
    station_id: Any
    accept_reject: int
    primary_label: str
    primary_score: float
    predicted_label: int
    reject_score: float
    top_defect_family: Optional[str] = None
    defect_family_probs: Mapping[str, float] = field(default_factory=dict)
    split: str = "unspecified"
    defect_scale: str = "unknown"
    evidence: PredictionEvidence = field(default_factory=PredictionEvidence)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.primary_label not in {"good", "bad"}:
            raise ValueError("HierarchicalPredictionRecord.primary_label must be 'good' or 'bad'.")
        if self.predicted_label not in (0, 1):
            raise ValueError("HierarchicalPredictionRecord.predicted_label must be 0 or 1.")
        if isinstance(self.evidence, Mapping):
            object.__setattr__(self, "evidence", PredictionEvidence(**self.evidence))


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    kind: str
    backend_family: str
    execution_backend: str
    status: str
    command: Tuple[str, ...] = ()
    resolved_profile_id: Optional[str] = None
    log_path: Optional[str] = None
    run_dir: Optional[str] = None
    metadata_path: Optional[str] = None
    retry_target: Optional[str] = None
    resume_target: Optional[str] = None
    created_at: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.command, list):
            object.__setattr__(self, "command", tuple(str(value) for value in self.command))


@dataclass(frozen=True)
class LatencyReport:
    backend_family: str
    runtime_engine: str
    accelerator: str
    batch_size: int
    image_shape: Tuple[int, int]
    iterations: int
    warmup_iterations: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    throughput_per_second: float
    peak_memory_mb: float
    target_ms: Optional[float] = None
    meets_target: Optional[bool] = None
    profile_id: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FailureRecord:
    failure_id: str
    stage: str
    variant: str
    status: str
    error_type: str
    error_message: str
    run_dir: str
    failure_dir: str
    traceback_path: str
    timestamp: str
    manifest_path: Optional[str] = None
    index_path: Optional[str] = None
    latest_checkpoint_path: Optional[str] = None
    best_checkpoint_path: Optional[str] = None
    epoch: int = 0
    global_step: int = 0
    offending_sample_ids: Tuple[str, ...] = ()
    partial_artifacts: Mapping[str, Any] = field(default_factory=dict)
    resume_metadata: Mapping[str, Any] = field(default_factory=dict)
    config_snapshot_path: Optional[str] = None
    metrics_path: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.offending_sample_ids, list):
            object.__setattr__(self, "offending_sample_ids", tuple(str(value) for value in self.offending_sample_ids))

    def __post_init__(self) -> None:
        if isinstance(self.offending_sample_ids, list):
            object.__setattr__(self, "offending_sample_ids", tuple(self.offending_sample_ids))

    def __post_init__(self) -> None:
        if isinstance(self.offending_sample_ids, list):
            object.__setattr__(self, "offending_sample_ids", tuple(str(value) for value in self.offending_sample_ids))


@dataclass(frozen=True)
class RunStatusRecord:
    run_dir: str
    stage: str
    variant: str
    status: str
    run_root: Optional[str] = None
    session_id: Optional[str] = None
    manifest_path: Optional[str] = None
    index_path: Optional[str] = None
    started_at: Optional[str] = None
    updated_at: Optional[str] = None
    completed_at: Optional[str] = None
    failed_at: Optional[str] = None
    latest_checkpoint_path: Optional[str] = None
    best_checkpoint_path: Optional[str] = None
    latest_usable_checkpoint_path: Optional[str] = None
    report_path: Optional[str] = None
    summary_path: Optional[str] = None
    metrics_path: Optional[str] = None
    epoch: int = 0
    global_step: int = 0
    model_version: Optional[str] = None
    distributed_strategy: Optional[str] = None
    extra_paths: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class ModelOutput:
    reject_score: Any
    accept_reject_logit: Any
    defect_family_probs: Any
    defect_logits: Any
    defect_heatmap: Any
    top_tiles: Any
    top_tile_indices: Any
    top_tile_boxes: Any
    station_decision: Optional[Any] = None
    station_temperature: Optional[Any] = None
    tile_logits: Optional[Any] = None
    tile_validity: Optional[Any] = None
    local_heatmap: Optional[Any] = None
    global_heatmap: Optional[Any] = None
    global_feature_map: Optional[Any] = None
    fused_embedding: Optional[Any] = None
    metadata: Optional[Mapping[str, Any]] = None


def geometry_mode_to_id(mode: GeometryMode) -> int:
    return mode.to_id()


def geometry_modes_to_tensor_values(modes: Sequence[GeometryMode]) -> Tuple[int, ...]:
    return tuple(mode.to_id() for mode in modes)


def station_id_to_int(station_id: Any) -> int:
    if isinstance(station_id, (int, np.integer)):
        return int(station_id)
    try:
        return int(station_id)
    except (TypeError, ValueError):
        return int(zlib.crc32(str(station_id).encode("utf-8")) & 0x7FFFFFFF)
