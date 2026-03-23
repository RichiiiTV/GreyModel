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
