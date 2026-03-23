from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple


@dataclass(frozen=True)
class GrayInspectConfig:
    name: str
    defect_families: Tuple[str, ...]
    num_stations: int
    num_geometry_modes: int = 2
    tile_size: Tuple[int, int] = (64, 64)
    tile_stride: Tuple[int, int] = (32, 32)
    stem_channels: Tuple[int, int] = (32, 64)
    global_hidden_dim: int = 160
    global_depth: int = 2
    global_heads: int = 4
    mlp_ratio: int = 4
    local_channels: Tuple[int, int, int] = (24, 48, 96)
    local_embedding_dim: int = 160
    conditioning_dim: int = 96
    fusion_dim: int = 160
    top_k_tiles: int = 5
    min_tile_valid_fraction: float = 0.2
    max_relative_height: int = 64
    max_relative_width: int = 64

    @property
    def num_defect_families(self) -> int:
        return len(self.defect_families)


def _normalize_defect_families(defect_families: Sequence[str], num_defect_families: int) -> Tuple[str, ...]:
    if defect_families:
        return tuple(defect_families)
    return tuple("defect_%d" % index for index in range(num_defect_families))


def build_base_config(
    num_defect_families: int,
    defect_families: Sequence[str] = (),
    num_stations: int = 32,
) -> GrayInspectConfig:
    return GrayInspectConfig(
        name="GrayInspect-H-Base",
        defect_families=_normalize_defect_families(defect_families, num_defect_families),
        num_stations=num_stations,
        tile_size=(64, 64),
        tile_stride=(32, 32),
        stem_channels=(32, 64),
        global_hidden_dim=192,
        global_depth=3,
        global_heads=6,
        local_channels=(24, 48, 96),
        local_embedding_dim=192,
        conditioning_dim=96,
        fusion_dim=192,
        top_k_tiles=5,
    )


def build_lite_config(
    num_defect_families: int,
    defect_families: Sequence[str] = (),
    num_stations: int = 32,
) -> GrayInspectConfig:
    return GrayInspectConfig(
        name="GrayInspect-H-Lite",
        defect_families=_normalize_defect_families(defect_families, num_defect_families),
        num_stations=num_stations,
        tile_size=(64, 64),
        tile_stride=(48, 48),
        stem_channels=(24, 48),
        global_hidden_dim=128,
        global_depth=2,
        global_heads=4,
        local_channels=(16, 32, 64),
        local_embedding_dim=128,
        conditioning_dim=64,
        fusion_dim=128,
        top_k_tiles=5,
    )
