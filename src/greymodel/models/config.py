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
    stem_channels: Tuple[int, int] = (32, 48)
    stage_channels: Tuple[int, int, int, int] = (64, 96, 160, 224)
    stage_depths: Tuple[int, int, int, int] = (2, 2, 4, 2)
    bifpn_channels: int = 160
    bifpn_repeats: int = 2
    global_hidden_dim: int = 224
    coarse_context_heads: int = 4
    coarse_context_mlp_ratio: int = 2
    local_channels: Tuple[int, int, int] = (32, 64, 96)
    local_embedding_dim: int = 160
    conditioning_dim: int = 96
    fusion_dim: int = 192
    top_k_tiles: int = 5
    min_tile_valid_fraction: float = 0.2
    max_global_feature_grid: int = 16
    activation_checkpointing: bool = False

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
    activation_checkpointing: bool = False,
    max_global_feature_grid: int = 16,
) -> GrayInspectConfig:
    return GrayInspectConfig(
        name="GrayInspect-H-Base",
        defect_families=_normalize_defect_families(defect_families, num_defect_families),
        num_stations=num_stations,
        tile_size=(64, 64),
        tile_stride=(32, 32),
        stem_channels=(32, 48),
        stage_channels=(64, 96, 160, 224),
        stage_depths=(2, 2, 4, 2),
        bifpn_channels=160,
        bifpn_repeats=2,
        global_hidden_dim=224,
        coarse_context_heads=4,
        coarse_context_mlp_ratio=2,
        local_channels=(32, 64, 96),
        local_embedding_dim=160,
        conditioning_dim=96,
        fusion_dim=192,
        top_k_tiles=5,
        max_global_feature_grid=max(8, int(max_global_feature_grid)),
        activation_checkpointing=bool(activation_checkpointing),
    )


def build_lite_config(
    num_defect_families: int,
    defect_families: Sequence[str] = (),
    num_stations: int = 32,
    activation_checkpointing: bool = False,
    max_global_feature_grid: int = 12,
) -> GrayInspectConfig:
    return GrayInspectConfig(
        name="GrayInspect-H-Lite",
        defect_families=_normalize_defect_families(defect_families, num_defect_families),
        num_stations=num_stations,
        tile_size=(64, 64),
        tile_stride=(48, 48),
        stem_channels=(24, 32),
        stage_channels=(32, 48, 80, 128),
        stage_depths=(1, 1, 2, 1),
        bifpn_channels=96,
        bifpn_repeats=1,
        global_hidden_dim=128,
        coarse_context_heads=4,
        coarse_context_mlp_ratio=2,
        local_channels=(24, 32, 48),
        local_embedding_dim=96,
        conditioning_dim=64,
        fusion_dim=128,
        top_k_tiles=5,
        max_global_feature_grid=max(6, int(max_global_feature_grid)),
        activation_checkpointing=bool(activation_checkpointing),
    )
