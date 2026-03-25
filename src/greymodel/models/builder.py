from __future__ import annotations

from .config import build_base_config, build_lite_config
from .grayinspect import GrayInspectH


def build_base_model(
    num_defect_families: int,
    defect_families=(),
    num_stations: int = 32,
    activation_checkpointing: bool = False,
    max_global_feature_grid: int = 16,
):
    return GrayInspectH(
        build_base_config(
            num_defect_families,
            defect_families=defect_families,
            num_stations=num_stations,
            activation_checkpointing=activation_checkpointing,
            max_global_feature_grid=max_global_feature_grid,
        )
    )


def build_lite_model(
    num_defect_families: int,
    defect_families=(),
    num_stations: int = 32,
    activation_checkpointing: bool = False,
    max_global_feature_grid: int = 12,
):
    return GrayInspectH(
        build_lite_config(
            num_defect_families,
            defect_families=defect_families,
            num_stations=num_stations,
            activation_checkpointing=activation_checkpointing,
            max_global_feature_grid=max_global_feature_grid,
        )
    )
