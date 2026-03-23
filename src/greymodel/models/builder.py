from __future__ import annotations

from .config import build_base_config, build_lite_config
from .grayinspect import GrayInspectH


def build_base_model(num_defect_families: int, defect_families=(), num_stations: int = 32):
    return GrayInspectH(build_base_config(num_defect_families, defect_families=defect_families, num_stations=num_stations))


def build_lite_model(num_defect_families: int, defect_families=(), num_stations: int = 32):
    return GrayInspectH(build_lite_config(num_defect_families, defect_families=defect_families, num_stations=num_stations))
