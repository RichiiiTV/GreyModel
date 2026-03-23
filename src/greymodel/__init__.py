"""Public package surface for the grayscale inspection stack."""

from .api import BaseModel, LiteModel
from .calibration import CalibratedStationDecision, StationCalibration, StationCalibrator
from .data import GreyInspectionDataset, collate_batch, group_samples_by_station
from .preprocessing import PreparedImage, preprocess_image, preprocess_sample, stack_prepared_images
from .synthetic import DefectInjection, inject_defect, inject_particle, inject_scratch, inject_streak
from .tiling import TileCoverage, TileGrid, build_tile_grid, compute_tile_coverage, verify_defect_coverage
from .types import GeometryMode, ModelInput, ModelOutput, Sample, StationConfig, TensorBatch, TopTilePrediction

__all__ = [
    "BaseModel",
    "CalibratedStationDecision",
    "DefectInjection",
    "GeometryMode",
    "GreyInspectionDataset",
    "LiteModel",
    "ModelInput",
    "ModelOutput",
    "PreparedImage",
    "Sample",
    "StationCalibration",
    "StationCalibrator",
    "StationConfig",
    "TensorBatch",
    "TileCoverage",
    "TileGrid",
    "TopTilePrediction",
    "build_tile_grid",
    "collate_batch",
    "compute_tile_coverage",
    "group_samples_by_station",
    "inject_defect",
    "inject_particle",
    "inject_scratch",
    "inject_streak",
    "preprocess_image",
    "preprocess_sample",
    "stack_prepared_images",
    "verify_defect_coverage",
]
