from __future__ import annotations

from dataclasses import is_dataclass

from greymodel import (
    BaseModel,
    GreyInspectionDataset,
    LiteModel,
    ModelInput,
    ModelOutput,
    PreparedImage,
    Sample,
    StationConfig,
    TensorBatch,
    TileCoverage,
    compute_tile_coverage,
    preprocess_image,
)


def test_public_data_models_exist_and_are_dataclasses() -> None:
    assert is_dataclass(Sample)
    assert is_dataclass(ModelInput)
    assert is_dataclass(ModelOutput)
    assert is_dataclass(StationConfig)
    assert is_dataclass(PreparedImage)
    assert is_dataclass(TensorBatch)
    assert is_dataclass(TileCoverage)


def test_public_data_models_have_expected_fields() -> None:
    sample_fields = Sample.__dataclass_fields__.keys()
    input_fields = ModelInput.__dataclass_fields__.keys()
    output_fields = ModelOutput.__dataclass_fields__.keys()
    station_fields = StationConfig.__dataclass_fields__.keys()

    assert "image_uint8" in sample_fields
    assert "station_id" in sample_fields
    assert "product_family" in sample_fields
    assert "geometry_mode" in sample_fields
    assert "accept_reject" in sample_fields
    assert "defect_tags" in sample_fields

    assert "image_uint8" in input_fields
    assert "station_id" in input_fields
    assert "geometry_mode" in input_fields

    assert "reject_score" in output_fields
    assert "accept_reject_logit" in output_fields
    assert "defect_family_probs" in output_fields
    assert "defect_heatmap" in output_fields
    assert "top_tiles" in output_fields

    assert "canvas_shape" in station_fields
    assert "pad_value" in station_fields
    assert "normalization_mean" in station_fields
    assert "normalization_std" in station_fields
    assert "tile_size" in station_fields
    assert "tile_stride" in station_fields
    assert "adapter_id" in station_fields
    assert "reject_threshold" in station_fields


def test_public_runtime_and_dataset_symbols_are_exported() -> None:
    assert BaseModel is not None
    assert LiteModel is not None
    assert GreyInspectionDataset is not None
    assert preprocess_image is not None
    assert compute_tile_coverage is not None
