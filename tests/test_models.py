from __future__ import annotations

import numpy as np

from greymodel import LiteModel, ModelInput, StationConfig, BaseModel


def _make_input(height: int, width: int, station_id: str, geometry_mode: str) -> ModelInput:
    image = np.zeros((height, width), dtype=np.uint8)
    return ModelInput(image_uint8=image, station_id=station_id, geometry_mode=geometry_mode)


def test_base_model_forward_returns_expected_shapes() -> None:
    model = BaseModel(num_defect_families=6)
    config = StationConfig(
        canvas_shape=(256, 704),
        pad_value=0,
        normalization_mean=0.5,
        normalization_std=0.25,
        tile_size=32,
        tile_stride=16,
        adapter_id="rect_a",
        reject_threshold=0.5,
    )

    output = model.forward(
        _make_input(225, 652, "station-01", "rect"),
        station_config=config,
    )

    assert output.accept_reject_logit.shape == ()
    assert output.reject_score.shape == ()
    assert output.defect_family_probs.shape == (6,)
    assert output.defect_heatmap.ndim == 2
    assert output.top_tiles.ndim == 2


def test_lite_model_forward_matches_public_contract() -> None:
    model = LiteModel(num_defect_families=6)
    config = StationConfig(
        canvas_shape=(256, 256),
        pad_value=0,
        normalization_mean=0.5,
        normalization_std=0.25,
        tile_size=32,
        tile_stride=16,
        adapter_id="square_a",
        reject_threshold=0.5,
    )

    output = model.forward(
        _make_input(256, 256, "station-02", "square"),
        station_config=config,
    )

    assert output.accept_reject_logit.shape == ()
    assert output.reject_score.shape == ()
    assert output.defect_family_probs.shape == (6,)
    assert output.defect_heatmap.ndim == 2
    assert output.top_tiles.ndim == 2
