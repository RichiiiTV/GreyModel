from __future__ import annotations

import numpy as np
import torch

from greymodel import LiteModel, ModelInput, StationConfig, BaseModel
from greymodel.models import build_base_model
from greymodel.models.grayinspect import RelativePositionBias2D
from greymodel.training import TrainingConfig, compute_masked_pretrain_objective
from greymodel.types import TensorBatch


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


def test_masked_pretrain_objective_skips_local_tile_branch(monkeypatch) -> None:
    model = build_base_model(num_defect_families=2)
    reconstruction_head = torch.nn.Conv2d(model.config.global_hidden_dim, 1, kernel_size=1)
    batch = TensorBatch(
        image=torch.zeros((1, 1, 128, 128), dtype=torch.float32),
        valid_mask=torch.ones((1, 1, 128, 128), dtype=torch.float32),
        station_id=torch.zeros((1,), dtype=torch.long),
        geometry_id=torch.zeros((1,), dtype=torch.long),
        metadata={},
    )

    def _unexpected_tile_extract(*_args, **_kwargs):
        raise AssertionError("The local tile branch should not run during masked pretraining.")

    monkeypatch.setattr(model, "_extract_tiles", _unexpected_tile_extract)
    loss, metrics, extras = compute_masked_pretrain_objective(model, reconstruction_head, batch, TrainingConfig())

    assert loss.ndim == 0
    assert "masked_reconstruction_loss" in metrics
    assert extras["output"].global_feature_map is not None


def test_relative_position_bias_resizes_beyond_configured_capacity() -> None:
    module = RelativePositionBias2D(num_heads=4, max_height=64, max_width=64)
    bias = module(height=80, width=2, device=torch.device("cpu"))

    assert bias.shape == (4, 160, 160)


def test_global_only_forward_bounds_oversized_token_grids() -> None:
    model = build_base_model(num_defect_families=2)
    batch = TensorBatch(
        image=torch.zeros((1, 1, 1056, 32), dtype=torch.float32),
        valid_mask=torch.ones((1, 1, 1056, 32), dtype=torch.float32),
        station_id=torch.zeros((1,), dtype=torch.long),
        geometry_id=torch.zeros((1,), dtype=torch.long),
        metadata={},
    )

    output = model(batch, return_mode="global_only")

    assert output.global_feature_map.shape[-2] <= model.config.max_relative_height
    assert output.global_feature_map.shape[-1] <= model.config.max_relative_width
