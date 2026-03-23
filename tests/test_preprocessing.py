from __future__ import annotations

import numpy as np

from greymodel import ModelInput, StationConfig, preprocess_image


def test_preprocess_preserves_aspect_ratio_and_emits_mask() -> None:
    image = np.arange(225 * 652, dtype=np.uint8).reshape(225, 652)
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
    model_input = ModelInput(
        image_uint8=image,
        station_id="station-01",
        geometry_mode="rect",
    )

    processed = preprocess_image(model_input, config)

    assert processed.image.shape[:2] == config.canvas_shape
    assert processed.valid_mask.shape == config.canvas_shape
    assert processed.image.dtype.kind == "f"
    assert processed.valid_mask.dtype == np.bool_
    assert processed.original_shape == image.shape
    assert processed.resized_shape[1] / processed.resized_shape[0] == image.shape[1] / image.shape[0]
    assert processed.valid_mask.any()


def test_preprocess_handles_square_and_rectangular_modes_consistently() -> None:
    image = np.full((128, 256), 127, dtype=np.uint8)
    rect_config = StationConfig(
        canvas_shape=(160, 320),
        pad_value=0,
        normalization_mean=0.5,
        normalization_std=0.25,
        tile_size=32,
        tile_stride=16,
        adapter_id="rect_a",
        reject_threshold=0.5,
    )
    square_config = StationConfig(
        canvas_shape=(256, 256),
        pad_value=0,
        normalization_mean=0.5,
        normalization_std=0.25,
        tile_size=32,
        tile_stride=16,
        adapter_id="square_a",
        reject_threshold=0.5,
    )

    rect = preprocess_image(
        ModelInput(image_uint8=image, station_id="station-01", geometry_mode="rect"),
        rect_config,
    )
    square = preprocess_image(
        ModelInput(image_uint8=image, station_id="station-01", geometry_mode="square"),
        square_config,
    )

    assert rect.image.ndim == 2
    assert square.image.ndim == 2
    assert rect.valid_mask.shape == rect_config.canvas_shape
    assert square.valid_mask.shape == square_config.canvas_shape

