from __future__ import annotations

from greymodel import StationConfig, compute_tile_coverage


def test_tile_coverage_captures_small_defects_with_context() -> None:
    config = StationConfig(
        canvas_shape=(225, 652),
        pad_value=0,
        normalization_mean=0.5,
        normalization_std=0.25,
        tile_size=32,
        tile_stride=16,
        adapter_id="rect_a",
        reject_threshold=0.5,
    )

    coverage = compute_tile_coverage(config)

    assert coverage.tile_size == 32
    assert coverage.tile_stride == 16
    assert coverage.tiles
    assert coverage.min_covered_context >= 5
    assert coverage.max_gap_pixels <= config.tile_stride
    assert coverage.covers_entire_canvas is True

