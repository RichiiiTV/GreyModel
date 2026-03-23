from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class TileGrid:
    image_size: Tuple[int, int]
    tile_size: Tuple[int, int]
    stride: Tuple[int, int]
    boxes: Tuple[Tuple[int, int, int, int], ...]
    grid_shape: Tuple[int, int]


@dataclass(frozen=True)
class TileCoverage:
    canvas_shape: Tuple[int, int]
    tile_size: int
    tile_stride: int
    tiles: Tuple[Tuple[int, int, int, int], ...]
    min_covered_context: int
    max_gap_pixels: int
    covers_entire_canvas: bool


def _axis_positions(length: int, tile: int, stride: int) -> List[int]:
    if tile <= 0 or stride <= 0:
        raise ValueError("Tile size and stride must be positive.")
    if tile > length:
        return [0]
    positions = list(range(0, max(length - tile + 1, 1), stride))
    last = length - tile
    if not positions:
        positions = [0]
    elif positions[-1] != last:
        positions.append(last)
    return positions


def build_tile_grid(
    image_size: Tuple[int, int],
    tile_size: Tuple[int, int],
    stride: Tuple[int, int],
) -> TileGrid:
    image_h, image_w = image_size
    tile_h, tile_w = tile_size
    stride_h, stride_w = stride
    y_positions = _axis_positions(image_h, tile_h, stride_h)
    x_positions = _axis_positions(image_w, tile_w, stride_w)
    boxes = []
    for y in y_positions:
        for x in x_positions:
            boxes.append((y, x, min(y + tile_h, image_h), min(x + tile_w, image_w)))
    return TileGrid(
        image_size=image_size,
        tile_size=tile_size,
        stride=stride,
        boxes=tuple(boxes),
        grid_shape=(len(y_positions), len(x_positions)),
    )


def tile_coverage_map(
    image_size: Tuple[int, int],
    tile_size: Tuple[int, int],
    stride: Tuple[int, int],
) -> np.ndarray:
    coverage = np.zeros(image_size, dtype=np.int32)
    grid = build_tile_grid(image_size, tile_size, stride)
    for y1, x1, y2, x2 in grid.boxes:
        coverage[y1:y2, x1:x2] += 1
    return coverage


def verify_defect_coverage(
    image_size: Tuple[int, int],
    tile_size: Tuple[int, int],
    stride: Tuple[int, int],
    defect_size: Tuple[int, int] = (5, 5),
) -> Tuple[bool, int]:
    image_h, image_w = image_size
    defect_h, defect_w = defect_size
    if defect_h > image_h or defect_w > image_w:
        return False, 0

    grid = build_tile_grid(image_size, tile_size, stride)
    uncovered = 0
    for defect_y in range(0, image_h - defect_h + 1):
        defect_y2 = defect_y + defect_h
        for defect_x in range(0, image_w - defect_w + 1):
            defect_x2 = defect_x + defect_w
            contains = any(
                y1 <= defect_y and defect_y2 <= y2 and x1 <= defect_x and defect_x2 <= x2
                for y1, x1, y2, x2 in grid.boxes
            )
            if not contains:
                uncovered += 1
    return uncovered == 0, uncovered


def boxes_to_numpy(boxes: Sequence[Tuple[int, int, int, int]]) -> np.ndarray:
    return np.asarray(boxes, dtype=np.int64)


def compute_tile_coverage(station_config) -> TileCoverage:
    tile_h, tile_w = station_config.tile_size_2d
    stride_h, stride_w = station_config.tile_stride_2d
    grid = build_tile_grid(station_config.canvas_shape, (tile_h, tile_w), (stride_h, stride_w))
    coverage = tile_coverage_map(station_config.canvas_shape, (tile_h, tile_w), (stride_h, stride_w))
    min_context = min(tile_h, tile_w) - max(stride_h, stride_w)
    covers_entire_canvas = bool(np.all(coverage > 0))
    return TileCoverage(
        canvas_shape=station_config.canvas_shape,
        tile_size=tile_h if tile_h == tile_w else min(tile_h, tile_w),
        tile_stride=stride_h if stride_h == stride_w else min(stride_h, stride_w),
        tiles=grid.boxes,
        min_covered_context=max(min_context, 0),
        max_gap_pixels=max(stride_h, stride_w),
        covers_entire_canvas=covers_entire_canvas,
    )
