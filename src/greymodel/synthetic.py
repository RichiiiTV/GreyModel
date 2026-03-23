from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class DefectInjection:
    kind: str
    box: Tuple[int, int, int, int]
    strength: int


def _clip_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0, 255).astype(np.uint8)


def inject_particle(
    image: np.ndarray,
    center: Tuple[int, int],
    radius: int = 2,
    intensity_delta: int = 80,
) -> Tuple[np.ndarray, DefectInjection]:
    canvas = image.astype(np.int16).copy()
    center_y, center_x = center
    yy, xx = np.ogrid[: canvas.shape[0], : canvas.shape[1]]
    mask = (yy - center_y) ** 2 + (xx - center_x) ** 2 <= radius ** 2
    canvas[mask] += intensity_delta
    y1 = max(center_y - radius, 0)
    x1 = max(center_x - radius, 0)
    y2 = min(center_y + radius + 1, canvas.shape[0])
    x2 = min(center_x + radius + 1, canvas.shape[1])
    return _clip_uint8(canvas), DefectInjection("particle", (y1, x1, y2, x2), intensity_delta)


def inject_scratch(
    image: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    thickness: int = 1,
    intensity_delta: int = 60,
) -> Tuple[np.ndarray, DefectInjection]:
    canvas = image.astype(np.int16).copy()
    y1, x1 = start
    y2, x2 = end
    steps = max(abs(y2 - y1), abs(x2 - x1), 1)
    ys = np.linspace(y1, y2, steps + 1).astype(np.int64)
    xs = np.linspace(x1, x2, steps + 1).astype(np.int64)
    for y, x in zip(ys, xs):
        min_y = max(y - thickness, 0)
        max_y = min(y + thickness + 1, canvas.shape[0])
        min_x = max(x - thickness, 0)
        max_x = min(x + thickness + 1, canvas.shape[1])
        canvas[min_y:max_y, min_x:max_x] += intensity_delta
    box = (
        max(min(y1, y2) - thickness, 0),
        max(min(x1, x2) - thickness, 0),
        min(max(y1, y2) + thickness + 1, canvas.shape[0]),
        min(max(x1, x2) + thickness + 1, canvas.shape[1]),
    )
    return _clip_uint8(canvas), DefectInjection("scratch", box, intensity_delta)


def inject_streak(
    image: np.ndarray,
    axis: int = 1,
    intensity_delta: int = -30,
    width: int = 4,
    position: Optional[int] = None,
) -> Tuple[np.ndarray, DefectInjection]:
    canvas = image.astype(np.int16).copy()
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1.")
    limit = canvas.shape[axis]
    position = limit // 2 if position is None else position
    start = max(position - width // 2, 0)
    stop = min(start + width, limit)
    if axis == 0:
        canvas[start:stop, :] += intensity_delta
        box = (start, 0, stop, canvas.shape[1])
    else:
        canvas[:, start:stop] += intensity_delta
        box = (0, start, canvas.shape[0], stop)
    return _clip_uint8(canvas), DefectInjection("streak", box, intensity_delta)


def inject_defect(
    image: np.ndarray,
    rng: np.random.Generator,
    kind: str = "particle",
) -> Tuple[np.ndarray, DefectInjection]:
    kind = kind.lower()
    if kind == "particle":
        center = (int(rng.integers(0, image.shape[0])), int(rng.integers(0, image.shape[1])))
        radius = int(rng.integers(1, 4))
        delta = int(rng.integers(40, 120))
        return inject_particle(image, center=center, radius=radius, intensity_delta=delta)
    if kind == "scratch":
        start = (int(rng.integers(0, image.shape[0])), int(rng.integers(0, image.shape[1])))
        end = (int(rng.integers(0, image.shape[0])), int(rng.integers(0, image.shape[1])))
        thickness = int(rng.integers(1, 3))
        delta = int(rng.integers(30, 90))
        return inject_scratch(image, start=start, end=end, thickness=thickness, intensity_delta=delta)
    if kind == "streak":
        axis = int(rng.integers(0, 2))
        width = int(rng.integers(2, 8))
        delta = int(rng.integers(-60, 60))
        return inject_streak(image, axis=axis, intensity_delta=delta, width=width)
    raise ValueError("Unsupported defect kind: %s" % kind)
