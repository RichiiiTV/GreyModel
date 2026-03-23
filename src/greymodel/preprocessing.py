from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np

from .types import GeometryMode, ModelInput, Sample, StationConfig, TensorBatch, station_id_to_int


@dataclass(frozen=True)
class PreparedImage:
    image: np.ndarray
    valid_mask: np.ndarray
    scale: float
    pad_offsets: Tuple[int, int]
    original_shape: Tuple[int, int]
    resized_shape: Tuple[int, int]
    canvas_shape: Tuple[int, int]
    station_id: object
    geometry_mode: GeometryMode
    image_uint8: np.ndarray

    @property
    def image_float32(self) -> np.ndarray:
        return self.image

    @property
    def canvas_size(self) -> Tuple[int, int]:
        return self.canvas_shape


def _resize_nearest(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_size
    src_h, src_w = image.shape
    if (src_h, src_w) == (target_h, target_w):
        return image.copy()
    y_idx = np.linspace(0, src_h - 1, target_h).astype(np.int64)
    x_idx = np.linspace(0, src_w - 1, target_w).astype(np.int64)
    return image[np.ix_(y_idx, x_idx)]


def aspect_fit_size(image_size: Tuple[int, int], canvas_size: Tuple[int, int]) -> Tuple[Tuple[int, int], float]:
    image_h, image_w = image_size
    canvas_h, canvas_w = canvas_size
    scale = min(float(canvas_h) / float(image_h), float(canvas_w) / float(image_w), 1.0)
    scaled_h = max(1, int(round(image_h * scale)))
    scaled_w = max(1, int(round(image_w * scale)))
    return (scaled_h, scaled_w), scale


def pad_to_canvas(
    image: np.ndarray,
    canvas_size: Tuple[int, int],
    pad_value: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    canvas_h, canvas_w = canvas_size
    image_h, image_w = image.shape
    if image_h > canvas_h or image_w > canvas_w:
        raise ValueError("Image must not exceed the target canvas after resizing.")

    padded = np.full((canvas_h, canvas_w), pad_value, dtype=np.uint8)
    valid_mask = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    offset_y = (canvas_h - image_h) // 2
    offset_x = (canvas_w - image_w) // 2
    padded[offset_y : offset_y + image_h, offset_x : offset_x + image_w] = image
    valid_mask[offset_y : offset_y + image_h, offset_x : offset_x + image_w] = 1.0
    return padded, valid_mask, (offset_y, offset_x)


def preprocess_image(model_input: ModelInput, station_config: StationConfig) -> PreparedImage:
    if station_config.geometry_mode is not None and station_config.geometry_mode != model_input.geometry_mode:
        raise ValueError("StationConfig.geometry_mode must match ModelInput.geometry_mode when provided.")

    target_size, scale = aspect_fit_size(model_input.image_uint8.shape, station_config.canvas_shape)
    resized = _resize_nearest(model_input.image_uint8, target_size)
    padded, valid_mask, offsets = pad_to_canvas(
        resized,
        station_config.canvas_shape,
        pad_value=station_config.pad_value,
    )
    image_float32 = padded.astype(np.float32)
    image_float32 = (image_float32 - station_config.normalization_mean) / max(
        station_config.normalization_std,
        1e-6,
    )
    image_float32 *= valid_mask
    return PreparedImage(
        image=image_float32,
        valid_mask=valid_mask.astype(np.bool_),
        scale=scale,
        pad_offsets=offsets,
        original_shape=tuple(int(v) for v in model_input.image_uint8.shape),
        resized_shape=tuple(int(v) for v in resized.shape),
        canvas_shape=station_config.canvas_shape,
        station_id=model_input.station_id,
        geometry_mode=model_input.geometry_mode,
        image_uint8=padded,
    )


def preprocess_sample(sample: Sample, station_config: StationConfig) -> PreparedImage:
    model_input = ModelInput(
        image_uint8=sample.image_uint8,
        station_id=sample.station_id,
        geometry_mode=sample.geometry_mode,
        metadata=sample.metadata,
    )
    return preprocess_image(model_input, station_config)


def stack_prepared_images(prepared_images: Sequence[PreparedImage], as_torch: bool = True) -> TensorBatch:
    if not prepared_images:
        raise ValueError("Expected at least one PreparedImage.")

    max_h = max(item.image_float32.shape[0] for item in prepared_images)
    max_w = max(item.image_float32.shape[1] for item in prepared_images)

    batch_images = []
    batch_masks = []
    station_ids = []
    geometry_ids = []

    for item in prepared_images:
        image = item.image
        mask = item.valid_mask.astype(np.float32)
        canvas = np.zeros((max_h, max_w), dtype=np.float32)
        canvas_mask = np.zeros((max_h, max_w), dtype=np.float32)
        canvas[: image.shape[0], : image.shape[1]] = image
        canvas_mask[: mask.shape[0], : mask.shape[1]] = mask
        batch_images.append(canvas)
        batch_masks.append(canvas_mask)
        station_ids.append(station_id_to_int(item.station_id))
        geometry_ids.append(item.geometry_mode.to_id())

    images_np = np.stack(batch_images, axis=0)[:, None, :, :]
    masks_np = np.stack(batch_masks, axis=0)[:, None, :, :]

    if not as_torch:
        return TensorBatch(
            image=images_np,
            valid_mask=masks_np,
            station_id=np.asarray(station_ids, dtype=np.int64),
            geometry_id=np.asarray(geometry_ids, dtype=np.int64),
            metadata={"canvas_size": (max_h, max_w)},
        )

    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required when as_torch=True.") from exc

    return TensorBatch(
        image=torch.from_numpy(images_np),
        valid_mask=torch.from_numpy(masks_np),
        station_id=torch.as_tensor(station_ids, dtype=torch.long),
        geometry_id=torch.as_tensor(geometry_ids, dtype=torch.long),
        metadata={"canvas_size": (max_h, max_w)},
    )


def preprocess_and_stack(
    samples: Iterable[Sample],
    station_configs: Sequence[StationConfig],
    as_torch: bool = True,
) -> TensorBatch:
    config_by_station = {config.station_id: config for config in station_configs}
    prepared = [preprocess_sample(sample, config_by_station[sample.station_id]) for sample in samples]
    return stack_prepared_images(prepared, as_torch=as_torch)
