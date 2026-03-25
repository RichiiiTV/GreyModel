from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math
import random
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .losses import combined_supervised_loss, masked_reconstruction_loss, tile_image_consistency_loss
from .types import TensorBatch


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for training utilities.") from exc
    return torch


@dataclass(frozen=True)
class TrainingBatch:
    model_input: object
    reject_targets: object
    defect_targets: object
    metadata: Optional[Mapping[str, object]] = None


@dataclass(frozen=True)
class TrainingConfig:
    defect_positive_weight: float = 1.0
    reject_positive_weight: float = 1.0
    focal_gamma: float = 2.0
    mask_ratio: float = 0.4
    gradient_clip_norm: float = 1.0
    epochs: int = 1
    steps_per_epoch: Optional[int] = None
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 0
    num_workers: int = 0
    prefetch_factor: int = 2
    persistent_workers: bool = False
    global_batch_size: Optional[int] = None
    per_device_batch_size: int = 4
    grad_accum_steps: int = 1
    precision: str = "auto"
    distributed_backend: str = "auto"
    distributed_strategy: str = "fsdp"
    activation_checkpointing: bool = True
    memory_report: bool = True
    seed: int = 17
    log_every_n_steps: int = 10
    val_every_n_steps: int = 0
    checkpoint_every_n_steps: int = 0
    keep_last_k_checkpoints: int = 2
    resume_from: Optional[str] = None
    station_balanced_sampling: bool = True
    show_progress: bool = True
    pretrain_crop_size: int = 512
    pretrain_num_crops: int = 1
    pretrain_crop_scales: Tuple[float, ...] = (0.75, 1.0, 1.25)
    pretrain_min_valid_fraction: float = 0.2
    max_global_feature_grid: int = 16
    channels_last: bool = False
    ema_decay: float = 0.0
    compile_model: bool = False

    def resolved_grad_accum_steps(self, world_size: int) -> int:
        base = max(1, self.per_device_batch_size * max(world_size, 1))
        if self.global_batch_size is None:
            return max(1, self.grad_accum_steps)
        if self.global_batch_size < base:
            raise ValueError("global_batch_size must be at least per_device_batch_size * world_size.")
        if self.global_batch_size % base != 0:
            raise ValueError("global_batch_size must be divisible by per_device_batch_size * world_size.")
        return max(1, int(self.global_batch_size // base))

    def effective_global_batch_size(self, world_size: int) -> int:
        return int(self.per_device_batch_size * max(world_size, 1) * self.resolved_grad_accum_steps(world_size))


def seed_everything(seed: int) -> None:
    torch = _require_torch()
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def station_balanced_index_order(station_ids: Iterable[int]):
    grouped: Dict[int, list] = {}
    for index, station_id in enumerate(station_ids):
        grouped.setdefault(int(station_id), []).append(index)

    max_length = max((len(indices) for indices in grouped.values()), default=0)
    interleaved = []
    for offset in range(max_length):
        for station_id in sorted(grouped):
            indices = grouped[station_id]
            if offset < len(indices):
                interleaved.append(indices[offset])
    return interleaved


def build_patch_mask(images, valid_mask, patch_size: int = 16, mask_ratio: float = 0.4):
    torch = _require_torch()
    batch, _, height, width = images.shape
    patch_h = max(height // patch_size, 1)
    patch_w = max(width // patch_size, 1)
    patch_mask = torch.zeros((batch, 1, height, width), device=images.device, dtype=images.dtype)
    num_patches = patch_h * patch_w
    num_masked = max(1, int(num_patches * mask_ratio))
    for batch_index in range(batch):
        indices = torch.randperm(num_patches, device=images.device)[:num_masked]
        for flat_idx in indices.tolist():
            row = flat_idx // patch_w
            col = flat_idx % patch_w
            y1 = row * patch_size
            x1 = col * patch_size
            y2 = min(y1 + patch_size, height)
            x2 = min(x1 + patch_size, width)
            patch_mask[batch_index, :, y1:y2, x1:x2] = 1.0
    patch_mask *= valid_mask
    return patch_mask


def _bounded_crop_size(base_size: int, scale: float, image_height: int, image_width: int) -> int:
    scaled_size = int(round(float(base_size) * float(scale)))
    limit = min(int(image_height), int(image_width))
    return max(16, min(limit, scaled_size))


def _valid_crop_bounds(valid_mask) -> Tuple[int, int, int, int]:
    torch = _require_torch()
    coordinates = torch.nonzero(valid_mask > 0.5, as_tuple=False)
    if coordinates.numel() == 0:
        height, width = int(valid_mask.shape[-2]), int(valid_mask.shape[-1])
        return 0, 0, height, width
    y_coords = coordinates[:, -2]
    x_coords = coordinates[:, -1]
    return (
        int(y_coords.min().item()),
        int(x_coords.min().item()),
        int(y_coords.max().item()) + 1,
        int(x_coords.max().item()) + 1,
    )


def _select_crop_window(mask_slice, crop_size: int, min_valid_fraction: float, max_attempts: int = 8) -> Tuple[int, int]:
    torch = _require_torch()
    height = int(mask_slice.shape[-2])
    width = int(mask_slice.shape[-1])
    if crop_size >= height or crop_size >= width:
        return 0, 0
    max_y = max(height - crop_size, 0)
    max_x = max(width - crop_size, 0)
    best_coords = (0, 0)
    best_valid = -1.0
    for _ in range(max_attempts):
        if max_y > 0:
            y1 = int(torch.randint(0, max_y + 1, (1,), device=mask_slice.device).item())
        else:
            y1 = 0
        if max_x > 0:
            x1 = int(torch.randint(0, max_x + 1, (1,), device=mask_slice.device).item())
        else:
            x1 = 0
        valid_fraction = float(mask_slice[:, y1 : y1 + crop_size, x1 : x1 + crop_size].float().mean().item())
        if valid_fraction >= float(min_valid_fraction):
            return y1, x1
        if valid_fraction > best_valid:
            best_valid = valid_fraction
            best_coords = (y1, x1)

    y_min, x_min, y_max, x_max = _valid_crop_bounds(mask_slice)
    if y_max - y_min > 0 and x_max - x_min > 0:
        center_y = (y_min + y_max) // 2
        center_x = (x_min + x_max) // 2
        y1 = min(max(center_y - crop_size // 2, 0), max_y)
        x1 = min(max(center_x - crop_size // 2, 0), max_x)
        return int(y1), int(x1)
    return best_coords


def sample_pretrain_crops(model_input: TensorBatch, config: TrainingConfig) -> TensorBatch:
    torch = _require_torch()
    images = model_input.image
    valid_mask = model_input.valid_mask
    batch, channels, height, width = images.shape
    crop_tensors = []
    crop_masks = []
    crop_station_ids = []
    crop_geometry_ids = []
    metadata = dict(model_input.metadata or {})
    crop_sizes = list(config.pretrain_crop_scales or (1.0,))
    crop_count = max(1, int(config.pretrain_num_crops))
    base_crop_size = max(16, int(config.pretrain_crop_size))

    for batch_index in range(batch):
        image_slice = images[batch_index : batch_index + 1]
        mask_slice = valid_mask[batch_index : batch_index + 1]
        for crop_index in range(crop_count):
            scale = crop_sizes[(batch_index * crop_count + crop_index) % len(crop_sizes)]
            crop_size = _bounded_crop_size(base_crop_size, scale, height, width)
            y1, x1 = _select_crop_window(mask_slice[0], crop_size, config.pretrain_min_valid_fraction)
            y2 = y1 + crop_size
            x2 = x1 + crop_size
            crop_tensors.append(image_slice[:, :, y1:y2, x1:x2])
            crop_masks.append(mask_slice[:, :, y1:y2, x1:x2])
            crop_station_ids.append(model_input.station_id[batch_index : batch_index + 1])
            crop_geometry_ids.append(model_input.geometry_id[batch_index : batch_index + 1])

    stacked_image = torch.cat(crop_tensors, dim=0)
    stacked_mask = torch.cat(crop_masks, dim=0)
    return TensorBatch(
        image=stacked_image,
        valid_mask=stacked_mask,
        station_id=torch.cat(crop_station_ids, dim=0),
        geometry_id=torch.cat(crop_geometry_ids, dim=0),
        metadata={
            **metadata,
            "pretrain_crop_size": base_crop_size,
            "pretrain_num_crops": crop_count,
            "pretrain_crop_scales": [float(value) for value in crop_sizes],
        },
    )


def estimate_batch_pixels(model_input: TensorBatch) -> int:
    image = model_input.image
    return int(image.shape[0] * image.shape[-2] * image.shape[-1])


def enforce_memory_guardrails(model_input: TensorBatch, config: TrainingConfig, stage: str) -> None:
    pixels = estimate_batch_pixels(model_input)
    if stage == "pretrain":
        max_crop_size = max(16, int(round(config.pretrain_crop_size * max(config.pretrain_crop_scales or (1.0,)))))
        limit = int(max_crop_size * max_crop_size * max(config.per_device_batch_size, 1) * max(config.pretrain_num_crops, 1))
    else:
        limit = int(
            max(config.max_global_feature_grid, 1)
            * max(config.max_global_feature_grid, 1)
            * 64
            * max(config.per_device_batch_size, 1)
        )
        limit = max(limit, pixels)
    if pixels > limit * 4:
        raise ValueError(
            "Runtime batch exceeded the configured memory guardrail for stage %s: %d pixels > %d."
            % (stage, pixels, limit * 4)
        )


def resolve_precision(training_config: TrainingConfig, device) -> str:
    precision = training_config.precision.lower()
    if precision == "auto":
        if getattr(device, "type", "cpu") == "cuda":
            import torch

            if torch.cuda.is_bf16_supported():
                return "bf16"
            return "fp16"
        return "fp32"
    return precision


def build_autocast_context(training_config: TrainingConfig, device):
    torch = _require_torch()
    precision = resolve_precision(training_config, device)
    if getattr(device, "type", "cpu") != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def build_grad_scaler(training_config: TrainingConfig, device):
    torch = _require_torch()
    precision = resolve_precision(training_config, device)
    enabled = getattr(device, "type", "cpu") == "cuda" and precision == "fp16"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def build_scheduler(optimizer, total_steps: int, warmup_steps: int = 0):
    torch = _require_torch()
    total_steps = max(int(total_steps), 1)
    warmup_steps = max(int(warmup_steps), 0)

    def _lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(warmup_steps, 1))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)


def compute_supervised_objective(model, batch: TrainingBatch, config: TrainingConfig):
    output = model(batch.model_input)
    loss_dict = combined_supervised_loss(
        output,
        batch.reject_targets,
        batch.defect_targets,
        batch.model_input.valid_mask,
        defect_positive_weight=config.defect_positive_weight,
        reject_positive_weight=config.reject_positive_weight,
        focal_gamma=config.focal_gamma,
    )
    return loss_dict["loss"], {key: value.detach() for key, value in loss_dict.items() if key != "loss"}, output


def compute_masked_pretrain_objective(model, reconstruction_head, model_input, config: TrainingConfig):
    torch = _require_torch()
    patch_mask = build_patch_mask(
        model_input.image,
        model_input.valid_mask,
        mask_ratio=config.mask_ratio,
    )
    masked_image = model_input.image * (1.0 - patch_mask)
    masked_input = type(model_input)(
        image=masked_image,
        valid_mask=model_input.valid_mask,
        station_id=model_input.station_id,
        geometry_id=model_input.geometry_id,
        metadata=model_input.metadata,
    )
    try:
        output = model(masked_input, return_mode="global_only")
    except TypeError:
        output = model(masked_input)
    reconstruction = reconstruction_head(output.global_feature_map)
    if reconstruction.shape[-2:] != model_input.image.shape[-2:]:
        reconstruction = torch.nn.functional.interpolate(
            reconstruction,
            size=model_input.image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    loss = masked_reconstruction_loss(reconstruction, model_input.image, patch_mask)
    metrics = {
        "masked_reconstruction_loss": loss.detach(),
        "masked_fraction": patch_mask.mean().detach(),
    }
    return loss, metrics, {"patch_mask": patch_mask, "reconstruction": reconstruction, "output": output}


def compute_domain_adaptation_objective(model, model_input, config: TrainingConfig):
    output = model(model_input)
    loss = tile_image_consistency_loss(output.local_heatmap, output.global_heatmap, model_input.valid_mask)
    return loss, {"domain_consistency_loss": loss.detach()}, output


def run_supervised_step(model, optimizer, batch: TrainingBatch, config: TrainingConfig):
    torch = _require_torch()
    model.train()
    optimizer.zero_grad()
    loss, metrics, _ = compute_supervised_objective(model, batch, config)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
    optimizer.step()
    payload = {"loss": loss.detach(), **metrics}
    return {key: value.item() if hasattr(value, "item") else value for key, value in payload.items()}


def run_masked_pretrain_step(model, reconstruction_head, optimizer, model_input, config: TrainingConfig):
    torch = _require_torch()
    model.train()
    reconstruction_head.train()
    optimizer.zero_grad()
    loss, metrics, _ = compute_masked_pretrain_objective(model, reconstruction_head, model_input, config)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(reconstruction_head.parameters()), config.gradient_clip_norm)
    optimizer.step()
    payload = {"loss": loss.detach(), **metrics}
    return {key: value.item() if hasattr(value, "item") else value for key, value in payload.items()}


def run_domain_adaptation_step(model, optimizer, model_input, config: TrainingConfig):
    torch = _require_torch()
    model.train()
    optimizer.zero_grad()
    loss, metrics, _ = compute_domain_adaptation_objective(model, model_input, config)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
    optimizer.step()
    payload = {"loss": loss.detach(), **metrics}
    return {key: value.item() if hasattr(value, "item") else value for key, value in payload.items()}
