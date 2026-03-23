from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional

from .losses import combined_supervised_loss, masked_reconstruction_loss, tile_image_consistency_loss


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


def run_supervised_step(model, optimizer, batch: TrainingBatch, config: TrainingConfig):
    torch = _require_torch()
    model.train()
    optimizer.zero_grad()
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
    loss = loss_dict["loss"]
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
    optimizer.step()
    return {key: value.item() if hasattr(value, "item") else value for key, value in loss_dict.items()}


def run_masked_pretrain_step(model, reconstruction_head, optimizer, model_input, config: TrainingConfig):
    torch = _require_torch()
    model.train()
    reconstruction_head.train()
    optimizer.zero_grad()
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
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(reconstruction_head.parameters()), config.gradient_clip_norm)
    optimizer.step()
    return {"masked_reconstruction_loss": float(loss.detach().item())}


def run_domain_adaptation_step(model, optimizer, model_input, config: TrainingConfig):
    torch = _require_torch()
    model.train()
    optimizer.zero_grad()
    output = model(model_input)
    loss = tile_image_consistency_loss(output.local_heatmap, output.global_heatmap, model_input.valid_mask)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
    optimizer.step()
    return {"domain_consistency_loss": float(loss.detach().item())}
