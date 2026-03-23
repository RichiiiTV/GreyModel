from __future__ import annotations

from typing import Dict, Mapping, Optional


def _require_torch():
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError("PyTorch is required for loss computation.") from exc
    return torch, F


def weighted_bce_with_logits(logits, targets, positive_weight: float = 1.0):
    torch, F = _require_torch()
    pos_weight = torch.as_tensor(positive_weight, device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


def binary_focal_with_logits(logits, targets, alpha: float = 0.25, gamma: float = 2.0):
    torch, F = _require_torch()
    probabilities = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    focal = alpha_t * ((1.0 - p_t) ** gamma) * ce
    return focal.mean()


def mil_pool_loss(tile_logits, image_targets, tile_validity=None):
    torch, F = _require_torch()
    if tile_validity is None:
        tile_validity = torch.ones_like(tile_logits, dtype=torch.bool)
    masked_logits = tile_logits.masked_fill(~tile_validity, -1e4)
    pooled = torch.logsumexp(masked_logits, dim=1)
    return F.binary_cross_entropy_with_logits(pooled, image_targets)


def heatmap_consistency_loss(local_heatmap, global_heatmap, valid_mask):
    torch, F = _require_torch()
    mask = valid_mask.expand_as(local_heatmap)
    local = torch.sigmoid(local_heatmap) * mask
    global_map = torch.sigmoid(global_heatmap) * mask
    return F.l1_loss(local, global_map)


def combined_supervised_loss(
    model_output,
    reject_targets,
    defect_targets,
    valid_mask,
    defect_positive_weight: float = 1.0,
    reject_positive_weight: float = 1.0,
    focal_gamma: float = 2.0,
) -> Mapping[str, object]:
    torch, _ = _require_torch()
    reject_targets = reject_targets.float().view_as(model_output.accept_reject_logit)
    defect_targets = defect_targets.float().view_as(model_output.defect_logits)
    reject_loss = weighted_bce_with_logits(
        model_output.accept_reject_logit,
        reject_targets,
        positive_weight=reject_positive_weight,
    )
    defect_loss = binary_focal_with_logits(
        model_output.defect_logits,
        defect_targets,
        alpha=0.25,
        gamma=focal_gamma,
    )
    tile_loss = mil_pool_loss(
        model_output.tile_logits,
        reject_targets.view(reject_targets.shape[0], -1).max(dim=1).values,
        tile_validity=model_output.tile_validity,
    )
    consistency = heatmap_consistency_loss(
        model_output.local_heatmap,
        model_output.global_heatmap,
        valid_mask,
    )
    total = reject_loss + defect_positive_weight * defect_loss + 0.5 * tile_loss + 0.25 * consistency
    return {
        "loss": total,
        "reject_loss": reject_loss.detach(),
        "defect_loss": defect_loss.detach(),
        "tile_loss": tile_loss.detach(),
        "consistency_loss": consistency.detach(),
    }


def masked_reconstruction_loss(reconstruction, original, patch_mask):
    torch, F = _require_torch()
    mask = patch_mask.expand_as(reconstruction)
    return F.mse_loss(reconstruction * mask, original * mask)


def tile_image_consistency_loss(tile_heatmap, image_heatmap, valid_mask):
    return heatmap_consistency_loss(tile_heatmap, image_heatmap, valid_mask)
