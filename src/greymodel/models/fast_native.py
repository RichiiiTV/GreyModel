from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from ..tiling import build_tile_grid
from ..types import ModelOutput


@dataclass(frozen=True)
class FastNativeConfig:
    defect_families: Tuple[str, ...]
    patch_size: Tuple[int, int] = (40, 40)
    patch_stride: Tuple[int, int] = (20, 20)
    top_k_patches: int = 4
    screen_channels: Tuple[int, int, int] = (16, 24, 32)
    patch_channels: Tuple[int, int, int] = (16, 24, 32)
    embedding_dim: int = 64
    uncertainty_good_max: float = 0.35
    uncertainty_bad_min: float = 0.65

    @property
    def num_defect_families(self) -> int:
        return max(len(self.defect_families), 1)


class _DepthwiseBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, inputs):
        return self.block(inputs)


class FastNativeCascade(nn.Module):
    def __init__(self, config: FastNativeConfig) -> None:
        super().__init__()
        self.config = config
        c1, c2, c3 = config.screen_channels
        p1, p2, p3 = config.patch_channels
        self.screen_encoder = nn.Sequential(
            _DepthwiseBlock(1, c1, stride=2),
            _DepthwiseBlock(c1, c2, stride=2),
            _DepthwiseBlock(c2, c3, stride=2),
        )
        self.screen_heatmap_head = nn.Conv2d(c3, 1, kernel_size=1)
        self.screen_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c3, 1),
        )
        self.patch_encoder = nn.Sequential(
            _DepthwiseBlock(1, p1, stride=2),
            _DepthwiseBlock(p1, p2, stride=2),
            _DepthwiseBlock(p2, p3, stride=2),
        )
        self.patch_projection = nn.Linear(p3, config.embedding_dim)
        self.patch_attention = nn.Linear(config.embedding_dim, 1)
        self.patch_classifier = nn.Linear(config.embedding_dim, 1)
        self.defect_head = nn.Linear(config.embedding_dim, config.num_defect_families)

    def screen_forward(self, images, valid_mask):
        features = self.screen_encoder(images)
        coarse_mask = F.interpolate(valid_mask, size=features.shape[-2:], mode="nearest")
        features = features * coarse_mask
        heatmap = self.screen_heatmap_head(features)
        logit = self.screen_classifier(features).squeeze(-1)
        return logit, heatmap

    def _extract_patch_tensor(self, images, valid_mask):
        height, width = int(images.shape[-2]), int(images.shape[-1])
        grid = build_tile_grid((height, width), self.config.patch_size, self.config.patch_stride)
        patches = []
        masks = []
        for y1, x1, y2, x2 in grid.boxes:
            patches.append(images[:, :, y1:y2, x1:x2])
            masks.append(valid_mask[:, :, y1:y2, x1:x2])
        patch_tensor = torch.stack(patches, dim=1)
        mask_tensor = torch.stack(masks, dim=1)
        patch_boxes = torch.as_tensor(grid.boxes, device=images.device, dtype=torch.long)
        return patch_tensor, mask_tensor, patch_boxes

    def refine_forward(self, images, valid_mask, coarse_heatmap):
        patch_tensor, mask_tensor, patch_boxes = self._extract_patch_tensor(images, valid_mask)
        batch_size, patch_count, channels, patch_h, patch_w = patch_tensor.shape
        pooled_heatmap = []
        for y1, x1, y2, x2 in patch_boxes.tolist():
            pooled_heatmap.append(coarse_heatmap[:, :, y1:y2, x1:x2].mean(dim=(-1, -2, -3)))
        patch_priors = torch.stack(pooled_heatmap, dim=1)
        patch_validity = mask_tensor.mean(dim=(-1, -2, -3)) > 0.2
        patch_priors = patch_priors.masked_fill(~patch_validity, -1e4)
        top_k = min(int(self.config.top_k_patches), int(patch_priors.shape[1]))
        top_scores, top_indices = torch.topk(patch_priors, k=top_k, dim=1)
        expanded_boxes = patch_boxes.unsqueeze(0).expand(batch_size, -1, -1)
        selected_boxes = torch.gather(expanded_boxes, 1, top_indices.unsqueeze(-1).expand(-1, -1, 4))
        selected_patches = torch.gather(
            patch_tensor,
            1,
            top_indices.view(batch_size, top_k, 1, 1, 1).expand(-1, -1, channels, patch_h, patch_w),
        )
        encoded = self.patch_encoder(selected_patches.reshape(batch_size * top_k, channels, patch_h, patch_w))
        pooled = encoded.mean(dim=(-1, -2))
        embeddings = self.patch_projection(pooled).reshape(batch_size, top_k, -1)
        attention_logits = self.patch_attention(embeddings).squeeze(-1)
        mil_weights = torch.softmax(attention_logits, dim=1)
        pooled_embedding = (embeddings * mil_weights.unsqueeze(-1)).sum(dim=1)
        mil_logit = self.patch_classifier(pooled_embedding).squeeze(-1)
        defect_logits = self.defect_head(pooled_embedding)
        top_tiles = torch.cat([selected_boxes.float(), top_scores.unsqueeze(-1)], dim=-1)
        return mil_logit, defect_logits, top_tiles, top_indices, selected_boxes

    def forward(self, batch):
        images = batch.image.float()
        valid_mask = batch.valid_mask.float()
        screen_logit, coarse_heatmap = self.screen_forward(images, valid_mask)
        coarse_heatmap_up = F.interpolate(coarse_heatmap, size=images.shape[-2:], mode="bilinear", align_corners=False)
        mil_logit, defect_logits, top_tiles, top_indices, top_boxes = self.refine_forward(images, valid_mask, coarse_heatmap_up)
        final_logit = 0.4 * screen_logit + 0.6 * mil_logit
        reject_score = torch.sigmoid(final_logit)
        defect_probs = torch.sigmoid(defect_logits)
        defect_heatmap = torch.sigmoid(coarse_heatmap_up) * valid_mask
        return ModelOutput(
            reject_score=reject_score,
            accept_reject_logit=final_logit,
            defect_family_probs=defect_probs,
            defect_logits=defect_logits,
            defect_heatmap=defect_heatmap,
            top_tiles=top_tiles,
            top_tile_indices=top_indices,
            top_tile_boxes=top_boxes,
            global_heatmap=coarse_heatmap_up,
            global_feature_map=coarse_heatmap,
            fused_embedding=None,
            metadata={"backend": "native_fast", "top_k_patches": int(self.config.top_k_patches)},
        )
