from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from ..tiling import build_tile_grid
from ..types import ModelOutput
from .conditioning import FeatureAffine, StationGeometryConditioner, VectorAffine
from .config import GrayInspectConfig


def _group_count(channels: int) -> int:
    if channels % 8 == 0:
        return 8
    if channels % 4 == 0:
        return 4
    return 1


def _masked_average(features, mask):
    masked = features * mask
    denom = mask.sum(dim=(-2, -1), keepdim=False).clamp_min(1.0)
    return masked.sum(dim=(-2, -1)) / denom


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, inputs):
        return self.block(inputs)


class RelativePositionBias2D(nn.Module):
    def __init__(self, num_heads: int, max_height: int, max_width: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.max_height = max_height
        self.max_width = max_width
        self.height_bias = nn.Parameter(torch.zeros(2 * max_height - 1, num_heads))
        self.width_bias = nn.Parameter(torch.zeros(2 * max_width - 1, num_heads))
        nn.init.trunc_normal_(self.height_bias, std=0.02)
        nn.init.trunc_normal_(self.width_bias, std=0.02)

    def forward(self, height: int, width: int, device) -> torch.Tensor:
        if height > self.max_height or width > self.max_width:
            raise ValueError("Global token grid exceeds configured relative-position capacity.")
        coords_h = torch.arange(height, device=device)
        coords_w = torch.arange(width, device=device)
        rel_h = coords_h[:, None] - coords_h[None, :] + self.max_height - 1
        rel_w = coords_w[:, None] - coords_w[None, :] + self.max_width - 1
        bh = self.height_bias.index_select(0, rel_h.reshape(-1)).view(height, height, self.num_heads)
        bw = self.width_bias.index_select(0, rel_w.reshape(-1)).view(width, width, self.num_heads)
        bias = bh[:, None, :, None, :] + bw[None, :, None, :, :]
        return bias.permute(4, 0, 1, 2, 3).reshape(self.num_heads, height * width, height * width)


class SelfAttention2D(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_height: int, max_width: int) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        if self.head_dim * num_heads != dim:
            raise ValueError("dim must be divisible by num_heads.")
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.rel_pos = RelativePositionBias2D(num_heads, max_height, max_width)

    def forward(self, tokens, spatial_shape: Tuple[int, int], token_mask=None):
        batch, length, channels = tokens.shape
        qkv = self.qkv(tokens).reshape(batch, length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        attn = torch.matmul(query * self.scale, key.transpose(-2, -1))
        attn = attn + self.rel_pos(spatial_shape[0], spatial_shape[1], tokens.device).unsqueeze(0)
        if token_mask is not None:
            key_mask = ~token_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(key_mask, -1e4)
        weights = torch.softmax(attn, dim=-1)
        context = torch.matmul(weights, value)
        context = context.transpose(1, 2).reshape(batch, length, channels)
        if token_mask is not None:
            context = context * token_mask.unsqueeze(-1).to(context.dtype)
        return self.out(context)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int, max_height: int, max_width: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention2D(dim, num_heads, max_height=max_height, max_width=max_width)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, tokens, spatial_shape, token_mask=None):
        tokens = tokens + self.attn(self.norm1(tokens), spatial_shape=spatial_shape, token_mask=token_mask)
        tokens = tokens + self.mlp(self.norm2(tokens))
        if token_mask is not None:
            tokens = tokens * token_mask.unsqueeze(-1).to(tokens.dtype)
        return tokens


class LocalTileEncoder(nn.Module):
    def __init__(self, channels: Sequence[int], embedding_dim: int) -> None:
        super().__init__()
        c1, c2, c3 = channels
        self.net = nn.Sequential(
            ConvNormAct(1, c1, stride=1),
            ConvNormAct(c1, c1, stride=2),
            ConvNormAct(c1, c2, stride=2),
            ConvNormAct(c2, c3, stride=2),
        )
        self.project = nn.Linear(c3, embedding_dim)

    def forward(self, tiles, tile_mask):
        features = self.net(tiles)
        pooled_mask = F.interpolate(tile_mask, size=features.shape[-2:], mode="nearest")
        pooled = _masked_average(features, pooled_mask)
        return self.project(pooled), features


class TileAttentionPool(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.attention = nn.Linear(embedding_dim, 1)

    def forward(self, tile_embeddings, tile_validity):
        scores = self.attention(tile_embeddings).squeeze(-1)
        scores = scores.masked_fill(~tile_validity, -1e4)
        weights = torch.softmax(scores, dim=1)
        pooled = (tile_embeddings * weights.unsqueeze(-1)).sum(dim=1)
        return pooled, weights


class GrayInspectH(nn.Module):
    def __init__(self, config: GrayInspectConfig) -> None:
        super().__init__()
        self.config = config
        stem_c1, stem_c2 = config.stem_channels
        self.conditioner = StationGeometryConditioner(
            num_stations=config.num_stations,
            num_geometry_modes=config.num_geometry_modes,
            embedding_dim=config.conditioning_dim,
        )
        self.stem = nn.Sequential(
            ConvNormAct(1, stem_c1, stride=1),
            ConvNormAct(stem_c1, stem_c1, stride=2),
            ConvNormAct(stem_c1, stem_c2, stride=2),
        )
        self.stem_affine = FeatureAffine(stem_c2, config.conditioning_dim)
        self.global_downsample = nn.Sequential(
            ConvNormAct(stem_c2, config.global_hidden_dim, stride=2),
            ConvNormAct(config.global_hidden_dim, config.global_hidden_dim, stride=2),
        )
        self.global_affine = FeatureAffine(config.global_hidden_dim, config.conditioning_dim)
        self.transformer = nn.ModuleList(
            [
                TransformerBlock(
                    dim=config.global_hidden_dim,
                    num_heads=config.global_heads,
                    mlp_ratio=config.mlp_ratio,
                    max_height=config.max_relative_height,
                    max_width=config.max_relative_width,
                )
                for _ in range(config.global_depth)
            ]
        )
        self.global_project = nn.Linear(config.global_hidden_dim, config.fusion_dim)
        self.local_encoder = LocalTileEncoder(config.local_channels, config.local_embedding_dim)
        self.local_affine = VectorAffine(config.local_embedding_dim, config.conditioning_dim)
        self.tile_classifier = nn.Linear(config.local_embedding_dim, 1)
        self.tile_pool = TileAttentionPool(config.local_embedding_dim)
        self.local_project = nn.Linear(config.local_embedding_dim, config.fusion_dim)
        self.condition_project = nn.Linear(config.conditioning_dim, config.fusion_dim)
        self.fusion_gate = nn.Sequential(
            nn.Linear(config.fusion_dim * 3, config.fusion_dim),
            nn.GELU(),
            nn.Linear(config.fusion_dim, config.fusion_dim),
        )
        self.reject_head = nn.Linear(config.fusion_dim, 1)
        self.defect_head = nn.Linear(config.fusion_dim, config.num_defect_families)
        self.global_heatmap_head = nn.Conv2d(config.global_hidden_dim, 1, kernel_size=1)

    def _extract_tiles(self, image, valid_mask):
        height, width = image.shape[-2:]
        grid = build_tile_grid((height, width), self.config.tile_size, self.config.tile_stride)
        tiles = []
        masks = []
        for y1, x1, y2, x2 in grid.boxes:
            tiles.append(image[:, :, y1:y2, x1:x2])
            masks.append(valid_mask[:, :, y1:y2, x1:x2])
        tile_tensor = torch.stack(tiles, dim=1)
        mask_tensor = torch.stack(masks, dim=1)
        boxes = torch.as_tensor(grid.boxes, device=image.device, dtype=torch.long)
        return tile_tensor, mask_tensor, boxes

    def _tile_scores_to_heatmap(self, tile_scores, tile_boxes, image_shape, valid_mask):
        batch = tile_scores.shape[0]
        height, width = image_shape
        heatmap = tile_scores.new_zeros((batch, 1, height, width))
        counts = tile_scores.new_zeros((batch, 1, height, width))
        for tile_index, box in enumerate(tile_boxes.tolist()):
            y1, x1, y2, x2 = box
            value = tile_scores[:, tile_index].view(batch, 1, 1, 1)
            heatmap[:, :, y1:y2, x1:x2] = heatmap[:, :, y1:y2, x1:x2] + value
            counts[:, :, y1:y2, x1:x2] = counts[:, :, y1:y2, x1:x2] + 1.0
        heatmap = heatmap / counts.clamp_min(1.0)
        return heatmap * valid_mask

    def _select_top_tiles(self, tile_logits, tile_boxes, tile_validity):
        masked_logits = tile_logits.masked_fill(~tile_validity, -1e4)
        top_scores, top_indices = torch.topk(masked_logits, k=min(self.config.top_k_tiles, tile_logits.shape[1]), dim=1)
        expanded_boxes = tile_boxes.unsqueeze(0).expand(tile_logits.shape[0], -1, -1)
        top_boxes = torch.gather(
            expanded_boxes,
            1,
            top_indices.unsqueeze(-1).expand(-1, -1, 4),
        )
        top_tiles = torch.cat([top_boxes.float(), top_scores.unsqueeze(-1)], dim=-1)
        return top_tiles, top_indices, top_boxes

    def forward(self, batch):
        image = batch.image.float()
        valid_mask = batch.valid_mask.float()
        station_id = batch.station_id.long()
        geometry_id = batch.geometry_id.long()

        conditioning = self.conditioner(station_id, geometry_id)

        stem = self.stem(image)
        stem = self.stem_affine(stem, conditioning)

        global_features = self.global_downsample(stem)
        global_features = self.global_affine(global_features, conditioning)
        global_mask = F.interpolate(valid_mask, size=global_features.shape[-2:], mode="nearest")
        tokens = global_features.flatten(2).transpose(1, 2)
        token_mask = global_mask.flatten(2).squeeze(1) > 0.5
        for block in self.transformer:
            tokens = block(tokens, spatial_shape=global_features.shape[-2:], token_mask=token_mask)
        global_features = tokens.transpose(1, 2).reshape_as(global_features)
        global_vector = _masked_average(global_features, global_mask)

        tile_tensor, tile_mask_tensor, tile_boxes = self._extract_tiles(image, valid_mask)
        batch_size, tile_count, channels, tile_h, tile_w = tile_tensor.shape
        encoded_tiles, _ = self.local_encoder(
            tile_tensor.reshape(batch_size * tile_count, channels, tile_h, tile_w),
            tile_mask_tensor.reshape(batch_size * tile_count, 1, tile_h, tile_w),
        )
        tile_embeddings = encoded_tiles.reshape(batch_size, tile_count, -1)
        tile_embeddings = self.local_affine(tile_embeddings, conditioning)
        tile_logits = self.tile_classifier(tile_embeddings).squeeze(-1)
        tile_validity = tile_mask_tensor.mean(dim=(-1, -2, -3)) >= self.config.min_tile_valid_fraction
        local_vector, _ = self.tile_pool(tile_embeddings, tile_validity)

        global_projected = self.global_project(global_vector)
        local_projected = self.local_project(local_vector)
        condition_projected = self.condition_project(conditioning)
        gate = torch.sigmoid(
            self.fusion_gate(torch.cat([global_projected, local_projected, condition_projected], dim=-1))
        )
        fused = gate * global_projected + (1.0 - gate) * local_projected + 0.1 * condition_projected

        reject_logit = self.reject_head(fused).squeeze(-1)
        defect_logits = self.defect_head(fused)
        reject_score = torch.sigmoid(reject_logit)
        defect_family_probs = torch.sigmoid(defect_logits)

        global_heatmap = self.global_heatmap_head(global_features)
        global_heatmap = F.interpolate(
            global_heatmap,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        local_heatmap = self._tile_scores_to_heatmap(tile_logits, tile_boxes, image.shape[-2:], valid_mask)
        defect_heatmap = torch.sigmoid(0.5 * global_heatmap + 0.5 * local_heatmap) * valid_mask
        top_tiles, top_indices, top_boxes = self._select_top_tiles(tile_logits, tile_boxes, tile_validity)
        return ModelOutput(
            reject_score=reject_score,
            accept_reject_logit=reject_logit,
            defect_family_probs=defect_family_probs,
            defect_logits=defect_logits,
            defect_heatmap=defect_heatmap,
            top_tiles=top_tiles,
            top_tile_indices=top_indices,
            top_tile_boxes=top_boxes,
            tile_logits=tile_logits,
            tile_validity=tile_validity,
            local_heatmap=local_heatmap,
            global_heatmap=global_heatmap,
            global_feature_map=global_features,
            fused_embedding=fused,
            metadata={"tile_boxes": tile_boxes},
        )
