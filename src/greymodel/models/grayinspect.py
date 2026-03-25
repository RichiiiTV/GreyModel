from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..tiling import build_tile_grid
from ..types import ModelOutput
from .conditioning import FeatureAffine, StationGeometryConditioner, VectorAffine
from .config import GrayInspectConfig


def _masked_average(features, mask):
    masked = features * mask
    denom = mask.sum(dim=(-2, -1), keepdim=False).clamp_min(1.0)
    return masked.sum(dim=(-2, -1)) / denom


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = float(eps)

    def forward(self, inputs):
        normalized = F.layer_norm(
            inputs.permute(0, 2, 3, 1),
            (inputs.shape[1],),
            self.weight,
            self.bias,
            self.eps,
        )
        return normalized.permute(0, 3, 1, 2)


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
        )

    def forward(self, inputs):
        return self.block(inputs)


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        hidden_dim = channels * mlp_ratio
        self.depthwise = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.norm = LayerNorm2d(channels)
        self.pointwise = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, channels, kernel_size=1),
        )
        self.scale = nn.Parameter(torch.ones(channels) * 1e-6)

    def forward(self, inputs):
        residual = inputs
        outputs = self.depthwise(inputs)
        outputs = self.norm(outputs)
        outputs = self.pointwise(outputs)
        return residual + self.scale.view(1, -1, 1, 1) * outputs


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            LayerNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, inputs):
        if inputs.shape[-2] < 2 or inputs.shape[-1] < 2:
            inputs = F.interpolate(
                inputs,
                size=(max(int(inputs.shape[-2]), 2), max(int(inputs.shape[-1]), 2)),
                mode="nearest",
            )
        return self.block(inputs)


class WeightedFeatureFusion(nn.Module):
    def __init__(self, channels: int, num_inputs: int) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32))
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, channels),
            nn.GELU(),
        )

    def forward(self, *inputs):
        normalized_weights = F.relu(self.weights)
        normalized_weights = normalized_weights / normalized_weights.sum().clamp_min(1e-6)
        stacked = torch.stack(list(inputs), dim=0)
        fused = (normalized_weights.view(-1, 1, 1, 1, 1) * stacked).sum(dim=0)
        return self.refine(fused)


class BiFPNBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.td3 = WeightedFeatureFusion(channels, 2)
        self.td2 = WeightedFeatureFusion(channels, 2)
        self.td1 = WeightedFeatureFusion(channels, 2)
        self.out2 = WeightedFeatureFusion(channels, 3)
        self.out3 = WeightedFeatureFusion(channels, 3)
        self.out4 = WeightedFeatureFusion(channels, 3)

    def forward(self, p1, p2, p3, p4):
        td4 = p4
        td3 = self.td3(p3, F.interpolate(td4, size=p3.shape[-2:], mode="nearest"))
        td2 = self.td2(p2, F.interpolate(td3, size=p2.shape[-2:], mode="nearest"))
        td1 = self.td1(p1, F.interpolate(td2, size=p1.shape[-2:], mode="nearest"))
        out1 = td1
        out2 = self.out2(p2, td2, F.adaptive_max_pool2d(out1, output_size=p2.shape[-2:]))
        out3 = self.out3(p3, td3, F.adaptive_max_pool2d(out2, output_size=p3.shape[-2:]))
        out4 = self.out4(p4, td4, F.adaptive_max_pool2d(out3, output_size=p4.shape[-2:]))
        return out1, out2, out3, out4


class CoarseContextBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int, mlp_ratio: int, max_grid: int) -> None:
        super().__init__()
        hidden_dim = channels * mlp_ratio
        self.num_heads = num_heads
        self.max_grid = int(max_grid)
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels),
        )

    def forward(self, features, valid_mask):
        target_h = min(int(features.shape[-2]), self.max_grid)
        target_w = min(int(features.shape[-1]), self.max_grid)
        pooled_features = F.adaptive_avg_pool2d(features, output_size=(target_h, target_w))
        pooled_mask = F.adaptive_max_pool2d(valid_mask, output_size=(target_h, target_w))
        tokens = pooled_features.flatten(2).transpose(1, 2)
        key_padding_mask = ~(pooled_mask.flatten(2).squeeze(1) > 0.5)
        attn_inputs = self.norm1(tokens)
        attn_outputs, _ = self.attn(attn_inputs, attn_inputs, attn_inputs, key_padding_mask=key_padding_mask, need_weights=False)
        tokens = tokens + attn_outputs
        tokens = tokens + self.mlp(self.norm2(tokens))
        context_map = tokens.transpose(1, 2).reshape(pooled_features.shape)
        if context_map.shape[-2:] != features.shape[-2:]:
            context_map = F.interpolate(context_map, size=features.shape[-2:], mode="bilinear", align_corners=False)
        return features + context_map * valid_mask


class LocalDefectEncoder(nn.Module):
    def __init__(self, channels: Sequence[int], embedding_dim: int) -> None:
        super().__init__()
        c1, c2, c3 = channels
        self.stem = ConvNormAct(1, c1, stride=1)
        self.stage1 = nn.Sequential(
            ConvNeXtBlock(c1),
            DownsampleBlock(c1, c2),
            ConvNeXtBlock(c2),
        )
        self.stage2 = nn.Sequential(
            DownsampleBlock(c2, c3),
            ConvNeXtBlock(c3),
            ConvNeXtBlock(c3),
        )
        self.project = nn.Linear(c3, embedding_dim)

    def forward(self, tiles, tile_mask):
        features = self.stem(tiles)
        features = self.stage1(features)
        features = self.stage2(features)
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
        self.conditioner = StationGeometryConditioner(
            num_stations=config.num_stations,
            num_geometry_modes=config.num_geometry_modes,
            embedding_dim=config.conditioning_dim,
        )

        stem_c1, stem_c2 = config.stem_channels
        s1, s2, s3, s4 = config.stage_channels
        d1, d2, d3, d4 = config.stage_depths

        self.stem = nn.Sequential(
            ConvNormAct(1, stem_c1, stride=2),
            ConvNormAct(stem_c1, stem_c2, stride=2),
        )
        self.stem_affine = FeatureAffine(stem_c2, config.conditioning_dim)

        self.stage1_proj = nn.Conv2d(stem_c2, s1, kernel_size=1)
        self.stage1 = nn.ModuleList([ConvNeXtBlock(s1) for _ in range(d1)])
        self.stage1_affine = FeatureAffine(s1, config.conditioning_dim)

        self.stage2_down = DownsampleBlock(s1, s2)
        self.stage2 = nn.ModuleList([ConvNeXtBlock(s2) for _ in range(d2)])
        self.stage2_affine = FeatureAffine(s2, config.conditioning_dim)

        self.stage3_down = DownsampleBlock(s2, s3)
        self.stage3 = nn.ModuleList([ConvNeXtBlock(s3) for _ in range(d3)])
        self.stage3_affine = FeatureAffine(s3, config.conditioning_dim)

        self.stage4_down = DownsampleBlock(s3, s4)
        self.stage4 = nn.ModuleList([ConvNeXtBlock(s4) for _ in range(d4)])
        self.stage4_affine = FeatureAffine(s4, config.conditioning_dim)

        self.pyramid_laterals = nn.ModuleList(
            [nn.Conv2d(channels, config.bifpn_channels, kernel_size=1) for channels in (s1, s2, s3, s4)]
        )
        self.bifpn = nn.ModuleList([BiFPNBlock(config.bifpn_channels) for _ in range(config.bifpn_repeats)])
        self.coarse_context = CoarseContextBlock(
            channels=config.bifpn_channels,
            num_heads=config.coarse_context_heads,
            mlp_ratio=config.coarse_context_mlp_ratio,
            max_grid=config.max_global_feature_grid,
        )
        self.global_context_project = nn.Conv2d(config.bifpn_channels, config.global_hidden_dim, kernel_size=1)
        self.global_project = nn.Linear(config.global_hidden_dim, config.fusion_dim)
        self.local_encoder = LocalDefectEncoder(config.local_channels, config.local_embedding_dim)
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
        self.global_heatmap_head = nn.Sequential(
            nn.Conv2d(config.bifpn_channels, config.bifpn_channels, kernel_size=3, padding=1, groups=config.bifpn_channels),
            nn.GELU(),
            nn.Conv2d(config.bifpn_channels, 1, kernel_size=1),
        )

    def _maybe_checkpoint(self, module, *inputs):
        if self.config.activation_checkpointing and self.training:
            return checkpoint(module, *inputs, use_reentrant=False)
        return module(*inputs)

    def _run_block_list(self, blocks, features):
        for block in blocks:
            features = self._maybe_checkpoint(block, features)
        return features

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
        top_boxes = torch.gather(expanded_boxes, 1, top_indices.unsqueeze(-1).expand(-1, -1, 4))
        top_tiles = torch.cat([top_boxes.float(), top_scores.unsqueeze(-1)], dim=-1)
        return top_tiles, top_indices, top_boxes

    def _encode_global(self, batch):
        image = batch.image.float()
        valid_mask = batch.valid_mask.float()
        station_id = batch.station_id.long()
        geometry_id = batch.geometry_id.long()
        conditioning = self.conditioner(station_id, geometry_id)

        stem = self.stem(image)
        stem = self.stem_affine(stem, conditioning)

        p1 = self.stage1_proj(stem)
        p1 = self._run_block_list(self.stage1, p1)
        p1 = self.stage1_affine(p1, conditioning)

        p2 = self.stage2_down(p1)
        p2 = self._run_block_list(self.stage2, p2)
        p2 = self.stage2_affine(p2, conditioning)

        p3 = self.stage3_down(p2)
        p3 = self._run_block_list(self.stage3, p3)
        p3 = self.stage3_affine(p3, conditioning)

        p4 = self.stage4_down(p3)
        p4 = self._run_block_list(self.stage4, p4)
        p4 = self.stage4_affine(p4, conditioning)

        pyramid = [layer(feature_map) for layer, feature_map in zip(self.pyramid_laterals, (p1, p2, p3, p4))]
        for block in self.bifpn:
            pyramid = list(self._maybe_checkpoint(block, *pyramid))
        p1_out, p2_out, p3_out, p4_out = pyramid

        coarse_mask = F.interpolate(valid_mask, size=p4_out.shape[-2:], mode="nearest")
        p4_context = self._maybe_checkpoint(self.coarse_context, p4_out, coarse_mask)
        bounded_h = min(int(p4_context.shape[-2]), int(self.config.max_global_feature_grid))
        bounded_w = min(int(p4_context.shape[-1]), int(self.config.max_global_feature_grid))
        pooled_context = F.adaptive_avg_pool2d(p4_context, output_size=(bounded_h, bounded_w))
        pooled_mask = F.adaptive_max_pool2d(coarse_mask, output_size=(bounded_h, bounded_w))
        global_features = self.global_context_project(pooled_context) * pooled_mask
        global_vector = _masked_average(global_features, pooled_mask)
        return image, valid_mask, conditioning, p1_out, global_features, global_vector

    def forward(self, batch, return_mode: str = "full"):
        image, valid_mask, conditioning, high_res_features, global_features, global_vector = self._encode_global(batch)

        if return_mode == "global_only":
            return ModelOutput(
                reject_score=None,
                accept_reject_logit=None,
                defect_family_probs=None,
                defect_logits=None,
                defect_heatmap=None,
                top_tiles=None,
                top_tile_indices=None,
                top_tile_boxes=None,
                global_feature_map=global_features,
                metadata={"forward_mode": "global_only", "global_grid": list(global_features.shape[-2:])},
            )
        if return_mode != "full":
            raise ValueError("Unsupported GrayInspectH return_mode %r." % return_mode)

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
        gate = torch.sigmoid(self.fusion_gate(torch.cat([global_projected, local_projected, condition_projected], dim=-1)))
        fused = gate * global_projected + (1.0 - gate) * local_projected + 0.1 * condition_projected

        reject_logit = self.reject_head(fused).squeeze(-1)
        defect_logits = self.defect_head(fused)
        reject_score = torch.sigmoid(reject_logit)
        defect_family_probs = torch.sigmoid(defect_logits)

        global_heatmap = self.global_heatmap_head(high_res_features)
        global_heatmap = F.interpolate(global_heatmap, size=image.shape[-2:], mode="bilinear", align_corners=False)
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
            metadata={"tile_boxes": tile_boxes, "global_grid": list(global_features.shape[-2:])},
        )
