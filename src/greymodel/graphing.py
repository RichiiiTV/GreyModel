from __future__ import annotations

from pathlib import Path
import shutil
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
import torch.fx as fx
from torch import nn

from .tiling import build_tile_grid
from .utils import ensure_dir, write_json


def _masked_average(features, mask):
    masked = features * mask
    denom = mask.sum(dim=(-2, -1), keepdim=False).clamp_min(1.0)
    return masked.sum(dim=(-2, -1)) / denom


class GraphExportAdapter(nn.Module):
    def __init__(self, model, image_shape: Tuple[int, int]) -> None:
        super().__init__()
        self.conditioner = model.conditioner
        self.stem = model.stem
        self.stem_affine = model.stem_affine
        self.global_downsample = model.global_downsample
        self.global_affine = model.global_affine
        self.transformer = model.transformer
        self.global_project = model.global_project
        self.local_encoder = model.local_encoder
        self.local_affine = model.local_affine
        self.tile_classifier = model.tile_classifier
        self.tile_pool = model.tile_pool
        self.local_project = model.local_project
        self.condition_project = model.condition_project
        self.fusion_gate = model.fusion_gate
        self.reject_head = model.reject_head
        self.defect_head = model.defect_head
        self.global_heatmap_head = model.global_heatmap_head
        self.min_tile_valid_fraction = model.config.min_tile_valid_fraction
        self.image_shape = tuple(int(value) for value in image_shape)
        self.tile_boxes = tuple(build_tile_grid(self.image_shape, model.config.tile_size, model.config.tile_stride).boxes)
        tile_mask_bank = torch.zeros(len(self.tile_boxes), 1, self.image_shape[0], self.image_shape[1])
        for tile_index, (y1, x1, y2, x2) in enumerate(self.tile_boxes):
            tile_mask_bank[tile_index, :, y1:y2, x1:x2] = 1.0
        self.register_buffer("tile_mask_bank", tile_mask_bank, persistent=False)
        self.register_buffer("tile_mask_counts", tile_mask_bank.sum(dim=0).clamp_min(1.0), persistent=False)
        self.relative_bias_names = []
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.image_shape[0], self.image_shape[1])
            global_features = self.global_downsample(self.stem(dummy))
        self.global_shape = tuple(int(value) for value in global_features.shape[-2:])
        with torch.no_grad():
            for index, block in enumerate(self.transformer):
                bias = block.attn.rel_pos(self.global_shape[0], self.global_shape[1], dummy.device).unsqueeze(0)
                name = "_rel_bias_%d" % index
                self.register_buffer(name, bias, persistent=False)
                self.relative_bias_names.append(name)

    def _extract_tiles(self, image, valid_mask):
        tiles = []
        masks = []
        for y1, x1, y2, x2 in self.tile_boxes:
            tiles.append(image[:, :, y1:y2, x1:x2])
            masks.append(valid_mask[:, :, y1:y2, x1:x2])
        return torch.stack(tiles, dim=1), torch.stack(masks, dim=1)

    def _tile_scores_to_heatmap(self, tile_scores, valid_mask):
        weights = tile_scores.view(tile_scores.shape[0], tile_scores.shape[1], 1, 1, 1)
        heatmap = (weights * self.tile_mask_bank.unsqueeze(0)).sum(dim=1)
        return (heatmap / self.tile_mask_counts) * valid_mask

    def _apply_transformer_block(self, block, tokens, token_mask, rel_bias):
        norm_tokens = block.norm1(tokens)
        batch, length, channels = norm_tokens.shape
        attn_module = block.attn
        qkv = attn_module.qkv(norm_tokens).reshape(batch, length, 3, attn_module.num_heads, attn_module.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        attn = torch.matmul(query * attn_module.scale, key.transpose(-2, -1))
        attn = attn + rel_bias
        if token_mask is not None:
            key_mask = ~token_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(key_mask, -1e4)
        weights = torch.softmax(attn, dim=-1)
        context = torch.matmul(weights, value)
        context = context.transpose(1, 2).reshape(batch, length, channels)
        if token_mask is not None:
            context = context * token_mask.unsqueeze(-1).to(context.dtype)
        tokens = tokens + attn_module.out(context)
        tokens = tokens + block.mlp(block.norm2(tokens))
        if token_mask is not None:
            tokens = tokens * token_mask.unsqueeze(-1).to(tokens.dtype)
        return tokens

    def _apply_vector_affine(self, vectors, conditioning, affine_module):
        scale = torch.tanh(affine_module.scale(conditioning)).unsqueeze(1)
        bias = affine_module.bias(conditioning).unsqueeze(1)
        return vectors * (1.0 + scale) + bias

    def forward(self, image, valid_mask, station_id, geometry_id):
        conditioning = self.conditioner(station_id, geometry_id)
        stem = self.stem(image)
        stem = self.stem_affine(stem, conditioning)

        global_features = self.global_downsample(stem)
        global_features = self.global_affine(global_features, conditioning)
        global_mask = F.interpolate(valid_mask, size=self.global_shape, mode="nearest")
        tokens = global_features.flatten(2).transpose(1, 2)
        token_mask = global_mask.flatten(2).squeeze(1) > 0.5
        for index, block in enumerate(self.transformer):
            tokens = self._apply_transformer_block(block, tokens, token_mask, getattr(self, self.relative_bias_names[index]))
        global_features = tokens.transpose(1, 2).reshape_as(global_features)
        global_vector = _masked_average(global_features, global_mask)

        tile_tensor, tile_mask_tensor = self._extract_tiles(image, valid_mask)
        batch_size, tile_count, channels, tile_h, tile_w = tile_tensor.shape
        encoded_tiles, _ = self.local_encoder(
            tile_tensor.reshape(batch_size * tile_count, channels, tile_h, tile_w),
            tile_mask_tensor.reshape(batch_size * tile_count, 1, tile_h, tile_w),
        )
        tile_embeddings = encoded_tiles.reshape(batch_size, tile_count, -1)
        tile_embeddings = self._apply_vector_affine(tile_embeddings, conditioning, self.local_affine)
        tile_logits = self.tile_classifier(tile_embeddings).squeeze(-1)
        tile_validity = tile_mask_tensor.mean(dim=(-1, -2, -3)) >= self.min_tile_valid_fraction
        local_vector, _ = self.tile_pool(tile_embeddings, tile_validity)

        global_projected = self.global_project(global_vector)
        local_projected = self.local_project(local_vector)
        condition_projected = self.condition_project(conditioning)
        gate = torch.sigmoid(self.fusion_gate(torch.cat([global_projected, local_projected, condition_projected], dim=-1)))
        fused = gate * global_projected + (1.0 - gate) * local_projected + 0.1 * condition_projected

        reject_logit = self.reject_head(fused).squeeze(-1)
        defect_logits = self.defect_head(fused)
        global_heatmap = self.global_heatmap_head(global_features)
        global_heatmap = F.interpolate(global_heatmap, size=self.image_shape, mode="bilinear", align_corners=False)
        local_heatmap = self._tile_scores_to_heatmap(tile_logits, valid_mask)
        defect_heatmap = torch.sigmoid(0.5 * global_heatmap + 0.5 * local_heatmap) * valid_mask
        return reject_logit, defect_logits, defect_heatmap


def _mermaid_from_graph(graph: fx.Graph) -> str:
    lines = ["graph TD"]
    for node in graph.nodes:
        label = "%s\\n%s" % (node.op, str(node.target))
        lines.append('  %s["%s"]' % (node.name, label.replace('"', "'")))
    for node in graph.nodes:
        for argument in node.all_input_nodes:
            lines.append("  %s --> %s" % (argument.name, node.name))
    return "\n".join(lines) + "\n"


def _dot_from_graph(graph: fx.Graph) -> str:
    lines = ["digraph GrayInspectH {"]
    for node in graph.nodes:
        label = "%s\\n%s" % (node.op, str(node.target))
        lines.append('  %s [label="%s"];' % (node.name, label.replace('"', "'")))
    for node in graph.nodes:
        for argument in node.all_input_nodes:
            lines.append("  %s -> %s;" % (argument.name, node.name))
    lines.append("}")
    return "\n".join(lines) + "\n"


def export_model_graph(model, output_dir: Path | str, image_shape: Tuple[int, int] = (256, 256)) -> Dict[str, Path]:
    output_dir = ensure_dir(Path(output_dir))
    adapter = GraphExportAdapter(model, image_shape=image_shape)
    traced = fx.symbolic_trace(adapter)
    graph_payload = {
        "nodes": [
            {
                "name": node.name,
                "op": node.op,
                "target": str(node.target),
                "inputs": [argument.name for argument in node.all_input_nodes],
            }
            for node in traced.graph.nodes
        ],
        "image_shape": list(image_shape),
        "model_name": getattr(getattr(model, "config", None), "name", model.__class__.__name__),
    }
    json_path = write_json(output_dir / "model_graph.json", graph_payload)
    mermaid_path = output_dir / "model_graph.mmd"
    mermaid_path.write_text(_mermaid_from_graph(traced.graph), encoding="utf-8")
    result = {"json_path": json_path, "mermaid_path": mermaid_path}
    if shutil.which("dot"):
        dot_path = output_dir / "model_graph.dot"
        dot_path.write_text(_dot_from_graph(traced.graph), encoding="utf-8")
        result["dot_path"] = dot_path
    return result
