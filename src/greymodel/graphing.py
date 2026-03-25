from __future__ import annotations

from pathlib import Path
import shutil
from typing import Dict, Tuple

import torch
import torch.fx as fx
import torch.nn.functional as F
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
        self.model = model
        self.image_shape = tuple(int(value) for value in image_shape)
        self.tile_boxes = tuple(build_tile_grid(self.image_shape, model.config.tile_size, model.config.tile_stride).boxes)
        tile_mask_bank = torch.zeros(len(self.tile_boxes), 1, self.image_shape[0], self.image_shape[1])
        for tile_index, (y1, x1, y2, x2) in enumerate(self.tile_boxes):
            tile_mask_bank[tile_index, :, y1:y2, x1:x2] = 1.0
        self.register_buffer("tile_mask_bank", tile_mask_bank, persistent=False)
        self.register_buffer("tile_mask_counts", tile_mask_bank.sum(dim=0).clamp_min(1.0), persistent=False)

    def _run_blocks(self, blocks, features):
        for block in blocks:
            features = block(features)
        return features

    def _vector_affine(self, vectors, conditioning, affine_module):
        scale = torch.tanh(affine_module.scale(conditioning)).unsqueeze(1)
        bias = affine_module.bias(conditioning).unsqueeze(1)
        return vectors * (1.0 + scale) + bias

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

    def forward(self, image, valid_mask, station_id, geometry_id):
        model = self.model
        conditioning = model.conditioner(station_id, geometry_id)

        stem = model.stem(image)
        stem = model.stem_affine(stem, conditioning)

        p1 = model.stage1_proj(stem)
        p1 = self._run_blocks(model.stage1, p1)
        p1 = model.stage1_affine(p1, conditioning)

        p2 = model.stage2_down.block(p1)
        p2 = self._run_blocks(model.stage2, p2)
        p2 = model.stage2_affine(p2, conditioning)

        p3 = model.stage3_down.block(p2)
        p3 = self._run_blocks(model.stage3, p3)
        p3 = model.stage3_affine(p3, conditioning)

        p4 = model.stage4_down.block(p3)
        p4 = self._run_blocks(model.stage4, p4)
        p4 = model.stage4_affine(p4, conditioning)

        pyramid = [layer(feature_map) for layer, feature_map in zip(model.pyramid_laterals, (p1, p2, p3, p4))]
        for block in model.bifpn:
            pyramid = list(block(*pyramid))
        p1_out, _, _, p4_out = pyramid

        coarse_mask = F.interpolate(valid_mask, size=p4_out.shape[-2:], mode="nearest")
        bounded_h = min(self.image_shape[0] // 32 if self.image_shape[0] >= 32 else 1, model.config.max_global_feature_grid)
        bounded_w = min(self.image_shape[1] // 32 if self.image_shape[1] >= 32 else 1, model.config.max_global_feature_grid)
        pooled_context = F.adaptive_avg_pool2d(p4_out, output_size=(max(1, bounded_h), max(1, bounded_w)))
        pooled_mask = F.adaptive_max_pool2d(coarse_mask, output_size=(max(1, bounded_h), max(1, bounded_w)))
        global_features = model.global_context_project(pooled_context) * pooled_mask
        global_vector = _masked_average(global_features, pooled_mask)

        global_projected = model.global_project(global_vector)
        tile_tensor, tile_mask_tensor = self._extract_tiles(image, valid_mask)
        tile_logits = (tile_tensor * tile_mask_tensor).mean(dim=(-1, -2, -3))
        local_heatmap = self._tile_scores_to_heatmap(tile_logits, valid_mask)
        local_scalar = local_heatmap.mean(dim=(-1, -2, -3), keepdim=False).unsqueeze(-1)
        local_projected = local_scalar.expand(-1, global_projected.shape[-1])
        condition_projected = model.condition_project(conditioning)
        gate = torch.sigmoid(model.fusion_gate(torch.cat([global_projected, local_projected, condition_projected], dim=-1)))
        fused = gate * global_projected + (1.0 - gate) * local_projected + 0.1 * condition_projected

        reject_logit = model.reject_head(fused).squeeze(-1)
        defect_logits = model.defect_head(fused)
        global_heatmap = model.global_heatmap_head(p1_out)
        global_heatmap = F.interpolate(global_heatmap, size=self.image_shape, mode="bilinear", align_corners=False)
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
