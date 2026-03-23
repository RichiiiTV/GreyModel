from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .preprocessing import preprocess_image, stack_prepared_images
from .tiling import build_tile_grid
from .types import ModelInput, ModelOutput, StationConfig


def _sigmoid(values):
    return 1.0 / (1.0 + np.exp(-values))


@dataclass
class _NumpyBackend:
    num_defect_families: int
    top_k_tiles: int

    def forward(self, model_input: ModelInput, station_config: StationConfig) -> ModelOutput:
        prepared = preprocess_image(model_input, station_config)
        image = prepared.image
        mask = prepared.valid_mask.astype(np.float32)
        grid = build_tile_grid(prepared.canvas_shape, station_config.tile_size_2d, station_config.tile_stride_2d)
        tile_scores = []
        for y1, x1, y2, x2 in grid.boxes:
            tile_mask = mask[y1:y2, x1:x2]
            if tile_mask.mean() <= 0:
                tile_scores.append(-10.0)
                continue
            tile = image[y1:y2, x1:x2]
            valid_pixels = tile[tile_mask.astype(bool)]
            score = float(np.std(valid_pixels) + abs(np.mean(valid_pixels)))
            tile_scores.append(score)
        tile_scores = np.asarray(tile_scores, dtype=np.float32)
        tile_max = float(tile_scores.max()) if tile_scores.size else 0.0
        tile_mean = float(tile_scores.mean()) if tile_scores.size else 0.0
        heatmap = np.zeros(prepared.canvas_shape, dtype=np.float32)
        counts = np.zeros(prepared.canvas_shape, dtype=np.float32)
        for score, (y1, x1, y2, x2) in zip(tile_scores, grid.boxes):
            heatmap[y1:y2, x1:x2] += score
            counts[y1:y2, x1:x2] += 1.0
        heatmap = np.divide(heatmap, np.maximum(counts, 1.0))
        heatmap *= mask
        reject_logit = np.float32(tile_max)
        reject_score = np.float32(_sigmoid(reject_logit))
        if self.num_defect_families:
            anchors = np.linspace(-0.5, 0.5, self.num_defect_families, dtype=np.float32)
            defect_logits = np.float32(tile_mean) + anchors
        else:
            defect_logits = np.zeros((0,), dtype=np.float32)
        defect_probs = _sigmoid(defect_logits).astype(np.float32)
        top_k = min(self.top_k_tiles, tile_scores.shape[0])
        top_indices = np.argsort(-tile_scores)[:top_k]
        top_boxes = np.asarray([grid.boxes[index] for index in top_indices], dtype=np.int64)
        top_scores = tile_scores[top_indices][:, None]
        top_tiles = np.concatenate([top_boxes.astype(np.float32), top_scores], axis=1) if top_k else np.zeros((0, 5), dtype=np.float32)
        return ModelOutput(
            reject_score=reject_score,
            accept_reject_logit=reject_logit,
            defect_family_probs=defect_probs,
            defect_logits=defect_logits,
            defect_heatmap=heatmap,
            top_tiles=top_tiles,
            top_tile_indices=top_indices,
            top_tile_boxes=top_boxes,
            metadata={"backend": "numpy"},
        )


class _TorchBackend:
    def __init__(self, builder, num_defect_families: int, defect_families=(), num_stations: int = 32) -> None:
        self.model = builder(
            num_defect_families=num_defect_families,
            defect_families=defect_families,
            num_stations=num_stations,
        )
        self.model.eval()

    def forward(self, model_input: ModelInput, station_config: StationConfig) -> ModelOutput:
        import torch

        prepared = preprocess_image(model_input, station_config)
        batch = stack_prepared_images([prepared], as_torch=True)
        batch.station_id = batch.station_id % max(self.model.config.num_stations, 1)
        with torch.no_grad():
            output = self.model(batch)
        return ModelOutput(
            reject_score=output.reject_score[0],
            accept_reject_logit=output.accept_reject_logit[0],
            defect_family_probs=output.defect_family_probs[0],
            defect_logits=output.defect_logits[0],
            defect_heatmap=output.defect_heatmap[0, 0],
            top_tiles=output.top_tiles[0],
            top_tile_indices=output.top_tile_indices[0],
            top_tile_boxes=output.top_tile_boxes[0],
            tile_logits=output.tile_logits[0] if output.tile_logits is not None else None,
            tile_validity=output.tile_validity[0] if output.tile_validity is not None else None,
            local_heatmap=output.local_heatmap[0, 0] if output.local_heatmap is not None else None,
            global_heatmap=output.global_heatmap[0, 0] if output.global_heatmap is not None else None,
            global_feature_map=output.global_feature_map[0] if output.global_feature_map is not None else None,
            fused_embedding=output.fused_embedding[0] if output.fused_embedding is not None else None,
            metadata={"backend": "torch"},
        )


class BaseModel:
    def __init__(self, num_defect_families: int, defect_families: Sequence[str] = (), num_stations: int = 32) -> None:
        self.num_defect_families = num_defect_families
        try:
            from .models import build_base_model

            self.backend = _TorchBackend(
                build_base_model,
                num_defect_families=num_defect_families,
                defect_families=defect_families,
                num_stations=num_stations,
            )
        except ImportError:
            self.backend = _NumpyBackend(num_defect_families=num_defect_families, top_k_tiles=5)

    def forward(self, model_input: ModelInput, station_config: StationConfig) -> ModelOutput:
        return self.backend.forward(model_input, station_config)

    __call__ = forward


class LiteModel:
    def __init__(self, num_defect_families: int, defect_families: Sequence[str] = (), num_stations: int = 32) -> None:
        self.num_defect_families = num_defect_families
        try:
            from .models import build_lite_model

            self.backend = _TorchBackend(
                build_lite_model,
                num_defect_families=num_defect_families,
                defect_families=defect_families,
                num_stations=num_stations,
            )
        except ImportError:
            self.backend = _NumpyBackend(num_defect_families=num_defect_families, top_k_tiles=5)

    def forward(self, model_input: ModelInput, station_config: StationConfig) -> ModelOutput:
        return self.backend.forward(model_input, station_config)

    __call__ = forward
