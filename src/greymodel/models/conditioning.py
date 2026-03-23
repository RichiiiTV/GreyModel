from __future__ import annotations

import torch
from torch import nn


class StationGeometryConditioner(nn.Module):
    def __init__(self, num_stations: int, num_geometry_modes: int, embedding_dim: int) -> None:
        super().__init__()
        self.station_embedding = nn.Embedding(max(num_stations, 1), embedding_dim)
        self.geometry_embedding = nn.Embedding(max(num_geometry_modes, 1), embedding_dim)
        self.proj = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

    def forward(self, station_ids, geometry_ids):
        embeddings = self.station_embedding(station_ids) + self.geometry_embedding(geometry_ids)
        return self.proj(embeddings)


class FeatureAffine(nn.Module):
    def __init__(self, channels: int, conditioning_dim: int) -> None:
        super().__init__()
        self.scale = nn.Linear(conditioning_dim, channels)
        self.bias = nn.Linear(conditioning_dim, channels)

    def forward(self, features, conditioning):
        scale = torch.tanh(self.scale(conditioning)).unsqueeze(-1).unsqueeze(-1)
        bias = self.bias(conditioning).unsqueeze(-1).unsqueeze(-1)
        return features * (1.0 + scale) + bias


class VectorAffine(nn.Module):
    def __init__(self, channels: int, conditioning_dim: int) -> None:
        super().__init__()
        self.scale = nn.Linear(conditioning_dim, channels)
        self.bias = nn.Linear(conditioning_dim, channels)

    def forward(self, vectors, conditioning):
        scale = torch.tanh(self.scale(conditioning))
        bias = self.bias(conditioning)
        while scale.dim() < vectors.dim():
            scale = scale.unsqueeze(1)
            bias = bias.unsqueeze(1)
        return vectors * (1.0 + scale) + bias
