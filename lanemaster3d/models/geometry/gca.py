from __future__ import annotations

import torch
from torch import Tensor, nn


class GeometryConsistencyAdapter(nn.Module):
    """几何一致性适配器，用于融合语义与几何特征。"""

    def __init__(self, in_dim: int, geo_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.feature_proj = nn.Linear(in_dim, hidden_dim)
        self.geo_proj = nn.Linear(geo_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.out = nn.Linear(hidden_dim, in_dim)

    def forward(self, anchor_feat: Tensor, geometry_feat: Tensor) -> Tensor:
        if anchor_feat.shape[:-1] != geometry_feat.shape[:-1]:
            raise ValueError("anchor_feat 与 geometry_feat 的前置维度必须一致")
        anchor_hidden = self.feature_proj(anchor_feat)
        geometry_hidden = self.geo_proj(geometry_feat)
        fusion_input = torch.cat([anchor_hidden, geometry_hidden], dim=-1)
        weight = self.gate(fusion_input)
        fused = anchor_hidden + weight * geometry_hidden
        return self.out(fused)

