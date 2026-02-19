from __future__ import annotations

import torch
from torch import Tensor, nn


class HeteroscedasticLaneUncertaintyLoss(nn.Module):
    """基于异方差高斯假设的点级不确定性损失。"""

    def __init__(self, logvar_min: float = -5.0, logvar_max: float = 4.0) -> None:
        super().__init__()
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)

    def forward(self, pred: Tensor, target: Tensor, logvar: Tensor, weight: Tensor | None = None) -> Tensor:
        if pred.shape != target.shape or pred.shape != logvar.shape:
            raise ValueError("pred、target、logvar 的形状必须一致")
        logvar_clip = logvar.clamp(min=self.logvar_min, max=self.logvar_max)
        precision = torch.exp(-logvar_clip)
        nll = 0.5 * (precision * (pred - target).square() + logvar_clip)
        if weight is None:
            return nll.mean()
        if weight.shape != pred.shape:
            raise ValueError("weight 的形状必须与 pred 一致")
        valid = weight.to(nll.dtype).clamp(min=0.0, max=1.0)
        denom = valid.sum().clamp_min(1.0)
        return (nll * valid).sum() / denom
