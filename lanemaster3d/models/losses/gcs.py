from __future__ import annotations

import torch
from torch import Tensor, nn


class GeometryConsistencySelfSupervisionLoss(nn.Module):
    """几何循环一致性损失。"""

    def __init__(self, lambda_consistency: float = 1.0, lambda_smooth: float = 0.2):
        super().__init__()
        self.lambda_consistency = lambda_consistency
        self.lambda_smooth = lambda_smooth

    def forward(
        self,
        forward_projected: Tensor,
        backward_reprojected: Tensor,
        visibility_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if forward_projected.shape != backward_reprojected.shape:
            raise ValueError("前向投影与反投影张量形状必须一致")
        diff = torch.abs(forward_projected - backward_reprojected)
        if visibility_mask is not None:
            mask = visibility_mask.to(diff.dtype).unsqueeze(-1)
            denom = mask.sum().clamp_min(1.0)
            consistency = (diff * mask).sum() / denom
        else:
            consistency = diff.mean()
        smooth = self._smoothness(forward_projected)
        total = self.lambda_consistency * consistency + self.lambda_smooth * smooth
        return {
            "loss_total": total,
            "loss_consistency": consistency,
            "loss_smooth": smooth,
        }

    def _smoothness(self, points: Tensor) -> Tensor:
        if points.size(-2) < 3:
            return points.new_tensor(0.0)
        second_order = points[..., 2:, :] - 2 * points[..., 1:-1, :] + points[..., :-2, :]
        return second_order.abs().mean()

