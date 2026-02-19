from __future__ import annotations

import torch

from lanemaster3d.models.losses import HeteroscedasticLaneUncertaintyLoss


def test_heteroscedastic_loss_shape_and_finite() -> None:
    loss_fn = HeteroscedasticLaneUncertaintyLoss(logvar_min=-4.0, logvar_max=3.0)
    pred = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    target = torch.tensor([[1.5, 2.0, 2.5]], dtype=torch.float32)
    logvar = torch.zeros_like(pred)
    mask = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32)
    loss = loss_fn(pred, target, logvar, mask)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert float(loss.item()) > 0.0


def test_heteroscedastic_loss_penalizes_overconfidence() -> None:
    loss_fn = HeteroscedasticLaneUncertaintyLoss(logvar_min=-6.0, logvar_max=6.0)
    pred = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    target = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    low_var = torch.tensor([[-5.0, -5.0]], dtype=torch.float32)
    high_var = torch.tensor([[2.0, 2.0]], dtype=torch.float32)
    mask = torch.ones_like(pred)
    loss_low = loss_fn(pred, target, low_var, mask)
    loss_high = loss_fn(pred, target, high_var, mask)
    assert float(loss_low.item()) > float(loss_high.item())
