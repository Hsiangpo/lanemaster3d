from __future__ import annotations

import pytest
import torch

from lanemaster3d.models.losses import GeometryConsistencySelfSupervisionLoss


def test_gcs_loss_outputs() -> None:
    criterion = GeometryConsistencySelfSupervisionLoss()
    pred = torch.randn(2, 24, 3)
    reproj = pred + 0.1
    outputs = criterion(pred, reproj)
    assert set(outputs.keys()) == {"loss_total", "loss_consistency", "loss_smooth"}
    assert outputs["loss_total"] > 0


def test_gcs_shape_mismatch() -> None:
    criterion = GeometryConsistencySelfSupervisionLoss()
    pred = torch.randn(2, 24, 3)
    reproj = torch.randn(2, 20, 3)
    with pytest.raises(ValueError):
        criterion(pred, reproj)

