from __future__ import annotations

import pytest
import torch

from lanemaster3d.models.geometry import GeometryConsistencyAdapter


def test_gca_forward_shape() -> None:
    model = GeometryConsistencyAdapter(in_dim=64, geo_dim=16, hidden_dim=32)
    anchor = torch.randn(2, 100, 64)
    geo = torch.randn(2, 100, 16)
    out = model(anchor, geo)
    assert out.shape == anchor.shape


def test_gca_shape_mismatch() -> None:
    model = GeometryConsistencyAdapter(in_dim=64, geo_dim=16, hidden_dim=32)
    anchor = torch.randn(2, 120, 64)
    geo = torch.randn(2, 100, 16)
    with pytest.raises(ValueError):
        model(anchor, geo)

