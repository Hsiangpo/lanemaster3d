from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from lanemaster3d.models import LaneMaster3DNet


def test_lane_master_net_anchor_projection_output() -> None:
    model = LaneMaster3DNet(
        hidden_dim=128,
        query_count=16,
        num_points=10,
        use_gca=True,
        backbone_name="resnet18",
        dynamic_anchor_enabled=True,
    )
    image = torch.randn(2, 3, 128, 256)
    out = model(image)
    assert out["pred_points"].shape == (2, 16, 10, 3)
    assert out["pred_scores"].shape == (2, 16)
    assert out["pred_vis"].shape == (2, 16, 10)
    assert out["pred_logits"].shape == (2, 16, 21)
    assert out["pred_logvar_x"].shape == (2, 16, 10)
    assert out["pred_logvar_z"].shape == (2, 16, 10)
    assert out["reg_proposals"].shape == (2, 16, 5 + 10 * 3 + 21)
    assert out["anchors"].shape == (2, 16, 5 + 10 * 3)
    assert out["anchor_priors"].shape == (2, 16, 10, 3)
    assert "ddp_aux" in out
    assert out["ddp_aux"].ndim == 0


def test_lane_master_net_physical_y_steps() -> None:
    model = LaneMaster3DNet(
        hidden_dim=64,
        query_count=8,
        num_points=6,
        use_gca=False,
        backbone_name="resnet18",
    )
    y_steps = model.y_steps.detach().cpu()
    assert y_steps.shape[0] == 6
    assert float(y_steps[0].item()) >= 3.0
    assert float(y_steps[-1].item()) >= 60.0
    assert torch.all(y_steps[1:] > y_steps[:-1])


def test_lane_master_net_with_projection_matrix() -> None:
    model = LaneMaster3DNet(
        hidden_dim=64,
        query_count=8,
        num_points=6,
        use_gca=True,
        backbone_name="resnet18",
    )
    image = torch.randn(1, 3, 128, 256)
    project = torch.tensor(
        [[[500.0, 0.0, 128.0, 0.0], [0.0, 500.0, 64.0, 0.0], [0.0, 0.0, 1.0, 0.0]]],
        dtype=torch.float32,
    )
    out = model(image, project_matrix=project)
    assert out["pred_points"].shape == (1, 8, 6, 3)


def test_lane_master_net_ddp_aux_covers_stage_cls_layer() -> None:
    model = LaneMaster3DNet(
        hidden_dim=64,
        query_count=8,
        num_points=6,
        num_category=22,
        use_gca=False,
        backbone_name="resnet18",
        iter_reg=1,
    )
    image = torch.randn(1, 3, 128, 256)
    out = model(image)
    model.zero_grad(set_to_none=True)
    out["ddp_aux"].backward()
    assert model.decode_stages[0].cls_layer.weight.grad is not None
    assert model.decode_stages[1].cls_layer.weight.grad is not None


def test_infer_build_model_should_respect_num_category() -> None:
    infer_path = Path(__file__).resolve().parents[1] / "tools" / "infer.py"
    spec = importlib.util.spec_from_file_location("lm3d_infer_module", infer_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = {
        "model": {
            "hidden_dim": 32,
            "query_count": 8,
            "num_points": 6,
            "num_category": 22,
            "iter_reg": 1,
            "anchor_feat_channels": 16,
            "feature_level": 2,
        },
        "innovation": {
            "gca": {"enabled": False},
            "dagp": {"enabled": False, "delta_scale": 0.25},
        },
    }
    model = module._build_model(config, torch.device("cpu"))
    assert model.num_category == 22
