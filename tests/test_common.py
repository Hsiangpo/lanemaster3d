from __future__ import annotations

import torch

from lanemaster3d.engine.common import build_project_matrix


def test_build_project_matrix_uses_src_image_hw_for_scaling() -> None:
    image = torch.zeros(1, 3, 720, 960, dtype=torch.float32)
    extrinsic = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    intrinsic = torch.tensor(
        [[[1000.0, 0.0, 1000.0], [0.0, 1000.0, 600.0], [0.0, 0.0, 1.0]]],
        dtype=torch.float32,
    )
    batch = {
        "image": image,
        "cam_extrinsic": extrinsic,
        "cam_intrinsic": intrinsic,
        "src_img_hw": torch.tensor([[1200.0, 1920.0]], dtype=torch.float32),
    }
    proj = build_project_matrix(batch)
    assert proj is not None
    expected = torch.tensor(
        [[[500.0, 0.0, 500.0, 0.0], [0.0, 600.0, 360.0, 0.0], [0.0, 0.0, 1.0, 0.0]]],
        dtype=torch.float32,
    )
    assert torch.allclose(proj, expected, atol=1e-4)


def test_build_project_matrix_without_src_hw_uses_input_image_size_fallback() -> None:
    image = torch.zeros(1, 3, 720, 960, dtype=torch.float32)
    extrinsic = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    intrinsic = torch.tensor(
        [[[1000.0, 0.0, 400.0], [0.0, 1000.0, 300.0], [0.0, 0.0, 1.0]]],
        dtype=torch.float32,
    )
    batch = {
        "image": image,
        "cam_extrinsic": extrinsic,
        "cam_intrinsic": intrinsic,
    }
    proj = build_project_matrix(batch)
    assert proj is not None
    expected = torch.tensor(
        [[[1000.0, 0.0, 400.0, 0.0], [0.0, 1000.0, 300.0, 0.0], [0.0, 0.0, 1.0, 0.0]]],
        dtype=torch.float32,
    )
    assert torch.allclose(proj, expected, atol=1e-4)
