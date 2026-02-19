from __future__ import annotations

from pathlib import Path

import torch

from lanemaster3d.command_builder import CommandBundle
from lanemaster3d.runtime import render_command
from tools import infer


def test_render_command() -> None:
    bundle = CommandBundle(env={"A": "1", "B": "2"}, cmd=["bash", "run.sh"], cwd=".")
    text = render_command(bundle)
    assert "A=1" in text
    assert "bash run.sh" in text


def test_render_command_with_cwd(tmp_path: Path) -> None:
    bundle = CommandBundle(env={}, cmd=["python", "demo.py"], cwd=str(tmp_path))
    text = render_command(bundle)
    assert str(tmp_path) in text
    assert "python demo.py" in text


class _DummyInferModel:
    def __call__(self, image: torch.Tensor, project_matrix=None) -> dict[str, torch.Tensor]:
        batch_size = int(image.shape[0])
        values = torch.arange(0, batch_size, dtype=torch.float32).view(batch_size, 1, 1, 1)
        pred_points = values.repeat(1, 1, 2, 3)
        pred_scores = values.view(batch_size, 1)
        return {"pred_points": pred_points, "pred_scores": pred_scores}


def test_run_inference_batches_collects_all_batches() -> None:
    loader = [
        {"image": torch.zeros(1, 3, 8, 8, dtype=torch.float32)},
        {"image": torch.zeros(2, 3, 8, 8, dtype=torch.float32)},
    ]
    output = infer._run_inference_batches(_DummyInferModel(), loader)
    assert output["pred_points"].shape[0] == 3
    assert output["pred_scores"].shape[0] == 3
