from __future__ import annotations

import inspect

from lanemaster3d.command_builder import build_eval_bundle, build_train_bundle
from lanemaster3d.engine.evaluator import evaluate_model


def test_build_train_bundle_single_gpu() -> None:
    bundle = build_train_bundle("configs/openlane/r50_960x720.py", "experiments/exp001", 1)
    assert bundle.cmd[0].endswith("python") or bundle.cmd[0].endswith("python.exe")
    assert bundle.cmd[1:3] == ["tools/train.py", "--config"]
    assert bundle.cmd[-2:] == ["--launcher", "none"]
    assert bundle.env["WORLD_SIZE"] == "1"


def test_build_train_bundle_ddp() -> None:
    bundle = build_train_bundle("configs/openlane/r50_960x720.py", "experiments/exp001", 2)
    assert bundle.cmd[:3] == ["torchrun", "--nproc_per_node", "2"]
    assert bundle.cmd[3] == "tools/train.py"
    assert bundle.cmd[-2:] == ["--launcher", "ddp"]
    assert bundle.env["WORLD_SIZE"] == "2"


def test_build_eval_bundle_single_gpu() -> None:
    bundle = build_eval_bundle("configs/openlane/r50_960x720.py", "best.pth", "outputs", 1)
    assert bundle.cmd[0].endswith("python") or bundle.cmd[0].endswith("python.exe")
    assert bundle.cmd[1:3] == ["tools/eval.py", "--config"]
    assert bundle.cmd[-2:] == ["--launcher", "none"]
    assert bundle.env["WORLD_SIZE"] == "1"


def test_build_eval_bundle_ddp() -> None:
    bundle = build_eval_bundle("configs/openlane/r50_960x720.py", "best.pth", "outputs", 2)
    assert bundle.cmd[:3] == ["torchrun", "--nproc_per_node", "2"]
    assert bundle.cmd[3] == "tools/eval.py"
    assert bundle.cmd[-2:] == ["--launcher", "ddp"]
    assert bundle.env["WORLD_SIZE"] == "2"


def test_evaluate_model_supports_launcher_parameter() -> None:
    params = inspect.signature(evaluate_model).parameters
    assert "launcher" in params
