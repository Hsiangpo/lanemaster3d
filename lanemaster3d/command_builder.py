from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class CommandBundle:
    env: Dict[str, str]
    cmd: List[str]
    cwd: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _build_env(gpus: int) -> Dict[str, str]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "4"
    env["WORLD_SIZE"] = str(gpus)
    return env


def build_train_bundle(config_path: str, work_dir: str, gpus: int) -> CommandBundle:
    if gpus > 1:
        cmd = [
            "torchrun",
            "--nproc_per_node",
            str(gpus),
            "tools/train.py",
            "--config",
            config_path,
            "--work-dir",
            work_dir,
            "--gpus",
            str(gpus),
            "--local-run",
            "--launcher",
            "ddp",
        ]
        return CommandBundle(env=_build_env(gpus), cmd=cmd, cwd=str(_repo_root()))
    cmd = [
        sys.executable,
        "tools/train.py",
        "--config",
        config_path,
        "--work-dir",
        work_dir,
        "--gpus",
        str(gpus),
        "--local-run",
        "--launcher",
        "none",
    ]
    return CommandBundle(env=_build_env(gpus), cmd=cmd, cwd=str(_repo_root()))


def build_eval_bundle(config_path: str, checkpoint: str, output_dir: str, gpus: int) -> CommandBundle:
    if gpus > 1:
        cmd = [
            "torchrun",
            "--nproc_per_node",
            str(gpus),
            "tools/eval.py",
            "--config",
            config_path,
            "--ckpt",
            checkpoint,
            "--out",
            output_dir,
            "--gpus",
            str(gpus),
            "--local-run",
            "--launcher",
            "ddp",
        ]
        return CommandBundle(env=_build_env(gpus), cmd=cmd, cwd=str(_repo_root()))
    cmd = [
        sys.executable,
        "tools/eval.py",
        "--config",
        config_path,
        "--ckpt",
        checkpoint,
        "--out",
        output_dir,
        "--gpus",
        str(gpus),
        "--local-run",
        "--launcher",
        "none",
    ]
    return CommandBundle(env=_build_env(gpus), cmd=cmd, cwd=str(_repo_root()))
