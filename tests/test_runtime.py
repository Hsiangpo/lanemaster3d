from __future__ import annotations

from pathlib import Path

from lanemaster3d.command_builder import CommandBundle
from lanemaster3d.runtime import render_command


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

