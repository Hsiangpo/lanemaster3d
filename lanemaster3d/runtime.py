from __future__ import annotations

import os
import shlex
import subprocess

from .command_builder import CommandBundle


def _render_env_delta(env: dict[str, str]) -> str:
    parts = []
    for key in sorted(env):
        old = os.environ.get(key)
        new = env[key]
        if old != new:
            parts.append(f"{key}={shlex.quote(new)}")
    return " ".join(parts)


def render_command(bundle: CommandBundle) -> str:
    env_part = _render_env_delta(bundle.env)
    cmd_part = " ".join(shlex.quote(item) for item in bundle.cmd)
    if env_part:
        return f"cd {shlex.quote(bundle.cwd)} && {env_part} {cmd_part}"
    return f"cd {shlex.quote(bundle.cwd)} && {cmd_part}"


def run_bundle(bundle: CommandBundle, dry_run: bool = False) -> int:
    printable = render_command(bundle)
    print(printable)
    if dry_run:
        return 0
    result = subprocess.run(
        bundle.cmd,
        cwd=bundle.cwd,
        env=bundle.env,
        check=False,
    )
    return result.returncode
