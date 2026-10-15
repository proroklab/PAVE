"""Calling the ``critic-sharing`` CLI from evaluation scripts.

Scripts run policies only through the CLI (never by importing training or
eval internals), as sub-processes started in the repo root with
repo-relative paths, so what gets recorded inside outputs does not depend on
the machine.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from evaluations._common.paths import REPO


def critic_sharing_cli() -> str:
    """The ``critic-sharing`` executable of the running interpreter's
    environment, so scripts work without an activated venv."""
    exe = Path(sys.executable).parent / "critic-sharing"
    if not exe.is_file():
        raise SystemExit(f"{exe} not found; install the package (uv sync).")
    return str(exe)


def rel(path: Path) -> str:
    """``path`` relative to the repo root when it lies inside it."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def run_cli(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run ``critic-sharing <args>`` in the repo root."""
    return subprocess.run([critic_sharing_cli(), *args], cwd=REPO, **kwargs)
