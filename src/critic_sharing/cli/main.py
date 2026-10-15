"""Entry point for the ``critic-sharing`` CLI.

Usage
-----
    critic-sharing train   leader   --scenario obstacle_arena [overrides]
    critic-sharing train   follower --scenario obstacle_arena [overrides]
    critic-sharing visualize value  --scenario obstacle_arena --model path
    critic-sharing visualize ood    --scenario obstacle_arena --model path
    critic-sharing eval             --scenario obstacle_arena

Install via ``pip install -e .`` then run ``critic-sharing --help``.
"""

from __future__ import annotations

# Silence noisy third-party warnings before importing transitive deps that
# emit them at import time (gym deprecation, torch CUDA driver mismatch).
import os
import warnings

os.environ.setdefault("PYTHONWARNINGS", "ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# gym 0.26+ prints its deprecation banner via stderr at import (not via
# warnings), so filter that out by capturing stderr while gym/SB3 import.
import sys as _sys

_orig_stderr = _sys.stderr


class _FilteredStderr:
    def __init__(self, target):
        self._t = target

    def write(self, s):
        if "Gym has been unmaintained" in s or "CUDA initialization" in s:
            return
        if "Please upgrade to Gymnasium" in s or "Users of this version of Gym" in s:
            return
        if "See the migration guide" in s:
            return
        self._t.write(s)

    def flush(self):
        self._t.flush()

    def __getattr__(self, name):
        return getattr(self._t, name)


_sys.stderr = _FilteredStderr(_orig_stderr)

import typer

from critic_sharing.cli.deploy import app as deploy_app
from critic_sharing.cli.train import app as train_app
from critic_sharing.cli.visualize import app as visualize_app
from critic_sharing.cli.eval import eval_compare
from critic_sharing.cli.finetune import finetune_leader
from critic_sharing.cli.competence import competence_cli

app = typer.Typer(
    name="critic-sharing",
    help="Leader/follower RL experiments with critic sharing.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)

app.add_typer(train_app, name="train", help="Train leader or follower agents.")
app.add_typer(visualize_app, name="visualize", help="Visualise value functions and OOD metrics.")
app.command(
    name="eval",
    help="Evaluate and compare strategies (ablations, baselines) for a scenario.",
)(eval_compare)
app.command(
    name="finetune-leader",
    help="Fine-tune a leader on the follower's rolled-out return (warm-start + PPO).",
)(finetune_leader)
app.command(
    name="competence",
    help="Standalone success rate of one policy in its training env.",
)(competence_cli)
app.add_typer(deploy_app, name="deploy", help="Deploy trained policies on real robots via ROS2.")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
