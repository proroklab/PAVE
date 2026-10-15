"""``critic-sharing competence``: standalone success rate of one policy."""

from __future__ import annotations

import json
from pathlib import Path

import typer


def competence_cli(
    scenario: str = typer.Option(..., help="Scenario name, e.g. warehouse."),
    role: str = typer.Option(..., help="leader or follower."),
    model: Path = typer.Option(..., exists=True, dir_okay=False,
                               help="PPO checkpoint (.zip)."),
    train_seed: int = typer.Option(..., help="Seed the policy was trained "
                                   "with; reset seeds are train_seed + 1e6 + k."),
    episodes: int = typer.Option(100, help="Number of episodes."),
    as_json: bool = typer.Option(False, "--json",
                                 help="Print one JSON object instead of the rate."),
) -> None:
    """Roll the policy alone, deterministically, in its training env and
    report the fraction of episodes in which its success event fires
    (see critic_sharing.training.competence)."""
    from critic_sharing.training.competence import competence

    rate = competence(scenario, role, model, train_seed=train_seed,
                      n_episodes=episodes)
    if as_json:
        typer.echo(json.dumps({"scenario": scenario, "role": role,
                               "model": str(model), "train_seed": train_seed,
                               "episodes": episodes, "success": rate}))
    else:
        typer.echo(f"{rate}")
