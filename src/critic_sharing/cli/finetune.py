"""``critic-sharing finetune-leader`` — fine-tune a leader on follower return.

Warm-starts (or from-scratch trains) a leader policy on
``L_norm + alpha * F_norm`` where ``F_norm`` is a single frozen-follower
Monte-Carlo rollout from the leader's terminal handoff. Cross-scenario via
``FinetuneHandoffAdapter`` — see ``training/leader_follower_finetune.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from critic_sharing.cli._common import resolve_model
from critic_sharing.training.leader_follower_finetune import _ADAPTERS


def finetune_leader(
    scenario: str = typer.Option(
        ..., help=f"Scenario to fine-tune. Supported: {sorted(_ADAPTERS)}"
    ),
    leader_model: Optional[Path] = typer.Option(
        None, help="Warm-start leader checkpoint (.zip). Default: latest. "
                   "Never overwritten — output goes to a separate namespace."
    ),
    follower_model: Optional[Path] = typer.Option(
        None, help="Frozen follower checkpoint (.zip) rolled out for the "
                   "terminal reward. Default: latest."
    ),
    alpha: float = typer.Option(
        1.0, help="Weight on the normalized follower return."
    ),
    total_timesteps: int = typer.Option(
        500_000, help="Training steps. Use more with --from-scratch."
    ),
    n_envs: int = typer.Option(4, help="Parallel envs (SubprocVecEnv)."),
    seed: int = typer.Option(42, help="Random seed."),
    from_scratch: bool = typer.Option(
        False, "--from-scratch",
        help="Train a fresh leader policy instead of warm-starting. "
             "--leader-model is then used ONLY to calibrate leader_range."
    ),
    reset_exploration: bool = typer.Option(
        False, "--reset-exploration",
        help="On warm-start: reset policy.log_std to 0 (σ=1.0), clear the "
             "Adam optimizer state, and bump ent_coef. Counters the "
             "exploration-collapse trap where converged narrow log_std + "
             "stale Adam moments + ent_coef=0 prevent PPO from discovering "
             "new behaviours under the follower-aware reward. Only meaningful "
             "for warm-start (no-op with --from-scratch)."
    ),
    finetune_ent_coef: float = typer.Option(
        0.01, help="Entropy bonus to use when --reset-exploration is set."
    ),
    eval_condition_prob: float = typer.Option(
        0.0, help="Fraction of training resets drawn from the eval condition "
                  "distribution (household-only; requires --fixed-table). "
                  "0=pure training spawn (default for the baseline), 1.0=every "
                  "episode matches eval."
    ),
    fixed_table: Optional[str] = typer.Option(
        None, help="Household-only: eval table preset to upsample (e.g. "
                   "'gap-center'). Only used when --eval-condition-prob > 0."
    ),
    calibration_episodes: Optional[int] = typer.Option(
        None, help="Override naive episodes used to calibrate leader/follower "
                   "ranges (default: cost config's calibration_episodes)."
    ),
    save_path: Optional[Path] = typer.Option(
        None, "--save-path",
        help="Extra copy of the fine-tuned model. Must NOT be the warm-start "
             "path (guarded). The run always lands in artifacts/"
             "<scenario>_leader_finetune/ regardless.",
    ),
    wandb_name: Optional[str] = typer.Option(None, help="Override wandb run name."),
) -> None:
    """Fine-tune a leader policy on the follower's rolled-out return."""
    if scenario not in _ADAPTERS:
        raise typer.BadParameter(
            f"Unsupported scenario {scenario!r}. Known: {sorted(_ADAPTERS)}"
        )

    leader_path = resolve_model(leader_model, scenario, "leader")
    follower_path = resolve_model(follower_model, scenario, "follower")
    typer.echo(f"Scenario          : {scenario}")
    typer.echo(f"Warm-start leader : {leader_path}"
               + ("  (calibration only — --from-scratch)" if from_scratch else ""))
    typer.echo(f"Frozen follower   : {follower_path}")

    from critic_sharing.training.leader_follower_finetune import finetune_agent

    saved = finetune_agent(
        scenario=scenario,
        leader_model_path=str(leader_path),
        follower_model_path=str(follower_path),
        total_timesteps=total_timesteps,
        n_envs=n_envs,
        seed=seed,
        alpha=alpha,
        from_scratch=from_scratch,
        reset_exploration=reset_exploration,
        finetune_ent_coef=finetune_ent_coef,
        eval_condition_prob=eval_condition_prob,
        fixed_table=fixed_table,
        calibration_episodes=calibration_episodes,
        save_path=str(save_path) if save_path is not None else None,
        wandb_name=wandb_name,
    )
    typer.echo(f"Fine-tuned leader saved to: {saved}")


__all__ = ["finetune_leader"]
