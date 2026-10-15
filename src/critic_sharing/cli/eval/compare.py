"""``critic-sharing eval`` — multi-mode comparison entry point.

This module used to expose ``compare`` as a Typer sub-command under an
``eval`` Typer app (``critic-sharing eval compare ...``). The
``compare`` sub-namespace has been removed; the function is now
registered directly in ``main.py`` as ``critic-sharing eval``.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import typer

from ._common import (
    EvalMode,
    Scenario,
    _print_summary_table,
    _resolve_vae,
    resolve_model,
)
from ._plotting import _plot_compare
from ._animation import _save_episode_mp4s
from .obstacle_arena import _compare_oa
from .mobile_arm import _compare_ma
from .household import _compare_hh
from .warehouse import (
    _compare_wh,
    _plot_warehouse_extras,
    _plot_wh_policy_follows_v,
    _plot_wh_v_argmax_agreement,
    _plot_wh_v_gain_over_baseline,
    _save_wh_v_at_gaps_plots,
)


# ---------------------------------------------------------------------------
# eval command (registered directly under the top-level CLI as
# ``critic-sharing eval``; no sub-namespace).
# ---------------------------------------------------------------------------


def eval_compare(
    scenario: Scenario = typer.Option(..., help="Scenario to compare."),
    leader_model: Optional[Path] = typer.Option(
        None, help="Leader model (.zip). Defaults to latest artifact."
    ),
    follower_model: Optional[Path] = typer.Option(
        None, help="Follower model (.zip). Defaults to latest artifact."
    ),
    vae: Optional[Path] = typer.Option(
        None,
        help="VAE checkpoint (.pt) — enables 'full' mode for obstacle_arena.",
    ),
    alpha: Optional[float] = typer.Option(
        None,
        help=(
            "Value factor on V_F: utility = r_leader_norm + alpha * "
            "trust * r_follower_norm. Applies regardless of whether OOD "
            "is wired in. Defaults to OnlineAdaptationCostConfig.alpha."
        ),
    ),
    episodes: int = typer.Option(50, help="Episodes per mode."),
    seed: int = typer.Option(0, help="Base random seed for episode generation."),
    follower_samples: int = typer.Option(
        1,
        help="Number of follower starting positions to average over per episode.",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        help="Output directory for CSV + figures. Default: ./eval_<scenario>.",
    ),
    modes: Optional[str] = typer.Option(
        None,
        help=(
            "Comma-separated list of modes to compare "
            "(e.g. 'unguided,value-only,value-and-ood'). "
            "Default: unguided,value-only."
        ),
    ),
    obstacle_distance: Optional[float] = typer.Option(
        None,
        help=(
            "obstacle_arena only: max initial distance from obstacle centre to "
            "the follower goal. Constrains scenarios to cases where the obstacle "
            "is near the goal. Default: no constraint (full arena)."
        ),
    ),
    fixed_obstacle: Optional[str] = typer.Option(
        None,
        help=(
            "obstacle_arena only: fix obstacle position for all episodes. "
            "Use 'gap-center' to place it midway between gap and goal, "
            "or 'x,y' for explicit coordinates (e.g. '0.0,-2.5')."
        ),
    ),
    workers: int = typer.Option(
        1,
        help=(
            "Number of parallel workers for episode execution. "
            "Each worker loads models and builds a controller once, then "
            "processes its share of episodes. Default: 1 (sequential)."
        ),
    ),
    no_trajectory: bool = typer.Option(
        False,
        "--no-trajectory",
        help=(
            "warehouse only: skip trajectory recording for each episode. "
            "Disables the per-episode MP4 animations and the arena panel "
            "of the V-vs-actual diagnostic plot. Speeds up large eval "
            "runs where you only want aggregate metrics."
        ),
    ),
    no_figures: bool = typer.Option(
        False,
        "--no-figures",
        help=(
            "Skip ALL figure generation (per-episode MP4s, summary box "
            "plots, scenario-specific diagnostic figures). The CSV is "
            "still written. Useful for paper-experiment headline runs "
            "where ``analyze.py`` regenerates publication figures from "
            "the CSV separately."
        ),
    ),
    follower_max_steps: Optional[int] = typer.Option(
        None,
        help=(
            "warehouse only: override the follower env's ``max_steps`` "
            "at eval time (and during calibration). Use to make the "
            "scenario harder without retraining: a smaller horizon "
            "truncates marginal navigation paths so naive S_F drops, "
            "exposing more headroom for V_F+OOD to fix. The follower "
            "policy itself does not see the horizon, so this is a clean "
            "exogenous knob. Defaults to the value in "
            "configs/warehouse/follower_env.yaml (300)."
        ),
    ),
) -> None:
    """Compare unguided / value-only / full leader guidance across N shared episodes.

    The follower always acts with its naive trained policy.  The same N random
    seeds are reused for every mode so episode difficulty is held constant.

    Starting positions are randomised independently each episode (e.g.
    obstacle_arena seeds the pusher and obstacle start positions per episode).

    Prints a summary table (mean +/- std per metric per mode) and saves:

      compare_<scenario>.csv                     per-episode rows for all modes
      compare_<scenario>_returns.png             box plots: leader / follower / total return
      compare_<scenario>_metrics.png             box plots: steps / cells / goal-reach rate
      compare_<scenario>_follower_improvement.png distribution of relative improvement
      top_episodes_<scenario>_<mode>.png         naive vs guided rollout for top episodes
    """
    from critic_sharing.config.resolver import load_run_config
    from critic_sharing.utils.experiment import init_experiment

    leader_model = resolve_model(leader_model, scenario.value, "leader")
    follower_model = resolve_model(follower_model, scenario.value, "follower")
    vae = _resolve_vae(vae, follower_model, scenario.value)

    # Load resolved config for logging
    resolved_cfg = load_run_config(scenario.value)

    # Build artifact references
    artifact_paths = {
        "leader_model": str(leader_model),
        "follower_model": str(follower_model),
    }
    if vae is not None:
        artifact_paths["vae"] = str(vae)

    # Determine experiment name
    exp_name = "compare-{s}".format(s=scenario.value)

    # Set up experiment directory (or use explicit --output-dir)
    if output_dir is not None:
        out_dir = output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        exp_paths = None
    else:
        exp_paths = init_experiment(
            name=exp_name,
            resolved_config=resolved_cfg,
            artifact_paths=artifact_paths,
            extra_metadata={
                "episodes": episodes,
                "seed": seed,
                "follower_samples": follower_samples,
                "modes": modes,
                "obstacle_distance": obstacle_distance,
                "fixed_obstacle": fixed_obstacle,
            },
        )
        out_dir = exp_paths.outputs_dir
        typer.echo("Experiment dir: {p}".format(p=exp_paths.experiment_dir))

    # Seed tree: a single master ``seed`` deterministically generates a
    # disjoint tree of sub-seeds via :class:`numpy.random.SeedSequence`,
    # exposed by :class:`critic_sharing.utils.seeding.SeedManager`.
    # ``eval_episode_seeds(n)`` returns ``(leader_seeds, follower_seeds,
    # planner_seeds)`` where each entry is reproducibly derived from the
    # master and mutually disjoint across (episode, role) triples. Two
    # runs with the SAME ``--seed`` produce bit-identical initial
    # conditions AND identical CEM-sample streams per episode; two runs
    # with DIFFERENT ``--seed`` produce statistically independent ones,
    # with no risk of integer-counter overlap.
    from critic_sharing.utils.seeding import SeedManager

    seed_mgr = SeedManager(master=int(seed))
    leader_seeds, follower_seeds, planner_seeds = seed_mgr.eval_episode_seeds(int(episodes))

    if modes is not None:
        # Parse user-specified modes
        mode_list: list[EvalMode] = []
        for m in modes.split(","):
            m = m.strip()
            try:
                mode_list.append(EvalMode(m))
            except ValueError:
                typer.echo(
                    "Unknown mode '{m}'. Valid: {valid}".format(
                        m=m,
                        valid=", ".join(e.value for e in EvalMode),
                    ),
                    err=True,
                )
                raise typer.Exit(code=1)
        run_modes = mode_list
    else:
        run_modes = [EvalMode.unguided, EvalMode.value_only]

    msg = "Comparing {n} modes \u00d7 {e} episodes on {s}...".format(
        n=len(run_modes), e=episodes, s=scenario.value
    )
    if follower_samples > 1:
        msg += " ({k} follower samples)".format(k=follower_samples)
    typer.echo(msg)

    if scenario == Scenario.obstacle_arena:
        if no_trajectory:
            typer.echo(
                "obstacle_arena: --no-trajectory set; skipping trajectory recording. "
                "Episode MP4s will not be saved.",
                err=True,
            )
        all_results = _compare_oa(
            leader_model=leader_model,
            follower_model=follower_model,
            vae=vae,
            alpha=alpha,
            modes=run_modes,
            leader_seeds=leader_seeds,
            follower_seeds=follower_seeds,
            planner_seeds=planner_seeds,
            n_follower_samples=follower_samples,
            obstacle_distance=obstacle_distance,
            fixed_obstacle=fixed_obstacle,
            workers=workers,
            record=not no_trajectory,
        )

    elif scenario in (
        Scenario.warehouse,
        Scenario.warehouse_pickup_blocked,
        Scenario.warehouse_pickup_unseen,
    ):
        if no_trajectory:
            typer.echo(
                "{s}: --no-trajectory set; skipping trajectory recording. "
                "Episode MP4s and arena panels will be empty.".format(
                    s=scenario.value,
                ),
                err=True,
            )
        # Auto-discover VAE if not explicitly given (looks for
        # follower_vae.pt next to follower model).
        wh_vae = _resolve_vae(vae, follower_model, scenario.value)
        all_results = _compare_wh(
            leader_model=leader_model,
            follower_model=follower_model,
            modes=run_modes,
            leader_seeds=leader_seeds,
            follower_seeds=follower_seeds,
            planner_seeds=planner_seeds,
            n_follower_samples=follower_samples,
            workers=workers,
            record=not no_trajectory,
            vae_path=str(wh_vae) if wh_vae is not None else None,
            alpha=alpha,
            follower_max_steps=follower_max_steps,
            scenario_name=scenario.value,
        )

    elif scenario == Scenario.mobile_arm:
        if no_trajectory:
            typer.echo(
                "mobile_arm: --no-trajectory set; skipping trajectory recording. "
                "Episode MP4s will not be saved.",
                err=True,
            )
        ma_vae = _resolve_vae(vae, follower_model, scenario.value)
        all_results = _compare_ma(
            leader_model=leader_model,
            follower_model=follower_model,
            modes=run_modes,
            leader_seeds=leader_seeds,
            follower_seeds=follower_seeds,
            planner_seeds=planner_seeds,
            n_follower_samples=follower_samples,
            workers=workers,
            record=not no_trajectory,
            vae_path=str(ma_vae) if ma_vae is not None else None,
            alpha=alpha,
        )

    elif scenario == Scenario.household:
        if no_trajectory:
            typer.echo(
                "household: --no-trajectory set; skipping trajectory recording. "
                "Episode MP4s will not be saved.",
                err=True,
            )
        hh_vae = _resolve_vae(vae, follower_model, scenario.value)
        all_results = _compare_hh(
            leader_model=leader_model,
            follower_model=follower_model,
            vae=hh_vae,
            alpha=alpha,
            modes=run_modes,
            leader_seeds=leader_seeds,
            follower_seeds=follower_seeds,
            planner_seeds=planner_seeds,
            n_follower_samples=follower_samples,
            workers=workers,
            record=not no_trajectory,
            fixed_table=fixed_obstacle,
        )

    else:
        typer.echo("Unknown scenario: {s}".format(s=scenario.value), err=True)
        raise typer.Exit(code=1)

    _print_summary_table(all_results, scenario)

    # Strip trajectory/grid data for CSV (not serializable)
    _TRAJ_KEYS = {
        "leader_traj",
        "follower_traj",
        "goal_pos",
        "leader_grids",
        "follower_grids",
        "leader_colored",
        "final_colored",
        "leader_path",
        "follower_path",
        "leader_frames",
        "follower_frames",
    }
    csv_results = [{k: v for k, v in r.items() if k not in _TRAJ_KEYS} for r in all_results]
    # Workers finish in arbitrary order; sort so identical results give a
    # byte-identical file. Columns keep their first-seen order.
    fieldnames = list(dict.fromkeys(k for r in csv_results for k in r))
    csv_results.sort(key=lambda r: (str(r.get("mode", "")), int(r.get("episode", 0))))
    csv_path = out_dir / "compare_{s}.csv".format(s=scenario.value)
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_results)
    typer.echo("CSV  \u2192 {p}".format(p=csv_path))

    if no_figures:
        typer.echo(
            "--no-figures: skipping all plot generation. CSV-only output.",
            err=True,
        )
    else:
        _plot_compare(all_results, scenario, out_dir)

        # Scenario-specific extra figures.
        if scenario in (
            Scenario.warehouse,
            Scenario.warehouse_pickup_blocked,
            Scenario.warehouse_pickup_unseen,
        ):
            _plot_warehouse_extras(all_results, out_dir)
            _plot_wh_v_argmax_agreement(all_results, out_dir)
            _plot_wh_policy_follows_v(all_results, out_dir)
            _plot_wh_v_gain_over_baseline(all_results, out_dir)
            _save_wh_v_at_gaps_plots(
                all_results, out_dir, max_workers=workers,
                scenario_name=scenario.value,
            )

        # --- episode MP4 animations ---
        # `workers` (the rollout-pool flag) is reused for the figure pool —
        # see the design note in utils.parallel.run_figures_parallel.
        _save_episode_mp4s(all_results, out_dir, scenario, max_workers=workers)
