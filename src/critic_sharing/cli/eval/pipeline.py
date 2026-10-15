"""Shared eval pipeline: adapter ABC + calibrate / spec-build / normalize.

Replaces four near-identical per-scenario orchestration functions with
one ``run_eval_pipeline(adapter, ...)`` + four small adapter classes.

Each scenario contributes an :class:`EvalScenarioAdapter` subclass that
knows three scenario-specific things:

1. How to construct the calibration env pair (so the unguided pre-pass
   can compute leader_range / follower_range).
2. How to sample per-episode initial conditions (obstacle_pos, table
   pose, etc.) — or ``[None] * n`` when no init conditions are needed.
3. How to build an :class:`EpisodeSpec` from a (mode, episode, seeds,
   condition, paths, alpha) tuple — basically just packs the
   scenario-specific fields onto the spec.

Optional adapter hooks let a scenario compute
``total_return_weighted = L_norm + alpha * trust * F_norm`` in the
post-merge step. Scenarios without a well-defined handoff state can
omit the hook and the column is simply absent in their output.

The pipeline writes the same column set across all four scenarios:

* ``leader_return``, ``follower_return``           (raw, always present)
* ``leader_return_norm`` / ``follower_return_norm`` (when calibration
                                                    ranges are non-None)
* ``ood_at_handoff``                                (when both
                                                    ``vae_path`` and
                                                    ``handoff_obs_for_ood``
                                                    are provided)
* ``total_return_weighted``                         (when ranges +
                                                    handoff hook are
                                                    both available)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type

import numpy as np
import typer

from critic_sharing.utils.parallel import (
    EpisodeSpec,
    EpisodeWorker,
    run_episodes_parallel,
)

from ._common import EvalMode


# ---------------------------------------------------------------------------
# Adapter ABC
# ---------------------------------------------------------------------------


class EvalScenarioAdapter(ABC):
    """Per-scenario hooks consumed by :func:`run_eval_pipeline`.

    Subclasses are usually constructed once per ``_compare_*`` call with
    the scenario's caller-side knobs (e.g. ``scenario_name`` for the
    warehouse-family variants, ``follower_max_steps`` for warehouse's
    eval-time horizon override, ``obstacle_distance`` / ``fixed_obstacle``
    for OA's init-condition sampling, ``fixed_table`` for household).
    """

    #: Used as the cost.yaml / planner.yaml lookup key.
    scenario_name: str = ""

    #: Concrete :class:`EpisodeWorker` subclass that runs episodes.
    worker_cls: Type[EpisodeWorker] = EpisodeWorker

    # ------------------------------------------------------------------
    # Required hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def build_calibration(self) -> Tuple[Any, Any, Any, Any]:
        """Return ``(leader_env, follower_env, leader_model, follower_model)``.

        Used by the calibration pre-pass: both envs are driven by their
        respective PPO models for ``n_episodes`` unguided rollouts and
        the returns are fed into ``calibrate_range``. The pipeline
        closes the envs when done.
        """

    @abstractmethod
    def sample_conditions(
        self, rng: np.random.Generator, n_episodes: int,
    ) -> List[Any]:
        """Return ``n_episodes`` scenario-specific init conditions.

        Each entry is whatever ``make_spec`` expects; if a scenario has
        no init-condition concept (warehouse, mobile_arm), return
        ``[None] * n_episodes``.
        """

    @abstractmethod
    def make_spec(
        self,
        *,
        episode: int,
        mode: str,
        leader_seed: int,
        follower_seed: int,
        planner_seed: int,
        condition: Any,
        paths: Dict[str, Optional[str]],
        alpha: Optional[float],
        record: bool,
    ) -> EpisodeSpec:
        """Build one :class:`EpisodeSpec` for the given cell."""

    @abstractmethod
    def worker_kwargs(
        self,
        *,
        leader_range: Optional[Tuple[float, float]],
        follower_range: Optional[Tuple[float, float]],
        paths: Dict[str, Optional[str]],
        alpha: Optional[float],
    ) -> Dict[str, Any]:
        """Build the ``worker_kwargs`` dict for :func:`run_episodes_parallel`.

        Should NOT include ``"mode"`` — that's grouped per-mode by the
        pool runner. Should NOT include the obsolete ``"seed"`` key —
        per-episode planner seed flows through :class:`EpisodeSpec`.
        """

    # ------------------------------------------------------------------
    # Optional hooks (defaults are conservative)
    # ------------------------------------------------------------------

    def supported_modes(self) -> set:
        """Modes this scenario will accept; mismatch is a hard error."""
        return {
            EvalMode.unguided,
            EvalMode.value_only,
            EvalMode.value_and_ood,
            EvalMode.uniform,
            EvalMode.sabotage,
        }

    def calibration_modes(self) -> set:
        """Skip calibration when requested modes don't intersect this."""
        return {
            EvalMode.value_only,
            EvalMode.value_and_ood,
            EvalMode.uniform,
            EvalMode.sabotage,
        }

    def build_ood_fn_for_post_processing(
        self, vae_path: Optional[str],
    ) -> Optional[Callable[[np.ndarray], np.ndarray]]:
        """Return ``ood_fn(leader_obs_batch) -> scores`` for trust-at-handoff.

        Default: ``None`` (skip the trust-weighted column).
        """
        return None

    def handoff_obs_for_ood(self, row: Dict[str, Any]) -> Optional[np.ndarray]:
        """Build a leader-shaped obs at the realized handoff state.

        Must return a 2-D array of shape ``(1, leader_obs_dim)`` so it
        feeds straight into the ``ood_fn`` returned by
        :meth:`build_ood_fn_for_post_processing`. Default ``None`` skips
        the trust computation.
        """
        return None


# ---------------------------------------------------------------------------
# Calibration pre-pass
# ---------------------------------------------------------------------------


def calibrate_ranges(
    adapter: EvalScenarioAdapter,
    modes: Sequence[EvalMode],
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """Return ``(leader_range, follower_range)`` per the adapter's contract.

    Both ranges are full-episode discounted naive returns; pool selection
    (successful-only vs all-episodes) is governed by
    ``cost.yaml::calibrate_on_success_only``. Returns ``(None, None)``
    when ``normalize_returns=false`` or no CEM mode was requested.
    """
    from critic_sharing.config.resolver import load_cost_config
    from critic_sharing.utils.calibration import calibrate_range

    cost_cfg = load_cost_config(adapter.scenario_name)
    if not getattr(cost_cfg, "normalize_returns", True):
        typer.echo(
            "{s}: normalize_returns=false; skipping calibration.".format(
                s=adapter.scenario_name,
            ),
            err=True,
        )
        return None, None
    if not (set(modes) & adapter.calibration_modes()):
        return None, None

    leader_gamma = float(getattr(cost_cfg, "leader_gamma", 0.99))
    n_eps = int(getattr(cost_cfg, "calibration_episodes", 30))
    success_only = bool(getattr(cost_cfg, "calibrate_on_success_only", True))
    pool_label = "successful" if success_only else "all"

    leader_env, follower_env, leader_model, follower_model = adapter.build_calibration()

    typer.echo(
        "  {s}  calibrating leader_range ({pool}) over {n} naive episodes...".format(
            s=adapter.scenario_name, pool=pool_label, n=n_eps,
        ),
        err=True,
    )
    leader_range = calibrate_range(
        leader_env, leader_model, n_episodes=n_eps, gamma=leader_gamma,
        success_only=success_only,
    )
    typer.echo(
        "    leader_range = ({lo:.3f}, {hi:.3f})".format(
            lo=leader_range[0], hi=leader_range[1],
        ),
        err=True,
    )

    typer.echo(
        "  {s}  calibrating follower_range ({pool}) over {n} naive episodes...".format(
            s=adapter.scenario_name, pool=pool_label, n=n_eps,
        ),
        err=True,
    )
    follower_range = calibrate_range(
        follower_env, follower_model, n_episodes=n_eps, gamma=leader_gamma,
        success_only=success_only,
    )
    typer.echo(
        "    follower_range = ({lo:.3f}, {hi:.3f})".format(
            lo=follower_range[0], hi=follower_range[1],
        ),
        err=True,
    )

    leader_env.close()
    follower_env.close()
    return leader_range, follower_range


# ---------------------------------------------------------------------------
# Post-merge normalization
# ---------------------------------------------------------------------------


def normalize_result_columns(
    results: List[Dict[str, Any]],
    *,
    adapter: EvalScenarioAdapter,
    leader_range: Optional[Tuple[float, float]],
    follower_range: Optional[Tuple[float, float]],
    alpha: float,
    ood_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> List[Dict[str, Any]]:
    """In-place: add ``leader_return_norm`` / ``follower_return_norm`` /
    ``ood_at_handoff`` / ``total_return_weighted`` columns.

    Skipped columns when the necessary inputs are missing:
    * ranges ``None`` → ``leader_return_norm`` / ``follower_return_norm``
      / ``total_return_weighted`` all absent.
    * ``ood_fn`` ``None`` or ``handoff_obs_for_ood`` returns ``None`` →
      ``ood_at_handoff`` absent and ``trust = 1`` for
      ``total_return_weighted`` (when ranges are still present).
    """
    from critic_sharing.config.resolver import load_cost_config

    cost_cfg = load_cost_config(adapter.scenario_name)
    clip_norm = bool(getattr(cost_cfg, "clip_normalized", True))

    def _norm(value: float, rng: Optional[Tuple[float, float]]) -> Optional[float]:
        if rng is None:
            return None
        lo, hi = rng
        span = max(float(hi) - float(lo), 1e-8)
        scaled = (float(value) - float(lo)) / span
        if clip_norm:
            scaled = float(np.clip(scaled, 0.0, 1.0))
        return float(scaled)

    for row in results:
        ln = _norm(row.get("leader_return", 0.0), leader_range)
        fn = _norm(row.get("follower_return", 0.0), follower_range)
        if ln is not None:
            row["leader_return_norm"] = ln
        if fn is not None:
            row["follower_return_norm"] = fn
        if ln is None or fn is None:
            continue

        trust = 1.0
        if ood_fn is not None:
            handoff_obs = adapter.handoff_obs_for_ood(row)
            if handoff_obs is not None:
                ood_score = float(
                    np.asarray(ood_fn(handoff_obs), dtype=np.float32).reshape(-1)[0]
                )
                trust = float(np.clip(1.0 - ood_score, 0.0, 1.0))
                row["ood_at_handoff"] = ood_score

        row["total_return_weighted"] = float(ln + float(alpha) * trust * fn)

    return results


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def run_eval_pipeline(
    adapter: EvalScenarioAdapter,
    *,
    leader_model: Path,
    follower_model: Path,
    vae: Optional[Path],
    alpha: Optional[float],
    modes: Sequence[EvalMode],
    leader_seeds: Sequence[int],
    follower_seeds: Sequence[int],
    planner_seeds: Sequence[int],
    n_follower_samples: int = 1,
    workers: int = 1,
    record: bool = True,
) -> List[Dict[str, Any]]:
    """End-to-end eval orchestration: calibrate → sample → dispatch → normalize.

    Identical control flow across all four in-scope scenarios; everything
    that differs lives in the ``adapter``.
    """
    # 1. Validate requested modes.
    supported = adapter.supported_modes()
    requested = set(modes)
    if not requested.issubset(supported):
        unsupported = requested - supported
        typer.echo(
            "{s}: modes not implemented yet: {u}. Supported: {S}.".format(
                s=adapter.scenario_name,
                u=", ".join(m.value for m in unsupported),
                S=", ".join(m.value for m in supported),
            ),
            err=True,
        )
        raise typer.Exit(code=1)
    if n_follower_samples != 1:
        typer.echo(
            "{s}: --follower-samples > 1 not implemented; using 1.".format(
                s=adapter.scenario_name,
            ),
            err=True,
        )

    # 2. Calibrate (no-op when normalize_returns=false or only unguided).
    leader_range, follower_range = calibrate_ranges(adapter, modes)

    # 3. Resolve alpha from cost.yaml if not explicitly passed.
    if alpha is None:
        from critic_sharing.config.resolver import load_cost_config

        alpha = float(getattr(
            load_cost_config(adapter.scenario_name), "alpha", 1.0,
        ))
    if vae is not None:
        typer.echo(
            "  {s}  VAE/OOD trust-weighting on: vae={p} alpha={l:.4f}".format(
                s=adapter.scenario_name, p=str(vae), l=float(alpha),
            ),
            err=True,
        )

    # 4. Sample per-episode init conditions (adapter-defined shape).
    init_rng = np.random.default_rng(leader_seeds[0] if leader_seeds else 0)
    conditions = adapter.sample_conditions(init_rng, len(leader_seeds))

    # 5. Build the (mode × episode) spec list.
    paths: Dict[str, Optional[str]] = {
        "leader_model": str(leader_model),
        "follower_model": str(follower_model),
        "vae": str(vae) if vae is not None else None,
    }
    specs: List[EpisodeSpec] = []
    for mode in modes:
        for ep, (ls, fs, ps, cond) in enumerate(
            zip(leader_seeds, follower_seeds, planner_seeds, conditions)
        ):
            specs.append(
                adapter.make_spec(
                    episode=ep,
                    mode=mode.value,
                    leader_seed=int(ls),
                    follower_seed=int(fs),
                    planner_seed=int(ps),
                    condition=cond,
                    paths=paths,
                    alpha=alpha,
                    record=bool(record),
                )
            )

    typer.echo(
        "  Dispatching {n} episode-mode pairs ({w} workers)...".format(
            n=len(specs), w=workers,
        )
    )

    # 6. Dispatch.
    raw_results = run_episodes_parallel(
        specs=specs,
        worker_cls=adapter.worker_cls,
        worker_kwargs=adapter.worker_kwargs(
            leader_range=leader_range,
            follower_range=follower_range,
            paths=paths,
            alpha=alpha,
        ),
        max_workers=workers,
    )

    # 7. Post-merge normalization + trust-at-handoff column.
    ood_fn = adapter.build_ood_fn_for_post_processing(paths.get("vae"))
    normalize_result_columns(
        raw_results,
        adapter=adapter,
        leader_range=leader_range,
        follower_range=follower_range,
        alpha=float(alpha),
        ood_fn=ood_fn,
    )
    return raw_results


__all__ = [
    "EvalScenarioAdapter",
    "calibrate_ranges",
    "normalize_result_columns",
    "run_eval_pipeline",
]
