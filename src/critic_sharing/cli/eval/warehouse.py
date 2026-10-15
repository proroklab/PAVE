"""Warehouse-specific eval logic for ``critic-sharing eval --scenario warehouse``.

Mode mapping (matches the shared :class:`EvalMode` vocabulary used by the
other scenarios):

* :attr:`EvalMode.unguided` — leader's PPO policy alone, no planner overlay.
* :attr:`EvalMode.value_only` — leader's PPO policy + CEM overlay maximising
  ``r_leader + V_follower(drop_pos)`` (cooperative).
* :attr:`EvalMode.sabotage` — leader's PPO policy + CEM overlay maximising
  ``r_leader - V_follower(drop_pos)`` (adversarial: leader still drops in a
  gap to grab its own goal_reward but picks the gap that hurts the follower
  most).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np
import typer

from critic_sharing.utils.parallel import (
    EpisodeSpec,
    EpisodeWorker,
    FigureTask,
    FigureWorker,
    run_episodes_parallel,
    run_figures_parallel,
)

from critic_sharing.planning.factory import MODE_SETTINGS

from ._common import EvalMode
from .pipeline import (
    EvalScenarioAdapter,
    calibrate_ranges,
    normalize_result_columns,
)


# ---------------------------------------------------------------------------
# CEM controller
# ---------------------------------------------------------------------------

def _build_wh_controller(
    leader_cfg,
    leader_model,
    value_fn: Callable[[np.ndarray], np.ndarray],
    seed: int,
    utility_type: str = "sum",
    leader_range: Optional[Tuple[float, float]] = None,
    follower_range: Optional[Tuple[float, float]] = None,
    vae_path: Optional[str] = None,
    alpha: float = 1.0,
    prior_kind: str = "policy",  # "policy" | "uniform"
    scenario_name: str = "warehouse",
):
    """Build the warehouse CEM controller.

    Returns ``(controller, cost, terminal_value_mode)``; see
    :func:`critic_sharing.planning.factory.build_cem_controller`. With a
    ``vae_path`` the cost applies the OOD trust factor.
    """
    import gymnasium as gym

    from critic_sharing.planning.factory import build_cem_controller
    from critic_sharing.scenarios.warehouse.leader_env import (
        warehouse_leader_dynamics_batch,
    )

    ood_metric = None
    if vae_path is not None:
        from critic_sharing.scenarios.warehouse.ood_query import (
            build_warehouse_ood_metric,
        )

        ood_metric = build_warehouse_ood_metric(vae_path)

    def dynamics_fn(obs_batch, act_batch):
        return warehouse_leader_dynamics_batch(obs_batch, act_batch, leader_cfg)

    action_space = gym.spaces.Box(
        low=np.array([
            -leader_cfg.max_speed,
            -leader_cfg.max_speed,
            -leader_cfg.max_yaw_rate,
        ], dtype=np.float32),
        high=np.array([
            leader_cfg.max_speed,
            leader_cfg.max_speed,
            leader_cfg.max_yaw_rate,
        ], dtype=np.float32),
        dtype=np.float32,
    )
    return build_cem_controller(
        scenario_name,
        leader_model=leader_model,
        follower_value_fn=value_fn,
        dynamics_fn=dynamics_fn,
        action_space=action_space,
        seed=seed,
        utility_type=utility_type,
        leader_range=leader_range,
        follower_range=follower_range,
        ood_metric=ood_metric,
        alpha=alpha,
        prior_kind=prior_kind,
    )


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

_DEFAULT_FOLLOWER_GAMMA = 0.99


def _rollout_follower_discounted_return(
    follower_env,
    follower_model,
    follower_seed: int,
    spawn_pos: np.ndarray,
    spawn_theta: float,
    object_pos: np.ndarray,
    goal: np.ndarray,
    gamma: float = _DEFAULT_FOLLOWER_GAMMA,
) -> float:
    """Roll the follower from ``(spawn, object_pos, goal)`` and return the
    discounted return — same scale as the PPO critic, so it is directly
    comparable to ``V_follower``."""
    obs, _ = follower_env.reset(seed=follower_seed, options={
        "agent_pos": spawn_pos,
        "agent_theta": spawn_theta,
        "object_pos": object_pos,
        "goal": goal,
    })
    g = 1.0
    total = 0.0
    done = False
    while not done:
        action, _ = follower_model.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = follower_env.step(action)
        total += g * float(r)
        g *= gamma
        done = term or trunc
    return total


def _run_episode_wh(
    leader_env,
    follower_env,
    leader_model,
    follower_model,
    follower_cfg,
    leader_seed: int,
    follower_seed: int,
    planner_seed: int,
    mode: str,
    *,
    record: bool = False,
    leader_range: Optional[Tuple[float, float]] = None,
    follower_range: Optional[Tuple[float, float]] = None,
    vae_path: Optional[str] = None,
    alpha: float = 1.0,
    scenario_name: str = "warehouse",
) -> dict:
    """One paired (leader → follower) episode. Returns standardized result row."""
    from critic_sharing.scenarios.warehouse import build_warehouse_follower_value_fn
    from critic_sharing.scenarios.warehouse.common import gap_centers

    # Pre-sample follower spawn + goal from follower_seed so the leader's planner
    # and the follower env see exactly the same fixed starting conditions.
    follower_env.reset(seed=follower_seed)
    f_spawn_pos = follower_env._agent_pos.copy()
    f_spawn_theta = float(follower_env._agent_theta)
    f_goal = follower_env._goal.copy()

    # Build per-episode V_follower closure once. Used by the planner when
    # mode==value_only, and always used to compute V at the 3 gap centres for
    # the diagnostics plot.
    value_fn = build_warehouse_follower_value_fn(
        follower_model,
        follower_cfg,
        f_spawn_pos,
        f_spawn_theta,
        f_goal,
    )

    # Query V at each gap centre (B=3 leader-shaped obs vectors, only the
    # object-pos columns are read — see build_warehouse_follower_value_fn).
    gaps = gap_centers(follower_cfg)  # (3, 2)
    leader_obs_at_gaps = np.zeros((gaps.shape[0], 7), dtype=np.float32)
    leader_obs_at_gaps[:, 4:6] = gaps
    v_at_gaps = np.asarray(value_fn(leader_obs_at_gaps), dtype=np.float32).reshape(-1)

    controller = None
    cost = None
    terminal_value_mode = "value"
    if mode in MODE_SETTINGS:  # planner modes; unguided runs the policy alone
        utility_type, use_ood, prior_kind = MODE_SETTINGS[mode]
        controller, cost, terminal_value_mode = _build_wh_controller(
            leader_cfg=leader_env.config,
            leader_model=leader_model,
            value_fn=value_fn,
            seed=planner_seed,
            utility_type=utility_type,
            leader_range=leader_range,
            follower_range=follower_range,
            vae_path=vae_path if use_ood else None,
            alpha=alpha,
            prior_kind=prior_kind,
            scenario_name=scenario_name,
        )

    # ---- Leader phase ----
    obs, _ = leader_env.reset(seed=leader_seed)
    leader_traj: List[np.ndarray] = [obs.copy()] if record else []
    leader_return = 0.0
    leader_steps = 0
    drop_pos = leader_env._object_pos.copy()
    gap_idx = -1
    if controller is not None:
        controller.reset()
    done = False
    # Track ``incurred + γ^t`` so the cost can reconstruct the from-step-0
    # discounted leader return on full-episode scale (matching
    # ``leader_range`` from full successful naive episodes). Both
    # terminal_value modes ride on this — only the partial-rollout side
    # of the equation differs (rollout extends with prior; value
    # bootstraps with V_leader at horizon).
    from critic_sharing.config.resolver import load_cost_config

    leader_gamma = float(
        getattr(load_cost_config(scenario_name), "leader_gamma", 0.99),
    )
    incurred_disc = 0.0
    gamma_pow_t = 1.0
    while not done:
        if controller is not None:
            if cost is not None:
                cost.set_incurred_return(incurred_disc, gamma_pow_t)
            if terminal_value_mode == "rollout":
                steps_remaining = int(leader_env.config.max_steps) - int(leader_env._steps)
                action = controller.command(obs, steps_remaining=steps_remaining)
            else:
                # "value" — no prior extension; cost bootstraps with
                # V_leader at horizon terminal.
                action = controller.command(obs)
        else:
            action, _ = leader_model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = leader_env.step(action)
        if record:
            leader_traj.append(obs.copy())
        leader_return += float(r)
        incurred_disc += gamma_pow_t * float(r)
        gamma_pow_t *= leader_gamma
        leader_steps += 1
        if info.get("dropped"):
            drop_pos = leader_env._object_pos.copy()
            gap_idx = int(info.get("gap_index", -1))
        done = term or trunc

    # ---- Follower phase ----
    obs, _ = follower_env.reset(seed=follower_seed, options={
        "agent_pos": f_spawn_pos,
        "agent_theta": f_spawn_theta,
        "object_pos": drop_pos,
        "goal": f_goal,
    })
    follower_traj: List[np.ndarray] = [obs.copy()] if record else []
    follower_return = 0.0
    follower_return_disc = 0.0
    follower_gamma_pow_t = 1.0
    follower_steps = 0
    on_goal = False
    info = {}
    done = False
    while not done:
        action, _ = follower_model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = follower_env.step(action)
        if record:
            follower_traj.append(obs.copy())
        follower_return += float(r)
        follower_return_disc += follower_gamma_pow_t * float(r)
        follower_gamma_pow_t *= leader_gamma
        follower_steps += 1
        on_goal = on_goal or bool(info.get("on_goal", False))
        done = term or trunc

    # Store discounted returns under the public ``*_return`` keys so the
    # downstream calibration-band normalization (``_norm`` against
    # ``leader_range`` / ``follower_range``, both of which are calibrated
    # in the discounted-return space via :func:`calibrate_range`) gets
    # matching units. Pre-2026-05-09 this stored undiscounted sums while
    # the bands were discounted; the resulting ``*_return_norm`` values
    # were inflated by the discount-factor mismatch and could exceed 1
    # for the naive baseline. The undiscounted sums are preserved under
    # ``*_return_undisc`` for descriptive use.
    result: dict = {
        "mode": mode,
        "leader_return": incurred_disc,
        "follower_return": follower_return_disc,
        "leader_return_undisc": leader_return,
        "follower_return_undisc": follower_return,
        "total_return": incurred_disc + follower_return_disc,
        "leader_steps": leader_steps,
        "follower_steps": follower_steps,
        "follower_on_goal": float(on_goal),
        # Binary leader-success signal for the cross_scenario S_L column.
        # gap_idx >= 0 means the leader actually dropped at one of the
        # three gaps; gap_idx == -1 means it ran out of time / hit a
        # wall before dropping.
        "leader_dropped_object": int(gap_idx >= 0),
        "drop_x": float(drop_pos[0]),
        "drop_y": float(drop_pos[1]),
        "gap_idx": gap_idx,
        "goal_x": float(f_goal[0]),
        "goal_y": float(f_goal[1]),
        "v_at_gap_0": float(v_at_gaps[0]),
        "v_at_gap_1": float(v_at_gaps[1]),
        "v_at_gap_2": float(v_at_gaps[2]),
    }
    if record:
        result["leader_traj"] = leader_traj
        result["follower_traj"] = follower_traj
    return result


# ---------------------------------------------------------------------------
# Worker for parallel execution
# ---------------------------------------------------------------------------

class WhEpisodeWorker(EpisodeWorker):
    """Loads PPO models + envs once per process, then runs episodes one by one.

    Each worker is bound to a single mode (set by the executor). The CEM
    controller is built per-episode in ``run_episode`` because the
    ``V_follower`` closure depends on the per-episode ``(spawn, goal)``;
    everything else (PPO models, env configs, instantiated envs) is cached
    in :meth:`setup`.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._leader_path: str = kwargs["leader_path"]
        self._follower_path: str = kwargs["follower_path"]
        # Calibration ranges from the main-process pre-pass — None when
        # normalize_returns=False or the user disabled normalization.
        self._leader_range: Optional[Tuple[float, float]] = kwargs.get("leader_range")
        self._follower_range: Optional[Tuple[float, float]] = kwargs.get("follower_range")
        # VAE/OOD trust-weighting. ``None`` → no trust factor applied.
        self._vae_path: Optional[str] = kwargs.get("vae_path")
        self._alpha: float = float(kwargs.get("alpha", 1.0))
        # Eval-time follower-horizon override. ``None`` keeps the
        # follower-env config's ``max_steps``; otherwise we
        # ``dataclasses.replace`` the loaded config in :meth:`setup`.
        self._follower_max_steps: Optional[int] = kwargs.get("follower_max_steps")
        # Scenario name (warehouse | warehouse_pickup_*). Used for config
        # loading paths so the worker reads from the right config tree.
        self._scenario_name: str = kwargs.get("scenario_name", "warehouse")

    def setup(self) -> None:
        from stable_baselines3 import PPO

        from critic_sharing.config.resolver import load_env_config, load_planner_config
        from critic_sharing.scenarios.warehouse import (
            WarehouseFollowerEnv,
            WarehouseLeaderEnv,
        )

        from dataclasses import replace as _dc_replace

        self._leader_cfg = load_env_config(self._scenario_name, "leader")
        self._follower_cfg = load_env_config(self._scenario_name, "follower")
        # Eval-time horizon override. The follower policy was trained on
        # the YAML's ``max_steps`` and doesn't see remaining time as obs,
        # so a smaller eval horizon just truncates marginal navigation
        # paths --- a clean exogenous "make it harder" knob.
        if self._follower_max_steps is not None:
            self._follower_cfg = _dc_replace(
                self._follower_cfg, max_steps=int(self._follower_max_steps),
            )
        self._leader_model = PPO.load(self._leader_path)
        self._follower_model = PPO.load(self._follower_path)
        self._leader_env = WarehouseLeaderEnv(self._leader_cfg)
        self._follower_env = WarehouseFollowerEnv(self._follower_cfg)
        # When terminal_value="rollout" the prior-extension batched
        # forward becomes the per-worker bottleneck; allow the user to
        # bump torch threads per worker via planner.yaml's
        # ``prior_extension_threads`` (set so num_workers * threads
        # stays at-or-below cpu_count).
        cem_cfg = load_planner_config(self._scenario_name)
        prior_threads = int(getattr(cem_cfg, "prior_extension_threads", 1))
        terminal_value_mode = getattr(cem_cfg, "terminal_value", "value")
        if prior_threads > 1 and terminal_value_mode == "rollout":
            try:
                import torch as th

                th.set_num_threads(prior_threads)
            except Exception:
                pass

    def run_episode(self, spec: EpisodeSpec) -> dict:
        f_seed = int(spec.follower_seeds[0]) if spec.follower_seeds else int(spec.leader_seed)
        return _run_episode_wh(
            leader_env=self._leader_env,
            follower_env=self._follower_env,
            leader_model=self._leader_model,
            follower_model=self._follower_model,
            follower_cfg=self._follower_cfg,
            leader_seed=int(spec.leader_seed),
            follower_seed=f_seed,
            planner_seed=int(spec.planner_seed),
            mode=spec.mode,
            record=bool(spec.record),
            leader_range=self._leader_range,
            follower_range=self._follower_range,
            vae_path=self._vae_path,
            alpha=self._alpha,
            scenario_name=self._scenario_name,
        )


# ---------------------------------------------------------------------------
# Top-level _compare_wh + WarehouseAdapter
# ---------------------------------------------------------------------------


class WarehouseAdapter(EvalScenarioAdapter):
    """:class:`EvalScenarioAdapter` for warehouse and its three variants
    (``warehouse_blocked`` / ``warehouse_pickup_blocked`` /
    ``warehouse_pickup_unseen``)."""

    worker_cls = WhEpisodeWorker

    def __init__(
        self,
        *,
        leader_path: str,
        follower_path: str,
        scenario_name: str = "warehouse",
        follower_max_steps: Optional[int] = None,
    ) -> None:
        self.scenario_name = scenario_name
        self._leader_path = leader_path
        self._follower_path = follower_path
        self._follower_max_steps = follower_max_steps

    def build_calibration(self):
        from stable_baselines3 import PPO

        from critic_sharing.config.resolver import load_env_config
        from critic_sharing.scenarios.warehouse import (
            WarehouseFollowerEnv,
            WarehouseLeaderEnv,
        )

        leader_cfg = load_env_config(self.scenario_name, "leader")
        follower_cfg = load_env_config(self.scenario_name, "follower")
        # Honor the eval-time follower-horizon override during calibration
        # so the follower_range reflects the same truncation pressure the
        # planner faces at run time. (Otherwise calibration would compute
        # against a 300-step naive whilst the cost evaluates against a
        # 150-step naive.)
        if self._follower_max_steps is not None:
            from dataclasses import replace as _dc_replace
            follower_cfg = _dc_replace(
                follower_cfg, max_steps=int(self._follower_max_steps),
            )
        leader_env = WarehouseLeaderEnv(leader_cfg)
        follower_env = WarehouseFollowerEnv(follower_cfg)
        leader_model = PPO.load(self._leader_path)
        follower_model = PPO.load(self._follower_path)
        return leader_env, follower_env, leader_model, follower_model

    def sample_conditions(self, rng, n_episodes):
        # Warehouse seeds env-reset directly from leader_seed/follower_seed;
        # no separate init-condition object.
        return [None] * n_episodes

    def make_spec(
        self, *, episode, mode, leader_seed, follower_seed, planner_seed,
        condition, paths, alpha, record,
    ):
        return EpisodeSpec(
            scenario=self.scenario_name,
            mode=mode,
            episode=episode,
            leader_seed=leader_seed,
            follower_seeds=[follower_seed],
            planner_seed=planner_seed,
            record=record,
            leader_model_path=paths["leader_model"],
            follower_model_path=paths["follower_model"],
        )

    def worker_kwargs(self, *, leader_range, follower_range, paths, alpha):
        return {
            "leader_path": paths["leader_model"],
            "follower_path": paths["follower_model"],
            "leader_range": leader_range,
            "follower_range": follower_range,
            "vae_path": paths.get("vae"),
            "alpha": float(alpha) if alpha is not None else None,
            "follower_max_steps": self._follower_max_steps,
            "scenario_name": self.scenario_name,
        }

    def build_ood_fn_for_post_processing(self, vae_path):
        if vae_path is None:
            return None
        from critic_sharing.scenarios.warehouse.ood_query import (
            build_warehouse_ood_metric,
        )
        return build_warehouse_ood_metric(vae_path)

    def handoff_obs_for_ood(self, row):
        # Leader-shape (1, 7); only columns 4-5 (object x/y at drop) are
        # consulted by the OOD closure.
        obs = np.zeros((1, 7), dtype=np.float32)
        obs[0, 4] = float(row.get("drop_x", 0.0))
        obs[0, 5] = float(row.get("drop_y", 0.0))
        return obs


def _compare_wh(
    *,
    leader_model: Path,
    follower_model: Path,
    modes: Sequence[EvalMode],
    leader_seeds: Sequence[int],
    follower_seeds: Sequence[int],
    planner_seeds: Sequence[int],
    n_follower_samples: int = 1,
    workers: int = 1,
    record: bool = True,
    vae_path: Optional[str] = None,
    alpha: Optional[float] = None,
    follower_max_steps: Optional[int] = None,
    scenario_name: str = "warehouse",
) -> list[dict]:
    """Run paired naive vs V-CEM episodes for the warehouse scenario.

    Uses the shared :mod:`pipeline` helpers (``calibrate_ranges`` +
    ``normalize_result_columns``) but keeps its own orchestration loop
    because warehouse has a serial diagnostic pre-pass between
    calibration and dispatch (the per-episode ``actual_returns_per_ep``
    and ``v_along_wall_per_ep`` arrays) that doesn't fit the
    :func:`run_eval_pipeline` shape.

    Returns a flat list of result dicts. Each row contains the standardized
    columns expected by ``_print_summary_table`` / ``_plot_compare`` plus
    warehouse-specific extras (``gap_idx``, ``drop_x``, ``drop_y``,
    ``goal_x``, ``goal_y``).
    """
    from stable_baselines3 import PPO

    from critic_sharing.config.resolver import load_env_config
    from critic_sharing.scenarios.warehouse import (
        WarehouseFollowerEnv,
    )
    from critic_sharing.scenarios.warehouse.common import gap_centers

    adapter = WarehouseAdapter(
        leader_path=str(leader_model),
        follower_path=str(follower_model),
        scenario_name=scenario_name,
        follower_max_steps=follower_max_steps,
    )

    requested = set(modes)
    if not requested.issubset(adapter.supported_modes()):
        unsupported = requested - adapter.supported_modes()
        typer.echo(
            "{s}: modes not implemented yet: {u}. Supported: {S}.".format(
                s=scenario_name,
                u=", ".join(m.value for m in unsupported),
                S=", ".join(m.value for m in adapter.supported_modes()),
            ),
            err=True,
        )
        raise typer.Exit(code=1)

    if n_follower_samples != 1:
        typer.echo(
            "warehouse: --follower-samples > 1 not implemented; using 1.",
            err=True,
        )

    follower_cfg = load_env_config(scenario_name, "follower")
    if follower_max_steps is not None:
        from dataclasses import replace as _dc_replace
        follower_cfg = _dc_replace(follower_cfg, max_steps=int(follower_max_steps))
        typer.echo(
            "  {s}  follower_max_steps override: {n}".format(
                s=scenario_name, n=int(follower_max_steps),
            ),
            err=True,
        )
    follower_obj = PPO.load(str(follower_model))
    follower_env = WarehouseFollowerEnv(follower_cfg)
    gaps = gap_centers(follower_cfg)

    # ---- Calibration (shared helper) ----
    leader_range, follower_range = calibrate_ranges(adapter, modes)

    # ---- Pre-pass (serial, main process) ----
    # 3 follower rollouts per episode → expected return when starting from
    # each gap centre. Mode-independent; merged into both modes' rows
    # after the parallel main pass returns.
    typer.echo(
        "  warehouse  pre-rollout: {n} episodes × {g} gaps...".format(
            n=len(follower_seeds), g=int(gaps.shape[0]),
        )
    )
    from critic_sharing.scenarios.warehouse import build_warehouse_follower_value_fn

    actual_returns_per_ep: dict[int, np.ndarray] = {}
    v_along_wall_per_ep: dict[int, dict[str, np.ndarray]] = {}

    # X-grid for the V-along-wall sweep (per-episode V(x, y=wall_y) curve).
    n_wall_x = 200
    wall_x_grid = np.linspace(
        float(follower_cfg.world_min) + float(follower_cfg.object_radius),
        float(follower_cfg.world_max) - float(follower_cfg.object_radius),
        n_wall_x,
        dtype=np.float32,
    )

    # Optional: VAE/OOD trust factor along the wall. Scenario-constant
    # (depends only on object position which is the only varying dim
    # along the wall sweep), so compute once and reuse for every episode
    # row's payload. ``None`` when ``vae_path`` is not supplied — the
    # downstream v_at_gaps figure will then gracefully skip its OOD and
    # discounted-V panels.
    ood_along_wall: Optional[np.ndarray] = None
    if vae_path is not None:
        from critic_sharing.scenarios.warehouse.ood_query import (
            build_warehouse_ood_metric,
        )

        ood_fn = build_warehouse_ood_metric(str(vae_path))
        leader_obs_at_wall_for_ood = np.zeros((n_wall_x, 7), dtype=np.float32)
        leader_obs_at_wall_for_ood[:, 4] = wall_x_grid
        leader_obs_at_wall_for_ood[:, 5] = float(follower_cfg.wall_y)
        ood_along_wall = np.asarray(
            ood_fn(leader_obs_at_wall_for_ood), dtype=np.float32,
        ).reshape(-1)

    try:
        for ep_idx, f_seed in enumerate(follower_seeds):
            follower_env.reset(seed=int(f_seed))
            spawn_pos = follower_env._agent_pos.copy()
            spawn_theta = float(follower_env._agent_theta)
            goal = follower_env._goal.copy()

            # Per-episode actual-return rollouts at each gap centre.
            rs = np.zeros(gaps.shape[0], dtype=np.float32)
            for i, gap in enumerate(gaps):
                rs[i] = _rollout_follower_discounted_return(
                    follower_env, follower_obj, int(f_seed),
                    spawn_pos, spawn_theta, gap, goal,
                )
            actual_returns_per_ep[ep_idx] = rs

            # Per-episode V_follower curve along y=wall_y. One batched PPO
            # forward pass — n_wall_x samples sharing the same (spawn, goal).
            value_fn = build_warehouse_follower_value_fn(
                follower_obj, follower_cfg, spawn_pos, spawn_theta, goal,
            )
            leader_obs_at_wall = np.zeros((n_wall_x, 7), dtype=np.float32)
            leader_obs_at_wall[:, 4] = wall_x_grid
            leader_obs_at_wall[:, 5] = float(follower_cfg.wall_y)
            v_curve = np.asarray(
                value_fn(leader_obs_at_wall), dtype=np.float32,
            ).reshape(-1)
            entry: dict[str, np.ndarray] = {
                "x": wall_x_grid.copy(),
                "v": v_curve,
            }
            if ood_along_wall is not None:
                # Trust-weighted V along wall — same form the planner cost
                # uses (V * (1 - OOD)). Lets the v_at_gaps figure show
                # the actual discounted V the planner would see.
                trust = np.clip(1.0 - ood_along_wall, 0.0, 1.0)
                entry["ood"] = ood_along_wall.copy()
                entry["v_disc"] = (v_curve * trust).astype(np.float32)
            v_along_wall_per_ep[ep_idx] = entry
    finally:
        follower_env.close()

    # ---- Build EpisodeSpec list ----
    specs: list[EpisodeSpec] = []
    for mode in modes:
        for ep_idx, (l_seed, f_seed, p_seed) in enumerate(
            zip(leader_seeds, follower_seeds, planner_seeds)
        ):
            specs.append(
                EpisodeSpec(
                    scenario=scenario_name,
                    mode=mode.value,
                    episode=ep_idx,
                    leader_seed=int(l_seed),
                    follower_seeds=[int(f_seed)],
                    planner_seed=int(p_seed),
                    record=bool(record),
                    leader_model_path=str(leader_model),
                    follower_model_path=str(follower_model),
                )
            )

    # Resolve alpha from cost.yaml if not explicitly passed.
    if alpha is None:
        from critic_sharing.config.resolver import load_cost_config

        alpha = float(getattr(load_cost_config(scenario_name), "alpha", 1.0))
    if vae_path is not None:
        typer.echo(
            "  {s}  VAE/OOD trust-weighting on: vae={p} alpha={l:.4f}".format(
                s=scenario_name, p=vae_path, l=float(alpha),
            ),
            err=True,
        )

    # ---- Main bulk in parallel ----
    raw_results = run_episodes_parallel(
        specs,
        worker_cls=WhEpisodeWorker,
        worker_kwargs={
            "leader_path": str(leader_model),
            "follower_path": str(follower_model),
            # Calibration ranges from the pre-pass — workers forward
            # them into JointUtilityCost when building per-episode CEM
            # controllers. ``None`` disables that side's normalization.
            "leader_range": leader_range,
            "follower_range": follower_range,
            # VAE/OOD trust-weighting (optional). vae_path=None means
            # no trust factor is applied; alpha is then irrelevant.
            "vae_path": str(vae_path) if vae_path else None,
            "alpha": float(alpha),
            # Eval-time follower horizon override. None → keep the
            # config's value. Workers honor this via dataclasses.replace
            # in :meth:`setup`.
            "follower_max_steps": (
                int(follower_max_steps) if follower_max_steps else None
            ),
            "scenario_name": scenario_name,
        },
        max_workers=int(workers) if workers else 1,
    )

    # ---- Post-merge ----
    all_results: list[dict] = []
    leader_seed_by_ep = {ep_idx: int(s) for ep_idx, s in enumerate(leader_seeds)}

    # ---- Merge warehouse-specific diagnostic columns into rows ----
    for row in raw_results:
        ep_idx = int(row.get("episode", -1))
        rs = actual_returns_per_ep.get(ep_idx)
        if rs is not None:
            row["actual_return_at_gap_0"] = float(rs[0])
            row["actual_return_at_gap_1"] = float(rs[1])
            row["actual_return_at_gap_2"] = float(rs[2])
        v_curve = v_along_wall_per_ep.get(ep_idx)
        if v_curve is not None:
            row["v_along_wall_x"] = v_curve["x"]
            row["v_along_wall_v"] = v_curve["v"]
            if "ood" in v_curve:
                row["ood_along_wall"] = v_curve["ood"]
            if "v_disc" in v_curve:
                row["v_disc_along_wall"] = v_curve["v_disc"]
        row.setdefault("seed", leader_seed_by_ep.get(ep_idx))
        all_results.append(row)

    # ---- Shared post-merge: leader_return_norm / follower_return_norm /
    #      ood_at_handoff / total_return_weighted ----
    ood_fn_for_table = adapter.build_ood_fn_for_post_processing(
        str(vae_path) if vae_path else None,
    )
    normalize_result_columns(
        all_results,
        adapter=adapter,
        leader_range=leader_range,
        follower_range=follower_range,
        alpha=float(alpha),
        ood_fn=ood_fn_for_table,
    )

    return all_results


# ---------------------------------------------------------------------------
# Animations
# ---------------------------------------------------------------------------

class WhEpisodeFigureWorker(FigureWorker):
    """Renders one warehouse 2-panel (leader | follower) MP4 per task."""

    def setup(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        from critic_sharing.config.resolver import load_env_config

        self._fps = int(self._kwargs.get("fps", 4))
        scenario_name = str(self._kwargs.get("scenario_name", "warehouse"))
        self._leader_cfg = load_env_config(scenario_name, "leader")
        self._follower_cfg = load_env_config(scenario_name, "follower")

    def render(self, task: FigureTask) -> None:
        import matplotlib.pyplot as plt
        from matplotlib import animation as mpl_animation

        from critic_sharing.scenarios.warehouse.render import (
            COLOR_AGENT_FOLLOWER,
            COLOR_AGENT_LEADER,
            COLOR_FOLLOWER_TRAIL,
            COLOR_LEADER_TRAIL,
            draw_arena,
            draw_goal,
            make_agent_patch,
            make_object_patch,
            update_agent_patch,
        )

        row: dict[str, Any] = task.payload  # type: ignore[assignment]
        leader_cfg = self._leader_cfg
        follower_cfg = self._follower_cfg

        leader_obs = np.asarray(row["leader_traj"], dtype=np.float32)
        follower_obs = np.asarray(row["follower_traj"], dtype=np.float32)

        fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
        ax_l, ax_f = axes
        # Surface follower's pickup_disabled_gap_indices on the leader panel
        # too — the leader env config doesn't carry it but the constraint is
        # a global property of the warehouse_pickup_blocked variant.
        pickup_disabled = getattr(follower_cfg, "pickup_disabled_gap_indices", ())
        draw_arena(ax_l, leader_cfg, pickup_disabled_gap_indices=pickup_disabled)
        draw_arena(ax_f, follower_cfg, pickup_disabled_gap_indices=pickup_disabled)
        goal = np.array([row["goal_x"], row["goal_y"]], dtype=np.float32)
        draw_goal(ax_f, follower_cfg, goal)

        l_agent_sq, l_agent_arrow = make_agent_patch(ax_l, leader_cfg, COLOR_AGENT_LEADER)
        l_object = make_object_patch(ax_l, leader_cfg)
        (l_trail,) = ax_l.plot([], [], color=COLOR_LEADER_TRAIL, linewidth=1.5, alpha=0.85, zorder=4)

        f_agent_sq, f_agent_arrow = make_agent_patch(ax_f, follower_cfg, COLOR_AGENT_FOLLOWER)
        f_object = make_object_patch(ax_f, follower_cfg)
        (f_trail,) = ax_f.plot([], [], color=COLOR_FOLLOWER_TRAIL, linewidth=1.5, alpha=0.85, zorder=4)

        drop = (float(row["drop_x"]), float(row["drop_y"]))
        n_l = len(leader_obs)
        n_f = len(follower_obs)
        total = n_l + n_f

        def _theta(o):
            return float(np.arctan2(o[2], o[3]))

        def _update(i, n_l=n_l, n_f=n_f, drop=drop):
            if i < n_l:
                o = leader_obs[i]
                update_agent_patch(l_agent_sq, l_agent_arrow, o[0:2], _theta(o), leader_cfg)
                l_object.center = (float(o[4]), float(o[5]))
                xs = leader_obs[: i + 1, 0]
                ys = leader_obs[: i + 1, 1]
                l_trail.set_data(xs, ys)
                f_agent_sq.set_visible(False)
                f_agent_arrow.set_visible(False)
                f_object.set_visible(False)
            else:
                j = i - n_l
                o_l = leader_obs[-1]
                update_agent_patch(l_agent_sq, l_agent_arrow, o_l[0:2], _theta(o_l), leader_cfg)
                l_object.center = drop
                o_f = follower_obs[j]
                f_agent_sq.set_visible(True)
                f_agent_arrow.set_visible(True)
                f_object.set_visible(True)
                update_agent_patch(f_agent_sq, f_agent_arrow, o_f[0:2], _theta(o_f), follower_cfg)
                f_object.center = (float(o_f[4]), float(o_f[5]))
                xs = follower_obs[: j + 1, 0]
                ys = follower_obs[: j + 1, 1]
                f_trail.set_data(xs, ys)
            return ()

        fig.tight_layout()
        anim = mpl_animation.FuncAnimation(
            fig, _update, frames=total,
            interval=1000 / max(1, self._fps), blit=False,
        )
        anim.save(str(task.output_path), writer=mpl_animation.FFMpegWriter(fps=self._fps))
        plt.close(fig)


def _wh_mp4_payload(row: dict) -> dict[str, Any]:
    """Pull the picklable subset of a recorded row that the worker needs.

    Strips e.g. trajectory tuples to numpy arrays and drops the keys the
    worker doesn't read (keeps pickle payload tight)."""
    return {
        "episode": int(row["episode"]),
        "mode": row["mode"],
        "leader_traj": np.asarray(row["leader_traj"], dtype=np.float32),
        "follower_traj": np.asarray(row["follower_traj"], dtype=np.float32),
        "goal_x": float(row["goal_x"]),
        "goal_y": float(row["goal_y"]),
        "drop_x": float(row["drop_x"]),
        "drop_y": float(row["drop_y"]),
        "gap_idx": int(row["gap_idx"]),
        "follower_on_goal": float(row["follower_on_goal"]),
        "leader_return": float(row["leader_return"]),
        "follower_return": float(row["follower_return"]),
    }


def _save_wh_episode_mp4s(
    all_results: List[dict],
    output_dir: Path,
    fps: int = 4,
    max_workers: int = 1,
    scenario_name: str = "warehouse",
) -> None:
    """Save 2-panel (leader | follower) MP4s for every recorded warehouse episode."""
    recorded = [r for r in all_results if "leader_traj" in r and "follower_traj" in r]
    if not recorded:
        return

    mp4_dir = output_dir / "episodes"
    mp4_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[FigureTask] = []
    for row in recorded:
        out_path = mp4_dir / "ep{ep:03d}_{mode}.mp4".format(
            ep=int(row["episode"]), mode=row["mode"],
        )
        tasks.append(FigureTask(output_path=out_path, payload=_wh_mp4_payload(row)))

    typer.echo(
        "Saving {n} warehouse episode MP4s to {d}/ ({w} workers)".format(
            n=len(tasks), d=mp4_dir, w=max(1, min(max_workers, len(tasks))),
        )
    )
    run_figures_parallel(
        tasks=tasks,
        worker_cls=WhEpisodeFigureWorker,
        worker_kwargs={"fps": int(fps), "scenario_name": scenario_name},
        max_workers=int(max_workers),
        progress_desc="WH episodes",
    )


# ---------------------------------------------------------------------------
# Warehouse-specific extras: gap-distribution + paired-diff figures
# ---------------------------------------------------------------------------

def _plot_warehouse_extras(
    all_results: List[dict],
    output_dir: Path,
) -> None:
    """Save warehouse-only figures: gap-choice distribution + paired diff scatter."""
    import matplotlib.pyplot as plt

    modes = sorted({r["mode"] for r in all_results})
    by_mode = {m: [r for r in all_results if r["mode"] == m] for m in modes}

    # ---- Gap-choice distribution ----
    fig, ax = plt.subplots(figsize=(7, 5))
    gap_categories = [0, 1, 2, -1]
    gap_labels = ["gap0", "gap1", "gap2", "miss"]
    gap_colors = ["#43A047", "#FB8C00", "#3949AB", "#9E9E9E"]
    bottom = np.zeros(len(modes))
    for cat, lab, col in zip(gap_categories, gap_labels, gap_colors):
        counts = []
        for m in modes:
            subset = by_mode[m]
            n = len(subset) or 1
            counts.append(100.0 * sum(1 for r in subset if r["gap_idx"] == cat) / n)
        ax.bar(modes, counts, bottom=bottom, label=lab, color=col)
        bottom += np.asarray(counts)
    ax.set_title("Gap-choice distribution")
    ax.set_ylabel("% of episodes")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = output_dir / "compare_warehouse_gap_distribution.png"
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    typer.echo("Gap distribution    → {p}".format(p=out_path))

    # ---- Paired diff scatter (only if both modes present) ----
    if EvalMode.unguided.value in by_mode and EvalMode.value_only.value in by_mode:
        # Use normalized follower return when calibration produced it,
        # so the scatter is in [0, 1] units consistent with the summary
        # table and headline returns figure.
        follower_key = (
            "follower_return_norm"
            if any("follower_return_norm" in r for r in all_results)
            else "follower_return"
        )
        un = {r["episode"]: r for r in by_mode[EvalMode.unguided.value]}
        gu = {r["episode"]: r for r in by_mode[EvalMode.value_only.value]}
        common = sorted(set(un) & set(gu))
        if common:
            xs = [un[e][follower_key] for e in common]
            ys = [gu[e][follower_key] for e in common]
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(xs, ys, alpha=0.6, color="#1565C0", edgecolor="#0D47A1")
            lo = min(min(xs), min(ys))
            hi = max(max(xs), max(ys))
            ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="y = x")
            n_better = sum(1 for x, y in zip(xs, ys) if y > x)
            ax.set_title(
                "Per-episode follower {fk}: {gu} vs {un}\n({n}/{tot} better with {gu})".format(
                    fk=follower_key,
                    gu=EvalMode.value_only.value,
                    un=EvalMode.unguided.value,
                    n=n_better,
                    tot=len(common),
                )
            )
            ax.set_xlabel("{u} {fk}".format(u=EvalMode.unguided.value, fk=follower_key))
            ax.set_ylabel("{v} {fk}".format(v=EvalMode.value_only.value, fk=follower_key))
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            out_path = output_dir / "compare_warehouse_paired_diff.png"
            fig.savefig(str(out_path), dpi=150)
            plt.close(fig)
            typer.echo("Paired diff scatter → {p}".format(p=out_path))


class WhVAtGapsFigureWorker(FigureWorker):
    """Renders one warehouse 3-panel V-at-gaps PNG per task.

    Pure render-from-data: all V values (``v_at_gap_*``, ``v_along_wall_*``)
    and rollout-return values (``actual_return_at_gap_*``) are precomputed
    in the main process by :func:`_compare_wh` and baked into each row.
    No model / env state is touched in the worker.
    """

    BAR_V = "#1565C0"
    BAR_ACTUAL = "#FB8C00"
    EDGE_CHOSEN = "#2E7D32"
    EDGE_ARGMAX = "#1976D2"
    EDGE_DEFAULT = "#37474F"

    def setup(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        from critic_sharing.config.resolver import load_env_config

        scenario_name = str(self._kwargs.get("scenario_name", "warehouse"))
        self._leader_cfg = load_env_config(scenario_name, "leader")
        self._follower_cfg = load_env_config(scenario_name, "follower")

    def render(self, task: FigureTask) -> None:
        import matplotlib.pyplot as plt

        from critic_sharing.scenarios.warehouse.common import gap_rects
        from critic_sharing.scenarios.warehouse.render import (
            COLOR_AGENT_FOLLOWER,
            COLOR_AGENT_LEADER,
            COLOR_FOLLOWER_TRAIL,
            COLOR_LEADER_TRAIL,
            COLOR_OBJECT,
            draw_arena,
            draw_goal,
        )

        row: dict[str, Any] = task.payload  # type: ignore[assignment]
        leader_cfg = self._leader_cfg
        follower_cfg = self._follower_cfg

        v_vals = [
            float(row.get("v_at_gap_0", 0.0)),
            float(row.get("v_at_gap_1", 0.0)),
            float(row.get("v_at_gap_2", 0.0)),
        ]
        actual_vals = [
            float(row.get("actual_return_at_gap_0", 0.0)),
            float(row.get("actual_return_at_gap_1", 0.0)),
            float(row.get("actual_return_at_gap_2", 0.0)),
        ]
        chosen = int(row.get("gap_idx", -1))
        argmax = int(np.argmax(v_vals))

        # ---- Layout: 3-row × 3-col gridspec ----
        # arena and bars span all 3 rows in cols 0 and 1; the right column
        # stacks 3 wall-slice subplots (V, OOD, V·(1-OOD)) sharing x-axis.
        # When OOD/discounted arrays are absent on the row payload, the
        # bottom two slices are hidden — the layout falls back to the
        # original 3-panel look (arena + bars + V-along-wall).
        has_ood = (
            row.get("ood_along_wall") is not None
            and row.get("v_disc_along_wall") is not None
        )
        fig = plt.figure(figsize=(16, 8) if has_ood else (16, 5.0))
        if has_ood:
            gs = fig.add_gridspec(
                3, 3,
                width_ratios=[1.4, 1.0, 1.4],
                height_ratios=[1, 1, 1],
                wspace=0.30, hspace=0.40,
            )
            ax_arena = fig.add_subplot(gs[:, 0])
            ax_bars = fig.add_subplot(gs[:, 1])
            ax_curve = fig.add_subplot(gs[0, 2])
            ax_ood = fig.add_subplot(gs[1, 2], sharex=ax_curve)
            ax_disc = fig.add_subplot(gs[2, 2], sharex=ax_curve)
        else:
            gs = fig.add_gridspec(1, 3, width_ratios=[1.4, 1.0, 1.4], wspace=0.30)
            ax_arena = fig.add_subplot(gs[0, 0])
            ax_bars = fig.add_subplot(gs[0, 1])
            ax_curve = fig.add_subplot(gs[0, 2])
            ax_ood = None
            ax_disc = None

        # ---- Left: arena with trajectories ----
        pickup_disabled = getattr(follower_cfg, "pickup_disabled_gap_indices", ())
        draw_arena(ax_arena, leader_cfg, pickup_disabled_gap_indices=pickup_disabled)
        goal = np.array([row.get("goal_x", 0.0), row.get("goal_y", 0.0)], dtype=np.float32)
        draw_goal(ax_arena, follower_cfg, goal)

        leader_obs = np.asarray(row.get("leader_traj", []), dtype=np.float32)
        follower_obs = np.asarray(row.get("follower_traj", []), dtype=np.float32)

        if leader_obs.size:
            obj_spawn = leader_obs[0, 4:6]
            ax_arena.plot(
                float(obj_spawn[0]), float(obj_spawn[1]),
                marker="o", markersize=10, markerfacecolor="none",
                markeredgecolor=COLOR_OBJECT, markeredgewidth=2,
                linestyle="None", zorder=6, label="object start",
            )
            ax_arena.plot(
                leader_obs[:, 0], leader_obs[:, 1],
                color=COLOR_LEADER_TRAIL, linewidth=1.6, alpha=0.9,
                zorder=4, label="leader path",
            )
            ax_arena.plot(
                float(leader_obs[0, 0]), float(leader_obs[0, 1]),
                marker="s", markersize=8, color=COLOR_AGENT_LEADER,
                markeredgecolor="#212121", linestyle="None", zorder=7,
                label="leader start",
            )

        drop = (float(row.get("drop_x", 0.0)), float(row.get("drop_y", 0.0)))
        ax_arena.plot(
            drop[0], drop[1],
            marker="o", markersize=9, color=COLOR_OBJECT,
            markeredgecolor="#311B92", markeredgewidth=1.5,
            linestyle="None", zorder=7, label="object drop",
        )

        if follower_obs.size:
            ax_arena.plot(
                follower_obs[:, 0], follower_obs[:, 1],
                color=COLOR_FOLLOWER_TRAIL, linewidth=1.6, alpha=0.9,
                zorder=4, label="follower path",
            )
            ax_arena.plot(
                float(follower_obs[0, 0]), float(follower_obs[0, 1]),
                marker="s", markersize=8, color=COLOR_AGENT_FOLLOWER,
                markeredgecolor="#212121", linestyle="None", zorder=7,
                label="follower start",
            )

        ax_arena.legend(loc="lower left", fontsize=7, framealpha=0.85)

        # ---- Middle: V vs actual return bars ----
        x = np.arange(3, dtype=np.float32)
        width = 0.4
        edges = []
        widths = []
        for i in range(3):
            if i == chosen:
                edges.append(self.EDGE_CHOSEN)
                widths.append(2.0)
            elif i == argmax:
                edges.append(self.EDGE_ARGMAX)
                widths.append(2.0)
            else:
                edges.append(self.EDGE_DEFAULT)
                widths.append(0.8)
        v_bars = ax_bars.bar(
            x - width / 2, v_vals, width,
            color=self.BAR_V, label="V_follower(gap)",
            edgecolor=edges, linewidth=widths,
        )
        a_bars = ax_bars.bar(
            x + width / 2, actual_vals, width,
            color=self.BAR_ACTUAL, label="actual rollout return",
            edgecolor=edges, linewidth=widths,
        )
        for bar, val in list(zip(v_bars, v_vals)) + list(zip(a_bars, actual_vals)):
            ax_bars.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                "{:.2f}".format(val),
                ha="center", va="bottom", fontsize=7,
            )
        ax_bars.axhline(0.0, color="#37474F", linewidth=0.6)
        ax_bars.set_xticks(x)
        ax_bars.set_xticklabels(["gap0", "gap1", "gap2"])
        ax_bars.set_ylabel("return (discounted)")
        ax_bars.legend(loc="best", fontsize=8)
        ax_bars.grid(True, axis="y", alpha=0.3)

        # ---- Right: V(x) along the wall ----
        x_curve = row.get("v_along_wall_x")
        v_curve = row.get("v_along_wall_v")
        if x_curve is not None and v_curve is not None:
            x_arr = np.asarray(x_curve, dtype=np.float32)
            v_arr = np.asarray(v_curve, dtype=np.float32)
            for x1, x2 in gap_rects(follower_cfg):
                ax_curve.axvspan(x1, x2, color="#43A047", alpha=0.18, zorder=1)
            ax_curve.plot(
                x_arr, v_arr,
                color="#1565C0", linewidth=1.6, zorder=3,
            )
            for i, gx in enumerate(sorted(float(x) for x in follower_cfg.gap_xs)):
                ax_curve.plot(
                    gx, v_vals[i],
                    marker="o", markersize=7, color="#1565C0",
                    markeredgecolor="#0D47A1", linestyle="None",
                    zorder=5,
                )
            if chosen >= 0:
                ax_curve.axvline(
                    float(row.get("drop_x", 0.0)),
                    color="#FB8C00", linestyle="--", linewidth=1.4,
                    label="chosen drop x", zorder=4,
                )
                ax_curve.legend(loc="best", fontsize=8)
            # Only label x on the bottom-most slice when stacking is active.
            if ax_ood is None:
                ax_curve.set_xlabel("x along wall (y={wy})".format(wy=follower_cfg.wall_y))
            ax_curve.set_ylabel("V_follower")
            ax_curve.grid(True, alpha=0.3)
        else:
            ax_curve.set_visible(False)

        # ---- Bottom-right pair (only when VAE wired in): OOD and V·(1-OOD) ----
        if ax_ood is not None and ax_disc is not None and has_ood:
            x_arr = np.asarray(row["v_along_wall_x"], dtype=np.float32)
            ood_arr = np.asarray(row["ood_along_wall"], dtype=np.float32)
            disc_arr = np.asarray(row["v_disc_along_wall"], dtype=np.float32)

            # OOD slice — same gap shading + chosen drop marker for visual
            # alignment with the V slice above.
            for x1, x2 in gap_rects(follower_cfg):
                ax_ood.axvspan(x1, x2, color="#43A047", alpha=0.18, zorder=1)
            ax_ood.plot(
                x_arr, ood_arr,
                color="#C62828", linewidth=1.6, zorder=3,
            )
            ax_ood.set_ylim(-0.05, 1.05)
            if chosen >= 0:
                ax_ood.axvline(
                    float(row.get("drop_x", 0.0)),
                    color="#FB8C00", linestyle="--", linewidth=1.4, zorder=4,
                )
            ax_ood.set_ylabel("OOD score")
            ax_ood.grid(True, alpha=0.3)
            # Hide x-tick labels on the middle subplot — bottom one carries
            # the shared x-axis label.
            plt.setp(ax_ood.get_xticklabels(), visible=False)

            # Discounted V slice — V·(1-OOD), the actual V_F contribution
            # the planner cost sees after trust-weighting.
            for x1, x2 in gap_rects(follower_cfg):
                ax_disc.axvspan(x1, x2, color="#43A047", alpha=0.18, zorder=1)
            ax_disc.plot(
                x_arr, disc_arr,
                color="#6A1B9A", linewidth=1.6, zorder=3,
            )
            for i, gx in enumerate(sorted(float(x) for x in follower_cfg.gap_xs)):
                # Mark each gap centre with the discounted V at that x —
                # interpolate from the curve so the marker sits on the
                # plotted line.
                gx_idx = int(np.argmin(np.abs(x_arr - float(gx))))
                ax_disc.plot(
                    float(x_arr[gx_idx]), float(disc_arr[gx_idx]),
                    marker="o", markersize=7, color="#6A1B9A",
                    markeredgecolor="#311B92", linestyle="None",
                    zorder=5,
                )
            if chosen >= 0:
                ax_disc.axvline(
                    float(row.get("drop_x", 0.0)),
                    color="#FB8C00", linestyle="--", linewidth=1.4, zorder=4,
                )
            ax_disc.set_xlabel("x along wall (y={wy})".format(wy=follower_cfg.wall_y))
            ax_disc.set_ylabel("V·(1-OOD)")
            ax_disc.grid(True, alpha=0.3)

        # tight_layout doesn't play well with custom gridspec width/height
        # ratios — let the gridspec wspace/hspace handle spacing.
        fig.savefig(str(task.output_path), dpi=110, bbox_inches="tight")
        plt.close(fig)


def _wh_v_at_gaps_payload(row: dict) -> dict[str, Any]:
    """Subset of a recorded row that the v_at_gaps worker reads.

    Trajectories are optional (drawn on left panel only); pre-computed V
    arrays + scalars are required by the middle / right panels."""
    payload: dict[str, Any] = {
        "episode": int(row["episode"]),
        "mode": row["mode"],
        "v_at_gap_0": float(row.get("v_at_gap_0", 0.0)),
        "v_at_gap_1": float(row.get("v_at_gap_1", 0.0)),
        "v_at_gap_2": float(row.get("v_at_gap_2", 0.0)),
        "actual_return_at_gap_0": float(row.get("actual_return_at_gap_0", 0.0)),
        "actual_return_at_gap_1": float(row.get("actual_return_at_gap_1", 0.0)),
        "actual_return_at_gap_2": float(row.get("actual_return_at_gap_2", 0.0)),
        "gap_idx": int(row.get("gap_idx", -1)),
        "drop_x": float(row.get("drop_x", 0.0)),
        "drop_y": float(row.get("drop_y", 0.0)),
        "goal_x": float(row.get("goal_x", 0.0)),
        "goal_y": float(row.get("goal_y", 0.0)),
        "follower_on_goal": float(row.get("follower_on_goal", 0.0)),
    }
    leader_traj = row.get("leader_traj")
    follower_traj = row.get("follower_traj")
    if leader_traj is not None:
        payload["leader_traj"] = np.asarray(leader_traj, dtype=np.float32)
    else:
        payload["leader_traj"] = np.zeros((0, 7), dtype=np.float32)
    if follower_traj is not None:
        payload["follower_traj"] = np.asarray(follower_traj, dtype=np.float32)
    else:
        payload["follower_traj"] = np.zeros((0, 7), dtype=np.float32)
    x_curve = row.get("v_along_wall_x")
    v_curve = row.get("v_along_wall_v")
    if x_curve is not None and v_curve is not None:
        payload["v_along_wall_x"] = np.asarray(x_curve, dtype=np.float32)
        payload["v_along_wall_v"] = np.asarray(v_curve, dtype=np.float32)
    # Optional VAE-derived along-wall arrays. Present iff eval was run
    # with a VAE (auto-discovered or explicit ``--vae``); the figure
    # renderer hides the OOD + V·(1-OOD) panels otherwise.
    ood_curve = row.get("ood_along_wall")
    disc_curve = row.get("v_disc_along_wall")
    if ood_curve is not None:
        payload["ood_along_wall"] = np.asarray(ood_curve, dtype=np.float32)
    if disc_curve is not None:
        payload["v_disc_along_wall"] = np.asarray(disc_curve, dtype=np.float32)
    return payload


def _save_wh_v_at_gaps_plots(
    all_results: List[dict],
    output_dir: Path,
    max_workers: int = 1,
    scenario_name: str = "warehouse",
) -> None:
    """Per-episode 3-panel diagnostic plot.

    * Left panel — arena with leader's object start, leader's full path,
      leader's drop position, follower's start, the goal, and the follower's
      full path (when trajectories were recorded).
    * Middle panel — V_follower(gap_i) vs actual discounted follower return
      (rolled out from gap_i with the same spawn/goal) as paired bars per gap.
    * Right panel — V_follower(x, wall_y) as a continuous function of x
      along the wall, with gap regions shaded and chosen drop position
      marked.

    Bar borders in the middle panel mark the leader's chosen gap (green,
    thick) and V's argmax (blue, thick) when not the chosen one.

    Output: ``output_dir/episodes/v_at_gaps/ep<NNN>_<mode>.png``.
    """
    if not all_results:
        return

    plot_dir = output_dir / "episodes" / "v_at_gaps"
    plot_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[FigureTask] = []
    for row in all_results:
        out_path = plot_dir / "ep{ep:03d}_{mode}.png".format(
            ep=int(row["episode"]), mode=row["mode"],
        )
        tasks.append(
            FigureTask(output_path=out_path, payload=_wh_v_at_gaps_payload(row)),
        )

    typer.echo(
        "Saving {n} arena+V-vs-actual plots to {p}/ ({w} workers)".format(
            n=len(tasks), p=plot_dir,
            w=max(1, min(max_workers, len(tasks))),
        )
    )
    run_figures_parallel(
        tasks=tasks,
        worker_cls=WhVAtGapsFigureWorker,
        worker_kwargs={"scenario_name": scenario_name},
        max_workers=int(max_workers),
        progress_desc="WH v_at_gaps",
    )


# ---------------------------------------------------------------------------
# Aggregate V-agreement metrics
# ---------------------------------------------------------------------------

def _v_at_gaps_array(row: dict) -> np.ndarray:
    return np.array(
        [row["v_at_gap_0"], row["v_at_gap_1"], row["v_at_gap_2"]],
        dtype=np.float32,
    )


def _actual_at_gaps_array(row: dict) -> np.ndarray:
    return np.array(
        [
            row["actual_return_at_gap_0"],
            row["actual_return_at_gap_1"],
            row["actual_return_at_gap_2"],
        ],
        dtype=np.float32,
    )


def _plot_wh_v_argmax_agreement(
    all_results: List[dict],
    output_dir: Path,
) -> None:
    """3×3 heatmap: V-argmax (rows) vs rollout-argmax (cols).

    Mode-independent (V and rollout values are computed once per episode and
    duplicated across mode rows), so we deduplicate by episode.
    """
    import matplotlib.pyplot as plt

    # Dedup by episode — V and rollout-returns are the same across modes.
    seen: dict[int, dict] = {}
    for row in all_results:
        ep = int(row.get("episode", -1))
        if ep not in seen:
            seen[ep] = row
    rows = list(seen.values())
    if not rows:
        return

    n_gaps = 3
    counts = np.zeros((n_gaps, n_gaps), dtype=np.int64)
    for row in rows:
        v_arg = int(np.argmax(_v_at_gaps_array(row)))
        r_arg = int(np.argmax(_actual_at_gaps_array(row)))
        counts[v_arg, r_arg] += 1
    total = int(counts.sum())
    pct = 100.0 * counts / max(total, 1)
    agree = int(np.trace(counts))
    pct_agree = 100.0 * agree / max(total, 1)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=max(pct.max(), 1.0))
    ax.set_xticks(range(n_gaps))
    ax.set_yticks(range(n_gaps))
    ax.set_xticklabels(["gap0", "gap1", "gap2"])
    ax.set_yticklabels(["gap0", "gap1", "gap2"])
    ax.set_xlabel("rollout argmax")
    ax.set_ylabel("V argmax")
    for i in range(n_gaps):
        for j in range(n_gaps):
            ax.text(
                j, i,
                "{p:.0f}%\n({c})".format(p=pct[i, j], c=int(counts[i, j])),
                ha="center", va="center",
                color="white" if pct[i, j] > 50 else "#212121",
                fontsize=9,
            )
    fig.colorbar(im, ax=ax, label="% of episodes")
    ax.set_title(
        "V-argmax vs rollout-argmax  ({a}/{t} agree, {p:.1f}%)".format(
            a=agree, t=total, p=pct_agree,
        ),
        fontsize=10,
    )
    fig.tight_layout()
    out = output_dir / "compare_warehouse_v_argmax_agreement.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    typer.echo("V-argmax agreement → {p}".format(p=out))


def _plot_wh_policy_follows_v(
    all_results: List[dict],
    output_dir: Path,
) -> None:
    """Per-mode bar: % of episodes where the chosen gap == V's argmax."""
    import matplotlib.pyplot as plt

    from ._common import _MODE_COLORS, _MODES_ORDER

    modes_present = sorted(
        {r["mode"] for r in all_results},
        key=lambda m: _MODES_ORDER.index(m) if m in _MODES_ORDER else 99,
    )
    if not modes_present:
        return

    rates: list[float] = []
    counts: list[tuple[int, int]] = []
    for mode in modes_present:
        subset = [r for r in all_results if r["mode"] == mode and int(r["gap_idx"]) >= 0]
        if not subset:
            rates.append(0.0)
            counts.append((0, 0))
            continue
        agree = sum(
            1 for r in subset
            if int(r["gap_idx"]) == int(np.argmax(_v_at_gaps_array(r)))
        )
        n = len(subset)
        rates.append(100.0 * agree / n)
        counts.append((agree, n))

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    colors = [_MODE_COLORS.get(m, "#888888") for m in modes_present]
    bars = ax.bar(modes_present, rates, color=colors, edgecolor="#37474F")
    for bar, val, (a, n) in zip(bars, rates, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            "{:.1f}%\n({a}/{n})".format(val, a=a, n=n),
            ha="center", va="bottom", fontsize=8,
        )
    # Random baseline reference (uniform over 3 gaps).
    ax.axhline(100.0 / 3.0, color="#9E9E9E", linewidth=1.2, linestyle="--", label="random (33.3%)")
    ax.set_ylabel("% chose == V argmax")
    ax.set_ylim(0, max(105, max(rates) * 1.2 if rates else 105))
    ax.set_title("Does the policy pick V's argmax gap?")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = output_dir / "compare_warehouse_policy_follows_v.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    typer.echo("Policy-follows-V    → {p}".format(p=out))


def _plot_wh_v_gain_over_baseline(
    all_results: List[dict],
    output_dir: Path,
) -> None:
    """Per-episode V gain of the adapted vs baseline mode.

    Per (b-iii): pick the first two modes ordered by ``_MODES_ORDER`` —
    first = baseline, second = adapted. Skipped (with a warning) when fewer
    than two modes are present.

    Left panel: histogram of ΔV = V(gap_adapted) − V(gap_baseline) per episode.
    Right panel: paired scatter of V(gap_baseline) vs V(gap_adapted) with y=x.
    """
    import matplotlib.pyplot as plt

    from ._common import _MODE_COLORS, _MODES_ORDER

    modes_present = sorted(
        {r["mode"] for r in all_results},
        key=lambda m: _MODES_ORDER.index(m) if m in _MODES_ORDER else 99,
    )
    if len(modes_present) < 2:
        typer.echo(
            "V-gain plot skipped: fewer than 2 modes present ({m}).".format(m=modes_present),
            err=True,
        )
        return
    baseline_mode, adapted_mode = modes_present[0], modes_present[1]

    by_mode_ep: dict[str, dict[int, dict]] = {
        m: {int(r["episode"]): r for r in all_results if r["mode"] == m}
        for m in (baseline_mode, adapted_mode)
    }
    common_eps = sorted(set(by_mode_ep[baseline_mode]) & set(by_mode_ep[adapted_mode]))
    if not common_eps:
        typer.echo("V-gain plot skipped: no common episodes between modes.", err=True)
        return

    # Use normalized follower return when calibration produced it — same
    # scale as the headline returns figure and the summary table.
    follower_key = (
        "follower_return_norm"
        if any("follower_return_norm" in r for r in all_results)
        else "follower_return"
    )

    v_baseline: list[float] = []
    v_adapted: list[float] = []
    actual_baseline: list[float] = []
    actual_adapted: list[float] = []
    for ep in common_eps:
        r_b = by_mode_ep[baseline_mode][ep]
        r_a = by_mode_ep[adapted_mode][ep]
        gb = int(r_b["gap_idx"])
        ga = int(r_a["gap_idx"])
        # When the chosen gap is "miss" (-1), use NaN so the episode is excluded
        # rather than silently treated as gap 0.
        v_b = float(_v_at_gaps_array(r_b)[gb]) if gb >= 0 else float("nan")
        v_a = float(_v_at_gaps_array(r_a)[ga]) if ga >= 0 else float("nan")
        v_baseline.append(v_b)
        v_adapted.append(v_a)
        actual_baseline.append(float(r_b[follower_key]))
        actual_adapted.append(float(r_a[follower_key]))
    arr_b = np.asarray(v_baseline, dtype=np.float32)
    arr_a = np.asarray(v_adapted, dtype=np.float32)
    actual_b = np.asarray(actual_baseline, dtype=np.float32)
    actual_a = np.asarray(actual_adapted, dtype=np.float32)
    valid = ~(np.isnan(arr_b) | np.isnan(arr_a))
    if not valid.any():
        typer.echo("V-gain plot skipped: no valid (non-miss) paired episodes.", err=True)
        return
    arr_b = arr_b[valid]
    arr_a = arr_a[valid]
    # The actual-return panel keeps every paired episode (including misses),
    # since "miss" outcomes still have a measured follower return — that's
    # part of the headline "did adapting help?" story.
    actual_b_all = actual_b
    actual_a_all = actual_a
    delta = arr_a - arr_b

    accent = _MODE_COLORS.get(adapted_mode, "#1565C0")

    fig, (ax_hist, ax_scatter, ax_actual) = plt.subplots(1, 3, figsize=(15, 4.5))

    # Histogram of ΔV.
    n_better = int(np.sum(delta > 0))
    n_equal = int(np.sum(delta == 0))
    n_worse = int(np.sum(delta < 0))
    n_total = int(len(delta))
    mean_delta = float(np.mean(delta))
    ax_hist.hist(delta, bins=20, color=accent, edgecolor="#0D47A1", alpha=0.85)
    ax_hist.axvline(0.0, color="#37474F", linestyle="--", linewidth=1.2)
    ax_hist.set_xlabel("ΔV = V(gap_{a}) − V(gap_{b})".format(a=adapted_mode, b=baseline_mode))
    ax_hist.set_ylabel("episodes")
    ax_hist.set_title(
        "Δ V distribution\n>0 in {bb}/{tt}; ==0 in {ee}/{tt}; <0 in {ww}/{tt};  mean Δ={md:+.3f}".format(
            bb=n_better, ee=n_equal, ww=n_worse, tt=n_total, md=mean_delta,
        ),
        fontsize=9,
    )
    ax_hist.grid(True, axis="y", alpha=0.3)

    # Paired V scatter.
    ax_scatter.scatter(arr_b, arr_a, alpha=0.6, color=accent, edgecolor="#0D47A1")
    lo = float(min(arr_b.min(), arr_a.min()))
    hi = float(max(arr_b.max(), arr_a.max()))
    ax_scatter.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="y = x")
    ax_scatter.set_xlabel("V(gap chosen by {b})".format(b=baseline_mode))
    ax_scatter.set_ylabel("V(gap chosen by {a})".format(a=adapted_mode))
    ax_scatter.set_title("Paired V — {a} vs {b}".format(a=adapted_mode, b=baseline_mode))
    ax_scatter.legend()
    ax_scatter.grid(True, alpha=0.3)

    # Paired actual-rollout-return scatter (includes miss episodes).
    n_better_actual = int(np.sum(actual_a_all > actual_b_all))
    n_eq_actual = int(np.sum(actual_a_all == actual_b_all))
    n_worse_actual = int(np.sum(actual_a_all < actual_b_all))
    n_total_actual = int(len(actual_a_all))
    ax_actual.scatter(actual_b_all, actual_a_all, alpha=0.6, color=accent, edgecolor="#0D47A1")
    if n_total_actual:
        lo_a = float(min(actual_b_all.min(), actual_a_all.min()))
        hi_a = float(max(actual_b_all.max(), actual_a_all.max()))
        ax_actual.plot([lo_a, hi_a], [lo_a, hi_a], "k--", alpha=0.5, label="y = x")
    ax_actual.set_xlabel("{fk} ({b})".format(fk=follower_key, b=baseline_mode))
    ax_actual.set_ylabel("{fk} ({a})".format(fk=follower_key, a=adapted_mode))
    ax_actual.set_title(
        "Actual {fk} — {a} vs {b}\n>{b} in {bb}/{tt};  ={b} in {ee}/{tt};  <{b} in {ww}/{tt}".format(
            fk=follower_key, a=adapted_mode, b=baseline_mode,
            bb=n_better_actual, ee=n_eq_actual, ww=n_worse_actual, tt=n_total_actual,
        ),
        fontsize=9,
    )
    ax_actual.legend()
    ax_actual.grid(True, alpha=0.3)

    fig.tight_layout()
    out = output_dir / "compare_warehouse_v_gain_over_baseline.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    typer.echo("V-gain vs baseline  → {p}".format(p=out))


__all__ = [
    "_compare_wh",
    "_plot_warehouse_extras",
    "_plot_wh_policy_follows_v",
    "_plot_wh_v_argmax_agreement",
    "_plot_wh_v_gain_over_baseline",
    "_save_wh_episode_mp4s",
    "_save_wh_v_at_gaps_plots",
]
