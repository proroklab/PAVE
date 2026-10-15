"""obstacle_arena-specific eval logic."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import typer

from critic_sharing.utils.parallel import EpisodeSpec, EpisodeWorker

from ._common import EvalMode
from .pipeline import EvalScenarioAdapter, run_eval_pipeline

# ---------------------------------------------------------------------------
# obstacle_arena — controller builder
# ---------------------------------------------------------------------------


def _build_oa_controller(
    *,
    push_model,
    nav_model,
    nav_config,
    mode: EvalMode,
    vae: Optional[Path],
    alpha: Optional[float] = None,
    seed: int,
    follower_spawn_pos: np.ndarray,
    scenario: str = "obstacle_arena",
    follower_model_path: Optional[Path] = None,
    leader_range: Optional[tuple] = None,
    follower_range: Optional[tuple] = None,
):
    """Build the leader controller for obstacle_arena.

    Returns ``(controller, cost, terminal_value_mode)`` — same contract
    as the warehouse and mobile_arm builders. ``unguided`` mode returns
    ``(None, None, "value")`` since there's nothing to plan with.

    The follower value function is **per-episode**: ``follower_spawn_pos``
    is baked into the closure, so V_F answers
    ``V(candidate_obstacle_pos | this_episode_follower_spawn)``. This
    matches the warehouse / mobile_arm V_F semantics — see
    :func:`build_obstacle_arena_follower_value_fn` for the obs-build
    contract. (The marginalized
    :class:`NavValueAggregator` lives on for the
    standalone visualize / training paths but no longer enters this
    eval path.)

    The CEM dynamics emit the push env's own per-step reward
    (wall-distance progress plus ``goal_reward`` on wall contact, see
    :func:`push_step_batch`), so the leader-side partial sum is on the
    same scale as the naive-policy returns ``leader_range`` is
    calibrated on.
    """
    if mode == EvalMode.unguided:
        return None, None, "value"

    import gymnasium as gym

    from critic_sharing.config.resolver import load_env_config
    from critic_sharing.planning.factory import MODE_SETTINGS, build_cem_controller
    from critic_sharing.scenarios.obstacle_arena import (
        build_obstacle_arena_follower_value_fn,
    )
    from critic_sharing.scenarios.obstacle_arena.push_env import push_step_batch

    push_config = load_env_config(scenario, "leader")
    value_fn = build_obstacle_arena_follower_value_fn(
        follower_model=nav_model,
        follower_config=nav_config,
        follower_spawn_pos=np.asarray(follower_spawn_pos, dtype=np.float32),
    )

    utility_type, use_ood, prior_kind = MODE_SETTINGS[mode.value]
    ood_metric = None
    if use_ood and vae is not None:
        from critic_sharing.scenarios.obstacle_arena.ood_query import (
            build_obstacle_arena_ood_metric,
        )

        ood_metric = build_obstacle_arena_ood_metric(str(vae))

    # Full leader obs in, (full obs, env reward, wall contact) out.
    def dynamics_fn(obs_batch, act_batch):
        agent = obs_batch[:, :2]
        obstacle = obs_batch[:, 2:4]
        next_a, next_o, rewards, dones = push_step_batch(agent, obstacle, act_batch, push_config)
        next_obs = np.concatenate([next_a, next_o], axis=1).astype(np.float32)
        return next_obs, rewards, dones

    action_space = gym.spaces.Box(
        low=-push_config.max_speed,
        high=push_config.max_speed,
        shape=(2,),
        dtype=np.float32,
    )
    return build_cem_controller(
        scenario,
        leader_model=push_model,
        follower_value_fn=value_fn,
        dynamics_fn=dynamics_fn,
        action_space=action_space,
        seed=seed,
        utility_type=utility_type,
        leader_range=leader_range,
        follower_range=follower_range,
        ood_metric=ood_metric,
        alpha=1.0 if alpha is None else float(alpha),
        prior_kind=prior_kind,
    )


# ---------------------------------------------------------------------------
# obstacle_arena — episode runner
# ---------------------------------------------------------------------------


def _run_episode_oa(
    push_env,
    nav_env,
    push_model,
    nav_model,
    controller,
    ep_seed: int,
    follower_seeds: Optional[list] = None,
    record: bool = False,
    obstacle_pos: Optional[np.ndarray] = None,
    leader_pos: Optional[np.ndarray] = None,
    follower_pos: Optional[np.ndarray] = None,
    cost=None,
    terminal_value_mode: str = "value",
    leader_gamma: float = 0.99,
) -> dict:
    """Run one leader->follower obstacle_arena episode. Returns raw metric dict.

    The leader phase is run once.  The follower phase is run once for each seed
    in follower_seeds (defaults to [ep_seed] for backward compatibility), and all
    follower metrics are averaged across those runs.

    When record=True the dict also contains trajectories for the FIRST follower
    seed only:
      leader_traj   list of (agent_pos, obstacle_pos) arrays, one per step
      follower_traj list of (nav_pos,   obstacle_pos) arrays, one per step
      goal_pos      np.ndarray -- goal position used by the first follower run
    """

    if follower_seeds is None:
        follower_seeds = [ep_seed]

    leader_traj: list = []

    # --- leader phase ---
    reset_options: dict = {}
    if obstacle_pos is not None:
        reset_options["obstacle_pos"] = obstacle_pos
    if leader_pos is not None:
        reset_options["agent_pos"] = leader_pos
    obs, _ = push_env.reset(seed=ep_seed, options=reset_options or None)
    if controller is not None:
        controller.reset()
    leader_return = 0.0
    leader_steps = 0
    done = False
    # See warehouse._run_episode_wh — same incurred / γ^t bookkeeping +
    # terminal_value gating. (obstacle_arena's dynamics_fn emits zero
    # per-step rewards so the only cost contribution is via terminal
    # bootstrap or extension, but we keep the bookkeeping uniform.)
    incurred_disc = 0.0
    gamma_pow_t = 1.0
    leader_info: dict = {}
    while not done:
        if record:
            leader_traj.append((obs[:2].copy(), obs[2:4].copy()))
        if controller is not None:
            if cost is not None:
                cost.set_incurred_return(incurred_disc, gamma_pow_t)
            if terminal_value_mode == "rollout":
                steps_remaining = int(push_env.config.max_steps) - int(push_env._steps)
                action = controller.command(obs, steps_remaining=steps_remaining)
            else:
                action = controller.command(obs)
        else:
            action, _ = push_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, leader_info = push_env.step(action)
        leader_return += float(reward)
        incurred_disc += gamma_pow_t * float(reward)
        gamma_pow_t *= float(leader_gamma)
        leader_steps += 1
        done = terminated or truncated
    if record:
        leader_traj.append((obs[:2].copy(), obs[2:4].copy()))
    final_obstacle_pos = obs[2:4].copy()
    # Env-logged leader-success event (the obstacle ended touching the
    # wall, which terminates the episode in PushObstacleEnv). Captured
    # at termination from the env's info dict so analyze.py can compute
    # S_L without going through return-based proxies.
    leader_hit_wall = bool(leader_info.get("hit_wall", False))

    # --- follower phase — averaged over follower_seeds ---
    f_returns: list[float] = []
    f_steps: list[int] = []
    f_reached: list[int] = []
    f_collided: list[int] = []
    follower_traj: list = []
    goal_pos = None

    f_returns_disc: list[float] = []
    for i, fs in enumerate(follower_seeds):
        nav_options: dict = {"obstacle_pos": final_obstacle_pos}
        if follower_pos is not None:
            nav_options["agent_pos"] = follower_pos
        obs, _ = nav_env.reset(seed=fs, options=nav_options)
        if i == 0:
            goal_pos = nav_env._goal.copy()
        fr = 0.0
        fr_disc = 0.0
        f_gamma_pow_t = 1.0
        fsteps = 0
        info: dict = {}
        done = False
        while not done:
            if record and i == 0:
                follower_traj.append((obs[:2].copy(), obs[2:4].copy()))
            action, _ = nav_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = nav_env.step(action)
            fr += float(reward)
            fr_disc += f_gamma_pow_t * float(reward)
            f_gamma_pow_t *= float(leader_gamma)
            fsteps += 1
            done = terminated or truncated
        if record and i == 0:
            follower_traj.append((obs[:2].copy(), obs[2:4].copy()))

        reached_goal = bool(info.get("reached_goal", False))
        f_returns.append(fr)
        f_returns_disc.append(fr_disc)
        f_steps.append(fsteps)
        f_reached.append(int(reached_goal))
        f_collided.append(int(terminated and not reached_goal))

    follower_return = float(np.mean(f_returns))
    follower_return_disc = float(np.mean(f_returns_disc))
    follower_steps = float(np.mean(f_steps))

    # See warehouse.py — store discounted returns so they match the
    # discounted calibration bands; preserve undiscounted sums under
    # ``*_return_undisc`` for descriptive use.
    result: dict = {
        "leader_return": incurred_disc,
        "follower_return": follower_return_disc,
        "leader_return_undisc": leader_return,
        "follower_return_undisc": follower_return,
        "total_return": incurred_disc + follower_return_disc,
        "leader_steps": leader_steps,
        "follower_steps": follower_steps,
        "leader_hit_wall": int(leader_hit_wall),
        "follower_reached_goal": float(np.mean(f_reached)),
        "follower_collided": float(np.mean(f_collided)),
    }
    if record:
        result["leader_traj"] = leader_traj
        result["follower_traj"] = follower_traj
        result["goal_pos"] = goal_pos
    return result


# ---------------------------------------------------------------------------
# obstacle_arena — parallel worker
# ---------------------------------------------------------------------------


class OAEpisodeWorker(EpisodeWorker):
    """Worker that loads heavy resources once and builds the CEM controller per episode.

    :meth:`setup` loads the PPO models and env instances (the heavy
    per-worker cost). :meth:`run_episode` constructs a fresh
    ``CEMController`` per episode with ``spec.planner_seed`` so the CEM
    sample stream is per-episode-reproducible and independent of parallel
    dispatch order.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._leader_path: str = kwargs["leader_path"]
        self._follower_path: str = kwargs["follower_path"]
        self._vae_path: Optional[str] = kwargs.get("vae_path")
        self._alpha: Optional[float] = kwargs.get("alpha")

        # Set by setup()
        self._push_model = None
        self._nav_model = None
        self._push_env = None
        self._nav_env = None
        self._nav_cfg = None
        self._leader_range = None
        self._follower_range = None
        self._leader_gamma = 0.99

    def setup(self) -> None:
        from stable_baselines3 import PPO

        from critic_sharing.config.resolver import (
            load_cost_config,
            load_env_config,
            load_planner_config,
        )
        from critic_sharing.scenarios.obstacle_arena.nav_env import OneObstacleReachEnv
        from critic_sharing.scenarios.obstacle_arena.push_env import PushObstacleEnv

        push_cfg = load_env_config("obstacle_arena", "leader")
        nav_cfg = load_env_config("obstacle_arena", "follower")
        self._nav_cfg = nav_cfg

        self._push_model = PPO.load(self._leader_path)
        self._nav_model = PPO.load(self._follower_path)
        self._push_env = PushObstacleEnv(push_cfg)
        self._nav_env = OneObstacleReachEnv(nav_cfg)

        self._leader_range = self._kwargs.get("leader_range")
        self._follower_range = self._kwargs.get("follower_range")
        self._leader_gamma = float(
            getattr(load_cost_config("obstacle_arena"), "leader_gamma", 0.99),
        )

        # Per-worker thread budget (rollout mode benefits from > 1; value
        # mode is dominated by tiny single forwards so 1 is fine). Set
        # here based on the planner-config flag regardless of the CEM
        # terminal_value mode used per-episode (the cost of one extra
        # thread when value is the mode is negligible).
        cem_cfg = load_planner_config("obstacle_arena")
        prior_threads = int(getattr(cem_cfg, "prior_extension_threads", 1))
        terminal_value = getattr(cem_cfg, "terminal_value", "value")
        if prior_threads > 1 and terminal_value == "rollout":
            try:
                import torch as th

                th.set_num_threads(prior_threads)
            except Exception:
                pass

    def run_episode(self, spec: EpisodeSpec) -> dict:
        # Per-episode V_F needs the realized follower spawn. Comes from
        # the adapter's sample_eval_conditions output (spec.follower_pos).
        # When unset (unguided shortcut), build_oa_controller returns
        # early and the value never reaches build_obstacle_arena_follower_value_fn.
        if spec.follower_pos is not None:
            follower_spawn_pos = np.array(spec.follower_pos, dtype=np.float32)
        else:
            follower_spawn_pos = np.zeros(2, dtype=np.float32)
        controller, cost, terminal_value_mode = _build_oa_controller(
            push_model=self._push_model,
            nav_model=self._nav_model,
            nav_config=self._nav_cfg,
            mode=EvalMode(self.mode),
            vae=Path(self._vae_path) if self._vae_path else None,
            alpha=self._alpha,
            seed=int(spec.planner_seed),
            follower_spawn_pos=follower_spawn_pos,
            follower_model_path=Path(self._follower_path),
            leader_range=self._leader_range,
            follower_range=self._follower_range,
        )
        return _run_episode_oa(
            self._push_env,
            self._nav_env,
            self._push_model,
            self._nav_model,
            controller,
            ep_seed=spec.leader_seed,
            follower_seeds=spec.follower_seeds,
            obstacle_pos=np.array(spec.obstacle_pos, dtype=np.float32)
            if spec.obstacle_pos
            else None,
            leader_pos=np.array(spec.leader_pos, dtype=np.float32) if spec.leader_pos else None,
            follower_pos=np.array(spec.follower_pos, dtype=np.float32)
            if spec.follower_pos
            else None,
            record=spec.record,
            cost=cost,
            terminal_value_mode=terminal_value_mode,
            leader_gamma=self._leader_gamma,
        )


# ---------------------------------------------------------------------------
# obstacle_arena — compare backend
# ---------------------------------------------------------------------------


_OA_CEM_MODES = {
    EvalMode.value_only,
    EvalMode.value_and_ood,
    EvalMode.uniform,
    EvalMode.sabotage,
}


class ObstacleArenaAdapter(EvalScenarioAdapter):
    """:class:`EvalScenarioAdapter` for the obstacle_arena scenario."""

    scenario_name = "obstacle_arena"
    worker_cls = OAEpisodeWorker

    def __init__(
        self,
        *,
        leader_path: str,
        follower_path: str,
        obstacle_distance: Optional[float] = None,
        fixed_obstacle: Optional[str] = None,
    ) -> None:
        self._leader_path = leader_path
        self._follower_path = follower_path
        self._obstacle_distance = obstacle_distance
        self._fixed_obstacle = fixed_obstacle

    def build_calibration(self):
        from stable_baselines3 import PPO

        from critic_sharing.config.resolver import load_env_config
        from critic_sharing.scenarios.obstacle_arena.nav_env import OneObstacleReachEnv
        from critic_sharing.scenarios.obstacle_arena.push_env import PushObstacleEnv

        leader_env = PushObstacleEnv(load_env_config("obstacle_arena", "leader"))
        follower_env = OneObstacleReachEnv(load_env_config("obstacle_arena", "follower"))
        leader_model = PPO.load(self._leader_path)
        follower_model = PPO.load(self._follower_path)
        return leader_env, follower_env, leader_model, follower_model

    def sample_conditions(self, rng, n_episodes):
        from critic_sharing.config.resolver import load_env_config
        from critic_sharing.scenarios.obstacle_arena.initial_conditions import (
            sample_eval_conditions,
        )

        push_cfg = load_env_config("obstacle_arena", "leader")
        nav_cfg = load_env_config("obstacle_arena", "follower")
        conditions = sample_eval_conditions(
            push_cfg=push_cfg,
            nav_cfg=nav_cfg,
            n_episodes=n_episodes,
            rng=rng,
            obstacle_distance=self._obstacle_distance,
            fixed_obstacle=self._fixed_obstacle,
        )

        if self._fixed_obstacle is not None and conditions:
            pos = conditions[0].obstacle_pos
            typer.echo(
                "  Fixed obstacle at ({x:.2f}, {y:.2f}) for all episodes".format(
                    x=float(pos[0]), y=float(pos[1]),
                )
            )
        elif self._obstacle_distance is not None:
            typer.echo(
                "  Constraining obstacle spawn within {d:.1f} of goal".format(
                    d=self._obstacle_distance,
                )
            )
        typer.echo(
            "  Pre-sampled {n} initial conditions (leader, obstacle, follower) for consistency".format(
                n=n_episodes,
            )
        )
        return conditions

    def make_spec(
        self, *, episode, mode, leader_seed, follower_seed, planner_seed,
        condition, paths, alpha, record,
    ):
        return EpisodeSpec(
            scenario="obstacle_arena",
            mode=mode,
            episode=episode,
            leader_seed=leader_seed,
            follower_seeds=[follower_seed],
            planner_seed=planner_seed,
            record=record,
            leader_model_path=paths["leader_model"],
            follower_model_path=paths["follower_model"],
            vae_path=paths.get("vae"),
            alpha=alpha,
            obstacle_pos=condition.obstacle_pos.tolist()
            if condition.obstacle_pos is not None else None,
            leader_pos=condition.leader_pos.tolist()
            if condition.leader_pos is not None else None,
            follower_pos=condition.follower_pos.tolist()
            if condition.follower_pos is not None else None,
        )

    def worker_kwargs(self, *, leader_range, follower_range, paths, alpha):
        return {
            "leader_path": paths["leader_model"],
            "follower_path": paths["follower_model"],
            "vae_path": paths.get("vae"),
            "alpha": alpha,
            "leader_range": leader_range,
            "follower_range": follower_range,
        }

    # OA doesn't store the realized handoff state in the result dict yet
    # (no obstacle_final_x/y column). Leaving handoff_obs_for_ood at
    # default (None) makes total_return_weighted fall back to
    # L_norm + alpha * F_norm — same as the no-VAE path. Wire a real
    # hook here once obstacle_final_pos is in the row.


def _compare_oa(
    *,
    leader_model: Path,
    follower_model: Path,
    vae: Optional[Path],
    alpha: Optional[float] = None,
    modes: list[EvalMode] = (),
    leader_seeds: list[int] = (),
    follower_seeds: list[int] = (),
    planner_seeds: list[int] = (),
    n_follower_samples: int = 1,
    obstacle_distance: Optional[float] = None,
    fixed_obstacle: Optional[str] = None,
    workers: int = 1,
    record: bool = True,
) -> list[dict]:
    """Thin wrapper: :class:`ObstacleArenaAdapter` + :func:`run_eval_pipeline`."""
    adapter = ObstacleArenaAdapter(
        leader_path=str(leader_model),
        follower_path=str(follower_model),
        obstacle_distance=obstacle_distance,
        fixed_obstacle=fixed_obstacle,
    )
    return run_eval_pipeline(
        adapter,
        leader_model=leader_model,
        follower_model=follower_model,
        vae=vae,
        alpha=alpha,
        modes=modes,
        leader_seeds=leader_seeds,
        follower_seeds=follower_seeds,
        planner_seeds=planner_seeds,
        n_follower_samples=n_follower_samples,
        workers=workers,
        record=record,
    )


