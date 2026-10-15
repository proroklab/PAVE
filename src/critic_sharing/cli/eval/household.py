"""household-specific eval logic.

Sequential leader-then-follower flow (mirrors obstacle_arena):

  1. Cleaner phase. Reset cleaner env from the shared
     ``EpisodeConditions``. Roll out the cleaner's PPO policy
     (``unguided``) — CEM modes raise ``NotImplementedError`` until
     wired. Stop on pickup or max_steps.
  2. Follower phase. Reset the navigator env using the **final table
     pose** the cleaner left behind. Roll out the navigator's PPO
     policy. Stop on goal-reach, table-collision, or max_steps.

The shared state across modes is the table's pose ``(t_x, t_y, θ_t)``
— that's what the cleaner controls and what V_F is conditioned on.

Trajectory recording captures cleaner pose + table pose per cleaner
step, and follower pose + table pose per follower step; the
``_animation.py`` worker turns these into multi-panel MP4s.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import typer

from critic_sharing.utils.parallel import EpisodeSpec, EpisodeWorker

from ._common import EvalMode
from .pipeline import EvalScenarioAdapter, run_eval_pipeline


# ---------------------------------------------------------------------------
# Controller builder
# ---------------------------------------------------------------------------

def _build_hh_controller(
    *,
    mode: EvalMode,
    cleaner_model,
    follower_model,
    seed: int,
    follower_spawn_pos: np.ndarray,
    follower_spawn_theta: float,
    follower_model_path: Optional[Path] = None,
    leader_range=None,
    follower_range=None,
    vae: Optional[Path] = None,
    alpha: Optional[float] = None,
):
    """Build the cleaner controller for household.

    Unguided → ``(None, None, "value")`` (the env's PPO policy steps the
    cleaner directly). Other modes build a CEMController that uses
    :func:`cleaner_dynamics_batch` to roll out candidate plans, a
    :class:`JointUtilityCost` blending leader + follower V_F (and the
    OOD trust factor when ``vae`` is provided), and the cleaner's PPO
    policy as the action prior.

    The follower value function is **per-episode**:
    ``follower_spawn_pos`` and ``follower_spawn_theta`` are baked into
    the closure so V_F answers
    ``V(candidate_table_pose | this_episode_follower_spawn_pose)``.
    Mirrors the warehouse / mobile_arm / OA V_F semantics — see
    :func:`build_household_follower_value_fn`.
    """

    if mode == EvalMode.unguided:
        return None, None, "value"

    import gymnasium as gym
    import torch as th

    from critic_sharing.config.resolver import (
        load_cost_config,
        load_env_config,
        load_planner_config,
    )
    from critic_sharing.planning.cem import CEMController
    from critic_sharing.planning.costs import JointUtilityCost
    from critic_sharing.planning.distributions import make_distribution
    from critic_sharing.planning.prior import PolicyPrior, UniformGaussianPrior
    from critic_sharing.scenarios.household import (
        build_household_follower_value_fn,
    )
    from critic_sharing.scenarios.household.push_env import cleaner_dynamics_batch

    clean_config = load_env_config("household", "leader")
    nav_config = load_env_config("household", "follower")
    cost_config = load_cost_config("household")
    cem_config = load_planner_config("household")
    terminal_value_mode = getattr(cem_config, "terminal_value", "value")

    # Per-episode V_F: bake the realized follower spawn pose into the
    # closure and query the nav PPO critic directly with the candidate
    # table pose from the leader rollout. Mirrors warehouse /
    # mobile_arm / obstacle_arena.
    value_fn = build_household_follower_value_fn(
        follower_model=follower_model,
        follower_config=nav_config,
        follower_spawn_pos=np.asarray(follower_spawn_pos, dtype=np.float32),
        follower_spawn_theta=float(follower_spawn_theta),
    )

    def follower_value_fn(obs_batch):
        return np.asarray(value_fn(obs_batch), dtype=np.float32).reshape(-1)

    utility_type = getattr(cost_config, "utility_type", "sum")
    leader_gamma = getattr(cost_config, "leader_gamma", 0.99)

    leader_value_fn = None
    if terminal_value_mode == "value":
        def _leader_value_fn(obs_batch):
            obs_t, _ = cleaner_model.policy.obs_to_tensor(
                np.asarray(obs_batch, dtype=np.float32),
            )
            with th.no_grad():
                values = cleaner_model.policy.predict_values(obs_t)
            return values.detach().cpu().numpy().reshape(-1)
        leader_value_fn = _leader_value_fn

    ood_metric = None
    effective_alpha = 1.0 if alpha is None else float(alpha)
    use_ood = vae is not None and mode in (
        EvalMode.value_and_ood, EvalMode.uniform, EvalMode.sabotage,
    )
    if use_ood:
        from critic_sharing.scenarios.household.ood_query import (
            build_household_ood_metric,
        )

        ood_metric = build_household_ood_metric(str(vae))

    cost = JointUtilityCost(
        follower_value_fn=follower_value_fn,
        leader_gamma=leader_gamma,
        utility_type=utility_type if mode != EvalMode.sabotage else "sabotage",
        leader_range=leader_range,
        follower_range=follower_range,
        leader_value_fn=leader_value_fn,
        ood_metric=ood_metric,
        alpha=effective_alpha,
    )

    def dynamics_fn(obs_batch, act_batch):
        return cleaner_dynamics_batch(obs_batch, act_batch, clean_config)

    action_space = gym.spaces.Box(
        low=np.array([-clean_config.max_speed, -clean_config.max_speed,
                      -clean_config.max_yaw_rate], dtype=np.float32),
        high=np.array([clean_config.max_speed, clean_config.max_speed,
                       clean_config.max_yaw_rate], dtype=np.float32),
        dtype=np.float32,
    )

    if mode == EvalMode.uniform:
        prior = UniformGaussianPrior(
            action_low=action_space.low,
            action_high=action_space.high,
            sigma_frac=1.0,
            rng=np.random.default_rng(seed),
        )
    else:
        prior = PolicyPrior(
            predict_fn=lambda obs: cleaner_model.predict(obs, deterministic=True)[0],
            predict_batch_fn=lambda obs_batch: cleaner_model.predict(
                np.asarray(obs_batch, dtype=np.float32), deterministic=True,
            )[0],
        )

    controller = CEMController(
        action_space=action_space,
        dynamics_fn=dynamics_fn,
        cost=cost,
        distribution=make_distribution(
            cem_config.distribution,
            noise_sigma=cem_config.noise_sigma,
            min_std=cem_config.min_std,
        ),
        horizon=cem_config.horizon,
        num_samples=cem_config.num_samples,
        num_iters=cem_config.num_iters,
        elite_frac=cem_config.elite_frac,
        replan_interval=cem_config.replan_interval,
        prior=prior,
        seed=seed,
    )
    return controller, cost, terminal_value_mode


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def _run_episode_hh(
    clean_env,
    nav_env,
    cleaner_model,
    follower_model,
    controller,
    *,
    ep_seed: int,
    follower_seeds: Optional[list] = None,
    record: bool = False,
    table_pos: Optional[np.ndarray] = None,
    table_theta: Optional[float] = None,
    cleaner_pos: Optional[np.ndarray] = None,
    cleaner_theta: Optional[float] = None,
    follower_pos: Optional[np.ndarray] = None,
    follower_theta: Optional[float] = None,
    cost=None,
    terminal_value_mode: str = "value",
    leader_gamma: float = 0.99,
) -> dict:
    """Run one sequential leader→follower household episode."""

    if follower_seeds is None:
        follower_seeds = [ep_seed]

    leader_traj: list = []

    # --- cleaner (leader) phase ---
    reset_options: dict = {}
    if table_pos is not None:
        reset_options["table_pos"] = table_pos
    if table_theta is not None:
        reset_options["table_theta"] = table_theta
    if cleaner_pos is not None:
        reset_options["cleaner_pos"] = cleaner_pos
    if cleaner_theta is not None:
        reset_options["cleaner_theta"] = cleaner_theta
    obs, _ = clean_env.reset(seed=ep_seed, options=reset_options or None)
    if controller is not None:
        controller.reset()

    leader_return = 0.0
    incurred_disc = 0.0
    gamma_pow_t = 1.0
    leader_steps = 0
    leader_info: dict = {}
    done = False
    while not done:
        if record:
            leader_traj.append({
                "cleaner_pos": obs[:2].copy(),
                "cleaner_theta_sin_cos": (float(obs[2]), float(obs[3])),
                "table_pos": obs[4:6].copy(),
                "table_theta_sin_cos": (float(obs[6]), float(obs[7])),
                "dirt_pos": obs[8:10].copy(),
            })
        if controller is not None:
            if cost is not None:
                cost.set_incurred_return(incurred_disc, gamma_pow_t)
            if terminal_value_mode == "rollout":
                steps_remaining = int(clean_env.config.max_steps) - int(clean_env._steps)
                action = controller.command(obs, steps_remaining=steps_remaining)
            else:
                action = controller.command(obs)
        else:
            action, _ = cleaner_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, leader_info = clean_env.step(action)
        leader_return += float(reward)
        incurred_disc += gamma_pow_t * float(reward)
        gamma_pow_t *= float(leader_gamma)
        leader_steps += 1
        done = terminated or truncated
    if record:
        leader_traj.append({
            "cleaner_pos": obs[:2].copy(),
            "cleaner_theta_sin_cos": (float(obs[2]), float(obs[3])),
            "table_pos": obs[4:6].copy(),
            "table_theta_sin_cos": (float(obs[6]), float(obs[7])),
            "dirt_pos": obs[8:10].copy(),
        })

    # Capture final table pose (sin, cos) → recover θ for follower reset.
    final_table_pos = obs[4:6].copy()
    final_table_sin, final_table_cos = float(obs[6]), float(obs[7])
    final_table_theta = float(np.arctan2(final_table_sin, final_table_cos))
    leader_picked = bool(leader_info.get("picked_up_dirt", False))

    # --- follower (navigator) phase ---
    f_returns: list[float] = []
    f_returns_disc: list[float] = []
    f_steps: list[int] = []
    f_reached: list[int] = []
    f_collided: list[int] = []
    follower_traj: list = []
    goal_pos = None

    for i, fs in enumerate(follower_seeds):
        nav_options: dict = {
            "table_pos": final_table_pos,
            "table_theta": final_table_theta,
            # follower spawns in upper half at deployment time.
            "spawn_region": "upper" if nav_env.config.wall_enabled else "full",
        }
        if follower_pos is not None:
            nav_options["follower_pos"] = follower_pos
        if follower_theta is not None:
            nav_options["follower_theta"] = follower_theta
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
                follower_traj.append({
                    "follower_pos": obs[:2].copy(),
                    "follower_theta_sin_cos": (float(obs[2]), float(obs[3])),
                    "table_pos": obs[4:6].copy(),
                    "table_theta_sin_cos": (float(obs[6]), float(obs[7])),
                })
            action, _ = follower_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = nav_env.step(action)
            fr += float(reward)
            fr_disc += f_gamma_pow_t * float(reward)
            f_gamma_pow_t *= float(leader_gamma)
            fsteps += 1
            done = terminated or truncated
        if record and i == 0:
            follower_traj.append({
                "follower_pos": obs[:2].copy(),
                "follower_theta_sin_cos": (float(obs[2]), float(obs[3])),
                "table_pos": obs[4:6].copy(),
                "table_theta_sin_cos": (float(obs[6]), float(obs[7])),
            })

        reached_goal = bool(info.get("reached_goal", False))
        collided = bool(info.get("collided", False))
        f_returns.append(fr)
        f_returns_disc.append(fr_disc)
        f_steps.append(fsteps)
        f_reached.append(int(reached_goal))
        f_collided.append(int(collided))

    follower_return = float(np.mean(f_returns))
    follower_return_disc = float(np.mean(f_returns_disc))
    follower_steps = float(np.mean(f_steps))

    result: dict = {
        "leader_return": incurred_disc,
        "follower_return": follower_return_disc,
        "leader_return_undisc": leader_return,
        "follower_return_undisc": follower_return,
        "total_return": incurred_disc + follower_return_disc,
        "leader_steps": leader_steps,
        "follower_steps": follower_steps,
        "leader_picked_up_dirt": int(leader_picked),
        "follower_reached_goal": float(np.mean(f_reached)),
        "follower_collided": float(np.mean(f_collided)),
    }
    if record:
        result["leader_traj"] = leader_traj
        result["follower_traj"] = follower_traj
        result["goal_pos"] = goal_pos
        result["final_table_pos"] = final_table_pos
        result["final_table_theta"] = final_table_theta
    return result


# ---------------------------------------------------------------------------
# Parallel worker
# ---------------------------------------------------------------------------

class HHEpisodeWorker(EpisodeWorker):
    """One worker per mode. Loads heavy resources on setup; builds the CEM
    controller per episode with ``spec.planner_seed``."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._leader_path: str = kwargs["leader_path"]
        self._follower_path: str = kwargs["follower_path"]
        self._vae_path: Optional[str] = kwargs.get("vae_path")
        self._alpha: Optional[float] = kwargs.get("alpha")
        self._cleaner_model = None
        self._follower_model = None
        self._clean_env = None
        self._nav_env = None
        self._leader_range = None
        self._follower_range = None
        self._leader_gamma = 0.99

    def setup(self) -> None:
        from stable_baselines3 import PPO

        from critic_sharing.config.resolver import load_cost_config, load_env_config
        from critic_sharing.scenarios.household.nav_env import HouseholdNavigatorEnv
        from critic_sharing.scenarios.household.push_env import HouseholdCleanerEnv

        clean_cfg = load_env_config("household", "leader")
        nav_cfg = load_env_config("household", "follower")

        self._cleaner_model = PPO.load(self._leader_path)
        self._follower_model = PPO.load(self._follower_path)
        self._clean_env = HouseholdCleanerEnv(clean_cfg)
        self._nav_env = HouseholdNavigatorEnv(nav_cfg)

        self._leader_range = self._kwargs.get("leader_range")
        self._follower_range = self._kwargs.get("follower_range")
        self._leader_gamma = float(
            getattr(load_cost_config("household"), "leader_gamma", 0.99),
        )

    def run_episode(self, spec: EpisodeSpec) -> dict:
        # Per-episode V_F needs the realized follower spawn pose.
        # Both spec.follower_pos and spec.follower_theta are populated
        # by the household adapter's sample_eval_conditions.
        if spec.follower_pos is not None:
            follower_spawn_pos = np.array(spec.follower_pos, dtype=np.float32)
        else:
            follower_spawn_pos = np.zeros(2, dtype=np.float32)
        follower_spawn_theta = (
            float(spec.follower_theta) if spec.follower_theta is not None else 0.0
        )
        controller, cost, terminal_value_mode = _build_hh_controller(
            mode=EvalMode(self.mode),
            cleaner_model=self._cleaner_model,
            follower_model=self._follower_model,
            seed=int(spec.planner_seed),
            follower_spawn_pos=follower_spawn_pos,
            follower_spawn_theta=follower_spawn_theta,
            follower_model_path=Path(self._follower_path),
            leader_range=self._leader_range,
            follower_range=self._follower_range,
            vae=Path(self._vae_path) if self._vae_path else None,
            alpha=self._alpha,
        )
        return _run_episode_hh(
            self._clean_env,
            self._nav_env,
            self._cleaner_model,
            self._follower_model,
            controller,
            ep_seed=spec.leader_seed,
            follower_seeds=spec.follower_seeds,
            record=spec.record,
            table_pos=np.array(spec.obstacle_pos, dtype=np.float32)
            if spec.obstacle_pos else None,
            table_theta=spec.obstacle_theta,
            cleaner_pos=np.array(spec.leader_pos, dtype=np.float32)
            if spec.leader_pos else None,
            cleaner_theta=spec.leader_theta,
            follower_pos=np.array(spec.follower_pos, dtype=np.float32)
            if spec.follower_pos else None,
            follower_theta=spec.follower_theta,
            cost=cost,
            terminal_value_mode=terminal_value_mode,
            leader_gamma=self._leader_gamma,
        )


# ---------------------------------------------------------------------------
# Compare backend
# ---------------------------------------------------------------------------

class HouseholdAdapter(EvalScenarioAdapter):
    """:class:`EvalScenarioAdapter` for the household scenario."""

    scenario_name = "household"
    worker_cls = HHEpisodeWorker

    def __init__(
        self,
        *,
        leader_path: str,
        follower_path: str,
        fixed_table: Optional[str] = None,
    ) -> None:
        self._leader_path = leader_path
        self._follower_path = follower_path
        self._fixed_table = fixed_table

    def build_calibration(self):
        from stable_baselines3 import PPO

        from critic_sharing.config.resolver import load_env_config
        from critic_sharing.scenarios.household.nav_env import HouseholdNavigatorEnv
        from critic_sharing.scenarios.household.push_env import HouseholdCleanerEnv

        leader_env = HouseholdCleanerEnv(load_env_config("household", "leader"))
        follower_env = HouseholdNavigatorEnv(load_env_config("household", "follower"))
        leader_model = PPO.load(self._leader_path)
        follower_model = PPO.load(self._follower_path)
        return leader_env, follower_env, leader_model, follower_model

    def sample_conditions(self, rng, n_episodes):
        from critic_sharing.config.resolver import load_env_config
        from critic_sharing.scenarios.household.initial_conditions import (
            sample_eval_conditions,
        )

        clean_cfg = load_env_config("household", "leader")
        nav_cfg = load_env_config("household", "follower")
        conditions = sample_eval_conditions(
            clean_cfg=clean_cfg,
            nav_cfg=nav_cfg,
            n_episodes=n_episodes,
            rng=rng,
            fixed_table=self._fixed_table,
        )
        typer.echo(
            "  Pre-sampled {n} initial conditions (table+cleaner+follower pose)".format(
                n=n_episodes,
            )
        )
        return conditions

    def make_spec(
        self, *, episode, mode, leader_seed, follower_seed, planner_seed,
        condition, paths, alpha, record,
    ):
        return EpisodeSpec(
            scenario="household",
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
            obstacle_pos=condition.table_pos.tolist(),
            obstacle_theta=float(condition.table_theta),
            leader_pos=condition.cleaner_pos.tolist(),
            leader_theta=float(condition.cleaner_theta),
            follower_pos=condition.follower_pos.tolist(),
            follower_theta=float(condition.follower_theta),
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

    # Household OOD is based on table pose (cols 4-7 of the cleaner obs).
    # The table doesn't move during the cleaner's episode, so the
    # initial table pose (carried on EpisodeSpec, also written into
    # the result row's spec_* columns when present) is a defensible
    # proxy. Left as default ``None`` until we wire a real handoff
    # column into _run_episode_hh's result dict.


def _compare_hh(
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
    workers: int = 1,
    record: bool = True,
    fixed_table: Optional[str] = None,
) -> list[dict]:
    """Thin wrapper: :class:`HouseholdAdapter` + :func:`run_eval_pipeline`."""
    adapter = HouseholdAdapter(
        leader_path=str(leader_model),
        follower_path=str(follower_model),
        fixed_table=fixed_table,
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


__all__ = [
    "HHEpisodeWorker",
    "HouseholdAdapter",
    "_build_hh_controller",
    "_compare_hh",
    "_run_episode_hh",
]
