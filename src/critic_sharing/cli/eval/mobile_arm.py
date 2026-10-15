"""Mobile_arm-specific eval logic for ``critic-sharing eval --scenario mobile_arm``.

Mode mapping:
* :attr:`EvalMode.unguided` — leader's PPO policy alone, no planner.
* :attr:`EvalMode.value_only` — leader + CEM overlay maximising
  ``r_leader + V_follower(transformed_obs)``.
* :attr:`EvalMode.sabotage` — leader + CEM overlay maximising
  ``r_leader - V_follower(transformed_obs)``.
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
    run_figures_parallel,
)

from critic_sharing.planning.factory import MODE_SETTINGS

from ._common import EvalMode
from .pipeline import EvalScenarioAdapter, run_eval_pipeline


_SUPPORTED_MODES = {
    EvalMode.unguided,
    EvalMode.value_only,
    EvalMode.value_and_ood,
    EvalMode.uniform,
    EvalMode.sabotage,
}
# CEM modes that consume JointUtilityCost (skip calibration when only
# unguided is requested).
_CEM_MODES = {
    EvalMode.value_only,
    EvalMode.value_and_ood,
    EvalMode.uniform,
    EvalMode.sabotage,
}


# ---------------------------------------------------------------------------
# CEM controller
# ---------------------------------------------------------------------------

def _build_ma_controller(
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
):
    """Build the mobile_arm CEM controller.

    Returns ``(controller, cost, terminal_value_mode)``; see
    :func:`critic_sharing.planning.factory.build_cem_controller`. With a
    ``vae_path`` the cost applies the OOD trust factor.
    """
    import gymnasium as gym

    from critic_sharing.planning.factory import build_cem_controller
    from critic_sharing.scenarios.mobile_arm.leader_env import (
        mobile_arm_leader_dynamics_batch,
    )

    ood_metric = None
    if vae_path is not None:
        from critic_sharing.scenarios.mobile_arm.ood_query import (
            build_mobile_arm_ood_metric,
        )

        ood_metric = build_mobile_arm_ood_metric(vae_path)

    def dynamics_fn(obs_batch, act_batch):
        return mobile_arm_leader_dynamics_batch(obs_batch, act_batch, leader_cfg)

    action_space = gym.spaces.Box(
        low=np.array([
            -leader_cfg.max_speed,
            -leader_cfg.max_speed,
            -leader_cfg.max_yaw_rate,
            -1.0,
        ], dtype=np.float32),
        high=np.array([
            leader_cfg.max_speed,
            leader_cfg.max_speed,
            leader_cfg.max_yaw_rate,
            1.0,
        ], dtype=np.float32),
        dtype=np.float32,
    )
    return build_cem_controller(
        "mobile_arm",
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

def _run_episode_ma(
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
) -> dict:
    """One paired (leader → follower) episode for mobile_arm."""
    from critic_sharing.scenarios.mobile_arm import build_mobile_arm_follower_value_fn
    from critic_sharing.scenarios.mobile_arm.common import world_to_base

    # ---- Pre-sample follower's initial joints (θ₁ random, θ₂/θ₃ default) ----
    follower_env.reset(seed=follower_seed)
    follower_yaw = float(follower_env._theta[0])
    # We don't capture the follower's per-episode object yet — that's set by
    # the leader's parking pose: object_world stays fixed across leader+follower.

    # Build per-episode V_follower closure baked over follower_yaw.
    value_fn = build_mobile_arm_follower_value_fn(
        follower_model,
        follower_cfg,
        follower_yaw,
    )

    controller = None
    cost = None
    terminal_value_mode = "value"
    if mode in MODE_SETTINGS:  # planner modes; unguided runs the policy alone
        utility_type, use_ood, prior_kind = MODE_SETTINGS[mode]
        controller, cost, terminal_value_mode = _build_ma_controller(
            leader_env.config, leader_model, value_fn, seed=planner_seed,
            utility_type=utility_type,
            leader_range=leader_range,
            follower_range=follower_range,
            vae_path=vae_path if use_ood else None,
            alpha=alpha,
            prior_kind=prior_kind,
        )

    # ---- Leader phase ----
    obs, _ = leader_env.reset(seed=leader_seed)
    object_world = leader_env._object_pos.copy()
    leader_traj: List[np.ndarray] = [obs.copy()] if record else []
    leader_return = 0.0
    leader_steps = 0
    if controller is not None:
        controller.reset()
    done = False
    info = {}
    # See warehouse._run_episode_wh — same incurred / γ^t bookkeeping +
    # terminal_value gating.
    from critic_sharing.config.resolver import load_cost_config

    leader_gamma = float(
        getattr(load_cost_config("mobile_arm"), "leader_gamma", 0.99),
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
        done = term or trunc
    leader_terminal_pos = leader_env._base_pos.copy()
    leader_terminal_yaw = float(leader_env._base_yaw)
    leader_stand_off_diff = float(info.get("stand_off_diff", 0.0))
    # Two flags from the leader env's info dict at the terminal step:
    #   leader_terminated_by_action — leader CHOSE to fire its terminate
    #     action (action[3] >= interact_threshold). Says nothing about
    #     position quality.
    #   leader_success — composite: terminated_by_action AND within ½
    #     sigma of the desired stand-off ring (i.e. earned >~78% of
    #     goal_reward). This is the actually-useful "leader_done_well"
    #     signal for the cross_scenario success-rate table.
    leader_terminated_by_action = bool(info.get("terminated_by_action", False))
    leader_success = bool(info.get("leader_success", False))

    # ---- Transform object to base frame for follower ----
    obj_base_2d = world_to_base(object_world, leader_terminal_pos, leader_terminal_yaw)
    object_pos_base = np.array([obj_base_2d[0], obj_base_2d[1], 0.0], dtype=np.float32)

    # ---- Follower phase ----
    follower_obs, _ = follower_env.reset(seed=follower_seed, options={
        "theta": np.array([
            float(follower_cfg.theta1_default),
            float(follower_cfg.theta2_default),
            float(follower_cfg.theta3_default),
            float(follower_cfg.theta4_default),
        ], dtype=np.float32),
        "object_pos": object_pos_base,
    })
    follower_traj: List[np.ndarray] = [follower_obs.copy()] if record else []
    follower_return = 0.0
    follower_return_disc = 0.0
    follower_gamma_pow_t = 1.0
    follower_steps = 0
    dropped = False
    info = {}
    done = False
    while not done:
        action, _ = follower_model.predict(follower_obs, deterministic=True)
        follower_obs, r, term, trunc, info = follower_env.step(action)
        if record:
            follower_traj.append(follower_obs.copy())
        follower_return += float(r)
        follower_return_disc += follower_gamma_pow_t * float(r)
        follower_gamma_pow_t *= leader_gamma
        follower_steps += 1
        dropped = dropped or bool(info.get("dropped", False))
        done = term or trunc

    # See warehouse.py — store discounted returns so they match the
    # discounted calibration bands; preserve undiscounted sums under
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
        "leader_stand_off_diff": leader_stand_off_diff,
        "leader_terminated_by_action": int(leader_terminated_by_action),
        "leader_success": int(leader_success),
        "leader_terminal_x": float(leader_terminal_pos[0]),
        "leader_terminal_y": float(leader_terminal_pos[1]),
        "leader_terminal_yaw": float(leader_terminal_yaw),
        "object_x_world": float(object_world[0]),
        "object_y_world": float(object_world[1]),
        "object_x_base": float(object_pos_base[0]),
        "object_y_base": float(object_pos_base[1]),
        "follower_yaw": follower_yaw,
        "follower_dropped": float(dropped),
    }
    if record:
        result["leader_traj"] = leader_traj
        result["follower_traj"] = follower_traj
    return result


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class MAEpisodeWorker(EpisodeWorker):
    """Loads PPO models + envs once per process, then runs episodes serially."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._leader_path: str = kwargs["leader_path"]
        self._follower_path: str = kwargs["follower_path"]
        self._leader_range: Optional[Tuple[float, float]] = kwargs.get("leader_range")
        self._follower_range: Optional[Tuple[float, float]] = kwargs.get("follower_range")
        self._vae_path: Optional[str] = kwargs.get("vae_path")
        self._alpha: float = float(kwargs.get("alpha", 1.0))

    def setup(self) -> None:
        from stable_baselines3 import PPO

        from critic_sharing.config.resolver import load_env_config, load_planner_config
        from critic_sharing.scenarios.mobile_arm import (
            MobileArmFollowerEnv,
            MobileArmLeaderEnv,
        )

        self._leader_cfg = load_env_config("mobile_arm", "leader")
        self._follower_cfg = load_env_config("mobile_arm", "follower")
        self._leader_model = PPO.load(self._leader_path)
        self._follower_model = PPO.load(self._follower_path)
        self._leader_env = MobileArmLeaderEnv(self._leader_cfg)
        self._follower_env = MobileArmFollowerEnv(self._follower_cfg)
        # See WhEpisodeWorker.setup — same per-worker thread bump for
        # terminal_value="rollout".
        cem_cfg = load_planner_config("mobile_arm")
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
        return _run_episode_ma(
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
        )


# ---------------------------------------------------------------------------
# Top-level _compare_ma
# ---------------------------------------------------------------------------

class MobileArmAdapter(EvalScenarioAdapter):
    """:class:`EvalScenarioAdapter` for the mobile_arm scenario."""

    scenario_name = "mobile_arm"
    worker_cls = MAEpisodeWorker

    def build_calibration(self):
        from stable_baselines3 import PPO

        from critic_sharing.config.resolver import load_env_config
        from critic_sharing.scenarios.mobile_arm import (
            MobileArmFollowerEnv,
            MobileArmLeaderEnv,
        )

        leader_env = MobileArmLeaderEnv(load_env_config("mobile_arm", "leader"))
        follower_env = MobileArmFollowerEnv(load_env_config("mobile_arm", "follower"))
        leader_model = PPO.load(self._leader_path)
        follower_model = PPO.load(self._follower_path)
        return leader_env, follower_env, leader_model, follower_model

    def __init__(self, *, leader_path: str, follower_path: str) -> None:
        self._leader_path = leader_path
        self._follower_path = follower_path

    def sample_conditions(self, rng, n_episodes):
        return [None] * n_episodes

    def make_spec(
        self, *, episode, mode, leader_seed, follower_seed, planner_seed,
        condition, paths, alpha, record,
    ):
        return EpisodeSpec(
            scenario="mobile_arm",
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
        }

    def build_ood_fn_for_post_processing(self, vae_path):
        if vae_path is None:
            return None
        from critic_sharing.scenarios.mobile_arm.ood_query import (
            build_mobile_arm_ood_metric,
        )
        return build_mobile_arm_ood_metric(vae_path)

    def handoff_obs_for_ood(self, row):
        # Leader-shape (1, 6): (terminal_x, terminal_y, sin yaw, cos yaw,
        #                       object_x_world, object_y_world)
        obs = np.zeros((1, 6), dtype=np.float32)
        obs[0, 0] = float(row.get("leader_terminal_x", 0.0))
        obs[0, 1] = float(row.get("leader_terminal_y", 0.0))
        yaw = float(row.get("leader_terminal_yaw", 0.0))
        obs[0, 2] = float(np.sin(yaw))
        obs[0, 3] = float(np.cos(yaw))
        obs[0, 4] = float(row.get("object_x_world", 0.0))
        obs[0, 5] = float(row.get("object_y_world", 0.0))
        return obs


def _compare_ma(
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
) -> list[dict]:
    """Run paired naive vs V-CEM episodes for the mobile_arm scenario.

    Thin wrapper around :func:`run_eval_pipeline` with a
    :class:`MobileArmAdapter`.
    """
    adapter = MobileArmAdapter(
        leader_path=str(leader_model),
        follower_path=str(follower_model),
    )
    return run_eval_pipeline(
        adapter,
        leader_model=leader_model,
        follower_model=follower_model,
        vae=Path(vae_path) if vae_path else None,
        alpha=alpha,
        modes=modes,
        leader_seeds=leader_seeds,
        follower_seeds=follower_seeds,
        planner_seeds=planner_seeds,
        n_follower_samples=n_follower_samples,
        workers=workers,
        record=record,
    )


# ---------------------------------------------------------------------------
# Animations
# ---------------------------------------------------------------------------

class MAEpisodeFigureWorker(FigureWorker):
    """Renders one mobile_arm 3-panel MP4 per task."""

    def setup(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        from critic_sharing.config.resolver import load_env_config

        self._fps = int(self._kwargs.get("fps", 4))
        self._leader_cfg = load_env_config("mobile_arm", "leader")
        self._follower_cfg = load_env_config("mobile_arm", "follower")

    def render(self, task: FigureTask) -> None:
        import matplotlib.pyplot as plt
        from matplotlib import animation as mpl_animation

        from critic_sharing.scenarios.mobile_arm.render import (
            COLOR_LEADER_TRAIL,
            draw_base_topdown,
            draw_desired_distance_ring,
            draw_follower_side,
            draw_follower_topdown,
            draw_leader_arena,
            draw_object_topdown,
        )

        row: dict[str, Any] = task.payload  # type: ignore[assignment]
        leader_cfg = self._leader_cfg
        follower_cfg = self._follower_cfg

        leader_obs = np.asarray(row["leader_traj"], dtype=np.float32)
        follower_obs = np.asarray(row["follower_traj"], dtype=np.float32)
        object_world = np.array(
            [row["object_x_world"], row["object_y_world"]],
            dtype=np.float32,
        )

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        ax_leader, ax_top, ax_side = axes

        n_l = leader_obs.shape[0]
        n_f = follower_obs.shape[0]
        total_frames = n_l + n_f

        def _theta(o):
            return float(np.arctan2(o[2], o[3]))

        def _setup(i):
            for ax in axes:
                ax.clear()
            # Leader panel
            draw_leader_arena(ax_leader, leader_cfg)
            draw_desired_distance_ring(
                ax_leader, object_world,
                float(leader_cfg.desired_distance),
                float(leader_cfg.gaussian_sigma),
            )
            draw_object_topdown(ax_leader, object_world, leader_cfg)
            if i < n_l:
                trail = leader_obs[: i + 1]
                ax_leader.plot(
                    trail[:, 0], trail[:, 1],
                    color=COLOR_LEADER_TRAIL,
                    linewidth=1.5, alpha=0.85, zorder=4,
                )
                draw_base_topdown(
                    ax_leader,
                    leader_obs[i, 0:2],
                    _theta(leader_obs[i]),
                    leader_cfg,
                )
            else:
                trail = leader_obs
                ax_leader.plot(
                    trail[:, 0], trail[:, 1],
                    color=COLOR_LEADER_TRAIL,
                    linewidth=1.5, alpha=0.85, zorder=4,
                )
                draw_base_topdown(
                    ax_leader,
                    leader_obs[-1, 0:2],
                    _theta(leader_obs[-1]),
                    leader_cfg,
                )

            if i < n_l:
                theta_static = np.array([
                    float(row["follower_yaw"]),
                    float(follower_cfg.theta2_default),
                    float(follower_cfg.theta3_default),
                    float(follower_cfg.theta4_default),
                ], dtype=np.float32)
                obj_static = np.array(
                    [row["object_x_base"], row["object_y_base"], 0.0],
                    dtype=np.float32,
                )
                draw_follower_topdown(ax_top, theta_static, obj_static, False, follower_cfg)
                draw_follower_side(ax_side, theta_static, obj_static, False, follower_cfg)
            else:
                j = i - n_l
                obs_f = follower_obs[j]
                theta_arr = np.array([
                    float(np.arctan2(obs_f[0], obs_f[1])),
                    float(np.arctan2(obs_f[2], obs_f[3])),
                    float(np.arctan2(obs_f[4], obs_f[5])),
                    float(np.arctan2(obs_f[6], obs_f[7])),
                ], dtype=np.float32)
                obj_b = np.array(
                    [obs_f[8], obs_f[9], obs_f[10]],
                    dtype=np.float32,
                )
                carry = bool(obs_f[11] >= 0.5)
                draw_follower_topdown(ax_top, theta_arr, obj_b, carry, follower_cfg)
                draw_follower_side(ax_side, theta_arr, obj_b, carry, follower_cfg)

        def _update(i):
            _setup(i)
            return ()

        fig.tight_layout()
        anim = mpl_animation.FuncAnimation(
            fig, _update, frames=total_frames,
            interval=1000 / max(1, self._fps), blit=False,
        )
        anim.save(
            str(task.output_path),
            writer=mpl_animation.FFMpegWriter(fps=self._fps),
        )
        plt.close(fig)


def _ma_mp4_payload(row: dict) -> dict[str, Any]:
    """Pull the picklable subset of a recorded mobile_arm row the worker reads."""
    return {
        "episode": int(row["episode"]),
        "mode": row["mode"],
        "leader_traj": np.asarray(row["leader_traj"], dtype=np.float32),
        "follower_traj": np.asarray(row["follower_traj"], dtype=np.float32),
        "object_x_world": float(row["object_x_world"]),
        "object_y_world": float(row["object_y_world"]),
        "object_x_base": float(row["object_x_base"]),
        "object_y_base": float(row["object_y_base"]),
        "follower_yaw": float(row["follower_yaw"]),
    }


def _save_ma_episode_mp4s(
    all_results: List[dict],
    output_dir: Path,
    fps: int = 4,
    max_workers: int = 1,
) -> None:
    """3-panel MP4 per episode: leader top-down | follower top-down | follower side."""
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
        tasks.append(FigureTask(output_path=out_path, payload=_ma_mp4_payload(row)))

    typer.echo(
        "Saving {n} mobile_arm episode MP4s to {d}/ ({w} workers)".format(
            n=len(tasks), d=mp4_dir, w=max(1, min(max_workers, len(tasks))),
        )
    )
    run_figures_parallel(
        tasks=tasks,
        worker_cls=MAEpisodeFigureWorker,
        worker_kwargs={"fps": int(fps)},
        max_workers=int(max_workers),
        progress_desc="MA episodes",
    )


__all__ = [
    "_compare_ma",
    "_save_ma_episode_mp4s",
]
