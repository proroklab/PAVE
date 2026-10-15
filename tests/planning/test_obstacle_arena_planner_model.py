"""The obstacle_arena CEM model must score plans with the env's own objective.

* The planner dynamics reproduce ``PushObstacleEnv.step``: next state,
  reward (wall-distance progress plus ``goal_reward`` on contact) and
  termination.
* The planner cost honours ``cost.yaml``'s ``clip_normalized``, like the
  warehouse and mobile_arm builders.
"""

from __future__ import annotations

import numpy as np
import pytest

from critic_sharing.cli.eval._common import EvalMode
from critic_sharing.cli.eval.obstacle_arena import _build_oa_controller
from critic_sharing.config.resolver import load_cost_config, load_env_config
from critic_sharing.scenarios.obstacle_arena.push_env import PushObstacleEnv

SCENARIO = "obstacle_arena"


def _controller(mode: EvalMode = EvalMode.value_only):
    # Models are only used inside closures (V_F, V_L, prior), never while
    # building, so the wiring can be checked without checkpoints.
    return _build_oa_controller(
        push_model=None,
        nav_model=None,
        nav_config=None,
        mode=mode,
        vae=None,
        alpha=4.0,
        seed=0,
        follower_spawn_pos=np.zeros(2, dtype=np.float32),
        scenario=SCENARIO,
        leader_range=(0.0, 1.0),
        follower_range=(0.0, 1.0),
    )


def _episode_transitions(env: PushObstacleEnv, rng: np.random.Generator, push: bool):
    """Yield (obs, action, next_obs, reward, terminated) along one episode.

    ``push=True`` drives the agent into the obstacle so episodes end on wall
    contact; ``push=False`` uses random (partly over-speed) actions.
    """
    obs, _ = env.reset(seed=int(rng.integers(1_000_000)))
    max_speed = float(env.config.max_speed)
    for _ in range(int(env.config.max_steps)):
        if push:
            direction = obs[2:4] - obs[:2]
            action = direction / (np.linalg.norm(direction) + 1e-8) * max_speed
        else:
            action = rng.uniform(-1.5 * max_speed, 1.5 * max_speed, size=2)
        action = np.asarray(action, dtype=np.float32)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        yield obs, action, next_obs, reward, terminated
        if terminated or truncated:
            return
        obs = next_obs


@pytest.mark.parametrize("push", [True, False])
def test_planner_dynamics_match_env_step(push):
    controller, _, _ = _controller()
    dynamics_fn = controller._dynamics_fn
    env = PushObstacleEnv(load_env_config(SCENARIO, "leader"))
    rng = np.random.default_rng(0)

    n_steps = n_terminal = 0
    for _ in range(10):
        for obs, action, next_obs, reward, terminated in _episode_transitions(env, rng, push):
            pred_obs, pred_reward, pred_done = dynamics_fn(obs[None], action[None])
            np.testing.assert_allclose(pred_obs[0], next_obs, atol=1e-6)
            assert pred_reward[0] == pytest.approx(reward, abs=1e-5)
            assert bool(pred_done[0]) == bool(terminated)
            n_steps += 1
            n_terminal += int(terminated)
    assert n_steps > 0
    if push:
        # The goal_reward branch must actually be exercised.
        assert n_terminal > 0


@pytest.mark.parametrize("mode", [EvalMode.value_only, EvalMode.value_and_ood])
def test_planner_cost_honours_clip_normalized(mode):
    _, cost, _ = _controller(mode)
    expected = bool(getattr(load_cost_config(SCENARIO), "clip_normalized", True))
    assert cost._clip_normalized is expected
