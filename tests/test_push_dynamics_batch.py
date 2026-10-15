import numpy as np

from critic_sharing.config.schema import PushEnvConfig
from critic_sharing.scenarios.obstacle_arena.push_env import PushObstacleEnv, pusher_dynamics_batch


def _env_step_transition(
    config: PushEnvConfig,
    agent_pos: np.ndarray,
    obstacle_pos: np.ndarray,
    action: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    env = PushObstacleEnv(config)
    env.reset(
        options={
            "agent_pos": np.asarray(agent_pos, dtype=np.float32),
            "obstacle_pos": np.asarray(obstacle_pos, dtype=np.float32),
        }
    )
    # Verify the env actually used the positions we passed (no nudging)
    obs = env._get_obs()
    np.testing.assert_allclose(obs[:2], agent_pos, atol=1e-5)
    np.testing.assert_allclose(obs[2:4], obstacle_pos, atol=1e-5)
    obs, _, _, _, _ = env.step(np.asarray(action, dtype=np.float32))
    env.close()
    return obs[:2].astype(np.float32), obs[2:4].astype(np.float32)


def test_pusher_dynamics_batch_matches_env_step_fixed_cases() -> None:
    config = PushEnvConfig(
        dt=0.1,
        max_speed=1.5,
        world_min=-5.0,
        world_max=5.0,
        agent_radius=0.25,
        obstacle_radius=0.8,
        push_strength=1.0,
    )


    # All positions below the wall (wall_y=0.0), no agent-obstacle overlap
    agent_pos = np.array(
        [
            [-3.0, -3.0],  # no contact (far apart)
            [-3.0, -2.0],  # no contact (different positions)
            [-2.0, -3.0],  # moderate distance
            [-4.0, -4.0],  # corner region
        ],
        dtype=np.float32,
    )
    obstacle_pos = np.array(
        [
            [-1.0, -3.0],  # 2.0 apart from agent — no contact
            [-1.0, -2.0],  # 2.0 apart — no contact
            [-2.0, -1.5],  # 1.5 apart — in contact range
            [-2.5, -4.0],  # 1.5 apart — in contact range
        ],
        dtype=np.float32,
    )
    actions = np.array(
        [
            [0.2, -0.1],
            [1.0, 0.0],
            [0.5, 0.5],
            [1.0, 0.5],
        ],
        dtype=np.float32,
    )

    batch_next_agent, batch_next_obstacle = pusher_dynamics_batch(
        agent_pos,
        obstacle_pos,
        actions,
        config,
    )

    for idx in range(agent_pos.shape[0]):
        env_next_agent, env_next_obstacle = _env_step_transition(
            config,
            agent_pos[idx],
            obstacle_pos[idx],
            actions[idx],
        )
        np.testing.assert_allclose(batch_next_agent[idx], env_next_agent, atol=1e-3, rtol=0.0)
        np.testing.assert_allclose(batch_next_obstacle[idx], env_next_obstacle, atol=1e-3, rtol=0.0)


def test_pusher_dynamics_batch_matches_env_step_random_cases() -> None:
    config = PushEnvConfig()
    rng = np.random.default_rng(1234)
    batch_size = 64
    min_dist = config.agent_radius + config.obstacle_radius + 0.1

    # Sample positions below the wall (wall_y=0.0), ensuring no overlap
    y_max = config.wall_y - config.obstacle_radius - 0.5
    y_min = config.world_min + 1.5

    agent_positions = []
    obstacle_positions = []
    for _ in range(batch_size):
        for _attempt in range(1000):
            ax = float(rng.uniform(-3.5, 3.5))
            ay = float(rng.uniform(y_min, y_max))
            ox = float(rng.uniform(-3.5, 3.5))
            oy = float(rng.uniform(y_min, y_max))
            a = np.array([ax, ay], dtype=np.float32)
            o = np.array([ox, oy], dtype=np.float32)
            if np.linalg.norm(a - o) >= min_dist:
                agent_positions.append(a)
                obstacle_positions.append(o)
                break

    agent_pos = np.array(agent_positions, dtype=np.float32)
    obstacle_pos = np.array(obstacle_positions, dtype=np.float32)
    actions = rng.normal(loc=0.0, scale=0.5, size=(batch_size, 2)).astype(np.float32)

    batch_next_agent, batch_next_obstacle = pusher_dynamics_batch(
        agent_pos,
        obstacle_pos,
        actions,
        config,
    )

    for idx in range(batch_size):
        env_next_agent, env_next_obstacle = _env_step_transition(
            config,
            agent_pos[idx],
            obstacle_pos[idx],
            actions[idx],
        )
        np.testing.assert_allclose(batch_next_agent[idx], env_next_agent, atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(
            batch_next_obstacle[idx], env_next_obstacle, atol=1e-3, rtol=1e-3
        )
