"""Tests for obstacle_arena environment correctness after simplification."""

import numpy as np

from critic_sharing.config.schema import BaseEnvConfig, NavEnvConfig, PushEnvConfig
from critic_sharing.scenarios.obstacle_arena.common import circle_collides_wall
from critic_sharing.scenarios.obstacle_arena.nav_env import OneObstacleReachEnv
from critic_sharing.scenarios.obstacle_arena.push_env import PushObstacleEnv


class TestWallMandatory:
    def test_base_config_wall_y_is_float(self):
        cfg = BaseEnvConfig()
        assert isinstance(cfg.wall_y, float)
        assert cfg.wall_y == 0.0

    def test_nav_config_inherits_wall(self):
        cfg = NavEnvConfig()
        assert isinstance(cfg.wall_y, float)

    def test_push_config_inherits_wall(self):
        cfg = PushEnvConfig()
        assert isinstance(cfg.wall_y, float)

    def test_nav_config_no_goal_radius(self):
        assert not hasattr(NavEnvConfig, "goal_radius") or "goal_radius" not in NavEnvConfig.__dataclass_fields__


class TestNavEnvSpawning:
    def test_follower_does_not_spawn_in_wall(self):
        """Follower can spawn above or below wall, but never inside it."""
        cfg = NavEnvConfig()
        env = OneObstacleReachEnv(cfg)
        for i in range(200):
            obs, _ = env.reset(seed=i)
            agent_pos = obs[:2]
            assert not circle_collides_wall(agent_pos, cfg.agent_radius, cfg), (
                "Follower spawned inside wall at pos=({x:.3f}, {y:.3f})".format(
                    x=float(agent_pos[0]), y=float(agent_pos[1])
                )
            )
        env.close()

    def test_follower_can_spawn_below_wall(self):
        """Over many resets, at least some spawns should be below the wall."""
        cfg = NavEnvConfig()
        env = OneObstacleReachEnv(cfg)
        below_count = 0
        n_resets = 200
        for i in range(n_resets):
            obs, _ = env.reset(seed=i)
            if obs[1] < cfg.wall_y:
                below_count += 1
        env.close()
        assert below_count > 0, (
            "Follower never spawned below wall in {n} resets".format(n=n_resets)
        )

    def test_obstacle_does_not_spawn_in_wall(self):
        """Obstacle must not spawn overlapping a wall segment."""
        cfg = NavEnvConfig()
        env = OneObstacleReachEnv(cfg)
        for i in range(100):
            obs, _ = env.reset(seed=i)
            obstacle_pos = obs[2:4]
            assert not circle_collides_wall(obstacle_pos, cfg.obstacle_radius, cfg), (
                "Obstacle spawned inside wall at pos=({x:.3f}, {y:.3f})".format(
                    x=float(obstacle_pos[0]), y=float(obstacle_pos[1])
                )
            )
        env.close()

    def test_no_spawn_collision_with_obstacle(self):
        cfg = NavEnvConfig()
        env = OneObstacleReachEnv(cfg)
        min_dist = cfg.obstacle_radius + cfg.agent_radius
        for i in range(100):
            obs, _ = env.reset(seed=i)
            agent_pos = obs[:2]
            obstacle_pos = obs[2:4]
            dist = np.linalg.norm(agent_pos - obstacle_pos)
            assert dist >= min_dist - 1e-3, (
                "Agent spawned overlapping obstacle: dist={d:.4f} < min_dist={m:.4f}".format(
                    d=dist, m=min_dist
                )
            )
        env.close()

    def test_goal_is_lower_boundary(self):
        """Follower reaches goal when touching the bottom wall."""
        cfg = NavEnvConfig()
        env = OneObstacleReachEnv(cfg)
        # Place agent near bottom boundary, obstacle far away
        obs, _ = env.reset(options={
            "agent_pos": np.array([0.0, cfg.world_min + cfg.agent_radius + 0.01], dtype=np.float32),
            "obstacle_pos": np.array([3.0, -3.0], dtype=np.float32),
        })
        # Step downward
        action = np.array([0.0, -cfg.max_speed], dtype=np.float32)
        _, _, terminated, _, info = env.step(action)
        assert info["reached_goal"] is True
        env.close()


class TestPushEnvSpawning:
    def test_leader_spawns_below_wall(self):
        cfg = PushEnvConfig()
        env = PushObstacleEnv(cfg)
        for i in range(100):
            obs, _ = env.reset(seed=i)
            agent_y = obs[1]
            assert agent_y < cfg.wall_y, (
                "Leader spawned at y={y} which is not below wall_y={wy}".format(
                    y=agent_y, wy=cfg.wall_y
                )
            )
        env.close()

    def test_obstacle_spawns_below_wall(self):
        cfg = PushEnvConfig()
        env = PushObstacleEnv(cfg)
        for i in range(100):
            obs, _ = env.reset(seed=i)
            obstacle_y = obs[3]
            assert obstacle_y < cfg.wall_y, (
                "Obstacle spawned at y={y} which is not below wall_y={wy}".format(
                    y=obstacle_y, wy=cfg.wall_y
                )
            )
        env.close()

    def test_no_spawn_collision_with_obstacle(self):
        cfg = PushEnvConfig()
        env = PushObstacleEnv(cfg)
        min_dist = cfg.obstacle_radius + cfg.agent_radius
        for i in range(100):
            obs, _ = env.reset(seed=i)
            agent_pos = obs[:2]
            obstacle_pos = obs[2:4]
            dist = np.linalg.norm(agent_pos - obstacle_pos)
            assert dist >= min_dist - 1e-3, (
                "Leader spawned overlapping obstacle: dist={d:.4f} < min_dist={m:.4f}".format(
                    d=dist, m=min_dist
                )
            )
        env.close()


class TestWallCollision:
    def test_wall_blocks_follower(self):
        """Follower outside the gap cannot cross the wall."""
        cfg = NavEnvConfig(gap_x=0.0, gap_width=2.0)
        env = OneObstacleReachEnv(cfg)
        # Place follower just above wall, outside the gap (x=3.0)
        obs, _ = env.reset(options={
            "agent_pos": np.array([3.0, cfg.wall_y + cfg.agent_radius + 0.05], dtype=np.float32),
            "obstacle_pos": np.array([-3.0, -3.0], dtype=np.float32),
        })
        # Step downward repeatedly
        action = np.array([0.0, -cfg.max_speed], dtype=np.float32)
        for _ in range(20):
            obs, _, terminated, _, _ = env.step(action)
            if terminated:
                break
            agent_y = obs[1]
            assert agent_y >= cfg.wall_y + cfg.agent_radius - 1e-3, (
                "Follower crossed wall: y={y}".format(y=agent_y)
            )
        env.close()

    def test_gap_allows_passage(self):
        """Follower inside the gap can cross the wall."""
        cfg = NavEnvConfig(gap_x=0.0, gap_width=2.0)
        env = OneObstacleReachEnv(cfg)
        # Place follower just above wall, inside the gap (x=0.0)
        obs, _ = env.reset(options={
            "agent_pos": np.array([0.0, cfg.wall_y + cfg.agent_radius + 0.05], dtype=np.float32),
            "obstacle_pos": np.array([-3.0, -3.0], dtype=np.float32),
        })
        # Step downward repeatedly until crossing or timeout
        action = np.array([0.0, -cfg.max_speed], dtype=np.float32)
        crossed = False
        for _ in range(10):
            obs, _, terminated, _, _ = env.step(action)
            if terminated:
                break
            if obs[1] < cfg.wall_y:
                crossed = True
                break
        assert crossed, (
            "Follower could not pass through gap after 10 steps: y={y}".format(y=obs[1])
        )
        env.close()
