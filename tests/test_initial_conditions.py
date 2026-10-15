"""Tests for initial condition generation."""

import numpy as np
import pytest

from critic_sharing.config.schema import NavEnvConfig, PushEnvConfig
from critic_sharing.scenarios.obstacle_arena.initial_conditions import (
    fixed_obstacle_position,
    sample_eval_conditions,
)


@pytest.fixture
def push_cfg():
    return PushEnvConfig()


@pytest.fixture
def nav_cfg():
    return NavEnvConfig()


class TestSampleEvalConditions:
    def test_returns_correct_count(self, push_cfg, nav_cfg):
        rng = np.random.default_rng(42)
        conditions = sample_eval_conditions(push_cfg, nav_cfg, 10, rng)
        assert len(conditions) == 10

    def test_same_seed_same_conditions(self, push_cfg, nav_cfg):
        c1 = sample_eval_conditions(push_cfg, nav_cfg, 5, np.random.default_rng(42))
        c2 = sample_eval_conditions(push_cfg, nav_cfg, 5, np.random.default_rng(42))
        for a, b in zip(c1, c2):
            np.testing.assert_array_equal(a.obstacle_pos, b.obstacle_pos)
            np.testing.assert_array_equal(a.leader_pos, b.leader_pos)
            np.testing.assert_array_equal(a.follower_pos, b.follower_pos)

    def test_leader_below_wall(self, push_cfg, nav_cfg):
        rng = np.random.default_rng(42)
        conditions = sample_eval_conditions(push_cfg, nav_cfg, 50, rng)
        for c in conditions:
            assert c.leader_pos[1] < push_cfg.wall_y, (
                "Leader at y={y} is not below wall_y={wy}".format(
                    y=c.leader_pos[1], wy=push_cfg.wall_y
                )
            )

    def test_follower_above_wall(self, push_cfg, nav_cfg):
        rng = np.random.default_rng(42)
        conditions = sample_eval_conditions(push_cfg, nav_cfg, 50, rng)
        for c in conditions:
            assert c.follower_pos[1] > nav_cfg.wall_y, (
                "Follower at y={y} is not above wall_y={wy}".format(
                    y=c.follower_pos[1], wy=nav_cfg.wall_y
                )
            )

    def test_obstacle_below_wall(self, push_cfg, nav_cfg):
        rng = np.random.default_rng(42)
        conditions = sample_eval_conditions(push_cfg, nav_cfg, 50, rng)
        for c in conditions:
            assert c.obstacle_pos[1] < push_cfg.wall_y, (
                "Obstacle at y={y} is not below wall_y={wy}".format(
                    y=c.obstacle_pos[1], wy=push_cfg.wall_y
                )
            )

    def test_no_leader_obstacle_overlap(self, push_cfg, nav_cfg):
        rng = np.random.default_rng(42)
        min_dist = push_cfg.obstacle_radius + push_cfg.agent_radius
        conditions = sample_eval_conditions(push_cfg, nav_cfg, 50, rng)
        for c in conditions:
            dist = np.linalg.norm(c.leader_pos - c.obstacle_pos)
            assert dist >= min_dist - 1e-3

    def test_no_follower_obstacle_overlap(self, push_cfg, nav_cfg):
        rng = np.random.default_rng(42)
        min_dist = nav_cfg.obstacle_radius + nav_cfg.agent_radius
        conditions = sample_eval_conditions(push_cfg, nav_cfg, 50, rng)
        for c in conditions:
            dist = np.linalg.norm(c.follower_pos - c.obstacle_pos)
            assert dist >= min_dist - 1e-3


class TestObstacleDistance:
    def test_obstacle_within_distance(self, push_cfg, nav_cfg):
        from critic_sharing.scenarios.obstacle_arena.common import nav_goal_position
        goal = nav_goal_position(nav_cfg)
        rng = np.random.default_rng(42)
        conditions = sample_eval_conditions(
            push_cfg, nav_cfg, 50, rng, obstacle_distance=3.0
        )
        for c in conditions:
            dist = np.linalg.norm(c.obstacle_pos - goal)
            assert dist <= 3.0 + 1e-3, (
                "Obstacle at dist={d:.2f} exceeds max_distance=3.0".format(d=dist)
            )


class TestFixedObstacle:
    def test_gap_center(self, push_cfg, nav_cfg):
        from critic_sharing.scenarios.obstacle_arena.common import nav_goal_position
        goal = nav_goal_position(nav_cfg)
        gap_pos = np.array([nav_cfg.gap_x, nav_cfg.wall_y], dtype=np.float32)
        expected = (gap_pos + goal) / 2.0

        rng = np.random.default_rng(42)
        conditions = sample_eval_conditions(
            push_cfg, nav_cfg, 5, rng, fixed_obstacle="gap-center"
        )
        for c in conditions:
            np.testing.assert_allclose(c.obstacle_pos, expected, atol=1e-5)

    def test_explicit_coordinates(self, nav_cfg):
        pos = fixed_obstacle_position("1.5,-2.0", nav_cfg)
        np.testing.assert_allclose(pos, [1.5, -2.0], atol=1e-5)

    def test_unknown_preset_raises(self, nav_cfg):
        with pytest.raises(ValueError):
            fixed_obstacle_position("unknown", nav_cfg)
