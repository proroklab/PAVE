"""Tests for the warehouse scenario envs and helpers."""

from __future__ import annotations

import numpy as np
import pytest

from critic_sharing.config.schema import (
    WarehouseFollowerEnvConfig,
    WarehouseLeaderEnvConfig,
)
from critic_sharing.scenarios.warehouse import (
    WarehouseFollowerEnv,
    WarehouseLeaderEnv,
    warehouse_leader_dynamics_batch,
)
from critic_sharing.scenarios.warehouse.common import (
    agent_inscribed_radius,
    drop_position,
    enforce_shelves,
    enforce_wall,
    gap_centers,
    gap_index_for_position,
    in_pickup_cone,
    point_in_any_shelf,
    sample_goal_on_shelf_face,
    shelf_face_segments,
    shelf_rects,
    snap_to_wall_if_in_gap,
    wall_segments,
)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

class TestGeometry:
    def test_wall_has_four_segments_around_three_gaps(self):
        cfg = WarehouseLeaderEnvConfig()
        segs = wall_segments(cfg)
        assert len(segs) == 4
        # Segments alternate solid/gap; each solid segment is non-empty.
        for x1, x2, _ in segs:
            assert x2 > x1

    def test_gap_centers_match_config(self):
        cfg = WarehouseLeaderEnvConfig()
        centers = gap_centers(cfg)
        assert centers.shape == (3, 2)
        assert sorted(c[0] for c in centers) == sorted(cfg.gap_xs)
        assert all(c[1] == cfg.wall_y for c in centers)

    def test_inscribed_circle_blocks_gaps(self):
        """Agent is too wide to fit through any gap, even centred on one.

        Single-step physics: per-step delta ≤ ``max_speed * dt`` is well within
        the wall's ``r`` band, so a step that would land *inside* the wall is
        always clamped. (Multi-step tunnelling is impossible by construction.)
        """
        cfg = WarehouseLeaderEnvConfig()
        r = agent_inscribed_radius(cfg)
        # 2r must exceed gap_width — design invariant.
        assert 2.0 * r > cfg.gap_width
        # Agent just below wall steps upward; new pos overlaps wall band.
        prev = np.array([0.0, cfg.wall_y - r - 0.05], dtype=np.float32)
        attempt = np.array([0.0, cfg.wall_y - r + 0.1], dtype=np.float32)
        clamped = enforce_wall(attempt, prev, r, cfg)
        assert clamped[1] <= cfg.wall_y - r + 1e-3

    def test_object_passes_through_gap(self):
        """A point at gap centre is recognised as in-gap."""
        cfg = WarehouseLeaderEnvConfig()
        for gx in cfg.gap_xs:
            p = np.array([gx, cfg.wall_y], dtype=np.float32)
            assert gap_index_for_position(p, cfg) >= 0

    def test_point_outside_gap_x(self):
        cfg = WarehouseLeaderEnvConfig()
        p = np.array([1.5, cfg.wall_y], dtype=np.float32)  # between gap centres
        assert gap_index_for_position(p, cfg) == -1

    def test_shelf_rects_match_config(self):
        cfg = WarehouseLeaderEnvConfig()
        rects = shelf_rects(cfg)
        assert len(rects) == 3
        for rect, sx in zip(rects, cfg.shelf_xs):
            xmin, xmax, _, _ = rect
            assert pytest.approx(xmin) == float(sx) - cfg.shelf_width / 2
            assert pytest.approx(xmax) == float(sx) + cfg.shelf_width / 2

    def test_shelves_block_walking_in(self):
        cfg = WarehouseFollowerEnvConfig()
        r = agent_inscribed_radius(cfg)
        attempt = np.array([0.0, 1.5], dtype=np.float32)  # centre of middle shelf
        out, collided = enforce_shelves(attempt, r, cfg)
        assert collided
        assert not point_in_any_shelf(out, cfg)

    def test_snap_to_wall(self):
        cfg = WarehouseLeaderEnvConfig()
        p = np.array([0.0, 0.25], dtype=np.float32)
        snapped = snap_to_wall_if_in_gap(p, cfg)
        assert snapped[1] == cfg.wall_y

        p_far = np.array([1.5, 0.0], dtype=np.float32)
        unsnapped = snap_to_wall_if_in_gap(p_far, cfg)
        assert np.allclose(unsnapped, p_far)


# ---------------------------------------------------------------------------
# Pickup cone
# ---------------------------------------------------------------------------

class TestPickupCone:
    def test_in_front_within_range(self):
        cfg = WarehouseLeaderEnvConfig()
        agent = np.array([0.0, 0.0])
        obj = np.array([0.5, 0.0])
        assert in_pickup_cone(agent, 0.0, obj, cfg)

    def test_too_far(self):
        cfg = WarehouseLeaderEnvConfig()
        agent = np.array([0.0, 0.0])
        obj = np.array([2.0, 0.0])  # > pickup_max_distance = 0.9
        assert not in_pickup_cone(agent, 0.0, obj, cfg)

    def test_behind(self):
        cfg = WarehouseLeaderEnvConfig()
        agent = np.array([0.0, 0.0])
        obj = np.array([-0.5, 0.0])
        assert not in_pickup_cone(agent, 0.0, obj, cfg)

    def test_at_cone_edge(self):
        cfg = WarehouseLeaderEnvConfig()
        agent = np.array([0.0, 0.0])
        # +60° (= +π/3) is the boundary; place object just inside.
        ang = cfg.pickup_cone_half_angle - 1e-3
        obj = np.array([0.5 * np.cos(ang), 0.5 * np.sin(ang)])
        assert in_pickup_cone(agent, 0.0, obj, cfg)

    def test_outside_cone(self):
        cfg = WarehouseLeaderEnvConfig()
        agent = np.array([0.0, 0.0])
        # 90° to the side is outside ±60° cone.
        obj = np.array([0.0, 0.5])
        assert not in_pickup_cone(agent, 0.0, obj, cfg)

    def test_drop_position_geometry(self):
        cfg = WarehouseLeaderEnvConfig()
        agent = np.array([1.0, 1.0])
        # Facing east → drop_x = 1.0 + drop_distance
        d = drop_position(agent, 0.0, cfg)
        assert d[0] == pytest.approx(1.0 + cfg.drop_distance)
        assert d[1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Goal sampling
# ---------------------------------------------------------------------------

class TestGoalSampling:
    def test_goal_lies_near_a_shelf_face(self):
        cfg = WarehouseFollowerEnvConfig()
        rng = np.random.default_rng(42)
        faces = shelf_face_segments(cfg)
        face_xs = sorted({float(p0[0]) for p0, _ in faces})
        for _ in range(50):
            g = sample_goal_on_shelf_face(rng, cfg, offset=0.1)
            # x is offset by 0.1 from a face — i.e. within 0.1 of some face_x.
            min_dx = min(abs(float(g[0]) - fx) for fx in face_xs)
            assert min_dx <= 0.1 + 1e-5
            assert cfg.shelf_y_min <= float(g[1]) <= cfg.shelf_y_max


# ---------------------------------------------------------------------------
# Leader env
# ---------------------------------------------------------------------------

class TestLeaderEnv:
    def test_reset_keeps_agent_and_object_in_lower_half(self):
        cfg = WarehouseLeaderEnvConfig()
        env = WarehouseLeaderEnv(cfg)
        for s in range(50):
            obs, _ = env.reset(seed=s)
            assert obs[1] <= cfg.wall_y - agent_inscribed_radius(cfg) + 1e-3
            assert obs[5] <= cfg.wall_y - cfg.object_radius + 1e-3
        env.close()

    def test_auto_pickup_when_object_in_cone(self):
        """Object 0.5 in front of agent → auto-pickup on the very next step."""
        cfg = WarehouseLeaderEnvConfig()
        env = WarehouseLeaderEnv(cfg)
        env.reset(seed=0, options={
            "agent_pos": np.array([0.0, -0.5], dtype=np.float32),
            "agent_theta": 0.0,
            "object_pos": np.array([0.5, -0.5], dtype=np.float32),
        })
        _, _, _, _, info = env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        assert info["picked_up"]
        assert env._carry
        env.close()

    def test_auto_drop_in_gap_terminates_with_goal_reward(self):
        cfg = WarehouseLeaderEnvConfig()
        env = WarehouseLeaderEnv(cfg)
        env.reset(seed=0, options={
            "agent_pos": np.array([0.0, -0.5], dtype=np.float32),
            "agent_theta": 0.0,
            "object_pos": np.array([0.5, -0.5], dtype=np.float32),
        })
        # Auto-pickup on first step.
        env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        assert env._carry
        # Approach wall facing up; auto-drop should fire as soon as the
        # would-be drop point falls inside gap 1.
        env._agent_theta = float(np.pi / 2)
        terminated = False
        for _ in range(15):
            _, r, term, _, info = env.step(np.array([0.0, cfg.max_speed, 0.0], dtype=np.float32))
            if info["dropped"]:
                assert info["gap_index"] >= 0
                assert term
                assert r > cfg.goal_reward - 1.0
                terminated = True
                break
        assert terminated, "auto-drop in gap never fired"
        env.close()

    def test_no_drop_when_carrying_and_not_facing_a_gap(self):
        """Carrying + far from any gap → no drop, no termination, no penalty."""
        cfg = WarehouseLeaderEnvConfig()
        env = WarehouseLeaderEnv(cfg)
        env.reset(seed=0, options={
            "agent_pos": np.array([0.0, -2.0], dtype=np.float32),
            "agent_theta": 0.0,
            "object_pos": np.array([0.5, -2.0], dtype=np.float32),
        })
        env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))  # auto-pickup
        assert env._carry
        # Agent at y=-2 facing east; drop point at (~0.55, -2.0) is far from any gap.
        _, _, term, _, info = env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        assert not info["dropped"]
        assert not term
        env.close()


# ---------------------------------------------------------------------------
# Leader CEM dynamics ↔ env step parity
# ---------------------------------------------------------------------------

class TestLeaderDynamicsParity:
    def _run_pair(self, options, actions):
        cfg = WarehouseLeaderEnvConfig()
        env = WarehouseLeaderEnv(cfg)
        env.reset(seed=42, options=options)
        env_obs_traj = [env._get_obs()]
        env_rewards = []
        for a in actions:
            o, r, t, tr, _ = env.step(a)
            env_obs_traj.append(o)
            env_rewards.append(r)
            if t or tr:
                break

        # Batch: replay from initial obs.
        b_obs = env_obs_traj[0][None, :].copy()
        batch_traj = [b_obs[0].copy()]
        batch_rewards = []
        for a in actions[: len(env_rewards)]:
            b_obs, b_r, _b_done = warehouse_leader_dynamics_batch(b_obs, a[None, :], cfg)
            batch_traj.append(b_obs[0].copy())
            batch_rewards.append(float(b_r[0]))
        env.close()
        return env_obs_traj, env_rewards, batch_traj, batch_rewards

    def test_translation_parity(self):
        actions = np.array([
            [1.0, 0.5, 0.2],
            [-0.5, 0.3, -0.3],
            [0.0, 0.0, 0.0],
        ], dtype=np.float32)
        opts = {
            "agent_pos": np.array([1.0, -2.0], dtype=np.float32),
            "agent_theta": 0.3,
            "object_pos": np.array([2.0, -3.0], dtype=np.float32),
        }
        env_traj, env_r, b_traj, b_r = self._run_pair(opts, actions)
        for et, bt in zip(env_traj, b_traj):
            np.testing.assert_allclose(et, bt, atol=1e-5)
        np.testing.assert_allclose(env_r, b_r, atol=1e-5)

    def test_pickup_drop_parity(self):
        # Auto-pickup on step 0 (object 0.5 in front), agent then heads north
        # facing up; auto-drop fires as soon as the drop point enters a gap.
        cfg = WarehouseLeaderEnvConfig()
        max_v = cfg.max_speed
        actions = np.array([
            [0.0, 0.0, 0.0],            # auto-pickup
            [0.0, 0.0, np.pi / 2 / 0.1],  # rotate to face up over one step
            [0.0, max_v, 0.0],
            [0.0, max_v, 0.0],
            [0.0, max_v, 0.0],
            [0.0, max_v, 0.0],
            [0.0, max_v, 0.0],
        ], dtype=np.float32)
        opts = {
            "agent_pos": np.array([0.0, -1.0], dtype=np.float32),
            "agent_theta": 0.0,
            "object_pos": np.array([0.5, -1.0], dtype=np.float32),
        }
        env_traj, env_r, b_traj, b_r = self._run_pair(opts, actions)
        for et, bt in zip(env_traj, b_traj):
            np.testing.assert_allclose(et, bt, atol=1e-4)
        np.testing.assert_allclose(env_r, b_r, atol=1e-4)


# ---------------------------------------------------------------------------
# Follower env
# ---------------------------------------------------------------------------

class TestFollowerEnv:
    def test_spawn_in_upper_half_outside_shelves(self):
        cfg = WarehouseFollowerEnvConfig()
        env = WarehouseFollowerEnv(cfg)
        for s in range(100):
            obs, _ = env.reset(seed=s)
            agent = obs[:2]
            assert agent[1] >= cfg.wall_y + agent_inscribed_radius(cfg) - 1e-3
            assert not point_in_any_shelf(agent, cfg)
        env.close()

    def test_object_spawns_in_a_gap(self):
        cfg = WarehouseFollowerEnvConfig()
        env = WarehouseFollowerEnv(cfg)
        for s in range(50):
            obs, _ = env.reset(seed=s)
            obj = obs[4:6]
            assert gap_index_for_position(obj, cfg) >= 0
        env.close()

    def test_options_object_pos_is_used(self):
        """Reproducibility: ``options['object_pos']`` is honored exactly."""
        cfg = WarehouseFollowerEnvConfig()
        env = WarehouseFollowerEnv(cfg)
        target = np.array([0.0, 0.0], dtype=np.float32)
        obs, _ = env.reset(seed=0, options={"object_pos": target})
        np.testing.assert_allclose(obs[4:6], target)
        env.close()

    def test_options_goal_is_used(self):
        cfg = WarehouseFollowerEnvConfig()
        env = WarehouseFollowerEnv(cfg)
        goal = np.array([2.5, 2.5], dtype=np.float32)
        obs, _ = env.reset(seed=0, options={"goal": goal})
        np.testing.assert_allclose(obs[7:9], goal)
        env.close()

    def test_auto_pickup_then_auto_goal_drop_terminates(self):
        cfg = WarehouseFollowerEnvConfig()
        env = WarehouseFollowerEnv(cfg)
        env.reset(seed=0, options={
            "agent_pos": np.array([0.0, 0.5], dtype=np.float32),
            "agent_theta": float(-np.pi / 2),
            "object_pos": np.array([0.0, 0.0], dtype=np.float32),
            "goal": np.array([2.5, 2.5], dtype=np.float32),
        })
        _, _, _, _, info = env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        assert info["picked_up"]
        # Warp to the goal vicinity facing the goal — auto-drop should fire.
        env._agent_pos = np.array([3.05, 2.5], dtype=np.float32)
        env._agent_theta = float(np.pi)
        _, r, term, _, info = env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        assert info["dropped"]
        assert info["on_goal"]
        assert term
        assert r > cfg.goal_reward - 1.0
        env.close()

    def test_spawn_near_goal_places_agent_next_to_goal_facing_it(self):
        """``spawn_near_goal=True`` puts the follower drop_distance from the goal."""
        from dataclasses import replace

        cfg = replace(WarehouseFollowerEnvConfig(), spawn_near_goal=True)
        env = WarehouseFollowerEnv(cfg)
        for s in range(20):
            goal = np.array([2.5, 2.5], dtype=np.float32) if s % 2 == 0 else np.array([-2.5, 3.0], dtype=np.float32)
            obs, _ = env.reset(seed=s, options={"goal": goal})
            agent = obs[:2]
            # The drop point of the spawned pose should land on the goal.
            sin_t, cos_t = float(obs[2]), float(obs[3])
            heading = np.array([cos_t, sin_t], dtype=np.float32)
            drop_pt = agent + heading * cfg.drop_distance
            assert np.allclose(drop_pt, goal, atol=1e-3), (
                "drop point {dp} should equal goal {g}".format(dp=drop_pt, g=goal)
            )
        env.close()

    def test_spawn_near_goal_options_pos_overrides(self):
        """Explicit ``options['agent_pos']`` still wins over spawn_near_goal."""
        from dataclasses import replace

        cfg = replace(WarehouseFollowerEnvConfig(), spawn_near_goal=True)
        env = WarehouseFollowerEnv(cfg)
        explicit = np.array([-3.0, 3.0], dtype=np.float32)
        obs, _ = env.reset(seed=0, options={
            "agent_pos": explicit,
            "agent_theta": 0.0,
            "goal": np.array([2.5, 2.5], dtype=np.float32),
        })
        np.testing.assert_allclose(obs[:2], explicit)
        env.close()

    def test_shelf_collision_emits_penalty(self):
        cfg = WarehouseFollowerEnvConfig()
        env = WarehouseFollowerEnv(cfg)
        env.reset(seed=0, options={
            "agent_pos": np.array([0.5, 1.0], dtype=np.float32),
            "agent_theta": float(np.pi),
            "object_pos": np.array([0.0, 0.0], dtype=np.float32),
            "goal": np.array([2.5, 2.5], dtype=np.float32),
        })
        _, r, _, _, info = env.step(np.array([-cfg.max_speed, 0.0, 0.0], dtype=np.float32))
        assert info["shelf_collision"]
        assert not point_in_any_shelf(env._agent_pos, cfg)
        env.close()


# ---------------------------------------------------------------------------
# Value query closure
# ---------------------------------------------------------------------------

class TestCarryPenaltySchedule:
    def test_linear_schedule_anchors_and_midpoint(self):
        """Warmup holds initial; midpoint of anneal gives midpoint value; plateau holds final."""
        from critic_sharing.training.warehouse_follower import CarryPenaltyScheduleCallback

        cb = CarryPenaltyScheduleCallback(
            initial_value=0.01,
            final_value=0.05,
            warmup_steps=500_000,
            anneal_steps=1_000_000,
        )
        assert cb.compute_value(0) == pytest.approx(0.01)
        assert cb.compute_value(499_999) == pytest.approx(0.01)
        assert cb.compute_value(500_000) == pytest.approx(0.01)
        assert cb.compute_value(1_000_000) == pytest.approx(0.03)  # midpoint of anneal
        assert cb.compute_value(1_500_000) == pytest.approx(0.05)
        assert cb.compute_value(2_000_000) == pytest.approx(0.05)  # plateau

    def test_disabled_when_warmup_and_anneal_both_zero(self):
        """warmup==0 AND anneal==0 should immediately return final value."""
        from critic_sharing.training.warehouse_follower import CarryPenaltyScheduleCallback

        cb = CarryPenaltyScheduleCallback(
            initial_value=0.01,
            final_value=0.05,
            warmup_steps=0,
            anneal_steps=0,
        )
        assert cb.compute_value(0) == pytest.approx(0.05)
        assert cb.compute_value(1_000_000) == pytest.approx(0.05)

    def test_env_setter_updates_step_reward(self):
        """``env.set_step_penalty_carrying(v)`` changes the step reward."""
        cfg = WarehouseFollowerEnvConfig()
        env = WarehouseFollowerEnv(cfg)
        env.reset(seed=0, options={
            "agent_pos": np.array([0.0, 1.5], dtype=np.float32),
            "agent_theta": 0.0,
            "object_pos": np.array([2.5, 1.5], dtype=np.float32),  # far away — no auto-pickup
            "goal": np.array([2.5, 2.5], dtype=np.float32),
        })
        env._carry = True
        env.set_step_penalty_carrying(0.01)
        _, r_low, _, _, _ = env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))

        env.reset(seed=0, options={
            "agent_pos": np.array([0.0, 1.5], dtype=np.float32),
            "agent_theta": 0.0,
            "object_pos": np.array([2.5, 1.5], dtype=np.float32),
            "goal": np.array([2.5, 2.5], dtype=np.float32),
        })
        env._carry = True
        env.set_step_penalty_carrying(0.05)
        _, r_high, _, _, _ = env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        env.close()

        # Same step except for the carry penalty value: difference should be ≈ 0.04.
        assert r_low - r_high == pytest.approx(0.04, abs=1e-6)


class TestValueQueryClosure:
    def test_lambda_extracts_object_pos_from_leader_obs(self):
        """The closure should ignore everything in leader obs except indices [4:6]."""
        from critic_sharing.scenarios.warehouse import build_warehouse_follower_value_fn

        # Stand-in PPO-like model: implement the minimal predict_values surface.
        class FakePolicy:
            class Dev:
                type = "cpu"
            device = Dev()

            def obs_to_tensor(self, obs):
                import torch as th
                return th.as_tensor(np.asarray(obs, dtype=np.float32)), None

            def predict_values(self, obs):
                # Return follower obs[:, 4] + obs[:, 5] (object x + y).
                return obs[:, 4:6].sum(dim=1, keepdim=True)

        class FakeModel:
            policy = FakePolicy()

        cfg = WarehouseFollowerEnvConfig()
        s_pos = np.array([0.0, 1.5], dtype=np.float32)
        goal = np.array([2.5, 2.5], dtype=np.float32)
        fn = build_warehouse_follower_value_fn(FakeModel(), cfg, s_pos, 0.0, goal)

        leader_obs = np.zeros((4, 7), dtype=np.float32)
        leader_obs[:, 4] = [1.0, 2.0, 3.0, -1.0]  # object_x
        leader_obs[:, 5] = [0.0, 0.5, -0.5, 1.0]  # object_y
        values = fn(leader_obs)
        np.testing.assert_allclose(values, [1.0, 2.5, 2.5, 0.0])
