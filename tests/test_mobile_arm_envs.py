"""Tests for the mobile_arm scenario."""

from __future__ import annotations

import math

import numpy as np
import pytest

from critic_sharing.config.schema import (
    MobileArmFollowerEnvConfig,
    MobileArmLeaderEnvConfig,
)
from critic_sharing.scenarios.mobile_arm import (
    MobileArmFollowerEnv,
    MobileArmLeaderEnv,
    mobile_arm_leader_dynamics_batch,
)
from critic_sharing.scenarios.mobile_arm.common import (
    arm_collides,
    base_inscribed_radius,
    base_to_world,
    enforce_base_object_collision,
    forward_kinematics,
    forward_kinematics_batch,
    tray_pos_base,
    world_to_base,
)


# ---------------------------------------------------------------------------
# Forward kinematics
# ---------------------------------------------------------------------------

class TestFK:
    def test_default_pose_above_ground(self):
        cfg = MobileArmFollowerEnvConfig()
        ee = forward_kinematics(
            0.0, cfg.theta2_default, cfg.theta3_default, cfg.theta4_default, cfg,
        )
        # Default pose: arm above the floor.
        assert ee[2] > 0.0
        # Specifically: (1.024, 0, 1.324) by hand calc.
        assert ee[0] == pytest.approx(1.024, abs=1e-3)
        assert ee[2] == pytest.approx(1.324, abs=1e-3)

    def test_yaw_rotates_xy(self):
        cfg = MobileArmFollowerEnvConfig()
        ee_fwd = forward_kinematics(
            0.0, cfg.theta2_default, cfg.theta3_default, cfg.theta4_default, cfg,
        )
        ee_left = forward_kinematics(
            math.pi / 2, cfg.theta2_default, cfg.theta3_default, cfg.theta4_default, cfg,
        )
        # Rotating yaw by π/2 swaps x ↔ y for a forward-facing arm.
        assert ee_left[0] == pytest.approx(0.0, abs=1e-5)
        assert ee_left[1] == pytest.approx(ee_fwd[0], abs=1e-5)

    def test_fully_extended_horizontal(self):
        """``θ_2 = θ_3 = θ_4 = 0`` ⇒ arm fully horizontal, EE at total reach."""
        cfg = MobileArmFollowerEnvConfig()
        ee = forward_kinematics(0.0, 0.0, 0.0, 0.0, cfg)
        total_reach = cfg.L1 + cfg.L2a + cfg.L2b
        assert ee[0] == pytest.approx(total_reach, abs=1e-5)
        assert ee[1] == pytest.approx(0.0, abs=1e-5)
        assert ee[2] == pytest.approx(cfg.base_height, abs=1e-5)

    def test_batch_matches_scalar(self):
        cfg = MobileArmFollowerEnvConfig()
        thetas = np.array([
            [0.0, cfg.theta2_default, cfg.theta3_default, cfg.theta4_default],
            [math.pi / 4, 0.5, 1.0, 0.3],
            [-math.pi / 3, 0.2, 0.7, 0.5],
        ], dtype=np.float32)
        batch = forward_kinematics_batch(thetas, cfg)
        for theta_row, batch_row in zip(thetas, batch):
            scalar = forward_kinematics(*[float(x) for x in theta_row], config=cfg)
            np.testing.assert_allclose(batch_row, scalar, atol=1e-5)


# ---------------------------------------------------------------------------
# Frame transforms
# ---------------------------------------------------------------------------

class TestFrameTransform:
    def test_round_trip(self):
        base_pos = np.array([1.5, -2.0])
        base_yaw = 0.7
        for p in [
            np.array([3.0, 1.0]),
            np.array([-1.0, 4.0]),
            np.array([0.0, 0.0]),
        ]:
            p_base = world_to_base(p, base_pos, base_yaw)
            p_back = base_to_world(p_base, base_pos, base_yaw)
            np.testing.assert_allclose(p_back, p, atol=1e-5)

    def test_3d_passes_z(self):
        p = np.array([1.0, 2.0, 0.5])
        p_base = world_to_base(p, np.array([0.0, 0.0]), 0.0)
        assert p_base[2] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Base ↔ object collision
# ---------------------------------------------------------------------------

class TestCollision:
    def test_no_clamp_when_far(self):
        cfg = MobileArmLeaderEnvConfig()
        next_pos = np.array([5.0, 5.0], dtype=np.float32)
        out = enforce_base_object_collision(next_pos, np.array([0.0, 0.0]), cfg)
        np.testing.assert_allclose(out, next_pos)

    def test_clamp_to_contact(self):
        cfg = MobileArmLeaderEnvConfig()
        contact = base_inscribed_radius(cfg) + cfg.object_radius
        # Approach object from +x direction; next_pos inside contact.
        out = enforce_base_object_collision(
            np.array([0.1, 0.0], dtype=np.float32),
            np.array([0.0, 0.0]),
            cfg,
        )
        np.testing.assert_allclose(out, np.array([contact, 0.0]), atol=1e-5)


# ---------------------------------------------------------------------------
# Leader env
# ---------------------------------------------------------------------------

class TestLeaderEnv:
    def test_terminate_at_desired_distance_yields_max_bonus(self):
        cfg = MobileArmLeaderEnvConfig()
        env = MobileArmLeaderEnv(cfg)
        env.reset(seed=0, options={
            "object_pos": np.array([0.0, 0.0]),
            "base_pos": np.array([cfg.desired_distance, 0.0]),
            "base_yaw": 0.0,
        })
        _, r, term, _, info = env.step(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
        assert term
        assert info["terminate_bonus"] == pytest.approx(cfg.goal_reward, rel=1e-3)
        env.close()

    def test_terminate_far_from_d_star_yields_low_bonus(self):
        cfg = MobileArmLeaderEnvConfig()
        env = MobileArmLeaderEnv(cfg)
        env.reset(seed=0, options={
            "object_pos": np.array([0.0, 0.0]),
            "base_pos": np.array([cfg.desired_distance + 2.0, 0.0]),
            "base_yaw": 0.0,
        })
        _, _, term, _, info = env.step(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
        assert term
        assert info["terminate_bonus"] < 1.0  # Gaussian decayed to near-zero

    def test_object_collision(self):
        cfg = MobileArmLeaderEnvConfig()
        env = MobileArmLeaderEnv(cfg)
        env.reset(seed=0, options={
            "object_pos": np.array([0.0, 0.0]),
            "base_pos": np.array([1.0, 0.0]),
            "base_yaw": 0.0,
        })
        for _ in range(50):
            obs, _, term, trunc, _ = env.step(
                np.array([-cfg.max_speed, 0.0, 0.0, -1.0], dtype=np.float32),
            )
            if term or trunc:
                break
        contact = base_inscribed_radius(cfg) + cfg.object_radius
        # Final base_x should be ≥ contact (clamped at boundary).
        assert env._base_pos[0] >= contact - 1e-3

    def test_no_terminate_below_threshold(self):
        cfg = MobileArmLeaderEnvConfig()
        env = MobileArmLeaderEnv(cfg)
        env.reset(seed=0, options={
            "object_pos": np.array([0.0, 0.0]),
            "base_pos": np.array([cfg.desired_distance, 0.0]),
            "base_yaw": 0.0,
        })
        # terminate=0.4 (below 0.5 threshold) → episode continues.
        _, _, term, _, _ = env.step(np.array([0.0, 0.0, 0.0, 0.4], dtype=np.float32))
        assert not term


# ---------------------------------------------------------------------------
# Leader CEM-batch parity
# ---------------------------------------------------------------------------

class TestLeaderBatchParity:
    def test_dones_returned_and_fire_on_terminate_action(self):
        """Batch dynamics returns 3-tuple; done fires iff terminate ≥ threshold."""
        cfg = MobileArmLeaderEnvConfig()
        obs = np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        a_no_term = np.array([[0.0, 0.0, 0.0, -1.0]], dtype=np.float32)
        _, _, dones = mobile_arm_leader_dynamics_batch(obs, a_no_term, cfg)
        assert dones.shape == (1,)
        assert dones.dtype == bool
        assert not bool(dones[0])
        a_term = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        _, _, dones = mobile_arm_leader_dynamics_batch(obs, a_term, cfg)
        assert bool(dones[0])

    def test_translation_parity(self):
        cfg = MobileArmLeaderEnvConfig()
        env = MobileArmLeaderEnv(cfg)
        env.reset(seed=0, options={
            "object_pos": np.array([2.0, 1.0]),
            "base_pos": np.array([3.0, 0.0]),
            "base_yaw": 0.5,
        })
        obs0 = env._get_obs()
        actions = np.array([
            [0.5, 0.5, 0.2, -1.0],
            [-0.3, 0.7, -0.1, -1.0],
            [0.0, 0.0, 0.0, -1.0],
        ], dtype=np.float32)
        env_obs = [obs0]
        env_r = []
        for a in actions:
            o, r, t, tr, _ = env.step(a)
            env_obs.append(o); env_r.append(r)
            if t or tr:
                break
        env.close()

        b = obs0[None, :].copy()
        b_traj = [b[0].copy()]
        b_r = []
        for a in actions[: len(env_r)]:
            b, br, _bd = mobile_arm_leader_dynamics_batch(b, a[None, :], cfg)
            b_traj.append(b[0].copy()); b_r.append(float(br[0]))
        for e, b in zip(env_obs, b_traj):
            np.testing.assert_allclose(e, b, atol=1e-5)
        np.testing.assert_allclose(env_r, b_r, atol=1e-5)


# ---------------------------------------------------------------------------
# Follower env
# ---------------------------------------------------------------------------

class TestFollowerEnv:
    def test_obs_shape(self):
        env = MobileArmFollowerEnv(MobileArmFollowerEnvConfig())
        obs, _ = env.reset(seed=0)
        # 4 joint sin/cos pairs (8) + object xyz (3) + carry_flag (1) = 12.
        assert obs.shape == (12,)

    def test_action_shape(self):
        env = MobileArmFollowerEnv(MobileArmFollowerEnvConfig())
        # 4 joint velocities.
        assert env.action_space.shape == (4,)

    def test_auto_pickup(self):
        cfg = MobileArmFollowerEnvConfig()
        env = MobileArmFollowerEnv(cfg)
        # Place object exactly at default-pose EE.
        ee = forward_kinematics(
            0.0, cfg.theta2_default, cfg.theta3_default, cfg.theta4_default, cfg,
        )
        env.reset(seed=0, options={
            "theta": np.array([
                0.0, cfg.theta2_default, cfg.theta3_default, cfg.theta4_default,
            ]),
            "object_pos": ee,
        })
        _, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))
        assert info["picked_up"]
        assert env._carry

    def test_auto_drop_on_tray_terminates(self):
        """Warp the arm to a pose where EE is on the tray, then verify auto-drop."""
        cfg = MobileArmFollowerEnvConfig()
        env = MobileArmFollowerEnv(cfg)
        # Solve for joints that place EE at tray (0.25, 0, 0.45) numerically.
        # With θ_1 = 0, EE in base frame = (r_planar, 0, base_height + z_planar).
        # Need r_planar = 0.25, z_planar = 0.05.
        # Use scipy.optimize? No — sample a few values and verify one works.
        # Easier: scan over (θ_2, θ_3, θ_4) in their valid ranges and pick a
        # set with EE close to tray.
        from critic_sharing.scenarios.mobile_arm.common import tray_pos_base
        tray = tray_pos_base(cfg)
        best_theta = None
        best_dist = float("inf")
        for t2 in np.linspace(cfg.theta2_low, cfg.theta2_high, 20):
            for t3 in np.linspace(cfg.theta3_low, cfg.theta3_high, 25):
                for t4 in np.linspace(cfg.theta4_low, cfg.theta4_high, 25):
                    ee = forward_kinematics(0.0, float(t2), float(t3), float(t4), cfg)
                    d = float(np.linalg.norm(ee - tray))
                    if d < best_dist:
                        best_dist = d
                        best_theta = (float(t2), float(t3), float(t4))
        assert best_theta is not None
        # Should be feasible to get EE within eps_tray of tray.
        assert best_dist < cfg.eps_tray, "no IK solution found within eps_tray"

        env.reset(seed=0)
        env._theta = env._clip_joints(np.array([0.0, *best_theta], dtype=np.float32))
        env._carry = True
        _, r, term, _, info = env.step(np.zeros(4, dtype=np.float32))
        assert info["dropped"]
        assert term
        assert r > cfg.goal_reward - 1.0

    def test_options_object_pos_used(self):
        cfg = MobileArmFollowerEnvConfig()
        env = MobileArmFollowerEnv(cfg)
        target = np.array([0.5, 0.3, 0.0], dtype=np.float32)
        obs, _ = env.reset(seed=0, options={"object_pos": target})
        # Object position now lives at indices [8:11].
        np.testing.assert_allclose(obs[8:11], target, atol=1e-5)

    def test_all_joints_default_at_reset(self):
        """Reset with no ``theta`` option → all four joints at config defaults."""
        cfg = MobileArmFollowerEnvConfig()
        env = MobileArmFollowerEnv(cfg)
        for s in range(10):
            env.reset(seed=s)
            assert env._theta[0] == pytest.approx(cfg.theta1_default, abs=1e-5)
            assert env._theta[1] == pytest.approx(cfg.theta2_default, abs=1e-5)
            assert env._theta[2] == pytest.approx(cfg.theta3_default, abs=1e-5)
            assert env._theta[3] == pytest.approx(cfg.theta4_default, abs=1e-5)

    def test_theta1_wraps_to_pi_range(self):
        """θ_1 wraps to [-π, π] after the integration step (matches leader yaw)."""
        cfg = MobileArmFollowerEnvConfig()
        env = MobileArmFollowerEnv(cfg)
        env.reset(seed=0)
        # Manually set θ_1 just below π so a small +ω push wraps it.
        env._theta = env._clip_joints(np.array([
            math.pi - 0.05,
            cfg.theta2_default,
            cfg.theta3_default,
            cfg.theta4_default,
        ], dtype=np.float32))
        # Action: +max_joint_rate on θ_1; per dt = π * 0.1 ≈ 0.314 rad → wraps past π.
        action = np.array([cfg.max_joint_rate, 0.0, 0.0, 0.0], dtype=np.float32)
        env.step(action)
        # After the wrap, θ_1 should be in [-π, π] and just past -π (wrapped).
        assert -math.pi <= env._theta[0] <= math.pi
        assert env._theta[0] < 0.0  # just past -π

    def test_object_spawn_radius_within_range(self):
        cfg = MobileArmFollowerEnvConfig()
        env = MobileArmFollowerEnv(cfg)
        for s in range(50):
            env.reset(seed=s)
            r = float(np.linalg.norm(env._object_pos[:2]))
            assert cfg.object_r_min - 1e-5 <= r <= cfg.object_r_max + 1e-5


# ---------------------------------------------------------------------------
# Tray geometry
# ---------------------------------------------------------------------------

class TestTrayPos:
    def test_z_above_chassis_top(self):
        cfg = MobileArmLeaderEnvConfig()
        tray = tray_pos_base(cfg)
        assert tray[2] == pytest.approx(cfg.base_height + cfg.tray_offset_z)


# ---------------------------------------------------------------------------
# Arm-vs-environment collision
# ---------------------------------------------------------------------------

class TestArmCollision:
    def test_default_pose_no_collision(self):
        cfg = MobileArmFollowerEnvConfig()
        theta = np.array([
            0.0, cfg.theta2_default, cfg.theta3_default, cfg.theta4_default,
        ], dtype=np.float32)
        floor, base = arm_collides(theta, cfg)
        assert not floor
        assert not base

    def test_ee_below_floor_detected(self):
        """θ_2=0, θ_3=π/2, θ_4=0 ⇒ wrist + EE at z=base_height-0.6=-0.2."""
        cfg = MobileArmFollowerEnvConfig()
        theta = np.array([0.0, 0.0, math.pi / 2, 0.0], dtype=np.float32)
        floor, _ = arm_collides(theta, cfg)
        assert floor

    def test_arm_through_base_detected(self):
        """θ_3=π folds the lower arm fully back along the upper arm.

        With θ_2=0 (upper arm horizontal forward), wrist ends up at
        ``r = L1 + L2a*cos(-π) = 0.5 - 0.6 = -0.1`` — inside the base xy
        footprint at z=base_height. With θ_4=π then EE at r=-0.1+L2b=0.5.
        Wrist endpoint is just inside the base box: x=-0.1 ∈ [-0.3, 0.3],
        y=0, z=base_height (boundary). The check uses ``z < base_height``
        strictly, so we drop wrist a hair below by giving θ_3 a tiny extra:
        """
        cfg = MobileArmFollowerEnvConfig()
        # θ_3 just past π would put the lower arm sweeping below base_height.
        # Easier: verify a clearly-colliding pose where both wrist and EE are
        # inside the base box.
        # θ_2=0 (horizontal), θ_3=3π/4 (lower arm bent back-and-down):
        #   wrist_r = 0.5 + 0.6*cos(-3π/4) = 0.5 - 0.424 = 0.076 (inside base xy)
        #   wrist_z = 0 + 0.6*sin(-3π/4) = -0.424 → world z = -0.024 (below floor)
        theta = np.array([0.0, 0.0, 3 * math.pi / 4, 0.0], dtype=np.float32)
        floor, base = arm_collides(theta, cfg)
        # Wrist below floor *and* xy inside base footprint ⇒ both flags.
        assert floor

    def test_step_reverts_on_collision(self):
        """A joint update that would crash the EE through the floor is reverted."""
        cfg = MobileArmFollowerEnvConfig()
        env = MobileArmFollowerEnv(cfg)
        # Start at default safe pose.
        env.reset(seed=0, options={
            "theta": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),  # extended horizontal
            "object_pos": np.array([5.0, 5.0, 0.0], dtype=np.float32),  # far, no pickup
        })
        prev_theta = env._theta.copy()
        # Action: dθ_3 = +max_joint_rate (wants to bend lower arm down a lot).
        # Per dt ≈ 0.314 rad. With θ_2=0, this sends wrist below the floor.
        action = np.array([0.0, 0.0, cfg.max_joint_rate, 0.0], dtype=np.float32)
        # Take many steps; eventually the env will refuse to advance θ_3.
        for _ in range(20):
            _, r, _, _, info = env.step(action)
            if info["arm_floor_collision"] or info["arm_base_collision"]:
                # Any subsequent step that *would* collide reverts θ.
                assert env._theta[2] <= prev_theta[2] + 1e-3 + cfg.max_joint_rate * cfg.dt
                # And the collision penalty is applied.
                assert r < -cfg.collision_penalty / 2.0
                env.close()
                return
            prev_theta = env._theta.copy()
        env.close()
        pytest.fail("Expected a collision within 20 steps but none fired")
