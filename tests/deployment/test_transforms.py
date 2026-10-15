"""Unit tests for the static SE(2) transform between real-world and sim frames."""

from __future__ import annotations

import numpy as np
import pytest

from critic_sharing.deployment.transforms import StaticSE2


def test_identity_position_is_unchanged():
    tf = StaticSE2()
    p = np.array([1.23, -4.56])
    np.testing.assert_allclose(tf.world_to_sim_position(p), p)


def test_identity_velocity_is_unchanged():
    tf = StaticSE2()
    v = np.array([0.7, -0.3])
    np.testing.assert_allclose(tf.sim_to_world_velocity(v), v)


def test_translation_only_position():
    tf = StaticSE2(tx=2.0, ty=-1.0, theta=0.0)
    # A point at real (3, 1) is at sim (1, 2) when sim origin is real (2, -1).
    np.testing.assert_allclose(tf.world_to_sim_position(np.array([3.0, 1.0])), [1.0, 2.0])


def test_translation_does_not_affect_velocity():
    tf = StaticSE2(tx=10.0, ty=-5.0, theta=0.0)
    v = np.array([0.4, 0.2])
    np.testing.assert_allclose(tf.sim_to_world_velocity(v), v)


def test_quarter_turn_position():
    # Sim frame rotated +90deg about the real-world origin.
    tf = StaticSE2(tx=0.0, ty=0.0, theta=np.pi / 2.0)
    # Real point (0, 1) is sim point (1, 0): R(-pi/2) @ (0, 1) = (1, 0).
    np.testing.assert_allclose(
        tf.world_to_sim_position(np.array([0.0, 1.0])),
        [1.0, 0.0],
        atol=1e-12,
    )


def test_quarter_turn_velocity():
    # Sim velocity (1, 0) becomes real (0, 1) under +90deg sim->real rotation.
    tf = StaticSE2(tx=0.0, ty=0.0, theta=np.pi / 2.0)
    np.testing.assert_allclose(
        tf.sim_to_world_velocity(np.array([1.0, 0.0])),
        [0.0, 1.0],
        atol=1e-12,
    )


@pytest.mark.parametrize("theta", [-1.3, 0.0, 0.7, 2.5])
def test_position_round_trip(theta: float):
    tf = StaticSE2(tx=0.4, ty=-2.1, theta=theta)
    p_sim = np.array([1.5, -0.25])
    p_real = StaticSE2(tx=tf.tx, ty=tf.ty, theta=tf.theta)
    # Forward: sim -> real via inverse formulas
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    p_real_xy = R @ p_sim + np.array([tf.tx, tf.ty])
    recovered = p_real.world_to_sim_position(p_real_xy)
    np.testing.assert_allclose(recovered, p_sim, atol=1e-12)


@pytest.mark.parametrize("theta", [-1.3, 0.0, 0.7, 2.5])
def test_velocity_round_trip(theta: float):
    tf = StaticSE2(tx=99.0, ty=99.0, theta=theta)  # translation must not matter
    v_sim = np.array([0.6, -0.4])
    v_real = tf.sim_to_world_velocity(v_sim)
    R_inv = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    np.testing.assert_allclose(R_inv @ v_real, v_sim, atol=1e-12)
