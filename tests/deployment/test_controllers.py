"""Unit tests for the ControllerAdapter implementations.

The PPO and CEM adapters are pure-Python (no ROS, no rclpy). PPO is
exercised with a tiny fake model that captures calls; CEM is exercised
with mocked CEMController + JointUtilityCost so we can assert the
reward bookkeeping math without spinning a real planner.
"""

from __future__ import annotations

import numpy as np
import pytest

from critic_sharing.config.schema import HouseholdCleanerEnvConfig
from critic_sharing.deployment.controllers import (
    CEMAdapter,
    ControllerAdapter,
    PPOPolicyAdapter,
)


# ---------------------------------------------------------------------------
# PPOPolicyAdapter
# ---------------------------------------------------------------------------

class _FakePPO:
    def __init__(self, action=(0.1, 0.2, 0.3)):
        self.action = np.asarray(action, dtype=np.float32)
        self.calls: list[tuple[np.ndarray, bool]] = []

    def predict(self, obs, deterministic):
        self.calls.append((np.asarray(obs).copy(), bool(deterministic)))
        return self.action, None


def test_ppo_adapter_rejects_none_model():
    with pytest.raises(ValueError):
        PPOPolicyAdapter(None)


def test_ppo_adapter_returns_predict_output():
    model = _FakePPO(action=(0.4, -0.2, 1.0))
    ad = PPOPolicyAdapter(model)
    out = ad.act(np.zeros(10, dtype=np.float32))
    np.testing.assert_allclose(out, [0.4, -0.2, 1.0])


def test_ppo_adapter_calls_predict_deterministic():
    model = _FakePPO()
    ad = PPOPolicyAdapter(model)
    obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                   dtype=np.float32)
    ad.act(obs)
    assert len(model.calls) == 1
    np.testing.assert_allclose(model.calls[0][0], obs)
    assert model.calls[0][1] is True


def test_ppo_adapter_implements_controller_interface():
    ad = PPOPolicyAdapter(_FakePPO())
    assert isinstance(ad, ControllerAdapter)
    # reset() is a no-op for PPO.
    ad.reset()


# ---------------------------------------------------------------------------
# CEMAdapter
# ---------------------------------------------------------------------------

class _MockCEMController:
    def __init__(self):
        self.reset_count = 0
        self.command_calls: list[tuple[np.ndarray, int | None]] = []

    def reset(self):
        self.reset_count += 1

    def command(self, obs, steps_remaining=None):
        self.command_calls.append((np.asarray(obs).copy(), steps_remaining))
        return np.array([0.05, -0.05, 0.1], dtype=np.float32)


class _MockJointCost:
    def __init__(self):
        self.calls: list[tuple[float, float]] = []

    def set_incurred_return(self, incurred, discount=1.0):
        self.calls.append((float(incurred), float(discount)))


def _cleaner_obs(cleaner_xy, dirt_xy, table_xy=(0.0, 0.0)):
    o = np.zeros(10, dtype=np.float32)
    o[0:2] = cleaner_xy
    o[2:4] = (np.sin(0.0), np.cos(0.0))  # yaw=0
    o[4:6] = table_xy
    o[6:8] = (np.sin(0.0), np.cos(0.0))
    o[8:10] = dirt_xy
    return o


def test_cem_adapter_rejects_none():
    with pytest.raises(ValueError):
        CEMAdapter(controller=None, cost=_MockJointCost(),
                   cleaner_env_config=HouseholdCleanerEnvConfig())
    with pytest.raises(ValueError):
        CEMAdapter(controller=_MockCEMController(), cost=None,
                   cleaner_env_config=HouseholdCleanerEnvConfig())


def test_cem_adapter_first_tick_no_reward_accumulated():
    """Tick 1 has no prev_obs → no reward delta. Cost sees (0, 1.0)."""
    ctrl, cost = _MockCEMController(), _MockJointCost()
    cfg = HouseholdCleanerEnvConfig()
    ad = CEMAdapter(controller=ctrl, cost=cost, cleaner_env_config=cfg)
    ad.reset()
    obs = _cleaner_obs(cleaner_xy=(0.0, 0.0), dirt_xy=(1.0, 0.0))
    ad.act(obs)
    assert cost.calls == [(0.0, 1.0)]


def test_cem_adapter_second_tick_accumulates_reward():
    """Cleaner closed 0.5 m toward dirt → progress reward 3.0 * 0.5 = 1.5,
    minus step_penalty 0.01 → 1.49. First call to cost is from tick 1
    (no prev_obs); second is the one we check.
    """
    ctrl, cost = _MockCEMController(), _MockJointCost()
    cfg = HouseholdCleanerEnvConfig()
    ad = CEMAdapter(controller=ctrl, cost=cost, cleaner_env_config=cfg,
                    leader_gamma=0.99)
    ad.reset()
    ad.act(_cleaner_obs(cleaner_xy=(0.0, 0.0), dirt_xy=(1.0, 0.0)))
    ad.act(_cleaner_obs(cleaner_xy=(0.5, 0.0), dirt_xy=(1.0, 0.0)))
    expected_reward = float(cfg.progress_weight) * 0.5 - float(cfg.step_penalty)
    assert cost.calls[-1] == pytest.approx((expected_reward, 0.99))


def test_cem_adapter_discount_compounds_correctly():
    ctrl, cost = _MockCEMController(), _MockJointCost()
    cfg = HouseholdCleanerEnvConfig()
    ad = CEMAdapter(controller=ctrl, cost=cost, cleaner_env_config=cfg,
                    leader_gamma=0.9)
    ad.reset()
    # Three identical-distance ticks → each reward is -step_penalty.
    obs = _cleaner_obs(cleaner_xy=(0.0, 0.0), dirt_xy=(1.0, 0.0))
    for _ in range(4):
        ad.act(obs)
    # After 4 calls: 3 transitions, each reward = -0.01.
    # incurred = -0.01 * (1 + 0.9 + 0.81) = -0.01 * 2.71 = -0.0271
    # gamma_pow_t = 0.9 ** 3 = 0.729
    expected = -0.01 * (1.0 + 0.9 + 0.81)
    np.testing.assert_allclose(cost.calls[-1][0], expected, atol=1e-6)
    np.testing.assert_allclose(cost.calls[-1][1], 0.9 ** 3, atol=1e-6)


def test_cem_adapter_reset_clears_state():
    ctrl, cost = _MockCEMController(), _MockJointCost()
    ad = CEMAdapter(controller=ctrl, cost=cost,
                    cleaner_env_config=HouseholdCleanerEnvConfig())
    ad.reset()
    ad.act(_cleaner_obs(cleaner_xy=(0.0, 0.0), dirt_xy=(1.0, 0.0)))
    ad.act(_cleaner_obs(cleaner_xy=(0.5, 0.0), dirt_xy=(1.0, 0.0)))
    assert cost.calls[-1] != (0.0, 1.0)  # confirm accumulated
    # Reset wipes the slate.
    ad.reset()
    ad.act(_cleaner_obs(cleaner_xy=(0.0, 0.0), dirt_xy=(1.0, 0.0)))
    assert cost.calls[-1] == (0.0, 1.0)
    assert ctrl.reset_count == 2  # once per ad.reset()


def test_cem_adapter_returns_command_output():
    ctrl, cost = _MockCEMController(), _MockJointCost()
    ad = CEMAdapter(controller=ctrl, cost=cost,
                    cleaner_env_config=HouseholdCleanerEnvConfig())
    ad.reset()
    out = ad.act(_cleaner_obs((0, 0), (1, 0)))
    np.testing.assert_allclose(out, [0.05, -0.05, 0.1])


def test_cem_adapter_forwards_steps_remaining():
    ctrl, cost = _MockCEMController(), _MockJointCost()
    ad = CEMAdapter(controller=ctrl, cost=cost,
                    cleaner_env_config=HouseholdCleanerEnvConfig())
    ad.reset()
    ad.act(_cleaner_obs((0, 0), (1, 0)), steps_remaining=42)
    assert ctrl.command_calls[-1][1] == 42


def test_cem_adapter_implements_controller_interface():
    ad = CEMAdapter(controller=_MockCEMController(), cost=_MockJointCost(),
                    cleaner_env_config=HouseholdCleanerEnvConfig())
    assert isinstance(ad, ControllerAdapter)
