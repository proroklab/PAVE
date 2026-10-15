"""Unit tests for :class:`evaluations._common.seeding.SeedManager`."""

from __future__ import annotations

import pytest

from evaluations._common.seeding import SeedManager


def test_named_purposes_stable():
    """Same master + same accessor = bit-identical seed."""
    a = SeedManager(master=42)
    b = SeedManager(master=42)
    assert a.training_seed() == b.training_seed()
    assert a.calibration_seed() == b.calibration_seed()
    assert a.planner_seed() == b.planner_seed()
    assert a.vae_seed() == b.vae_seed()
    assert a.rendering_seed() == b.rendering_seed()


def test_named_purposes_disjoint():
    """Different purposes return different seeds."""
    m = SeedManager(master=42)
    purposes = {
        m.training_seed(),
        m.calibration_seed(),
        m.planner_seed(),
        m.vae_seed(),
        m.rendering_seed(),
    }
    assert len(purposes) == 5, "named purposes collided"


def test_different_masters_differ():
    """Different masters yield different seeds for the same purpose."""
    a = SeedManager(master=42)
    b = SeedManager(master=43)
    assert a.training_seed() != b.training_seed()
    assert a.eval_episode_seeds(10) != b.eval_episode_seeds(10)


def test_eval_episode_seeds_idempotent():
    """Calling ``eval_episode_seeds(n)`` twice returns identical lists."""
    m = SeedManager(master=42)
    out1 = m.eval_episode_seeds(7)
    out2 = m.eval_episode_seeds(7)
    assert out1 == out2


def test_eval_episode_seeds_disjoint_within_episode():
    """Within an episode, leader, follower, and planner seeds all differ."""
    m = SeedManager(master=42)
    ls, fs, ps = m.eval_episode_seeds(20)
    for l, f, p in zip(ls, fs, ps):
        assert len({l, f, p}) == 3


def test_eval_episode_seeds_no_intra_collisions():
    """Across episodes, every (leader, follower, planner) seed is distinct."""
    m = SeedManager(master=42)
    ls, fs, ps = m.eval_episode_seeds(100)
    all_seeds = ls + fs + ps
    assert len(set(all_seeds)) == len(all_seeds)


def test_eval_episode_seeds_disjoint_from_named():
    """Episode seeds (all three streams) don't collide with the named-purpose seeds."""
    m = SeedManager(master=42)
    named = {
        m.training_seed(), m.calibration_seed(), m.planner_seed(),
        m.vae_seed(), m.rendering_seed(),
    }
    ls, fs, ps = m.eval_episode_seeds(100)
    assert not (named & set(ls + fs + ps))


def test_eval_episode_seeds_returns_triple():
    """The new contract is a 3-tuple."""
    m = SeedManager(master=42)
    out = m.eval_episode_seeds(5)
    assert isinstance(out, tuple) and len(out) == 3
    leader, follower, planner = out
    assert len(leader) == len(follower) == len(planner) == 5


def test_cell_seed_stable():
    """Same master + same labels → same seed."""
    a = SeedManager(master=42)
    b = SeedManager(master=42)
    assert a.cell_seed("warehouse", "unguided", "0") == b.cell_seed("warehouse", "unguided", "0")


def test_cell_seed_order_sensitive():
    """Order of labels matters."""
    m = SeedManager(master=42)
    assert m.cell_seed("a", "b") != m.cell_seed("b", "a")


def test_cell_seed_label_sensitive():
    """Different labels → different seeds."""
    m = SeedManager(master=42)
    s1 = m.cell_seed("warehouse", "unguided", "0")
    s2 = m.cell_seed("warehouse", "value-only", "0")
    s3 = m.cell_seed("warehouse", "unguided", "1")
    assert len({s1, s2, s3}) == 3


def test_cell_seed_disjoint_from_named():
    """Cell seeds don't accidentally collide with named-purpose seeds."""
    m = SeedManager(master=42)
    named = {m.training_seed(), m.calibration_seed(), m.planner_seed(), m.vae_seed()}
    cell_samples = {m.cell_seed("scn", "mode", str(i)) for i in range(50)}
    assert not (named & cell_samples)


def test_cell_seed_requires_label():
    m = SeedManager(master=42)
    with pytest.raises(ValueError):
        m.cell_seed()


def test_unknown_purpose_raises():
    """Internal ``_purpose_root`` rejects unknown purpose names."""
    m = SeedManager(master=42)
    with pytest.raises(ValueError):
        m._purpose_root("not-a-purpose")
