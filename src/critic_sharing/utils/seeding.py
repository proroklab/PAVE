"""Single-source-of-randomness for the evaluation pipeline.

Every Python-callable entry point that consumes "a seed" should
ultimately get its seed from a :class:`SeedManager`, which builds a
deterministic, non-colliding tree of :class:`numpy.random.SeedSequence`
nodes from one master integer.

Why this exists
---------------
The previous pipeline mixed three independent sources of randomness:

* Training scripts: ``seed=42`` default per agent, hard-coded.
* ``eval``: ``rng = np.random.default_rng(seed); rng.integers(...)``
  produced ``episodes * 2`` *adjacent* integers, then handed them to
  ``env.reset(seed=...)``. Two evals with overlapping seed ranges saw
  bitwise-identical episodes.
* The CEM planner: separate ``seed`` keyword, also numerically close
  to the eval seed.

``SeedSequence.spawn(n)`` is the modern fix. Each call returns ``n``
sequences that are guaranteed never to collide with each other or with
any further ``.spawn()`` calls on the same tree. A
:class:`SeedManager` partitions the tree by *purpose* (training,
calibration, eval-episodes, planner) so the names match the code
that consumes them.

Usage
-----
::

    mgr = SeedManager(master=42)
    train_seed = mgr.training_seed()
    calib_seed = mgr.calibration_seed()
    leader_seeds, follower_seeds, planner_seeds = mgr.eval_episode_seeds(n=100)
    planner_seed = mgr.planner_seed()

The hierarchy is **stable**: same ``master`` → same outputs for every
call, regardless of order. Tested in ``tests/evaluations/test_seeding.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.random import SeedSequence


# Named purposes. The integer index is what disambiguates one named
# purpose's seed tree from another (via the SeedSequence ``spawn_key``
# parameter). Appending is fine; reordering breaks reproducibility.
_PURPOSES = (
    "training",
    "calibration",
    "eval_episodes",
    "planner",
    "vae",          # follower-VAE / kNN training (when ``--detector vae``).
    "rendering",    # MP4 / figure RNG.
)
_PURPOSE_INDEX = {name: i for i, name in enumerate(_PURPOSES)}


def _u32(ss: SeedSequence) -> int:
    """Materialise a 32-bit positive integer from a :class:`SeedSequence`."""
    return int(ss.generate_state(1, dtype=np.uint32)[0])


@dataclass
class SeedManager:
    """Single source of truth for every seed in an evaluation pipeline.

    ``master`` is the only knob exposed to the CLI/user. Every named
    accessor re-derives its sub-sequence deterministically from the
    master via the SeedSequence ``spawn_key`` — calling the same
    accessor twice on the same manager always returns the same value.
    """

    master: int

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _purpose_root(self, purpose: str) -> SeedSequence:
        """Return a fresh root SeedSequence keyed to ``purpose``.

        Same master + same purpose → bit-identical SeedSequence every
        call. Different purposes are guaranteed disjoint by
        ``_PURPOSE_INDEX`` (used as the SeedSequence ``spawn_key``).
        """
        if purpose not in _PURPOSE_INDEX:
            raise ValueError(f"Unknown purpose {purpose!r}")
        return SeedSequence(int(self.master), spawn_key=(_PURPOSE_INDEX[purpose],))

    # ------------------------------------------------------------------
    # Named-purpose accessors (one integer each).
    # ------------------------------------------------------------------

    def training_seed(self) -> int:
        return _u32(self._purpose_root("training"))

    def calibration_seed(self) -> int:
        return _u32(self._purpose_root("calibration"))

    def planner_seed(self) -> int:
        return _u32(self._purpose_root("planner"))

    def vae_seed(self) -> int:
        return _u32(self._purpose_root("vae"))

    def rendering_seed(self) -> int:
        return _u32(self._purpose_root("rendering"))

    # ------------------------------------------------------------------
    # Episode batches — one seed per (episode, role).
    # ------------------------------------------------------------------

    def eval_episode_seeds(
        self, n: int,
    ) -> Tuple[list[int], list[int], list[int]]:
        """Return ``(leader_seeds, follower_seeds, planner_seeds)`` of length ``n`` each.

        Leader, follower, and planner seeds are siblings within each
        episode's sub-sequence, so the same episode's three streams are
        decoupled from each other but the *triple* is reproducible from
        the master. The planner_seed is consumed by the per-episode
        CEMController so its sample RNG is independent of env-reset
        randomness and of parallel-dispatch order.
        """
        per_episode = self._purpose_root("eval_episodes").spawn(int(n))
        leader_seeds: list[int] = []
        follower_seeds: list[int] = []
        planner_seeds: list[int] = []
        for ep_ss in per_episode:
            leader_child, follower_child, planner_child = ep_ss.spawn(3)
            leader_seeds.append(_u32(leader_child))
            follower_seeds.append(_u32(follower_child))
            planner_seeds.append(_u32(planner_child))
        return leader_seeds, follower_seeds, planner_seeds

    # ------------------------------------------------------------------
    # Free-form sub-seeds (per-cell within a grid sweep).
    # ------------------------------------------------------------------

    def cell_seed(self, *labels: str) -> int:
        """Derive a stable seed for an experiment-grid cell.

        ``labels`` are deterministically hashed into the SeedSequence
        ``spawn_key``. Identical labels + master return the same seed;
        different labels never collide with each other or with the
        named-purpose children (the named purposes use small integer
        spawn_keys, the cell seeds use 32-bit hashes that won't collide
        with single-digit integers).
        """
        if not labels:
            raise ValueError("cell_seed needs at least one label")
        spawn_key = tuple(_label_to_uint(label) for label in labels)
        ss = SeedSequence(int(self.master), spawn_key=spawn_key)
        return _u32(ss)


def _label_to_uint(label: str) -> int:
    """Hash a string label into a non-negative 32-bit int (deterministic)."""

    import hashlib

    digest = hashlib.blake2b(label.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big", signed=False)


__all__ = ["SeedManager"]
