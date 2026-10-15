"""Naive-policy range calibration for CEM cost normalization.

The :class:`critic_sharing.planning.costs.JointUtilityCost` accepts
``leader_range=(lo, hi)`` and ``follower_range=(lo, hi)`` to map the
leader's discounted return and the follower's predicted V into a common
``[0, 1]`` band before combining them. Without normalization the two
sides live on wildly different scales and one dominates the other.

The numbers ``(lo, hi)`` come from a calibration pass: roll the **naive**
policy in the env's natural reset distribution for ``n_episodes``
episodes, filter to the **successful** ones (``info[env.SUCCESS_INFO_KEY]``
ever-True during the episode), and take ``(min, max)`` of the discounted
returns.

Calibrating from natural env resets — not from the leader-induced initial
states the follower would face during a joint rollout — is deliberate.
The follower's V is trained on the natural reset distribution; we want
the range it operates over, not the constrained subset the leader's
policy happens to produce.

Each env that participates in CEM normalization must expose a class
attribute :attr:`SUCCESS_INFO_KEY` naming the per-step ``info`` boolean
that flips True on success. Composite success conditions (e.g. warehouse
leader = ``dropped AND gap_index >= 0``) are composed inside the env's
``step()`` method so this helper stays generic — it only reads one key.

Zero-success calibrations are a hard error: no successful naive episode
means the calibration data is degenerate, the resulting range is
meaningless, and we'd silently inject garbage into the CEM cost. The
caller's checkpoint is the most likely culprit; surfacing it loudly is
the right behavior.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


class CalibrationError(RuntimeError):
    """Raised by :func:`calibrate_range` when the calibration pass produced
    no successful episodes (or the env doesn't expose ``SUCCESS_INFO_KEY``).

    Carries enough context that the user can decide whether to bump
    ``n_episodes``, swap the checkpoint, or disable normalization via
    ``cost.yaml``'s ``normalize_returns: false``.
    """


def calibrate_range(
    env,
    model,
    n_episodes: int = 30,
    gamma: float = 0.99,
    *,
    seed_offset: int = 10_000,
    deterministic: bool = True,
    success_only: bool = True,
) -> Tuple[float, float]:
    """Roll the naive policy and return ``(min, max)`` of returns.

    Two semantics, switched by ``success_only``:

    * ``True`` (default, legacy): only successful episodes contribute.
      The resulting band is "what naive achieves when it succeeds" —
      narrow and highly discriminative for above-naive plans, but
      failed CEM plans normalize to large negative numbers (with
      ``clip_normalized=False``). Asymmetric reward shaping
      (``goal_reward >> step_penalty``) accentuates this asymmetry.

    * ``False``: every episode contributes. ``lo`` ≈ "naive's worst
      failure return", ``hi`` ≈ "naive's best success return". This
      is the natural ``[0, 1]`` mapping for the joint utility cost:
      failed plans land near 0, successful ones near 1, gradient is
      smooth, total-return variance shrinks.

    Args:
        env: a gym env instance with class attribute :attr:`SUCCESS_INFO_KEY`
            (a string naming the per-step ``info`` boolean that latches
            True on success). Reset and stepped repeatedly here.
            Required even when ``success_only=False`` so the helper can
            still report success-rate context in error messages.
        model: an SB3-style policy with ``predict(obs, deterministic) ->
            (action, _)``.
        n_episodes: number of naive rollouts. With ``success_only=True``
            and a ~50% success rate, expect ~n/2 useful samples; bump
            to 60–100 to tighten the bounds. With ``success_only=False``
            every episode is a sample.
        gamma: discount factor used when computing per-episode discounted
            return. Should match the policy's training gamma so the
            calibration return is on the same scale as the cost's
            ``r_leader`` / V_follower output.
        seed_offset: integer offset added to the calibration episode index
            to produce ``env.reset(seed=...)``. Picked deterministically
            from the index so calibration is reproducible. Distinct from
            normal eval seeds (default 10_000) to avoid coupling.
        deterministic: passed through to ``model.predict``. Almost always
            True for eval-time naive baselines.
        success_only: see semantics above. Default ``True`` for legacy
            behavior; controlled at the call-site via cost.yaml's
            ``calibrate_on_success_only``.

    Returns:
        ``(lo, hi)`` — the min and max of the discounted returns over
        the chosen subset.

    Raises:
        CalibrationError: if ``env`` doesn't expose ``SUCCESS_INFO_KEY``,
            or if ``success_only=True`` and the rollouts produced zero
            successful episodes.
    """
    success_key = getattr(env, "SUCCESS_INFO_KEY", None)
    if not success_key:
        raise CalibrationError(
            "env {cls} does not expose SUCCESS_INFO_KEY — calibration cannot "
            "filter to successful episodes. Add a class attribute naming the "
            "per-step info key that flips True on success.".format(
                cls=type(env).__name__,
            )
        )

    all_returns: list[float] = []
    successful_returns: list[float] = []
    n_total = int(n_episodes)
    for ep in range(n_total):
        obs, _ = env.reset(seed=int(seed_offset + ep))
        success = False
        ret = 0.0
        g = 1.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, terminated, truncated, info = env.step(action)
            ret += g * float(r)
            g *= float(gamma)
            if bool(info.get(success_key, False)):
                success = True
            done = bool(terminated) or bool(truncated)
        all_returns.append(float(ret))
        if success:
            successful_returns.append(float(ret))

    sample_pool = successful_returns if success_only else all_returns

    if not sample_pool:
        raise CalibrationError(
            "calibrate_range({cls}): 0/{n} naive episodes succeeded "
            "(success_key={k!r}, success_only={so}). The naive policy "
            "produced no usable samples for the chosen calibration mode. "
            "Likely causes: wrong checkpoint, env regression, or "
            "under-trained policy. Workarounds: bump n_episodes, swap "
            "the checkpoint, set calibrate_on_success_only=false to "
            "include failures, or set normalize_returns=false in "
            "cost.yaml.".format(
                cls=type(env).__name__,
                n=n_total,
                k=success_key,
                so=success_only,
            )
        )

    arr = np.asarray(sample_pool, dtype=np.float32)
    return float(arr.min()), float(arr.max())


def calibrate_horizon_window_range(
    env,
    model,
    horizon: int,
    n_episodes: int = 30,
    gamma: float = 0.99,
    *,
    seed_offset: int = 10_000,
    deterministic: bool = True,
) -> Tuple[float, float]:
    """Roll naive policy and return ``(min, max)`` of horizon-H window
    discounted-sums over successful trajectories.

    Each successful episode of length ``T`` contributes ``T - H + 1``
    rolling windows; each window's value is
    ``sum_{j=0..H-1} γ^j · r_{i+j}`` for window starting at step ``i``.
    Together they form the empirical distribution of "horizon-H
    discounted sums on successful naive trajectories".

    This is the right scale to calibrate against when the cost (e.g.
    :class:`JointUtilityCost`) only sees the partial-horizon CEM rollout
    (no prior extension to episode end). Successful trajectories shorter
    than ``horizon`` contribute their single full-trajectory discounted
    return, treated as a window of effective length ``T < H``.

    Args:
        env: gym env with ``SUCCESS_INFO_KEY`` class attribute.
        model: SB3-style policy.
        horizon: CEM horizon ``H`` to match the cost's partial-rollout
            scale. Read from ``planner.yaml``'s ``horizon`` field.
        n_episodes / gamma / seed_offset / deterministic: see
            :func:`calibrate_range`.

    Returns:
        ``(lo, hi)`` over the pool of windows from all successful episodes.

    Raises:
        CalibrationError: zero successful episodes (same contract as
        :func:`calibrate_range`).
    """
    success_key = getattr(env, "SUCCESS_INFO_KEY", None)
    if not success_key:
        raise CalibrationError(
            "env {cls} does not expose SUCCESS_INFO_KEY".format(
                cls=type(env).__name__,
            )
        )

    H = int(horizon)
    if H < 1:
        raise ValueError("horizon must be >= 1; got {h}".format(h=horizon))
    g = float(gamma)
    # Pre-compute γ^[0..H-1] once.
    horizon_gammas = np.power(g, np.arange(H, dtype=np.float32))

    successful_count = 0
    all_windows: list[float] = []

    for ep in range(int(n_episodes)):
        obs, _ = env.reset(seed=int(seed_offset + ep))
        success = False
        rewards: list[float] = []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, terminated, truncated, info = env.step(action)
            rewards.append(float(r))
            if bool(info.get(success_key, False)):
                success = True
            done = bool(terminated) or bool(truncated)
        if not success:
            continue
        successful_count += 1

        T = len(rewards)
        rew_arr = np.asarray(rewards, dtype=np.float32)
        if T < H:
            # Episode shorter than horizon — emit a single
            # discounted-sum-over-full-trajectory data point. (Cost will
            # truncate to T < H steps in this case anyway.)
            window_disc = float(
                (rew_arr * horizon_gammas[:T]).sum()
            )
            all_windows.append(window_disc)
            continue
        # Rolling H-step discounted sums via convolution. We compute
        # ``r_i * γ^j`` for window starting at i, summed over j∈[0,H).
        # Equivalently, slide a length-H γ^[0..H-1] kernel over rewards.
        # ``np.convolve(..., mode='valid')`` returns T-H+1 windows.
        # Note: convolve flips the kernel, so we pre-flip horizon_gammas.
        kernel = horizon_gammas[::-1]
        windows = np.convolve(rew_arr, kernel, mode="valid")
        all_windows.extend(windows.tolist())

    if successful_count == 0 or not all_windows:
        raise CalibrationError(
            "calibrate_horizon_window_range({cls}): 0/{n} naive episodes "
            "succeeded (success_key={k!r}). Same recovery options as "
            "calibrate_range — bump n_episodes, swap the checkpoint, or "
            "set normalize_returns=false in cost.yaml.".format(
                cls=type(env).__name__,
                n=int(n_episodes),
                k=success_key,
            )
        )

    arr = np.asarray(all_windows, dtype=np.float32)
    return float(arr.min()), float(arr.max())


__all__ = [
    "CalibrationError",
    "calibrate_horizon_window_range",
    "calibrate_range",
]
