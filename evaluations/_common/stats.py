"""Statistical primitives for evaluation aggregation.

Three independent groups — none of these functions call each other,
so the call tree is one level deep. Built so an ``analyze.py`` can
read top-to-bottom and never recurse.

  * :func:`paired_wilcoxon`   — paired non-parametric significance
  * :func:`sig_marker`        — p-value → asterisk string
  * :func:`mean_std`          — mean and std with NaN-safe defaults
  * :func:`percent_improvement` — relative delta vs baseline
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    """Return ``(mean, std)``. Empty input → ``(nan, nan)``."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=0))


def paired_wilcoxon(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the two-sided paired-Wilcoxon p-value of ``a − b``.

    Uses ``scipy.stats.wilcoxon`` with ``zero_method="wilcox"`` (drop
    zero-differences, the classical recipe). Empty inputs or all-zero
    differences return ``nan``.

    The signed paired test is appropriate when ``a`` and ``b`` are the
    per-episode metric for the SAME (episode, scenario) cell under two
    different modes — the eval framework's "shared seeds across modes"
    guarantee means episode i in mode A is the same physical
    initial-condition as episode i in mode B.
    """
    from scipy.stats import wilcoxon

    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if a_arr.size == 0 or a_arr.shape != b_arr.shape:
        return float("nan")
    diffs = a_arr - b_arr
    if np.allclose(diffs, 0.0):
        return float("nan")
    try:
        result = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
        return float(result.pvalue)
    except ValueError:
        return float("nan")


def sig_marker(p_value: float) -> str:
    """Return the conventional asterisk marker for ``p_value``.

    Thresholds: ``p < 0.001`` → ``"***"``, ``0.001 ≤ p < 0.01`` → ``"**"``,
    ``0.01 ≤ p < 0.05`` → ``"*"``, otherwise empty string.
    Returns empty string for ``nan``.
    """
    if not np.isfinite(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def percent_improvement(treated: float, baseline: float) -> float:
    """Return ``(treated - baseline) / |baseline|`` as a percentage.

    Returns ``nan`` if ``baseline`` is zero (relative comparison is
    ill-defined). Negative baselines produce a sign-flipped result
    (i.e. "improvement" is always in the direction of larger numbers).
    """
    if baseline == 0.0 or not np.isfinite(baseline):
        return float("nan")
    return 100.0 * (treated - baseline) / abs(baseline)


__all__ = [
    "mean_std",
    "paired_wilcoxon",
    "percent_improvement",
    "sig_marker",
]
