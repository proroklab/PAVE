"""Unit tests for :mod:`evaluations._common.stats`."""

from __future__ import annotations

import math

import numpy as np

from evaluations._common.stats import (
    mean_std,
    paired_wilcoxon,
    percent_improvement,
    sig_marker,
)


def test_mean_std_empty():
    m, s = mean_std([])
    assert math.isnan(m) and math.isnan(s)


def test_mean_std_basic():
    m, s = mean_std([1.0, 2.0, 3.0])
    assert m == 2.0
    assert s == np.std([1.0, 2.0, 3.0], ddof=0)


def test_paired_wilcoxon_significant():
    """``a`` systematically larger than ``b`` → p < 0.05 on enough samples."""
    rng = np.random.default_rng(0)
    a = rng.normal(loc=1.0, scale=0.5, size=30)
    b = rng.normal(loc=0.0, scale=0.5, size=30)
    p = paired_wilcoxon(a, b)
    assert p < 0.05


def test_paired_wilcoxon_zero_diff():
    """All-equal samples return NaN (no signal)."""
    p = paired_wilcoxon([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert math.isnan(p)


def test_paired_wilcoxon_shape_mismatch():
    p = paired_wilcoxon([1.0, 2.0], [1.0])
    assert math.isnan(p)


def test_sig_marker_thresholds():
    assert sig_marker(0.0001) == "***"
    assert sig_marker(0.005) == "**"
    assert sig_marker(0.04) == "*"
    assert sig_marker(0.5) == ""
    assert sig_marker(float("nan")) == ""


def test_percent_improvement_basic():
    assert percent_improvement(11.0, 10.0) == 10.0
    assert percent_improvement(9.0, 10.0) == -10.0


def test_percent_improvement_zero_baseline():
    assert math.isnan(percent_improvement(1.0, 0.0))


def test_percent_improvement_negative_baseline():
    # ``treated=-1, baseline=-2`` → ``(-1 - -2) / 2`` = +50% (improvement)
    assert percent_improvement(-1.0, -2.0) == 50.0
