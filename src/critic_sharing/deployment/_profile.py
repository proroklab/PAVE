"""Lightweight per-phase profiler for the deployment control loop.

Usage::

    prof = PhaseTimer()
    with prof.measure("spin"):
        rclpy.spin_once(node)
    with prof.measure("redraw"):
        _redraw()
    ...
    print(prof.report())

The profiler accumulates wall-clock samples per named phase. Overhead is
~1 µs per ``measure``, negligible at 10 Hz. Disabled when ``enabled=False``
— ``measure`` becomes a no-op context manager.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, List


class PhaseTimer:
    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled)
        self._samples: Dict[str, List[float]] = {}

    @contextmanager
    def measure(self, phase: str):
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self._samples.setdefault(phase, []).append(dt_ms)

    def record(self, phase: str, dt_ms: float) -> None:
        if not self.enabled:
            return
        self._samples.setdefault(phase, []).append(float(dt_ms))

    def report(self, target_period_ms: float | None = None) -> str:
        if not self._samples:
            return "(profiler: no samples)"
        lines = ["", "=" * 70,
                 f"{'phase':<22} {'n':>5} {'mean':>8} {'p50':>8} {'p90':>8} {'p99':>8} {'max':>8}",
                 "-" * 70]
        for phase, samples in sorted(self._samples.items()):
            samples_sorted = sorted(samples)
            n = len(samples_sorted)
            mean = sum(samples_sorted) / n
            p50 = samples_sorted[int(n * 0.50)]
            p90 = samples_sorted[min(int(n * 0.90), n - 1)]
            p99 = samples_sorted[min(int(n * 0.99), n - 1)]
            maxv = samples_sorted[-1]
            lines.append(
                f"{phase:<22} {n:>5d} {mean:>7.1f}ms {p50:>7.1f}ms "
                f"{p90:>7.1f}ms {p99:>7.1f}ms {maxv:>7.1f}ms"
            )
        if target_period_ms is not None:
            lines.append("-" * 70)
            lines.append(f"target tick period: {target_period_ms:.1f}ms")
        lines.append("=" * 70)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        out = {}
        for phase, samples in self._samples.items():
            if not samples:
                continue
            samples_sorted = sorted(samples)
            n = len(samples_sorted)
            out[phase] = {
                "n": n,
                "mean_ms": sum(samples_sorted) / n,
                "p50_ms": samples_sorted[int(n * 0.50)],
                "p90_ms": samples_sorted[min(int(n * 0.90), n - 1)],
                "p99_ms": samples_sorted[min(int(n * 0.99), n - 1)],
                "max_ms": samples_sorted[-1],
            }
        return out


__all__ = ["PhaseTimer"]
