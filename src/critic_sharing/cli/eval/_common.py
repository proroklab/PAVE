"""Shared enums, constants, and helpers for the eval package."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import typer

from critic_sharing.cli._common import resolve_model  # noqa: F401 (re-exported)


def _resolve_vae(explicit: Optional[Path], follower_model: Path, scenario: str) -> Optional[Path]:
    """Return OOD-detector checkpoint path: explicit > auto-discovered.

    Auto-discovery searches in order:

    1. ``<follower_model_dir>/follower_ood.pt`` — generic name written
       by ``critic-sharing train ood`` (current default).
    2. ``<follower_model_dir>/follower_vae.pt`` — legacy generic name
       written by the old ``critic-sharing train vae`` (still
       auto-loaded for back-compat with existing artefacts on disk /
       on Drive).
    3. ``<follower_model_dir>/nav_env_vae.pt`` — even older
       obstacle_arena-inline-trained name from before ``train vae``
       existed (kept for the published-paper checkpoints).

    The function name remains ``_resolve_vae`` so callers don't have
    to be touched; the *resolved* file is consumed by
    ``build_*_ood_metric``, which inspects the kNN payload first and
    only falls back to the VAE branch when the kNN payload is absent.
    """
    if explicit is not None:
        return explicit
    candidates = [
        follower_model.parent / "follower_ood.pt",
        follower_model.parent / "follower_vae.pt",
    ]
    if scenario == "obstacle_arena":
        candidates.append(follower_model.parent / "nav_env_vae.pt")
    for path in candidates:
        if path.exists():
            tag = "OOD checkpoint" if path.name == "follower_ood.pt" else "OOD checkpoint (legacy)"
            typer.echo(f"Auto-discovered {tag}: {path}")
            return path
    return None


class Scenario(str, Enum):
    obstacle_arena = "obstacle_arena"
    warehouse = "warehouse"
    # Warehouse variant: pickup auto-skipped at one gap (geometry
    # unchanged) at training AND deployment.
    warehouse_pickup_blocked = "warehouse_pickup_blocked"
    # Same training as warehouse_pickup_blocked but deployment enables
    # pickup at the disabled gap.
    warehouse_pickup_unseen = "warehouse_pickup_unseen"
    mobile_arm = "mobile_arm"
    household = "household"


class EvalMode(str, Enum):
    unguided = "unguided"  # π_L only — no adaptation
    # joint utility with follower critic; V_F at full weight, NO OOD trust
    # factor regardless of VAE availability. r_F = α · V_F_norm.
    value_only = "value-only"
    # joint utility with follower critic + OOD trust factor.
    # r_F = α · (1 − OOD) · V_F_norm. Requires a VAE.
    value_and_ood = "value-and-ood"
    # value_and_ood ablation that strips the leader's policy contribution
    # from the planner: CEM is seeded with a state-independent wide
    # Gaussian (mean=0, σ=½ action range), and prior-extension rollouts
    # use the same Gaussian. Tests "is V_F+OOD doing the work, or is the
    # leader's prior?".
    uniform = "uniform"
    sabotage = "sabotage"  # maximise leader, minimise follower


# Display helpers
_MODES_ORDER = [
    "unguided",
    "value-only",
    "value-and-ood",
    "uniform",
    "sabotage",
]
_MODE_COLORS = {
    "unguided": "#78909C",
    "value-only": "#1565C0",
    "value-and-ood": "#2E7D32",
    "uniform": "#EF6C00",
    "sabotage": "#C62828",
}
_MODE_LABELS = {
    "unguided": "Naive",
    "value-only": r"$V_F$",
    "value-and-ood": "PAVE",
    "uniform": "Uniform",
    "sabotage": "Sabotage",
}


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def _print_summary_table(results: list[dict], scenario: Scenario) -> None:
    # Per-scenario row split. warehouse-only for now: each mode row gets
    # split into "(drop)" and "(no-drop)" sub-rows along gap_idx >= 0.
    # ``split_predicate`` returns the drop_success bool; ``split_label``
    # is the human-readable suffix.
    split_predicate = None
    if scenario in (
        Scenario.warehouse,
        Scenario.warehouse_pickup_blocked,
        Scenario.warehouse_pickup_unseen,
    ):
        split_predicate = lambda r: int(r.get("gap_idx", -1)) >= 0
        split_label = lambda b: "(drop)" if b else "(no-drop)"

    def _build_groups():
        modes = sorted(
            set(r["mode"] for r in results),
            key=lambda m: _MODES_ORDER.index(m) if m in _MODES_ORDER else 99,
        )
        base = [(m, None) for m in modes]
        base_label = lambda g: g[0]
        base_filter = lambda g: lambda r: r["mode"] == g[0]

        if split_predicate is None:
            return base, base_label, base_filter

        # Expand each base group into [base × {True, False}]. drop=True
        # rows render first per mode, no-drop second.
        expanded: list[tuple] = []
        for g in base:
            expanded.append((g, True))
            expanded.append((g, False))

        def expanded_label(g: tuple) -> str:
            base_g, success = g
            return "{m} {s}".format(m=base_label(base_g), s=split_label(success))

        def expanded_filter(g: tuple):
            base_g, success = g
            base_f = base_filter(base_g)
            return lambda r: base_f(r) and (split_predicate(r) == success)

        return expanded, expanded_label, expanded_filter

    groups, group_label, row_filter = _build_groups()

    # Pick column source. For warehouse and mobile_arm (the scenarios
    # whose ``_compare_*`` precomputes ``leader_return_norm`` /
    # ``follower_return_norm`` from the calibration ranges), display the
    # normalized values — directly comparable across episodes and matched
    # to the CEM cost's own [0, 1] mapping. Falls back to absolute
    # returns when no normalization is available (e.g.
    # cost_config.normalize_returns=False, or only unguided was
    # requested so calibration didn't run).
    has_norm = (
        scenario in {
            Scenario.warehouse,
            Scenario.warehouse_pickup_blocked,
            Scenario.warehouse_pickup_unseen,
            Scenario.mobile_arm,
        }
        and any(("leader_return_norm" in r) for r in results)
    )
    if has_norm:
        # Detect whether normalization is unclipped: a value strictly
        # outside [0, 1] on any row is the tell. Marks normalized columns
        # with a trailing asterisk + footnote so the researcher knows the
        # numbers can extrapolate beyond the calibration envelope.
        unclipped = any(
            (
                ("leader_return_norm" in r and not (0.0 <= r["leader_return_norm"] <= 1.0))
                or (
                    "follower_return_norm" in r
                    and not (0.0 <= r["follower_return_norm"] <= 1.0)
                )
            )
            for r in results
        )
        norm_marker = "*" if unclipped else ""
        leader_key = "leader_return_norm"
        follower_key = "follower_return_norm"
        leader_label = "leader_ret_n" + norm_marker
        follower_label = "follower_ret_n" + norm_marker
        # Total: prefer ``total_return_weighted`` (the actual
        # CEM-minimized objective: L_norm + λ·trust·F_norm with the trust
        # factor read off the realized handoff state). Falls back to the
        # plain L_norm + F_norm when the per-scenario _compare_* didn't
        # precompute the weighted version. The label flags which form
        # is shown:
        #   total_ret_w  — weighted (planner objective)
        #   total_ret_n  — plain normalized sum
        if any(("total_return_weighted" in r) for r in results):
            total_key = "total_return_weighted"
            total_label = "total_ret_w" + norm_marker
        else:
            for r in results:
                if "leader_return_norm" in r and "follower_return_norm" in r:
                    r["total_return_norm"] = float(
                        r["leader_return_norm"] + r["follower_return_norm"]
                    )
            total_key = "total_return_norm"
            total_label = "total_ret_n" + norm_marker
    else:
        unclipped = False
        leader_key = "leader_return"
        follower_key = "follower_return"
        total_key = "total_return"
        leader_label = "leader_ret"
        follower_label = "follower_ret"
        total_label = "total_ret"

    if scenario == Scenario.obstacle_arena:
        columns = [
            (leader_key, leader_label),
            (follower_key, follower_label),
            (total_key, total_label),
            ("follower_steps", "f_steps"),
            ("follower_reached_goal", "goal_rate"),
        ]
    elif scenario in (
        Scenario.warehouse,
        Scenario.warehouse_pickup_blocked,
        Scenario.warehouse_pickup_unseen,
    ):
        columns = [
            (leader_key, leader_label),
            (follower_key, follower_label),
            (total_key, total_label),
            ("follower_steps", "f_steps"),
            ("follower_on_goal", "goal_rate"),
        ]
    elif scenario == Scenario.mobile_arm:
        columns = [
            (leader_key, leader_label),
            (follower_key, follower_label),
            (total_key, total_label),
            ("leader_stand_off_diff", "stand_off"),
            ("follower_dropped", "drop_rate"),
        ]
    elif scenario == Scenario.household:
        columns = [
            (leader_key, leader_label),
            (follower_key, follower_label),
            (total_key, total_label),
            ("follower_steps", "f_steps"),
            ("leader_picked_up_dirt", "pickup_rate"),
            ("follower_reached_goal", "goal_rate"),
        ]
    else:
        columns = [
            ("leader_return", "leader_ret"),
            ("follower_return", "follower_ret"),
            ("total_return", "total_ret"),
            ("leader_cells", "l_cells"),
            ("follower_cells", "f_cells"),
        ]

    # --- compute per-episode improvement over unguided baseline ---
    by_group = {g: [r for r in results if row_filter(g)(r)] for g in groups}
    # Baseline lookup uses the same return key as the column display, so
    # warehouse with normalized columns gets normalized improvements
    # (and the legacy absolute path stays bit-identical for non-warehouse).
    unguided_leader: dict[int, float] = {}
    unguided_follower: dict[int, float] = {}
    for r in results:
        if r["mode"] == EvalMode.unguided.value:
            unguided_leader[r["episode"]] = float(r.get(leader_key, r["leader_return"]))
            unguided_follower[r["episode"]] = float(
                r.get(follower_key, r["follower_return"])
            )

    has_baseline = bool(unguided_leader)
    improvement_cols = [
        ("l_impr_abs", "l_impr(abs)"),
        ("l_impr_rel", "l_impr(%)"),
        ("f_impr_abs", "f_impr(abs)"),
        ("f_impr_rel", "f_impr(%)"),
    ]

    # Helper: a row's "mode key" - used to detect the unguided baseline
    # row irrespective of whether ``g`` is a plain (mode, None) tuple or a
    # ((mode, None), drop_success) tuple after warehouse splitting.
    def _mode_of(g) -> str:
        if split_predicate is not None:
            return g[0][0]
        return g[0]

    col_w = 18
    # Wider mode column for warehouse split labels like
    # "value-only (no-drop)".
    mode_w = 24 if split_predicate is not None else 14
    all_columns = columns + (improvement_cols if has_baseline else [])
    header = "{:<{mw}}".format("mode", mw=mode_w) + "".join(
        "{:>{cw}}".format(h, cw=col_w) for _, h in all_columns
    )
    sep = "-" * len(header)
    typer.echo("\n" + sep)
    typer.echo(header)
    typer.echo(sep)
    for g in groups:
        subset = by_group[g]
        if not subset:
            continue
        row = "{:<{mw}}".format(group_label(g), mw=mode_w)
        for key, _ in columns:
            vals = [r[key] for r in subset]
            row += "{:>{cw}}".format(
                "{:.2f}\u00b1{:.2f}".format(float(np.mean(vals)), float(np.std(vals))),
                cw=col_w,
            )
        if has_baseline:
            if _mode_of(g) == EvalMode.unguided.value:
                for _ in improvement_cols:
                    row += "{:>{cw}}".format("(baseline)", cw=col_w)
            else:
                eps = 1e-8
                l_abs = np.array([
                    float(r.get(leader_key, r["leader_return"]))
                    - unguided_leader.get(r["episode"], 0.0)
                    for r in subset
                ])
                l_rel = np.array([
                    100.0 * (
                        float(r.get(leader_key, r["leader_return"]))
                        - unguided_leader.get(r["episode"], 0.0)
                    ) / (abs(unguided_leader.get(r["episode"], 0.0)) + eps)
                    for r in subset
                ])
                f_abs = np.array([
                    float(r.get(follower_key, r["follower_return"]))
                    - unguided_follower.get(r["episode"], 0.0)
                    for r in subset
                ])
                f_rel = np.array([
                    100.0 * (
                        float(r.get(follower_key, r["follower_return"]))
                        - unguided_follower.get(r["episode"], 0.0)
                    ) / (abs(unguided_follower.get(r["episode"], 0.0)) + eps)
                    for r in subset
                ])
                for arr in (l_abs, l_rel, f_abs, f_rel):
                    fmt = "{:+.2f}\u00b1{:.2f}".format(float(np.mean(arr)), float(np.std(arr)))
                    row += "{:>{cw}}".format(fmt, cw=col_w)
        typer.echo(row)
    typer.echo(sep)
    if unclipped:
        typer.echo(
            "* unclipped normalized return — values can extrapolate "
            "above 1 (better than calibration ceiling) or below 0 "
            "(worse than calibration floor); cost.yaml::clip_normalized=False"
        )
    typer.echo("")

    # --- Split by goal-reached (obstacle_arena only) ---
    if scenario == Scenario.obstacle_arena:
        modes_for_split = [g[0] for g in groups]
        _print_goal_split_table(results, modes_for_split, col_w, mode_w)


def _print_goal_split_table(
    results: list[dict],
    modes: list[str],
    col_w: int,
    mode_w: int,
) -> None:
    """Print follower stats split by goal-reached vs not-reached."""
    split_cols = [
        ("follower_return", "follower_ret"),
        ("follower_steps", "f_steps"),
    ]
    for label, filter_fn in [
        ("GOAL REACHED", lambda r: r.get("follower_reached_goal", 0) > 0.5),
        ("GOAL NOT REACHED", lambda r: r.get("follower_reached_goal", 0) <= 0.5),
    ]:
        count_col = "{:>{cw}}".format("count", cw=col_w)
        header = (
            "{:<{mw}}".format("mode", mw=mode_w)
            + count_col
            + "".join("{:>{cw}}".format(h, cw=col_w) for _, h in split_cols)
        )
        sep = "-" * len(header)
        typer.echo("  {label}".format(label=label))
        typer.echo("  " + sep)
        typer.echo("  " + header)
        typer.echo("  " + sep)
        for m in modes:
            subset = [r for r in results if r["mode"] == m and filter_fn(r)]
            n = len(subset)
            row = "{:<{mw}}".format(m, mw=mode_w)
            row += "{:>{cw}}".format(str(n), cw=col_w)
            if n > 0:
                for key, _ in split_cols:
                    vals = [r[key] for r in subset]
                    row += "{:>{cw}}".format(
                        "{:.2f}\u00b1{:.2f}".format(float(np.mean(vals)), float(np.std(vals))),
                        cw=col_w,
                    )
            else:
                for _ in split_cols:
                    row += "{:>{cw}}".format("n/a", cw=col_w)
            typer.echo("  " + row)
        typer.echo("  " + sep + "\n")
