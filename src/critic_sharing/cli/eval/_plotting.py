"""Comparison figures for the eval package."""

from __future__ import annotations

from pathlib import Path

import typer

from ._common import EvalMode, Scenario, _MODE_COLORS, _MODE_LABELS, _MODES_ORDER


def _resolve_return_keys(results: list[dict]) -> tuple[str, str, str, str, str]:
    """Pick the right (leader, follower, total) keys + axis labels.

    Returns a 5-tuple ``(leader_key, follower_key, total_key,
    axis_label, total_title)``.

    When per-row ``leader_return_norm``/``follower_return_norm`` exist
    (set by the per-scenario ``_compare_*`` post-merge when
    calibration ran), all return plots use the normalized columns —
    consistent with what the summary table shows.

    The total key prefers ``total_return_weighted`` (= the planner
    objective ``L_norm + λ·trust·F_norm`` set on rows when a VAE was
    wired in). Otherwise it computes ``total_return_norm = L_norm +
    F_norm`` inline on each row so the rest of the plotting code can
    just read the key. Falls back to absolute-return columns when
    normalization didn't run.
    """
    has_norm = any("leader_return_norm" in r for r in results)
    if not has_norm:
        return (
            "leader_return",
            "follower_return",
            "total_return",
            "return",
            "Total return (leader + follower)",
        )

    if any("total_return_weighted" in r for r in results):
        total_key = "total_return_weighted"
        total_title = "Total — planner objective (L_norm + λ·trust·F_norm)"
    else:
        for r in results:
            if "leader_return_norm" in r and "follower_return_norm" in r:
                r.setdefault(
                    "total_return_norm",
                    float(r["leader_return_norm"] + r["follower_return_norm"]),
                )
        total_key = "total_return_norm"
        total_title = "Total — L_norm + F_norm"
    return (
        "leader_return_norm",
        "follower_return_norm",
        total_key,
        "normalized return",
        total_title,
    )


def _plot_compare(results: list[dict], scenario: Scenario, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    modes  = sorted(
        set(r["mode"] for r in results),
        key=lambda m: _MODES_ORDER.index(m) if m in _MODES_ORDER else 99,
    )
    by_mode = {m: [r for r in results if r["mode"] == m] for m in modes}
    n_ep    = len(next(iter(by_mode.values())))

    def _vals(mode: str, key: str) -> list:
        return [r[key] for r in by_mode[mode]]

    def _violinplot(ax, key: str, ylabel: str, title: str) -> None:
        import numpy as np

        data = [_vals(m, key) for m in modes]

        # Violin plots need non-zero variance to fit a KDE. For all-equal
        # groups (e.g. box_at_goal rate == 1.0 everywhere) we fall back to a
        # boxplot for that group — violin would raise LinAlgError.
        def _nontrivial(values: list[float]) -> bool:
            if len(values) < 2:
                return False
            arr = np.asarray(values, dtype=float)
            return float(np.std(arr)) > 1e-9

        positions = np.arange(1, len(modes) + 1)

        for pos, m, values in zip(positions, modes, data):
            color = _MODE_COLORS.get(m, "#888888")
            if _nontrivial(values):
                parts = ax.violinplot(
                    [values],
                    positions=[pos],
                    widths=0.7,
                    showmedians=True,
                    showextrema=True,
                )
                for body in parts["bodies"]:
                    body.set_facecolor(color)
                    body.set_edgecolor("#333333")
                    body.set_alpha(0.75)
                for part_key in ("cmedians", "cbars", "cmins", "cmaxes"):
                    if part_key in parts:
                        parts[part_key].set_color("#333333")
            else:
                # Degenerate group — draw a dot at the constant value.
                const_val = float(values[0]) if values else 0.0
                ax.scatter(
                    [pos],
                    [const_val],
                    color=color,
                    edgecolors="#333333",
                    s=60,
                    zorder=3,
                )

        ax.set_xticks(positions)
        ax.set_xticklabels([_MODE_LABELS.get(m, m) for m in modes], fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    subtitle = "{s}  —  {n} ep / mode".format(s=scenario.value, n=n_ep)

    # Pick normalized return keys when calibration produced them (warehouse,
    # mobile_arm); fall back to absolute returns for other scenarios.
    leader_key, follower_key, total_key, return_label, total_title = (
        _resolve_return_keys(results)
    )

    # --- Figure 1: returns ---
    fig1, axes1 = plt.subplots(1, 3, figsize=(13, 4))
    fig1.suptitle("Returns  |  " + subtitle, fontsize=11)
    _violinplot(axes1[0], leader_key,   return_label, "Leader return")
    _violinplot(axes1[1], follower_key, return_label, "Follower return")
    _violinplot(axes1[2], total_key,    return_label, total_title)
    plt.tight_layout()
    out1 = output_dir / "compare_{s}_returns.png".format(s=scenario.value)
    fig1.savefig(str(out1), dpi=150)
    plt.close(fig1)
    typer.echo("Returns figure  → {p}".format(p=out1))

    # --- Figure 2: episode-level metrics ---
    if scenario == Scenario.obstacle_arena:
        metric_triples = [
            ("leader_steps",         "steps", "Leader steps"),
            ("follower_steps",        "steps", "Follower steps"),
            ("follower_reached_goal", "rate",  "Follower goal-reach rate"),
        ]
    elif scenario in (
        Scenario.warehouse,
        Scenario.warehouse_pickup_blocked,
        Scenario.warehouse_pickup_unseen,
    ):
        metric_triples = [
            ("leader_steps",        "steps", "Leader steps"),
            ("follower_steps",      "steps", "Follower steps"),
            ("follower_on_goal",    "rate",  "Follower goal-reach rate"),
        ]
    elif scenario == Scenario.mobile_arm:
        metric_triples = [
            ("leader_stand_off_diff", "m",    "Leader stand-off |d - d*|"),
            ("follower_steps",        "steps","Follower steps"),
            ("follower_dropped",      "rate", "Tray-drop rate"),
        ]
    else:  # Scenario.household
        metric_triples = [
            ("leader_steps",          "steps", "Cleaner steps"),
            ("follower_steps",        "steps", "Follower steps"),
            ("leader_picked_up_dirt", "rate",  "Cleaner pickup rate"),
        ]

    fig2, axes2 = plt.subplots(1, 3, figsize=(13, 4))
    fig2.suptitle("Episode metrics  |  " + subtitle, fontsize=11)
    for ax, (key, ylabel, title) in zip(axes2, metric_triples):
        _violinplot(ax, key, ylabel, title)
    plt.tight_layout()
    out2 = output_dir / "compare_{s}_metrics.png".format(s=scenario.value)
    fig2.savefig(str(out2), dpi=150)
    plt.close(fig2)
    typer.echo("Metrics figure  → {p}".format(p=out2))

    # --- Figure 3a / 3b: relative improvement over unguided ---
    # Scenario-agnostic: one PNG per role (leader / follower) with one violin
    # per guided mode. Per-episode relative improvement is matched on
    # ``episode`` against the unguided baseline.
    guided_modes = [m for m in modes if m != EvalMode.unguided.value]
    if guided_modes and EvalMode.unguided.value in by_mode:
        import numpy as np

        # Use the same return-key resolution as the headline returns
        # figure — improvement is "(adapted_norm - baseline_norm) /
        # |baseline_norm|" when normalization is on, else absolute
        # returns. Keeps both figures interpreting the same numbers.
        unguided_leader_by_ep: dict[int, float] = {
            r["episode"]: float(r[leader_key])
            for r in by_mode[EvalMode.unguided.value]
        }
        unguided_follower_by_ep: dict[int, float] = {
            r["episode"]: float(r[follower_key])
            for r in by_mode[EvalMode.unguided.value]
        }
        eps = 1e-8

        def _plot_improvement(
            role: str,
            role_key: str,
            unguided_by_ep: dict[int, float],
            return_key: str,
        ) -> None:
            mode_rel: dict[str, np.ndarray] = {}
            for m in guided_modes:
                rel_vals: list[float] = []
                for r in by_mode[m]:
                    base = unguided_by_ep.get(r["episode"])
                    if base is None:
                        # Episode missing from baseline (e.g. seed mismatch)
                        # — skip rather than KeyError.
                        continue
                    rel_vals.append((r[return_key] - base) / (abs(base) + eps))
                mode_rel[m] = np.array(rel_vals)

            fig, ax = plt.subplots(figsize=(7, 5))
            fig.suptitle(
                "{role} improvement over unguided  |  {sub}".format(
                    role=role, sub=subtitle,
                ),
                fontsize=11,
            )

            positions = np.arange(1, len(guided_modes) + 1)
            for pos, m in zip(positions, guided_modes):
                rel = mode_rel[m]
                color = _MODE_COLORS.get(m, "#888888")
                if len(rel) >= 2 and float(np.std(rel)) > 1e-9:
                    parts = ax.violinplot(
                        [rel],
                        positions=[pos],
                        widths=0.7,
                        showmedians=True,
                        showextrema=True,
                    )
                    for body in parts["bodies"]:
                        body.set_facecolor(color)
                        body.set_edgecolor("#333333")
                        body.set_alpha(0.75)
                    for part_key in ("cmedians", "cbars", "cmins", "cmaxes"):
                        if part_key in parts:
                            parts[part_key].set_color("#333333")
                else:
                    const_val = float(rel[0]) if len(rel) else 0.0
                    ax.scatter(
                        [pos],
                        [const_val],
                        color=color,
                        edgecolors="#333333",
                        s=60,
                        zorder=3,
                    )

            ax.axhline(0.0, color="#333333", linewidth=1.2, linestyle="--")

            labels = []
            for m in guided_modes:
                rel = mode_rel[m]
                median = float(np.median(rel)) if len(rel) else 0.0
                labels.append(
                    "{l}\n(median {v:+.2f})".format(
                        l=_MODE_LABELS.get(m, m), v=median
                    )
                )
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_ylabel(
                "Relative {role} return improvement".format(role=role.lower()),
                fontsize=9,
            )
            ax.grid(axis="y", linestyle="--", alpha=0.4)

            plt.tight_layout()
            out_path = output_dir / "compare_{s}_{r}_improvement.png".format(
                s=scenario.value, r=role_key,
            )
            fig.savefig(str(out_path), dpi=150)
            plt.close(fig)
            typer.echo("{role} improvement figure → {p}".format(
                role=role, p=out_path,
            ))

        _plot_improvement("Leader", "leader", unguided_leader_by_ep, leader_key)
        _plot_improvement("Follower", "follower", unguided_follower_by_ep, follower_key)

    # --- Figure 5: value estimate vs actual follower return ---
    has_value_diag = any(
        r.get("follower_value_estimate") is not None for r in results
    )
    if has_value_diag:
        import numpy as np

        fig5, ax5 = plt.subplots(figsize=(7, 7))
        fig5.suptitle(
            "Value estimate vs actual follower return  |  " + subtitle,
            fontsize=11,
        )

        for m in modes:
            rows = by_mode[m]
            v_est = [r["follower_value_estimate"] for r in rows if r.get("follower_value_estimate") is not None]
            v_act = [r["follower_actual_return"] for r in rows if r.get("follower_value_estimate") is not None]
            if not v_est:
                continue
            v_est_arr = np.array(v_est, dtype=float)
            v_act_arr = np.array(v_act, dtype=float)
            color = _MODE_COLORS.get(m, "#888888")
            label = _MODE_LABELS.get(m, m)
            ax5.scatter(
                v_est_arr, v_act_arr,
                c=color, label=label, alpha=0.6, edgecolors="white", linewidths=0.5, s=40,
            )

        all_ve = [r["follower_value_estimate"] for r in results if r.get("follower_value_estimate") is not None]
        all_va = [r["follower_actual_return"] for r in results if r.get("follower_value_estimate") is not None]
        if all_ve and all_va:
            lo = min(min(all_ve), min(all_va))
            hi = max(max(all_ve), max(all_va))
            margin = (hi - lo) * 0.1 if hi > lo else 1.0
            ax5.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                     "k--", alpha=0.3, linewidth=1, label="perfect")
            ax5.set_xlim(lo - margin, hi + margin)
            ax5.set_ylim(lo - margin, hi + margin)
        ax5.set_xlabel("V(s) estimate (critic)", fontsize=10)
        ax5.set_ylabel("Actual follower return", fontsize=10)
        ax5.set_aspect("equal")
        ax5.legend(fontsize=8)
        ax5.grid(linestyle="--", alpha=0.4)
        plt.tight_layout()
        out5 = output_dir / "compare_{s}_value_diagnostic.png".format(s=scenario.value)
        fig5.savefig(str(out5), dpi=150)
        plt.close(fig5)
        typer.echo("Value diagnostic → {p}".format(p=out5))


