"""Generate one LaTeX **reward-parameter** table per (scenario, role).

For each (scenario, role) pair, resolves the env config and dumps only the
fields that appear in the reward computation of the corresponding env
(``self.config.<field>`` references on any ``reward = ... `` / ``reward +=
...`` line). Bookkeeping, geometry, and dynamics fields are filtered out.

    python -m evaluations.paper.appendix.generate_env_tables

Outputs: ``evaluations/reports/appendix/env_tables/env_<scenario>_<role>.tex``.
The companion ``appendix_envs.tex`` wrapper places leader/follower
side-by-side per scenario.
"""

from __future__ import annotations

import dataclasses as dc
from typing import Any

from evaluations._common.latex import row, write_table
from evaluations._common.paths import APPENDIX

# (scenario, role) pairs to document — the paper scenarios, both agents.
SCENARIO_ROLES: tuple[tuple[str, str], ...] = (
    ("mobile_arm", "follower"),
    ("mobile_arm", "leader"),
    ("obstacle_arena", "follower"),
    ("obstacle_arena", "leader"),
    ("warehouse", "follower"),
    ("warehouse", "leader"),
    ("warehouse_pickup_blocked", "follower"),
    ("warehouse_pickup_blocked", "leader"),
    ("warehouse_pickup_unseen", "follower"),
    ("warehouse_pickup_unseen", "leader"),
    ("household", "follower"),
    ("household", "leader"),
)

# Reward-determining fields per (scenario, role). Sourced by static
# inspection of each env's step() reward expression in src/.
#
# Update this table when an env's reward formula changes — anything not
# listed here is omitted from the appendix table.
_REWARD_FIELDS: dict[tuple[str, str], set[str]] = {
    ("obstacle_arena", "leader"):
        {"goal_reward"},
    ("obstacle_arena", "follower"):
        {"progress_weight", "distance_weight", "collision_penalty", "goal_reward"},
    ("mobile_arm", "leader"):
        {"step_penalty", "progress_weight", "goal_reward", "gaussian_sigma"},
    ("mobile_arm", "follower"):
        {"step_penalty", "progress_weight", "pickup_reward",
         "collision_penalty", "goal_reward"},
    ("warehouse", "leader"):
        {"step_penalty", "step_penalty_carrying", "progress_weight",
         "pickup_reward", "goal_reward"},
    ("warehouse", "follower"):
        {"step_penalty", "step_penalty_carrying", "progress_weight",
         "collision_penalty", "pickup_reward", "goal_reward"},
    ("household", "leader"):
        {"step_penalty", "progress_weight", "wall_collision_penalty",
         "pickup_reward"},
    ("household", "follower"):
        {"progress_weight", "distance_weight", "collision_penalty",
         "goal_reward"},
}
# Warehouse variants share reward structure with the base warehouse.
for _variant in ("warehouse_pickup_blocked", "warehouse_pickup_unseen"):
    _REWARD_FIELDS[(_variant, "leader")] = _REWARD_FIELDS[("warehouse", "leader")]
    _REWARD_FIELDS[(_variant, "follower")] = _REWARD_FIELDS[("warehouse", "follower")]


def _esc(text: str) -> str:
    return text.replace("_", r"\_")


def _fmt_val(v: Any) -> str:
    """Render a config value as a LaTeX table cell."""
    if v is None:
        return "--"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v == 0.0:
            return "0"
        return f"{v:.4g}"
    if isinstance(v, (tuple, list)):
        return "(" + ",\\ ".join(_fmt_val(x) for x in v) + ")"
    if isinstance(v, str):
        return "\\texttt{" + _esc(v) + "}"
    return _esc(str(v))


def main() -> int:
    from critic_sharing.config.resolver import load_env_config

    out_dir = APPENDIX / "env_tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing per-(scenario, role) reward tables to: {out_dir}")

    for scenario, role in SCENARIO_ROLES:
        cfg = load_env_config(scenario, role)
        keep = _REWARD_FIELDS.get((scenario, role))
        if keep is None:
            raise RuntimeError(
                f"No reward-field set declared for ({scenario}, {role}); "
                f"add an entry to _REWARD_FIELDS."
            )

        cfg_fields = {f.name for f in dc.fields(cfg)}
        missing = keep - cfg_fields
        if missing:
            raise RuntimeError(
                f"({scenario}, {role}): reward fields {missing} not on "
                f"{type(cfg).__name__}. The env or the filter is stale."
            )

        rows = [
            row([f"\\texttt{{{_esc(f.name)}}}", _fmt_val(getattr(cfg, f.name))])
            for f in dc.fields(cfg)
            if f.name in keep
        ]
        cls = type(cfg).__name__
        write_table(
            out_dir / f"env_{scenario}_{role}.tex",
            column_spec="lc",
            header=row(["Parameter", "Value"]),
            rows=rows,
            caption=(
                f"\\scenario{{{_esc(scenario)}}} {role} reward parameters "
                f"(\\texttt{{{_esc(cls)}}})."
            ),
            label=f"tab:env_{scenario}_{role}",
            use_resizebox=False,
        )
        print(f"  wrote env_{scenario}_{role}.tex  ({len(rows)} field(s))")

    print(f"\n{len(SCENARIO_ROLES)} tables written.")
    print("Remember: \\scenario{} + booktabs must be defined in the preamble.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
