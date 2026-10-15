"""Matplotlib rendering helpers for the warehouse scenario.

Centralised so the eval CLI, notebooks, and tests share one drawing source.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from critic_sharing.config.schema import _WarehouseBaseEnvConfig
from critic_sharing.scenarios.warehouse.common import (
    gap_rects,
    shelf_rects,
    wall_segments,
)


# Colours (kept consistent across the scenario's plots).
COLOR_WALL = "#424242"
COLOR_SHELF = "#90A4AE"
COLOR_GAP = "#B0BEC5"
COLOR_AGENT_LEADER = "#E65100"
COLOR_AGENT_FOLLOWER = "#1565C0"
COLOR_OBJECT = "#6A1B9A"
COLOR_GOAL = "#43A047"
COLOR_LEADER_TRAIL = "#FFB74D"
COLOR_FOLLOWER_TRAIL = "#64B5F6"


def draw_arena(
    ax,
    config: _WarehouseBaseEnvConfig,
    *,
    show_gaps: bool = True,
    show_shelves: bool = True,
    show_world_box: bool = True,
    pickup_disabled_gap_indices: Sequence[int] | None = None,
) -> None:
    """Draw walls, shelves, gaps and the world frame on ``ax``.

    Pure scenery — no agent/object/goal.

    If ``pickup_disabled_gap_indices`` is given (or the config has
    ``pickup_disabled_gap_indices`` set), the listed gaps get a hatched
    red block straddling the wall to flag that pickup is impossible there
    — used by the ``warehouse_pickup_blocked`` viz.
    """
    import matplotlib.patches as mpatches

    if show_world_box:
        ax.set_xlim(config.world_min, config.world_max)
        ax.set_ylim(config.world_min, config.world_max)
        ax.set_aspect("equal")
        ax.add_patch(
            mpatches.Rectangle(
                (config.world_min, config.world_min),
                config.world_max - config.world_min,
                config.world_max - config.world_min,
                fill=False,
                edgecolor="#37474F",
                linewidth=1.5,
                zorder=1,
            )
        )

    # Wall segments (outside any gap).
    for x1, x2, wy in wall_segments(config):
        ax.plot([x1, x2], [wy, wy], color=COLOR_WALL, linewidth=3.0, solid_capstyle="butt", zorder=2)

    # Highlight the 3 gap regions on the wall line.
    if show_gaps:
        wy = float(config.wall_y)
        for x1, x2 in gap_rects(config):
            ax.plot([x1, x2], [wy, wy], color=COLOR_GAP, linewidth=3.0, linestyle=":", zorder=2)

    if show_shelves:
        for xmin, xmax, ymin, ymax in shelf_rects(config):
            ax.add_patch(
                mpatches.Rectangle(
                    (xmin, ymin),
                    xmax - xmin,
                    ymax - ymin,
                    facecolor=COLOR_SHELF,
                    edgecolor="#37474F",
                    linewidth=1.0,
                    alpha=0.9,
                    zorder=3,
                )
            )

    # Pickup-disabled gap markers (warehouse_pickup_blocked only — empty for
    # other variants). Drawn as a wide red bar sitting in the corridor
    # between the wall (y=wall_y) and the shelves (y≈wall_y+0.8) with an
    # ✗ label, signalling "pickup impossible here".
    disabled = pickup_disabled_gap_indices
    if disabled is None:
        disabled = getattr(config, "pickup_disabled_gap_indices", ())
    disabled_set = {int(i) for i in disabled}
    if disabled_set:
        wy = float(config.wall_y)
        gw = float(config.gap_width)
        # Block is ~2× gap width and sits in the corridor just above the wall.
        block_width = gw * 2.2
        block_height = 0.18
        # Block sits just above the wall (small clearance so it visually
        # reads as "blocking the gap" without overlapping the wall line).
        block_y0 = wy + 0.06
        for i, (x1, x2) in enumerate(gap_rects(config)):
            if i in disabled_set:
                gap_center_x = 0.5 * (x1 + x2)
                ax.add_patch(
                    mpatches.Rectangle(
                        (gap_center_x - block_width / 2, block_y0),
                        block_width,
                        block_height,
                        facecolor="#E53935",
                        edgecolor="#7F0000",
                        linewidth=1.5,
                        alpha=0.85,
                        zorder=5,
                    )
                )
                ax.text(
                    gap_center_x,
                    block_y0 + block_height / 2,
                    r"$\times$",
                    color="white",
                    ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    zorder=6,
                )


def make_agent_patch(
    ax,
    config: _WarehouseBaseEnvConfig,
    color: str,
    *,
    alpha: float = 0.85,
    z: int = 6,
):
    """Create and add a square + heading-arrow representing the agent.

    Returns ``(square_patch, arrow_line)`` — both Artists; reposition each frame
    via :func:`update_agent_patch`.
    """
    import matplotlib.patches as mpatches

    side = float(config.agent_side)
    sq = mpatches.Rectangle(
        (-side / 2, -side / 2),
        side, side,
        facecolor=color, edgecolor="#212121", alpha=alpha, zorder=z,
    )
    ax.add_patch(sq)
    (arrow,) = ax.plot(
        [], [], color="#212121", linewidth=1.5, zorder=z + 1
    )
    return sq, arrow


def update_agent_patch(
    sq,
    arrow,
    pos: np.ndarray,
    theta: float,
    config: _WarehouseBaseEnvConfig,
) -> None:
    """Reposition / re-rotate an agent (square + heading arrow)."""
    import matplotlib.transforms as mtransforms

    side = float(config.agent_side)
    # Recreate transform: translate so centre is at pos, rotate by theta around it.
    sq.set_xy((-side / 2, -side / 2))
    tf = (
        mtransforms.Affine2D()
        .rotate(float(theta))
        .translate(float(pos[0]), float(pos[1]))
    )
    sq.set_transform(tf + sq.axes.transData)

    arrow_len = side * 0.6
    x0, y0 = float(pos[0]), float(pos[1])
    x1 = x0 + arrow_len * float(np.cos(theta))
    y1 = y0 + arrow_len * float(np.sin(theta))
    arrow.set_data([x0, x1], [y0, y1])


def make_object_patch(
    ax,
    config: _WarehouseBaseEnvConfig,
    *,
    z: int = 5,
):
    """Create a circle representing the object. Reposition via ``patch.center =``."""
    import matplotlib.patches as mpatches

    p = mpatches.Circle(
        (0, 0),
        float(config.object_radius),
        facecolor=COLOR_OBJECT,
        edgecolor="#311B92",
        linewidth=1.0,
        alpha=0.95,
        zorder=z,
    )
    ax.add_patch(p)
    return p


def draw_goal(
    ax,
    config: _WarehouseBaseEnvConfig,
    goal: np.ndarray,
    *,
    z: int = 7,
) -> None:
    """Draw the goal as a star marker with a tolerance ring."""
    import matplotlib.patches as mpatches

    ax.plot(
        float(goal[0]),
        float(goal[1]),
        marker="*",
        color=COLOR_GOAL,
        markersize=14,
        markeredgecolor="#1B5E20",
        zorder=z + 1,
    )
    tol = getattr(config, "goal_tolerance", None)
    if tol is not None:
        ax.add_patch(
            mpatches.Circle(
                (float(goal[0]), float(goal[1])),
                float(tol),
                facecolor="none",
                edgecolor=COLOR_GOAL,
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
                zorder=z,
            )
        )


__all__ = [
    "COLOR_AGENT_FOLLOWER",
    "COLOR_AGENT_LEADER",
    "COLOR_FOLLOWER_TRAIL",
    "COLOR_GOAL",
    "COLOR_LEADER_TRAIL",
    "COLOR_OBJECT",
    "draw_arena",
    "draw_goal",
    "make_agent_patch",
    "make_object_patch",
    "update_agent_patch",
]
