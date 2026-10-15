"""Rotated-rectangle rendering for the household scenario.

Draws the cleaner (blue), table (brown, with long-axis line so the
orientation is visually unambiguous), follower (green), dirt (red
dot), and — when enabled — the wall with its single gap.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from critic_sharing.scenarios.household.common import (
    HouseholdEnvConfig,
    draw_rect,
    draw_wall,
)

CLEANER_FACE = "#42A5F5"
CLEANER_EDGE = "#0D47A1"
TABLE_FACE = "#A1887F"
TABLE_EDGE = "#3E2723"
FOLLOWER_FACE = "#66BB6A"
FOLLOWER_EDGE = "#1B5E20"
DIRT_COLOR = "#D32F2F"
GOAL_COLOR = "#FFA000"


def draw_arena(ax, config: HouseholdEnvConfig) -> None:
    """Set axis limits and draw the static world (walls if enabled)."""

    ax.set_xlim(config.world_x_min, config.world_x_max)
    ax.set_ylim(config.world_y_min, config.world_y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#9E9E9E")
    draw_wall(ax, config)


def draw_goal_line(ax, config: HouseholdEnvConfig) -> None:
    """Draw the centered goal segment on the lower boundary.

    The goal spans ``|x - mid_x| <= goal_x_half_extent`` (default: the
    centered half of the boundary).
    """

    mid_x = 0.5 * (float(config.world_x_min) + float(config.world_x_max))
    hx = getattr(config, "goal_x_half_extent", None)
    if hx is None:
        hx = 0.25 * (float(config.world_x_max) - float(config.world_x_min))
    hx = float(hx)
    ax.plot(
        [mid_x - hx, mid_x + hx],
        [config.world_y_min, config.world_y_min],
        color=GOAL_COLOR, linewidth=3.0, linestyle=":",
    )


def draw_table(ax, center, theta: float, half_extents: Tuple[float, float]) -> None:
    """Draw the table rectangle with a line along its long axis."""

    draw_rect(ax, center, float(theta), half_extents,
              face=TABLE_FACE, edge=TABLE_EDGE, alpha=0.85, linewidth=1.5)
    # Long-axis marker: pick the longer half-extent and draw a stroke
    # from -hx to +hx (or hy → hy) along that axis.
    hx, hy = float(half_extents[0]), float(half_extents[1])
    along_x = hx >= hy
    half_along = hx if along_x else hy
    c, s = float(np.cos(theta)), float(np.sin(theta))
    if along_x:
        dx, dy = c * half_along, s * half_along
    else:
        dx, dy = -s * half_along, c * half_along
    cx, cy = float(center[0]), float(center[1])
    ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy],
            color=TABLE_EDGE, linewidth=2.0, alpha=0.9)


def draw_cleaner(ax, center, theta: float, half_extents: Tuple[float, float]) -> None:
    draw_rect(ax, center, float(theta), half_extents,
              face=CLEANER_FACE, edge=CLEANER_EDGE, alpha=0.85, linewidth=1.5)
    # Heading arrow from centroid forward along local x.
    hx = float(half_extents[0])
    c, s = float(np.cos(theta)), float(np.sin(theta))
    cx, cy = float(center[0]), float(center[1])
    ax.annotate(
        "", xy=(cx + 1.6 * hx * c, cy + 1.6 * hx * s),
        xytext=(cx, cy),
        arrowprops=dict(arrowstyle="-|>", color=CLEANER_EDGE,
                        linewidth=1.2, mutation_scale=12),
    )


def draw_follower(ax, center, theta: float, half_extents: Tuple[float, float]) -> None:
    draw_rect(ax, center, float(theta), half_extents,
              face=FOLLOWER_FACE, edge=FOLLOWER_EDGE, alpha=0.85, linewidth=1.5)
    hx = float(half_extents[0])
    c, s = float(np.cos(theta)), float(np.sin(theta))
    cx, cy = float(center[0]), float(center[1])
    ax.annotate(
        "", xy=(cx + 1.6 * hx * c, cy + 1.6 * hx * s),
        xytext=(cx, cy),
        arrowprops=dict(arrowstyle="-|>", color=FOLLOWER_EDGE,
                        linewidth=1.2, mutation_scale=12),
    )


def draw_dirt(ax, center) -> None:
    cx, cy = float(center[0]), float(center[1])
    ax.plot(cx, cy, marker="o", color=DIRT_COLOR, markersize=10,
            markeredgecolor="#7F0000", zorder=5)


def render_state(
    ax,
    config: HouseholdEnvConfig,
    cleaner_pos: Optional[np.ndarray] = None,
    cleaner_theta: Optional[float] = None,
    table_pos: Optional[np.ndarray] = None,
    table_theta: Optional[float] = None,
    follower_pos: Optional[np.ndarray] = None,
    follower_theta: Optional[float] = None,
    dirt_pos: Optional[np.ndarray] = None,
    cleaner_half_extents: Optional[Tuple[float, float]] = None,
    table_half_extents: Optional[Tuple[float, float]] = None,
    follower_half_extents: Optional[Tuple[float, float]] = None,
) -> None:
    """One-shot render of an arbitrary household state onto ``ax``."""

    draw_arena(ax, config)
    draw_goal_line(ax, config)
    if table_pos is not None and table_theta is not None:
        draw_table(ax, table_pos, table_theta,
                   table_half_extents or config.table_half_extents)
    if dirt_pos is not None:
        draw_dirt(ax, dirt_pos)
    if cleaner_pos is not None and cleaner_theta is not None:
        draw_cleaner(ax, cleaner_pos, cleaner_theta,
                     cleaner_half_extents or config.cleaner_half_extents)
    if follower_pos is not None and follower_theta is not None:
        draw_follower(ax, follower_pos, follower_theta,
                      follower_half_extents or config.follower_half_extents)


__all__ = [
    "draw_arena",
    "draw_cleaner",
    "draw_dirt",
    "draw_follower",
    "draw_goal_line",
    "draw_table",
    "render_state",
]
