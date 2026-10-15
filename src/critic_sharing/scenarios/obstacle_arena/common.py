"""Shared helpers for the obstacle-arena scenario."""

from __future__ import annotations

import numpy as np

from critic_sharing.config.schema import BaseEnvConfig, NavEnvConfig, PushEnvConfig


def agent_bounds(config: NavEnvConfig | PushEnvConfig) -> tuple[float, float]:
    low = float(config.world_min + config.agent_radius)
    high = float(config.world_max - config.agent_radius)
    if low > high:
        raise ValueError(
            "Invalid world/agent radius: world_min + agent_radius > world_max - agent_radius."
        )
    return low, high


def push_obstacle_bounds(config: PushEnvConfig) -> tuple[float, float]:
    low = float(config.world_min + config.obstacle_radius + 1e-3)
    high = float(config.world_max - config.obstacle_radius - 1e-3)
    if low > high:
        raise ValueError(
            "Invalid obstacle radius: world_min + obstacle_radius > world_max - obstacle_radius."
        )
    return low, high


def nav_goal_position(config: NavEnvConfig) -> np.ndarray:
    mid_x = 0.5 * (config.world_min + config.world_max)
    return np.array([mid_x, config.world_min], dtype=np.float32)


# ---------------------------------------------------------------------------
# Wall-with-gap helpers
# ---------------------------------------------------------------------------

def wall_segments(
    config: BaseEnvConfig,
) -> list[tuple[float, float, float]]:
    """Return wall line segments as [(x_start, x_end, wall_y), ...].

    The gap is centred at ``config.gap_x`` with full width ``config.gap_width``.
    """
    wy = float(config.wall_y)
    half_gap = float(config.gap_width) / 2.0
    gap_left = float(config.gap_x) - half_gap
    gap_right = float(config.gap_x) + half_gap
    segments: list[tuple[float, float, float]] = []
    # Left segment: world_min to gap_left
    if gap_left > config.world_min:
        segments.append((float(config.world_min), gap_left, wy))
    # Right segment: gap_right to world_max
    if gap_right < config.world_max:
        segments.append((gap_right, float(config.world_max), wy))
    return segments


def circle_collides_wall(
    pos: np.ndarray,
    radius: float,
    config: BaseEnvConfig,
) -> bool:
    """Check if a circle at ``pos`` with ``radius`` overlaps any wall segment.

    Collision test: the circle overlaps a horizontal segment (x1, x2) at wall_y
    if the vertical distance is < radius AND the circle's horizontal extent
    overlaps the segment.
    """
    x, y = float(pos[0]), float(pos[1])
    r = float(radius)
    wy = float(config.wall_y)
    if abs(y - wy) >= r:
        return False
    for x1, x2, _ in wall_segments(config):
        # Check horizontal overlap: circle spans [x-r, x+r], segment spans [x1, x2]
        if x + r > x1 and x - r < x2:
            return True
    return False


def enforce_wall(
    next_pos: np.ndarray,
    prev_pos: np.ndarray,
    radius: float,
    config: BaseEnvConfig,
) -> np.ndarray:
    """Clamp ``next_pos`` so the circle does not cross the wall.

    If the entity is in the gap horizontally, it can pass through freely.
    Otherwise, clamp the y-coordinate to stay on the same side of the wall
    as ``prev_pos``.

    Returns the (possibly clamped) position.
    """
    wy = float(config.wall_y)
    r = float(radius)
    nx, ny = float(next_pos[0]), float(next_pos[1])
    py = float(prev_pos[1])

    # Check if entity is in the gap (horizontally clear of all segments)
    in_gap = True
    for x1, x2, _ in wall_segments(config):
        if nx + r > x1 and nx - r < x2:
            in_gap = False
            break
    if in_gap:
        return next_pos

    # Entity overlaps a wall segment — clamp y to stay on the same side
    if abs(ny - wy) < r:
        if py >= wy:
            # Was above wall, keep above
            ny = wy + r
        else:
            # Was below wall, keep below
            ny = wy - r
        return np.array([nx, ny], dtype=np.float32)
    return next_pos


def enforce_wall_batch(
    next_pos: np.ndarray,
    prev_pos: np.ndarray,
    radius: float,
    config: BaseEnvConfig,
) -> np.ndarray:
    """Batched version of :func:`enforce_wall` for shape ``(B, 2)``.

    Fully vectorised — no Python loops over the batch dimension.
    """
    wy = float(config.wall_y)
    r = float(radius)
    result = next_pos.copy()
    segments = wall_segments(config)
    if not segments:
        return result

    nx = next_pos[:, 0]  # (B,)
    ny = next_pos[:, 1]  # (B,)
    py = prev_pos[:, 1]  # (B,)

    # Check if each position overlaps any wall segment (not in gap)
    overlaps_segment = np.zeros(next_pos.shape[0], dtype=bool)
    for x1, x2, _ in segments:
        overlaps_segment |= (nx + r > x1) & (nx - r < x2)

    # Only clamp positions that overlap a wall segment AND are close to wall_y
    needs_clamp = overlaps_segment & (np.abs(ny - wy) < r)

    # Clamp: stay on the same side of the wall as prev_pos
    was_above = py >= wy
    result[needs_clamp & was_above, 1] = wy + r
    result[needs_clamp & ~was_above, 1] = wy - r
    return result


def draw_wall(ax, config: BaseEnvConfig, color: str = "#424242", linewidth: float = 3.0) -> None:
    """Draw wall segments on a matplotlib axes (skipping the gap)."""
    for x1, x2, wy in wall_segments(config):
        ax.plot([x1, x2], [wy, wy], color=color, linewidth=linewidth, solid_capstyle="butt")


__all__ = [
    "agent_bounds",
    "circle_collides_wall",
    "draw_wall",
    "enforce_wall",
    "enforce_wall_batch",
    "nav_goal_position",
    "push_obstacle_bounds",
    "wall_segments",
]
