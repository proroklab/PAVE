"""Spawn samplers for the household scenario.

The cleaner and the table always spawn in the lower half (``y <
wall_y``) if the wall is enabled, else uniformly in the full arena.
Orientations are sampled uniformly from ``[-π, π)`` so V_F / OOD
observe the full table-pose distribution.

The follower spawns uniformly in the full arena at training time
(matches the obstacle_arena convention of "Follower can spawn on
either side of the wall (important for training experience below the
wall)"). Deployment-time restriction to the upper half is applied
externally via the ``options={"spawn_region": "upper"}`` knob honoured
by ``NavigatorEnv.reset``.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np

from critic_sharing.scenarios.household.common import (
    aabb_half_extents,
    rect_corners,
    rect_overlaps_wall,
    sat_overlap,
)

SpawnRegion = Literal["full", "lower", "upper"]


def _centroid_bounds(
    half_extents: Tuple[float, float], theta: float, config,
    region: SpawnRegion,
) -> Tuple[float, float, float, float]:
    """Orientation-aware spawn bounds for the centroid (per-axis world).

    The wall-thickness ``wall_thickness`` is added to the half-plane
    margin so spawns don't start inside the wall strip.
    """

    x_min = float(config.world_x_min)
    x_max = float(config.world_x_max)
    y_min = float(config.world_y_min)
    y_max = float(config.world_y_max)
    mid_x = 0.5 * (x_min + x_max)
    mid_y = 0.5 * (y_min + y_max)
    aabb_hw, aabb_hh = aabb_half_extents(half_extents, theta)

    if 2 * aabb_hw > x_max - x_min:
        low_x = high_x = mid_x
    else:
        low_x = x_min + aabb_hw
        high_x = x_max - aabb_hw

    if 2 * aabb_hh > y_max - y_min:
        low_y = high_y = mid_y
    else:
        low_y = y_min + aabb_hh
        high_y = y_max - aabb_hh

    if getattr(config, "wall_enabled", False):
        wy = float(config.wall_y)
        half_thick = float(getattr(config, "wall_thickness", 0.0)) / 2.0
        if region == "lower":
            high_y = min(high_y, wy - half_thick - aabb_hh - 1e-3)
        elif region == "upper":
            low_y = max(low_y, wy + half_thick + aabb_hh + 1e-3)
    return low_x, high_x, low_y, high_y


def sample_pose(
    rng: np.random.Generator,
    half_extents: Tuple[float, float],
    config,
    region: SpawnRegion = "full",
    max_tries: int = 1000,
) -> Tuple[np.ndarray, float]:
    """Sample a (centroid, theta) pose for a rectangle.

    Picks a random orientation first, then samples the centroid within
    the orientation-aware AABB bounds. Falls back to the arena midpoint
    along any axis where the OBB's AABB exceeds the arena width.
    Returns a numpy ``(2,)`` centroid and theta in ``[-π, π)``.
    """

    for _ in range(max_tries):
        theta = float(rng.uniform(-np.pi, np.pi))
        low_x, high_x, low_y, high_y = _centroid_bounds(
            half_extents, theta, config, region,
        )
        if high_y < low_y or high_x < low_x:
            # Orientation doesn't fit the region — try another theta.
            continue
        x = float(rng.uniform(low_x, high_x)) if high_x > low_x else low_x
        y = float(rng.uniform(low_y, high_y)) if high_y > low_y else low_y
        center = np.array([x, y], dtype=np.float32)
        if not rect_overlaps_wall(center, theta, half_extents, config):
            return center, theta
    # Last-ditch: axis-aligned at arena midpoint.
    mid_x = 0.5 * (float(config.world_x_min) + float(config.world_x_max))
    mid_y = 0.5 * (float(config.world_y_min) + float(config.world_y_max))
    return np.array([mid_x, mid_y], dtype=np.float32), 0.0


def sample_pose_no_overlap(
    rng: np.random.Generator,
    half_extents: Tuple[float, float],
    occupants: list[tuple[np.ndarray, float, Tuple[float, float]]],
    config,
    region: SpawnRegion = "full",
    max_tries: int = 1000,
) -> Tuple[np.ndarray, float]:
    """Sample a pose that does not SAT-overlap any of ``occupants``.

    ``occupants`` is a list of ``(center, theta, half_extents)`` tuples
    representing existing rectangles to avoid.
    """

    for _ in range(max_tries):
        center, theta = sample_pose(rng, half_extents, config, region=region,
                                    max_tries=64)
        ours = rect_corners(center, theta, half_extents)
        clear = True
        for (oc, ot, oh) in occupants:
            theirs = rect_corners(oc, ot, oh)
            overlapping, _, _ = sat_overlap(ours, theirs)
            if overlapping:
                clear = False
                break
        if clear:
            return center, theta
    # Last-ditch: return the most recent sample even if it overlaps.
    return center, theta


from dataclasses import dataclass


@dataclass(frozen=True)
class EpisodeConditions:
    """Initial state for one household episode.

    The ``table_pos`` is also the dirt's frozen location (5a-(i)). All
    orientations are in radians, wrapped to ``[-π, π)``.
    """

    table_pos: np.ndarray       # (2,)
    table_theta: float
    cleaner_pos: np.ndarray     # (2,)
    cleaner_theta: float
    follower_pos: np.ndarray    # (2,)
    follower_theta: float


def fixed_table_position(
    name: str, clean_cfg, nav_cfg,
) -> tuple[np.ndarray, "Optional[float] | list[float]"]:
    """Return ``(table_pos, table_theta)`` for a named preset.

    The returned ``theta`` may be:

    * ``None`` — sample uniformly from ``[-π, π)`` per episode.
    * a float — fixed orientation.
    * a list of floats — sample one element per episode (used by
      ``"gap-center"`` to alternate between axis-aligned 0° and 90°).

    Supported presets:
      - ``"gap-center"``: centroid at the midpoint of the lower half
        (y = midpoint between wall_bottom and world_y_min), x = gap_x.
        Orientation alternates between 0 and π/2 per episode.
      - ``"x,y"``: explicit coordinates, theta = 0.
      - ``"x,y,theta"``: explicit coordinates and orientation in radians.
    """

    if name == "gap-center":
        half_thick = float(getattr(clean_cfg, "wall_thickness", 0.0)) / 2.0
        wall_bottom = float(clean_cfg.wall_y) - half_thick
        y_floor = float(clean_cfg.world_y_min)
        # Centre of the lower half (between wall_bottom and arena floor).
        y = 0.5 * (wall_bottom + y_floor)
        return (
            np.array([float(clean_cfg.gap_x), y], dtype=np.float32),
            [0.0, float(np.pi / 2)],
        )
    parts = name.split(",")
    if len(parts) == 2:
        return np.array([float(parts[0]), float(parts[1])], dtype=np.float32), 0.0
    if len(parts) == 3:
        return (
            np.array([float(parts[0]), float(parts[1])], dtype=np.float32),
            float(parts[2]),
        )
    raise ValueError(f"Unknown fixed_table preset: {name!r}")


def sample_eval_conditions(
    clean_cfg,
    nav_cfg,
    n_episodes: int,
    rng: np.random.Generator,
    fixed_table: Optional[str] = None,
) -> list[EpisodeConditions]:
    """Pre-generate consistent initial conditions for evaluation.

    Same conditions are reused across guidance modes so episode
    difficulty is held constant.

    The table, cleaner, and dirt spawn in the lower half when the
    wall is enabled (else full arena); the follower spawns in the
    upper half (else full arena).

    Parameters
    ----------
    fixed_table
        If a preset name (``"gap-center"``, ``"x,y"``, ``"x,y,theta"``),
        all episodes use that exact table pose. ``None`` samples a
        fresh pose every episode.
    """

    clean_region = "lower" if getattr(clean_cfg, "wall_enabled", False) else "full"
    nav_region = "upper" if getattr(nav_cfg, "wall_enabled", False) else "full"

    table_he = tuple(clean_cfg.table_half_extents)
    cleaner_he = tuple(clean_cfg.cleaner_half_extents)
    follower_he = tuple(nav_cfg.follower_half_extents)

    fixed_pos: np.ndarray | None = None
    fixed_theta: "Optional[float] | list[float]" = None
    if fixed_table is not None:
        fixed_pos, fixed_theta = fixed_table_position(fixed_table, clean_cfg, nav_cfg)

    out: list[EpisodeConditions] = []
    for _ in range(n_episodes):
        if fixed_pos is not None:
            table_pos = fixed_pos.copy()
            if fixed_theta is None:
                table_theta = float(rng.uniform(-np.pi, np.pi))
            elif isinstance(fixed_theta, (list, tuple)):
                # Pick one orientation from the allowed set per episode.
                idx = int(rng.integers(0, len(fixed_theta)))
                table_theta = float(fixed_theta[idx])
            else:
                table_theta = float(fixed_theta)
        else:
            # Table spawns in the lower half (same as cleaner).
            table_pos, table_theta = sample_pose(
                rng, table_he, clean_cfg, region=clean_region,
            )

        if fixed_pos is not None:
            # At eval time with a fixed-table preset, constrain the
            # cleaner to spawn directly below the table so it
            # approaches from below (rather than anywhere in the
            # lower half).
            cleaner_pos, cleaner_theta = sample_pose_below_table(
                rng, cleaner_he,
                table_pos, table_theta, table_he,
                clean_cfg,
            )
        else:
            cleaner_pos, cleaner_theta = sample_pose_no_overlap(
                rng, cleaner_he,
                [(table_pos, table_theta, table_he)],
                clean_cfg, region=clean_region,
            )
        follower_pos, follower_theta = sample_pose_no_overlap(
            rng, follower_he,
            [(table_pos, table_theta, table_he),
             (cleaner_pos, cleaner_theta, cleaner_he)],
            nav_cfg, region=nav_region,
        )
        out.append(EpisodeConditions(
            table_pos=table_pos.astype(np.float32),
            table_theta=float(table_theta),
            cleaner_pos=cleaner_pos.astype(np.float32),
            cleaner_theta=float(cleaner_theta),
            follower_pos=follower_pos.astype(np.float32),
            follower_theta=float(follower_theta),
        ))
    return out


def sample_pose_below_table(
    rng: np.random.Generator,
    half_extents: Tuple[float, float],
    table_pos: np.ndarray,
    table_theta: float,
    table_half_extents: Tuple[float, float],
    config,
    x_margin: float = 0.2,
    y_margin: float = 0.02,
    max_tries: int = 1000,
) -> Tuple[np.ndarray, float]:
    """Sample a pose strictly below the table, near its x-footprint.

    Used at eval time to constrain the cleaner spawn region so it
    approaches the table from below rather than from any direction.

    The box is::

        x ∈ [table_x ± table_aabb_hw ± x_margin]
        y ∈ [world_y_min + aabb_hh,
             (table_y − table_aabb_hh) − aabb_hh − y_margin]

    Orientation-aware: re-rolls ``theta`` if the y-range collapses.
    Falls back to ``sample_pose_no_overlap`` if no feasible sample is
    found within ``max_tries``.
    """

    table_aabb_hw, table_aabb_hh = aabb_half_extents(table_half_extents, table_theta)
    table_bottom = float(table_pos[1]) - table_aabb_hh

    x_lo = max(float(config.world_x_min),
               float(table_pos[0]) - table_aabb_hw - x_margin)
    x_hi = min(float(config.world_x_max),
               float(table_pos[0]) + table_aabb_hw + x_margin)

    for _ in range(max_tries):
        theta = float(rng.uniform(-np.pi, np.pi))
        aabb_hw, aabb_hh = aabb_half_extents(half_extents, theta)
        y_lo = float(config.world_y_min) + aabb_hh
        y_hi = table_bottom - aabb_hh - y_margin
        if y_hi <= y_lo:
            continue  # infeasible at this θ
        x_lo_t = max(float(config.world_x_min) + aabb_hw, x_lo)
        x_hi_t = min(float(config.world_x_max) - aabb_hw, x_hi)
        if x_hi_t <= x_lo_t:
            continue
        x = float(rng.uniform(x_lo_t, x_hi_t))
        y = float(rng.uniform(y_lo, y_hi))
        center = np.array([x, y], dtype=np.float32)

        # Reject overlap with the table and with any wall rects.
        ours = rect_corners(center, theta, half_extents)
        tc = rect_corners(table_pos, table_theta, table_half_extents)
        ov, _, _ = sat_overlap(ours, tc)
        if ov:
            continue
        if rect_overlaps_wall(center, theta, half_extents, config):
            continue
        return center, theta
    # Fallback: relax the constraint, just spawn anywhere in the lower half.
    return sample_pose_no_overlap(
        rng, half_extents,
        [(table_pos, table_theta, table_half_extents)],
        config, region="lower",
    )


__all__ = [
    "EpisodeConditions",
    "fixed_table_position",
    "sample_eval_conditions",
    "sample_pose",
    "sample_pose_below_table",
    "sample_pose_no_overlap",
]
