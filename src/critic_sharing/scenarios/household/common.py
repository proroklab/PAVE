"""Shared helpers for the household scenario.

The household world has three rigid bodies: cleaner, table, and
follower — all oriented rectangles with explicit yaw and full SAT
contact resolution. The wall is a pair of static rectangles (left +
right segments) framing a single gap; agents and the table SAT-collide
with them just like with each other.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from critic_sharing.config.schema import HouseholdCleanerEnvConfig, HouseholdNavEnvConfig

HouseholdEnvConfig = HouseholdCleanerEnvConfig | HouseholdNavEnvConfig


# A static OBB obstacle, used for wall segments.
StaticRect = Tuple[np.ndarray, float, Tuple[float, float]]  # (center, theta, half_extents)


# ---------------------------------------------------------------------------
# Rotated-rectangle primitives
# ---------------------------------------------------------------------------

def rect_corners(
    center: np.ndarray, theta: float, half_extents: Tuple[float, float],
) -> np.ndarray:
    """Return the 4 corners of a rotated rectangle as ``(4, 2)``.

    ``half_extents`` is ``(hx, hy)`` in the rectangle's local frame
    (hx along local x, hy along local y). ``theta`` rotates the local
    frame counter-clockwise by ``theta`` radians.
    """

    hx, hy = float(half_extents[0]), float(half_extents[1])
    cx, cy = float(center[0]), float(center[1])
    c, s = float(np.cos(theta)), float(np.sin(theta))
    local = np.array(
        [
            [+hx, +hy],
            [+hx, -hy],
            [-hx, -hy],
            [-hx, +hy],
        ],
        dtype=np.float32,
    )
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    return (local @ R.T) + np.array([cx, cy], dtype=np.float32)


def rect_inscribed_radius(half_extents: Tuple[float, float]) -> float:
    return float(min(half_extents[0], half_extents[1]))


def rect_circumscribed_radius(half_extents: Tuple[float, float]) -> float:
    hx, hy = float(half_extents[0]), float(half_extents[1])
    return float(np.hypot(hx, hy))


def aabb_half_extents(half_extents: Tuple[float, float], theta: float) -> Tuple[float, float]:
    c, s = abs(float(np.cos(theta))), abs(float(np.sin(theta)))
    hx, hy = float(half_extents[0]), float(half_extents[1])
    return c * hx + s * hy, s * hx + c * hy


def point_in_rect(point: np.ndarray, center: np.ndarray, theta: float,
                  half_extents: Tuple[float, float]) -> bool:
    px = float(point[0]) - float(center[0])
    py = float(point[1]) - float(center[1])
    c, s = float(np.cos(-theta)), float(np.sin(-theta))
    lx = c * px - s * py
    ly = s * px + c * py
    return bool(abs(lx) <= half_extents[0] and abs(ly) <= half_extents[1])


# ---------------------------------------------------------------------------
# SAT (Separating Axis Theorem) for rotated rectangles
# ---------------------------------------------------------------------------

def _rect_axes(corners: np.ndarray) -> np.ndarray:
    edge_x = corners[1] - corners[0]
    edge_y = corners[3] - corners[0]
    edge_x = edge_x / (np.linalg.norm(edge_x) + 1e-12)
    edge_y = edge_y / (np.linalg.norm(edge_y) + 1e-12)
    return np.stack([edge_x, edge_y], axis=0)


def _project(corners: np.ndarray, axis: np.ndarray) -> Tuple[float, float]:
    proj = corners @ axis
    return float(proj.min()), float(proj.max())


def sat_overlap(
    corners_a: np.ndarray, corners_b: np.ndarray,
) -> Tuple[bool, np.ndarray, float]:
    """Test SAT overlap between two OBBs. Returns ``(overlapping, mtv, depth)``.

    The MTV points from A toward B; depth ≥ 0.
    """

    axes = np.concatenate([_rect_axes(corners_a), _rect_axes(corners_b)], axis=0)
    centroid_a = corners_a.mean(axis=0)
    centroid_b = corners_b.mean(axis=0)
    direction = centroid_b - centroid_a

    best_depth = np.inf
    best_axis = np.array([0.0, 0.0], dtype=np.float32)
    for axis in axes:
        amin, amax = _project(corners_a, axis)
        bmin, bmax = _project(corners_b, axis)
        overlap = min(amax, bmax) - max(amin, bmin)
        if overlap <= 0.0:
            return False, np.array([0.0, 0.0], dtype=np.float32), 0.0
        if overlap < best_depth:
            best_depth = overlap
            best_axis = axis if float(np.dot(axis, direction)) >= 0.0 else -axis
    return True, best_axis.astype(np.float32), float(best_depth)


def sat_contact_point(
    corners_a: np.ndarray, corners_b: np.ndarray, mtv_normal: np.ndarray,
) -> np.ndarray:
    centroid_b = corners_b.mean(axis=0)
    bmin, bmax = _project(corners_b, mtv_normal)
    proj_a = corners_a @ mtv_normal
    inside = (proj_a >= bmin) & (proj_a <= bmax)
    candidates = corners_a[inside]
    if candidates.size > 0:
        return candidates.mean(axis=0).astype(np.float32)
    centroid_a = corners_a.mean(axis=0)
    return ((centroid_a + centroid_b) * 0.5).astype(np.float32)


# ---------------------------------------------------------------------------
# Wall as static SAT rectangles
# ---------------------------------------------------------------------------

def wall_rects(config: HouseholdEnvConfig) -> List[StaticRect]:
    """Return the wall as a list of static ``(center, theta, half_extents)`` OBBs.

    Two axis-aligned rectangles span the wall's full thickness, with a
    ``gap_width``-wide opening centred at ``gap_x``. Empty when
    ``wall_enabled`` is False.
    """

    if not getattr(config, "wall_enabled", False):
        return []
    wy = float(config.wall_y)
    half_thick = float(config.wall_thickness) / 2.0
    half_gap = float(config.gap_width) / 2.0
    gap_left = float(config.gap_x) - half_gap
    gap_right = float(config.gap_x) + half_gap
    rects: List[StaticRect] = []
    # Left segment: world_x_min → gap_left
    if gap_left > config.world_x_min:
        cx = 0.5 * (float(config.world_x_min) + gap_left)
        hx = 0.5 * (gap_left - float(config.world_x_min))
        rects.append((np.array([cx, wy], dtype=np.float32), 0.0, (hx, half_thick)))
    # Right segment: gap_right → world_x_max
    if gap_right < config.world_x_max:
        cx = 0.5 * (gap_right + float(config.world_x_max))
        hx = 0.5 * (float(config.world_x_max) - gap_right)
        rects.append((np.array([cx, wy], dtype=np.float32), 0.0, (hx, half_thick)))
    return rects


def rect_overlaps_wall(
    center: np.ndarray, theta: float, half_extents: Tuple[float, float],
    config: HouseholdEnvConfig,
) -> bool:
    """Does the rotated rectangle SAT-overlap any wall segment?"""

    rects = wall_rects(config)
    if not rects:
        return False
    corners = rect_corners(np.asarray(center, dtype=np.float32),
                           float(theta), half_extents)
    for (wcenter, wtheta, whe) in rects:
        wcorners = rect_corners(wcenter, wtheta, whe)
        overlapping, _, _ = sat_overlap(corners, wcorners)
        if overlapping:
            return True
    return False


def resolve_static_collision(
    center: np.ndarray, theta: float, half_extents: Tuple[float, float],
    static_rects: List[StaticRect],
) -> Tuple[np.ndarray, bool]:
    """Push the moving rect along the SAT MTV away from any static obstacle.

    Returns ``(new_center, was_clamped)``. Iterates up to a few passes
    so that being pinned between two static rects (e.g. wall corner)
    still converges to a non-overlapping position.
    """

    new_center = np.asarray(center, dtype=np.float32).copy()
    was_clamped = False
    if not static_rects:
        return new_center, was_clamped
    for _ in range(4):
        any_overlap = False
        moving = rect_corners(new_center, float(theta), half_extents)
        for (scenter, stheta, she) in static_rects:
            static = rect_corners(scenter, float(stheta), she)
            overlapping, mtv, depth = sat_overlap(static, moving)
            if not overlapping:
                continue
            # MTV points from static toward moving — push moving along +MTV.
            new_center = new_center + (float(depth) + 1e-4) * mtv
            any_overlap = True
            was_clamped = True
        if not any_overlap:
            break
    return new_center.astype(np.float32), was_clamped


def enforce_wall_rect(
    next_center: np.ndarray, next_theta: float,
    prev_center: np.ndarray, half_extents: Tuple[float, float],
    config: HouseholdEnvConfig,
) -> Tuple[np.ndarray, bool]:
    """Compatibility shim: SAT-resolve against the wall rects.

    ``prev_center`` is no longer used (the SAT-MTV pushback always
    moves the body along the contact normal, naturally preserving the
    starting side).
    """

    return resolve_static_collision(next_center, next_theta, half_extents,
                                    wall_rects(config))


# ---------------------------------------------------------------------------
# World-bound clamp (per-axis)
# ---------------------------------------------------------------------------

def clamp_to_world(
    center: np.ndarray, half_extents: Tuple[float, float],
    config: HouseholdEnvConfig, theta: float = 0.0,
) -> Tuple[np.ndarray, bool]:
    """Clamp centroid so the rotated AABB stays inside the per-axis bounds.

    Returns ``(clamped_center, hit_boundary)``.
    """

    corners = rect_corners(np.asarray(center, dtype=np.float32),
                           float(theta), half_extents)
    x_min = float(config.world_x_min)
    x_max = float(config.world_x_max)
    y_min = float(config.world_y_min)
    y_max = float(config.world_y_max)
    mid_x = 0.5 * (x_min + x_max)
    mid_y = 0.5 * (y_min + y_max)

    cx, cy = float(center[0]), float(center[1])
    new_cx, new_cy = cx, cy

    aabb_xmin = float(corners[:, 0].min())
    aabb_xmax = float(corners[:, 0].max())
    if aabb_xmax - aabb_xmin > x_max - x_min:
        new_cx = mid_x
    else:
        if aabb_xmin < x_min:
            new_cx = cx + (x_min - aabb_xmin)
        elif aabb_xmax > x_max:
            new_cx = cx - (aabb_xmax - x_max)

    aabb_ymin = float(corners[:, 1].min())
    aabb_ymax = float(corners[:, 1].max())
    if aabb_ymax - aabb_ymin > y_max - y_min:
        new_cy = mid_y
    else:
        if aabb_ymin < y_min:
            new_cy = cy + (y_min - aabb_ymin)
        elif aabb_ymax > y_max:
            new_cy = cy - (aabb_ymax - y_max)

    clamped = np.array([new_cx, new_cy], dtype=np.float32)
    hit = bool(not np.allclose(clamped, center, atol=1e-6))
    return clamped, hit


def nav_goal_position(config: HouseholdEnvConfig) -> np.ndarray:
    mid_x = 0.5 * (float(config.world_x_min) + float(config.world_x_max))
    return np.array([mid_x, float(config.world_y_min)], dtype=np.float32)


# ---------------------------------------------------------------------------
# Matplotlib helpers
# ---------------------------------------------------------------------------

def draw_wall(ax, config: HouseholdEnvConfig, color: str = "#424242",
              edge: str = "#212121", alpha: float = 0.85) -> None:
    for (wcenter, wtheta, whe) in wall_rects(config):
        draw_rect(ax, wcenter, wtheta, whe, face=color, edge=edge,
                  alpha=alpha, linewidth=1.0)


def draw_rect(ax, center: np.ndarray, theta: float,
              half_extents: Tuple[float, float],
              face: str = "#90CAF9", edge: str = "#1565C0", alpha: float = 0.7,
              linewidth: float = 1.5) -> None:
    corners = rect_corners(np.asarray(center, dtype=np.float32), float(theta),
                           half_extents)
    closed = np.vstack([corners, corners[:1]])
    ax.fill(closed[:, 0], closed[:, 1], facecolor=face, edgecolor=edge,
            alpha=alpha, linewidth=linewidth)


# ---------------------------------------------------------------------------
# Batched (vectorised) primitives for CEM planning dynamics.
# ---------------------------------------------------------------------------

def rect_corners_batched(
    center: np.ndarray, theta: np.ndarray, half_extents: Tuple[float, float],
) -> np.ndarray:
    hx, hy = float(half_extents[0]), float(half_extents[1])
    local = np.array(
        [[+hx, +hy], [+hx, -hy], [-hx, -hy], [-hx, +hy]],
        dtype=np.float32,
    )
    c = np.cos(theta).astype(np.float32)
    s = np.sin(theta).astype(np.float32)
    rotated = np.empty((center.shape[0], 4, 2), dtype=np.float32)
    rotated[:, :, 0] = local[None, :, 0] * c[:, None] - local[None, :, 1] * s[:, None]
    rotated[:, :, 1] = local[None, :, 0] * s[:, None] + local[None, :, 1] * c[:, None]
    return rotated + center[:, None, :].astype(np.float32)


def aabb_half_extents_batched(
    half_extents: Tuple[float, float], theta: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    c = np.abs(np.cos(theta)).astype(np.float32)
    s = np.abs(np.sin(theta)).astype(np.float32)
    hx, hy = float(half_extents[0]), float(half_extents[1])
    return c * hx + s * hy, s * hx + c * hy


def clamp_to_world_batched(
    center: np.ndarray, theta: np.ndarray, half_extents: Tuple[float, float],
    config: HouseholdEnvConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    x_min = float(config.world_x_min)
    x_max = float(config.world_x_max)
    y_min = float(config.world_y_min)
    y_max = float(config.world_y_max)
    mid_x = 0.5 * (x_min + x_max)
    mid_y = 0.5 * (y_min + y_max)
    aabb_hw, aabb_hh = aabb_half_extents_batched(half_extents, theta)

    cx = center[:, 0].astype(np.float32)
    cy = center[:, 1].astype(np.float32)

    too_wide_x = 2 * aabb_hw > (x_max - x_min)
    new_cx = np.where(too_wide_x, mid_x, cx)
    new_cx = np.where((~too_wide_x) & (cx - aabb_hw < x_min),
                      x_min + aabb_hw, new_cx)
    new_cx = np.where((~too_wide_x) & (cx + aabb_hw > x_max),
                      x_max - aabb_hw, new_cx)

    too_wide_y = 2 * aabb_hh > (y_max - y_min)
    new_cy = np.where(too_wide_y, mid_y, cy)
    new_cy = np.where((~too_wide_y) & (cy - aabb_hh < y_min),
                      y_min + aabb_hh, new_cy)
    new_cy = np.where((~too_wide_y) & (cy + aabb_hh > y_max),
                      y_max - aabb_hh, new_cy)

    clamped = np.stack([new_cx, new_cy], axis=1).astype(np.float32)
    hit = np.any(np.abs(clamped - center.astype(np.float32)) > 1e-6, axis=1)
    return clamped, hit


def _rect_axes_batched(corners: np.ndarray) -> np.ndarray:
    edge_x = corners[:, 1] - corners[:, 0]
    edge_y = corners[:, 3] - corners[:, 0]
    nx = np.linalg.norm(edge_x, axis=1, keepdims=True) + 1e-12
    ny = np.linalg.norm(edge_y, axis=1, keepdims=True) + 1e-12
    return np.stack([edge_x / nx, edge_y / ny], axis=1).astype(np.float32)


def sat_overlap_batched(
    corners_a: np.ndarray, corners_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    B = corners_a.shape[0]
    axes = np.concatenate(
        [_rect_axes_batched(corners_a), _rect_axes_batched(corners_b)], axis=1,
    )
    proj_a = np.einsum("baj,bcj->bac", axes, corners_a)
    proj_b = np.einsum("baj,bcj->bac", axes, corners_b)
    amin, amax = proj_a.min(axis=2), proj_a.max(axis=2)
    bmin, bmax = proj_b.min(axis=2), proj_b.max(axis=2)
    overlap = np.minimum(amax, bmax) - np.maximum(amin, bmin)
    overlapping = (overlap > 0).all(axis=1)

    overlap_safe = np.where(overlapping[:, None], overlap, np.inf)
    best_idx = overlap_safe.argmin(axis=1)
    rows = np.arange(B)
    best_axis = axes[rows, best_idx]
    best_depth = np.where(overlapping, overlap[rows, best_idx], 0.0).astype(np.float32)

    centroid_a = corners_a.mean(axis=1)
    centroid_b = corners_b.mean(axis=1)
    direction = centroid_b - centroid_a
    signs = np.sign((best_axis * direction).sum(axis=1, keepdims=True))
    signs = np.where(signs == 0, 1.0, signs)
    best_axis = (best_axis * signs).astype(np.float32)
    return overlapping, best_axis, best_depth


def sat_contact_point_batched(
    corners_a: np.ndarray, corners_b: np.ndarray, mtv_normal: np.ndarray,
) -> np.ndarray:
    proj_a = np.einsum("bj,bcj->bc", mtv_normal, corners_a)
    proj_b = np.einsum("bj,bcj->bc", mtv_normal, corners_b)
    bmin = proj_b.min(axis=1, keepdims=True)
    bmax = proj_b.max(axis=1, keepdims=True)
    inside = (proj_a >= bmin) & (proj_a <= bmax)
    count = inside.sum(axis=1).astype(np.float32)
    safe_count = np.where(count > 0, count, 1.0)
    weighted = (corners_a * inside[:, :, None]).sum(axis=1)
    candidate = weighted / safe_count[:, None]
    centroid_a = corners_a.mean(axis=1)
    centroid_b = corners_b.mean(axis=1)
    fallback = 0.5 * (centroid_a + centroid_b)
    return np.where((count > 0)[:, None], candidate, fallback).astype(np.float32)


def point_in_rect_batched(
    point: np.ndarray, center: np.ndarray, theta: np.ndarray,
    half_extents: Tuple[float, float],
) -> np.ndarray:
    dx = point[:, 0] - center[:, 0]
    dy = point[:, 1] - center[:, 1]
    c = np.cos(-theta)
    s = np.sin(-theta)
    lx = c * dx - s * dy
    ly = s * dx + c * dy
    return (np.abs(lx) <= half_extents[0]) & (np.abs(ly) <= half_extents[1])


def resolve_static_collision_batched(
    center: np.ndarray, theta: np.ndarray, half_extents: Tuple[float, float],
    static_rects: List[StaticRect],
) -> Tuple[np.ndarray, np.ndarray]:
    """Batched: push moving rects out of each static rect via SAT-MTV."""

    B = center.shape[0]
    new_center = center.astype(np.float32).copy()
    was_clamped = np.zeros(B, dtype=bool)
    if not static_rects:
        return new_center, was_clamped
    for _ in range(4):
        moving = rect_corners_batched(new_center, theta, half_extents)
        any_pass = False
        for (scenter, stheta, she) in static_rects:
            static_center_b = np.broadcast_to(
                scenter.astype(np.float32), (B, 2),
            ).copy()
            static_theta_b = np.full(B, float(stheta), dtype=np.float32)
            static = rect_corners_batched(static_center_b, static_theta_b, she)
            ov, mtv, depth = sat_overlap_batched(static, moving)
            if not ov.any():
                continue
            push = (depth + 1e-4) * ov  # (B,)
            new_center = new_center + push[:, None] * mtv
            was_clamped |= ov
            any_pass = True
        if not any_pass:
            break
    return new_center.astype(np.float32), was_clamped


# Backward-compat wrapper (kept so older call sites keep working).
def enforce_wall_rect_batched(
    next_center: np.ndarray, next_theta: np.ndarray,
    prev_center: np.ndarray, half_extents: Tuple[float, float],
    config: HouseholdEnvConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    return resolve_static_collision_batched(
        next_center, next_theta, half_extents, wall_rects(config),
    )


__all__ = [
    "StaticRect",
    "aabb_half_extents",
    "aabb_half_extents_batched",
    "clamp_to_world",
    "clamp_to_world_batched",
    "draw_rect",
    "draw_wall",
    "enforce_wall_rect",
    "enforce_wall_rect_batched",
    "nav_goal_position",
    "point_in_rect",
    "point_in_rect_batched",
    "rect_circumscribed_radius",
    "rect_corners",
    "rect_corners_batched",
    "rect_inscribed_radius",
    "rect_overlaps_wall",
    "resolve_static_collision",
    "resolve_static_collision_batched",
    "sat_contact_point",
    "sat_contact_point_batched",
    "sat_overlap",
    "sat_overlap_batched",
    "wall_rects",
]
