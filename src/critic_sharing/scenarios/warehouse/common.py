"""Shared geometry helpers for the warehouse scenario.

The world contains:
  * a horizontal wall at ``y = wall_y`` with three gaps (dropoff zones),
    agent-impassable everywhere (gaps are too narrow for the inscribed-circle
    agent), object-passable through gaps;
  * three vertical shelves in the upper half (axis-aligned rectangles), fully
    untraversable by both agents.

Agents are squares with explicit yaw, but collisions are computed against the
inscribed circle ``r = side / 2``. Yaw is used only for the pickup cone and the
drop placement direction.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from critic_sharing.config.schema import (
    _WarehouseBaseEnvConfig,
)


WarehouseConfig = _WarehouseBaseEnvConfig  # type alias used in signatures


# ---------------------------------------------------------------------------
# Basic accessors
# ---------------------------------------------------------------------------

def agent_inscribed_radius(config: WarehouseConfig) -> float:
    return float(config.agent_side) / 2.0


def agent_bounds(config: WarehouseConfig) -> Tuple[float, float]:
    r = agent_inscribed_radius(config)
    low = float(config.world_min) + r
    high = float(config.world_max) - r
    if low > high:
        raise ValueError("Invalid world bounds: world too small for agent.")
    return low, high


def object_bounds(config: WarehouseConfig) -> Tuple[float, float]:
    low = float(config.world_min) + float(config.object_radius) + 1e-3
    high = float(config.world_max) - float(config.object_radius) - 1e-3
    if low > high:
        raise ValueError("Invalid world bounds: world too small for object.")
    return low, high


# ---------------------------------------------------------------------------
# Wall + 3 gaps
# ---------------------------------------------------------------------------

def wall_segments(config: WarehouseConfig) -> List[Tuple[float, float, float]]:
    """Return solid wall segments as ``[(x_start, x_end, wall_y), ...]``.

    Built by partitioning the line ``y = wall_y`` around the 3 gaps. The agent
    cannot pass through any of these segments. Gaps fall in between.
    """
    wy = float(config.wall_y)
    half = float(config.gap_width) / 2.0
    gap_xs = sorted(float(x) for x in config.gap_xs)
    boundaries: List[float] = [float(config.world_min)]
    for gx in gap_xs:
        boundaries.append(gx - half)
        boundaries.append(gx + half)
    boundaries.append(float(config.world_max))
    segments: List[Tuple[float, float, float]] = []
    for i in range(0, len(boundaries), 2):
        x_start = boundaries[i]
        x_end = boundaries[i + 1]
        if x_end > x_start:
            segments.append((x_start, x_end, wy))
    return segments


def gap_rects(config: WarehouseConfig) -> List[Tuple[float, float]]:
    """Return ``[(gap_x_min, gap_x_max), ...]`` for each gap, sorted by x."""
    half = float(config.gap_width) / 2.0
    return [(float(gx) - half, float(gx) + half) for gx in sorted(config.gap_xs)]


def gap_centers(config: WarehouseConfig) -> np.ndarray:
    """Return ``(num_gaps, 2)`` array of gap centres at ``y = wall_y``."""
    wy = float(config.wall_y)
    return np.array(
        [(float(gx), wy) for gx in sorted(config.gap_xs)],
        dtype=np.float32,
    )


def gap_index_for_position(point: np.ndarray, config: WarehouseConfig) -> int:
    """Return the gap index (0..N-1) containing ``point`` or ``-1`` if none.

    A drop counts as in-gap iff its x is inside any gap range AND its y is
    within ``drop_distance`` of ``wall_y`` (the "object settles into the gap"
    semantics — the leader can only drop close enough when standing close
    to the wall and aiming through it).
    """
    y_tol = float(config.drop_distance)
    if abs(float(point[1]) - float(config.wall_y)) > y_tol:
        return -1
    px = float(point[0])
    half = float(config.gap_width) / 2.0
    sorted_gaps = sorted(enumerate(config.gap_xs), key=lambda t: t[1])
    for new_idx, (orig_idx, gx) in enumerate(sorted_gaps):
        if abs(px - float(gx)) <= half:
            return new_idx
    return -1


def snap_to_wall_if_in_gap(point: np.ndarray, config: WarehouseConfig) -> np.ndarray:
    """If ``point`` qualifies as in-gap, snap its y to ``wall_y``.

    Used so a successful drop always places the object exactly on the wall
    line — ensures the follower's pickup geometry is well-defined regardless
    of the leader's exact drop angle.
    """
    if gap_index_for_position(point, config) >= 0:
        out = np.asarray(point, dtype=np.float32).copy()
        out[1] = float(config.wall_y)
        return out
    return np.asarray(point, dtype=np.float32)


# ---------------------------------------------------------------------------
# Wall enforcement (agent-impassable everywhere, including gaps)
# ---------------------------------------------------------------------------

def enforce_wall(
    next_pos: np.ndarray,
    prev_pos: np.ndarray,
    radius: float,
    config: WarehouseConfig,
) -> np.ndarray:
    """Clamp a circle position so it never crosses the wall.

    Unlike obstacle_arena's variant, gaps do *not* let the agent pass through:
    the gap is sized so the inscribed circle cannot fit. The clamp logic only
    cares whether the circle's y-extent overlaps any wall segment.
    """
    wy = float(config.wall_y)
    r = float(radius)
    nx, ny = float(next_pos[0]), float(next_pos[1])
    py = float(prev_pos[1])
    if abs(ny - wy) >= r:
        return next_pos
    # Determine if the circle horizontally overlaps any wall segment.
    overlaps = False
    for x1, x2, _ in wall_segments(config):
        if nx + r > x1 and nx - r < x2:
            overlaps = True
            break
    if not overlaps:
        return next_pos
    out = np.array([nx, ny], dtype=np.float32)
    out[1] = wy + r if py >= wy else wy - r
    return out


def enforce_agent_object_collision(
    next_pos: np.ndarray,
    object_pos: np.ndarray,
    agent_radius: float,
    config: WarehouseConfig,
) -> np.ndarray:
    """Clamp ``next_pos`` so the agent never overlaps the (uncarried) object.

    Inscribed-circle collision: if proposed position is within
    ``agent_radius + object_radius`` of the object centre, push it back to
    the contact boundary along the line from object → next_pos.
    """
    p = np.asarray(next_pos, dtype=np.float32).copy()
    o = np.asarray(object_pos, dtype=np.float32)
    contact = float(agent_radius) + float(config.object_radius)
    diff = p - o
    dist = float(np.linalg.norm(diff))
    if dist >= contact:
        return p
    if dist < 1e-8:
        diff = np.array([1.0, 0.0], dtype=np.float32)
        dist = 1.0
    return (o + diff / dist * contact).astype(np.float32)


def enforce_agent_object_collision_batch(
    next_pos: np.ndarray,
    object_pos: np.ndarray,
    carry: np.ndarray,
    agent_radius: float,
    config: WarehouseConfig,
) -> np.ndarray:
    """Vectorised :func:`enforce_agent_object_collision`. Skips carrying rows."""
    p = np.asarray(next_pos, dtype=np.float32).copy()
    o = np.asarray(object_pos, dtype=np.float32)
    if o.ndim == 1:
        o = np.broadcast_to(o, p.shape)
    contact = float(agent_radius) + float(config.object_radius)
    diff = p - o
    dist = np.linalg.norm(diff, axis=1)
    mask = (~np.asarray(carry, dtype=bool)) & (dist < contact)
    if not np.any(mask):
        return p
    safe_dist = np.where(dist > 1e-8, dist, 1.0)
    direction = diff / safe_dist[:, None]
    direction = np.where(
        (dist > 1e-8)[:, None],
        direction,
        np.array([1.0, 0.0], dtype=np.float32),
    )
    p[mask] = (o[mask] + direction[mask] * contact).astype(np.float32)
    return p


def enforce_wall_batch(
    next_pos: np.ndarray,
    prev_pos: np.ndarray,
    radius: float,
    config: WarehouseConfig,
) -> np.ndarray:
    """Vectorised :func:`enforce_wall` for ``(B, 2)`` arrays."""
    wy = float(config.wall_y)
    r = float(radius)
    result = next_pos.copy()
    segments = wall_segments(config)
    if not segments:
        return result
    nx = next_pos[:, 0]
    ny = next_pos[:, 1]
    py = prev_pos[:, 1]
    near_wall = np.abs(ny - wy) < r
    overlaps = np.zeros(next_pos.shape[0], dtype=bool)
    for x1, x2, _ in segments:
        overlaps |= (nx + r > x1) & (nx - r < x2)
    needs_clamp = near_wall & overlaps
    was_above = py >= wy
    result[needs_clamp & was_above, 1] = wy + r
    result[needs_clamp & ~was_above, 1] = wy - r
    return result


# ---------------------------------------------------------------------------
# Shelves
# ---------------------------------------------------------------------------

def shelf_rects(config: WarehouseConfig) -> List[Tuple[float, float, float, float]]:
    """Return ``[(xmin, xmax, ymin, ymax), ...]`` for each shelf."""
    half_w = float(config.shelf_width) / 2.0
    ymin = float(config.shelf_y_min)
    ymax = float(config.shelf_y_max)
    return [
        (float(sx) - half_w, float(sx) + half_w, ymin, ymax)
        for sx in config.shelf_xs
    ]


def _push_circle_out_of_rect(
    cx: np.ndarray,
    cy: np.ndarray,
    radius: float,
    rect: Tuple[float, float, float, float],
    eps: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Push circles out of one inflated axis-aligned rectangle.

    ``cx, cy`` are arrays of shape ``(B,)``. Returns ``(new_cx, new_cy, was_inside)``.
    The inflated rect is the rect grown by ``radius`` (Minkowski sum with circle):
    a circle centre inside the inflated rect overlaps the rect.
    """
    xmin, xmax, ymin, ymax = rect
    ix_min = xmin - radius
    ix_max = xmax + radius
    iy_min = ymin - radius
    iy_max = ymax + radius
    inside = (cx > ix_min) & (cx < ix_max) & (cy > iy_min) & (cy < iy_max)
    if not np.any(inside):
        return cx, cy, inside
    push_left = cx - ix_min
    push_right = ix_max - cx
    push_down = cy - iy_min
    push_up = iy_max - cy
    pushes = np.stack([push_left, push_right, push_down, push_up], axis=-1)
    min_idx = np.argmin(pushes, axis=-1)
    new_cx = cx.copy()
    new_cy = cy.copy()
    sel_left = inside & (min_idx == 0)
    sel_right = inside & (min_idx == 1)
    sel_down = inside & (min_idx == 2)
    sel_up = inside & (min_idx == 3)
    new_cx[sel_left] = ix_min - eps
    new_cx[sel_right] = ix_max + eps
    new_cy[sel_down] = iy_min - eps
    new_cy[sel_up] = iy_max + eps
    return new_cx, new_cy, inside


def enforce_shelves_batch(
    next_pos: np.ndarray,
    radius: float,
    config: WarehouseConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Push circles out of all shelves. Returns ``(new_pos, collided_mask)``."""
    pos = next_pos.copy()
    collided = np.zeros(pos.shape[0], dtype=bool)
    for rect in shelf_rects(config):
        new_x, new_y, inside = _push_circle_out_of_rect(pos[:, 0], pos[:, 1], radius, rect)
        pos[:, 0] = new_x
        pos[:, 1] = new_y
        collided |= inside
    return pos, collided


def enforce_shelves(
    next_pos: np.ndarray,
    radius: float,
    config: WarehouseConfig,
) -> Tuple[np.ndarray, bool]:
    """Per-step :func:`enforce_shelves_batch`."""
    pos_b = next_pos.reshape(1, 2)
    out, col = enforce_shelves_batch(pos_b, radius, config)
    return out[0].astype(np.float32), bool(col[0])


def point_in_any_shelf(point: np.ndarray, config: WarehouseConfig) -> bool:
    """Check if a point (e.g. dropped object centre) lies inside any shelf."""
    px, py = float(point[0]), float(point[1])
    for xmin, xmax, ymin, ymax in shelf_rects(config):
        if xmin <= px <= xmax and ymin <= py <= ymax:
            return True
    return False


def point_in_any_wall_segment(
    point: np.ndarray,
    config: WarehouseConfig,
    margin: float = 0.0,
) -> bool:
    """Check if a point lies inside a wall segment (within ``margin`` of wall_y)."""
    px, py = float(point[0]), float(point[1])
    if abs(py - float(config.wall_y)) > margin:
        return False
    for x1, x2, _ in wall_segments(config):
        if x1 <= px <= x2:
            return True
    return False


def shelf_face_segments(
    config: WarehouseConfig,
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Return left/right shelf faces as ``[((x, y_min), (x, y_max)), ...]``.

    Two faces per shelf (left at ``shelf_x - shelf_width/2``, right at the
    matching positive offset), spanning the shelf's y-extent. Used for goal
    sampling.
    """
    half_w = float(config.shelf_width) / 2.0
    ymin = float(config.shelf_y_min)
    ymax = float(config.shelf_y_max)
    faces: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for sx in config.shelf_xs:
        sx_f = float(sx)
        faces.append(((sx_f - half_w, ymin), (sx_f - half_w, ymax)))
        faces.append(((sx_f + half_w, ymin), (sx_f + half_w, ymax)))
    return faces


def sample_goal_on_shelf_face(
    rng: np.random.Generator,
    config: WarehouseConfig,
    *,
    offset: float = 0.0,
) -> np.ndarray:
    """Sample a goal point uniformly along the union of shelf left/right faces.

    Sampling is weighted by face length so each unit of perimeter is equally
    likely. ``offset`` shifts the goal outward from the face by that distance
    (positive = into the aisle); set to 0 to place on the face.
    """
    faces = shelf_face_segments(config)
    lengths = np.array([abs(p1[1] - p0[1]) for p0, p1 in faces], dtype=np.float64)
    total = float(lengths.sum())
    u = float(rng.uniform(0.0, total))
    cum = 0.0
    chosen_face = faces[-1]
    for face, length in zip(faces, lengths):
        cum += float(length)
        if u <= cum:
            chosen_face = face
            break
    p0, p1 = chosen_face
    y = float(rng.uniform(p0[1], p1[1]))
    x = float(p0[0])
    # Offset perpendicular to the face. Faces are vertical: positive offset
    # pushes outward (away from the shelf centre).
    half_w = float(config.shelf_width) / 2.0
    # Determine which side of the shelf this face is on by checking sign of
    # (face_x - shelf_centre_x) — we don't know the centre here, so infer
    # from any shelf with matching face: positive face_x relative to shelf
    # centre means right face; negative means left.
    direction = 0.0
    for sx in config.shelf_xs:
        if abs((float(sx) - half_w) - x) < 1e-6:
            direction = -1.0
            break
        if abs((float(sx) + half_w) - x) < 1e-6:
            direction = +1.0
            break
    x = x + direction * float(offset)
    return np.array([x, y], dtype=np.float32)


# ---------------------------------------------------------------------------
# Pickup cone & drop placement
# ---------------------------------------------------------------------------

def _wrap_to_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def in_pickup_cone(
    agent_pos: np.ndarray,
    agent_theta: float,
    object_pos: np.ndarray,
    config: WarehouseConfig,
) -> bool:
    """Return True if ``object_pos`` lies inside the agent's pickup cone."""
    dx = float(object_pos[0]) - float(agent_pos[0])
    dy = float(object_pos[1]) - float(agent_pos[1])
    dist = float(np.hypot(dx, dy))
    if dist > float(config.pickup_max_distance):
        return False
    if dist < 1e-8:
        return True
    angle_to_obj = float(np.arctan2(dy, dx))
    diff = _wrap_to_pi(angle_to_obj - float(agent_theta))
    return abs(diff) <= float(config.pickup_cone_half_angle)


def drop_position(
    agent_pos: np.ndarray,
    agent_theta: float,
    config: WarehouseConfig,
) -> np.ndarray:
    """Return the would-be drop position in front of the agent."""
    d = float(config.drop_distance)
    return np.array(
        [
            float(agent_pos[0]) + d * float(np.cos(agent_theta)),
            float(agent_pos[1]) + d * float(np.sin(agent_theta)),
        ],
        dtype=np.float32,
    )


def clamp_drop_position(
    drop_pt: np.ndarray,
    leader_side: str,
    config: WarehouseConfig,
) -> np.ndarray:
    """Clamp a drop point that lands inside a wall segment.

    If the drop point falls inside a wall segment (outside any gap, near
    ``wall_y``), snap it to the side the agent came from so the dropped object
    is recoverable. ``leader_side`` is ``"below"`` or ``"above"`` (relative to
    the wall).
    """
    obj_r = float(config.object_radius)
    if not point_in_any_wall_segment(drop_pt, config, margin=obj_r):
        return drop_pt
    out = drop_pt.copy().astype(np.float32)
    wy = float(config.wall_y)
    if leader_side == "below":
        out[1] = wy - obj_r - 1e-3
    else:
        out[1] = wy + obj_r + 1e-3
    return out


__all__ = [
    "WarehouseConfig",
    "agent_bounds",
    "agent_inscribed_radius",
    "clamp_drop_position",
    "drop_position",
    "enforce_agent_object_collision",
    "enforce_agent_object_collision_batch",
    "enforce_shelves",
    "enforce_shelves_batch",
    "enforce_wall",
    "enforce_wall_batch",
    "gap_centers",
    "gap_index_for_position",
    "gap_rects",
    "in_pickup_cone",
    "object_bounds",
    "point_in_any_shelf",
    "point_in_any_wall_segment",
    "sample_goal_on_shelf_face",
    "shelf_face_segments",
    "shelf_rects",
    "snap_to_wall_if_in_gap",
    "wall_segments",
]
