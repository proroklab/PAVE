"""Shared geometry helpers for the mobile_arm scenario.

Two coordinate frames matter:

* **World frame** — used by the leader env. The mobile base lives in 2D
  on the ground plane; the object also lives at ``z = 0`` here.
* **Base frame** — used by the follower env. Origin at the base centre,
  x-axis along the base's forward direction, z-axis up. The arm is
  mounted at ``(0, 0, base_height)`` in this frame.

The arm is a planar 4-DOF chain. Sign convention: each ``θ_i`` is a
*positive bend forward/down*. Cumulative angles from horizontal compound
**subtractively**::

    φ_upper = θ_2
    φ_L2a   = θ_2 − θ_3
    φ_L2b   = θ_2 − θ_3 − θ_4

Forward kinematics in base frame::

    r_planar = L1·cos(φ_upper) + L2a·cos(φ_L2a) + L2b·cos(φ_L2b)
    z_planar = L1·sin(φ_upper) + L2a·sin(φ_L2a) + L2b·sin(φ_L2b)
    ee_x_base =  r_planar · cos(θ_1)
    ee_y_base =  r_planar · sin(θ_1)
    ee_z      =  base_height + z_planar

The arm has no self-collision and no ground/base collision — joint
limits are the only kinematic constraints.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from critic_sharing.config.schema import _MobileArmBaseEnvConfig

MobileArmConfig = _MobileArmBaseEnvConfig


# ---------------------------------------------------------------------------
# Basic accessors
# ---------------------------------------------------------------------------

def base_inscribed_radius(config: MobileArmConfig) -> float:
    """Half of ``base_size`` — the inscribed-circle radius for collision."""
    return float(config.base_size) / 2.0


def base_world_bounds(config: MobileArmConfig) -> Tuple[float, float]:
    r = base_inscribed_radius(config)
    return float(config.world_min) + r, float(config.world_max) - r


def tray_pos_base(config: MobileArmConfig) -> np.ndarray:
    """Tray location in the follower's base-frame coords.

    ``tray_offset_z`` is the height *above the chassis top* (i.e., above
    ``base_height``). The returned z is the absolute z (matching FK's
    ``ee_z = base_height + z_planar`` convention).
    """
    return np.array(
        [
            config.tray_offset_x,
            config.tray_offset_y,
            float(config.base_height) + float(config.tray_offset_z),
        ],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Forward kinematics
# ---------------------------------------------------------------------------

def forward_kinematics(
    theta1: float,
    theta2: float,
    theta3: float,
    theta4: float,
    config: MobileArmConfig,
) -> np.ndarray:
    """Return the EE position ``(x, y, z)`` in **base frame**.

    See module-level docstring for the joint-sign convention. With
    ``θ_2 = π/2`` the upper arm points straight up; positive ``θ_3, θ_4``
    bend each subsequent segment forward/down relative to its parent.
    """
    L1 = float(config.L1)
    L2a = float(config.L2a)
    L2b = float(config.L2b)
    phi_upper = theta2
    phi_L2a = theta2 - theta3
    phi_L2b = theta2 - theta3 - theta4
    r_planar = (
        L1 * np.cos(phi_upper)
        + L2a * np.cos(phi_L2a)
        + L2b * np.cos(phi_L2b)
    )
    z_planar = (
        L1 * np.sin(phi_upper)
        + L2a * np.sin(phi_L2a)
        + L2b * np.sin(phi_L2b)
    )
    return np.array(
        [
            r_planar * np.cos(theta1),
            r_planar * np.sin(theta1),
            float(config.base_height) + z_planar,
        ],
        dtype=np.float32,
    )


def forward_kinematics_batch(
    theta: np.ndarray,
    config: MobileArmConfig,
) -> np.ndarray:
    """Vectorised :func:`forward_kinematics`. Input ``(B, 4)`` joints, output ``(B, 3)`` EE."""
    theta = np.asarray(theta, dtype=np.float32)
    if theta.ndim != 2 or theta.shape[1] != 4:
        raise ValueError("theta must be (B, 4)")
    L1 = float(config.L1)
    L2a = float(config.L2a)
    L2b = float(config.L2b)
    t1, t2, t3, t4 = theta[:, 0], theta[:, 1], theta[:, 2], theta[:, 3]
    phi_upper = t2
    phi_L2a = t2 - t3
    phi_L2b = t2 - t3 - t4
    r_planar = (
        L1 * np.cos(phi_upper)
        + L2a * np.cos(phi_L2a)
        + L2b * np.cos(phi_L2b)
    )
    z_planar = (
        L1 * np.sin(phi_upper)
        + L2a * np.sin(phi_L2a)
        + L2b * np.sin(phi_L2b)
    )
    out = np.empty((theta.shape[0], 3), dtype=np.float32)
    out[:, 0] = r_planar * np.cos(t1)
    out[:, 1] = r_planar * np.sin(t1)
    out[:, 2] = float(config.base_height) + z_planar
    return out


# ---------------------------------------------------------------------------
# Joint endpoints (in base frame)
# ---------------------------------------------------------------------------

def joint_endpoints(
    theta1: float,
    theta2: float,
    theta3: float,
    theta4: float,
    config: MobileArmConfig,
) -> np.ndarray:
    """Return ``(4, 3)`` array of (shoulder, elbow, wrist, EE) in base frame.

    ``shoulder = (0, 0, base_height)``; the other three follow the FK chain.
    Used by the arm-collision check: the linear segments between consecutive
    endpoints are the arm's physical extent.
    """
    L1 = float(config.L1)
    L2a = float(config.L2a)
    L2b = float(config.L2b)
    base_h = float(config.base_height)
    phi_upper = theta2
    phi_L2a = theta2 - theta3
    phi_L2b = theta2 - theta3 - theta4
    cos_yaw = np.cos(theta1)
    sin_yaw = np.sin(theta1)

    elbow_r = L1 * np.cos(phi_upper)
    wrist_r = elbow_r + L2a * np.cos(phi_L2a)
    ee_r = wrist_r + L2b * np.cos(phi_L2b)
    elbow_z = base_h + L1 * np.sin(phi_upper)
    wrist_z = elbow_z + L2a * np.sin(phi_L2a)
    ee_z = wrist_z + L2b * np.sin(phi_L2b)

    return np.array(
        [
            [0.0, 0.0, base_h],                                  # shoulder
            [elbow_r * cos_yaw, elbow_r * sin_yaw, elbow_z],     # elbow
            [wrist_r * cos_yaw, wrist_r * sin_yaw, wrist_z],     # wrist
            [ee_r * cos_yaw,    ee_r * sin_yaw,    ee_z],        # EE
        ],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Arm collision (against the floor and the base)
# ---------------------------------------------------------------------------

def _point_inside_base_box(point: np.ndarray, config: MobileArmConfig) -> bool:
    half = float(config.base_size) / 2.0
    base_h = float(config.base_height)
    return bool(
        -half <= float(point[0]) <= half
        and -half <= float(point[1]) <= half
        and float(point[2]) < base_h
    )


def arm_collides(
    theta: np.ndarray,
    config: MobileArmConfig,
) -> Tuple[bool, bool]:
    """Endpoint-only arm-vs-environment collision check.

    Returns ``(floor_collision, base_collision)`` — whether any of
    ``(elbow, wrist, EE)`` is below the floor (``z < 0``) or inside the
    chassis box (xy in footprint AND ``z < base_height``).

    Shoulder is fixed at ``(0, 0, base_height)`` and never collides — skipped.
    """
    pts = joint_endpoints(
        float(theta[0]), float(theta[1]), float(theta[2]), float(theta[3]),
        config,
    )
    floor_hit = False
    base_hit = False
    # Skip index 0 (shoulder) — fixed at the corner of the base.
    for i in (1, 2, 3):
        p = pts[i]
        if float(p[2]) < 0.0:
            floor_hit = True
        if _point_inside_base_box(p, config):
            base_hit = True
    return floor_hit, base_hit


# ---------------------------------------------------------------------------
# Frame transforms
# ---------------------------------------------------------------------------

def world_to_base(
    point_world: np.ndarray,
    base_pos: np.ndarray,
    base_yaw: float,
) -> np.ndarray:
    """Transform a 2D or 3D point from world frame to the base's local frame.

    ``base_pos`` is the base centre in world frame (2-vector). ``base_yaw``
    is the base's orientation in world frame. Returns a ``(2,)`` or ``(3,)``
    array matching the input.
    """
    p = np.asarray(point_world, dtype=np.float32)
    bp = np.asarray(base_pos, dtype=np.float32)
    dx = float(p[0]) - float(bp[0])
    dy = float(p[1]) - float(bp[1])
    cy = float(np.cos(base_yaw))
    sy = float(np.sin(base_yaw))
    px = dx * cy + dy * sy
    py = -dx * sy + dy * cy
    if p.shape[0] == 3:
        return np.array([px, py, float(p[2])], dtype=np.float32)
    return np.array([px, py], dtype=np.float32)


def world_to_base_batch(
    points_world: np.ndarray,
    base_pos: np.ndarray,
    base_yaw: np.ndarray,
) -> np.ndarray:
    """Vectorised :func:`world_to_base`. ``points_world`` is ``(B, 2)`` or ``(B, 3)``;
    ``base_pos`` is ``(B, 2)``; ``base_yaw`` is ``(B,)``. Returns ``(B, ...)``.
    """
    p = np.asarray(points_world, dtype=np.float32)
    bp = np.asarray(base_pos, dtype=np.float32)
    yaw = np.asarray(base_yaw, dtype=np.float32)
    dx = p[:, 0] - bp[:, 0]
    dy = p[:, 1] - bp[:, 1]
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    out = np.empty_like(p)
    out[:, 0] = dx * cy + dy * sy
    out[:, 1] = -dx * sy + dy * cy
    if p.shape[1] == 3:
        out[:, 2] = p[:, 2]
    return out


def base_to_world(
    point_base: np.ndarray,
    base_pos: np.ndarray,
    base_yaw: float,
) -> np.ndarray:
    """Inverse of :func:`world_to_base`."""
    p = np.asarray(point_base, dtype=np.float32)
    bp = np.asarray(base_pos, dtype=np.float32)
    cy = float(np.cos(base_yaw))
    sy = float(np.sin(base_yaw))
    px = float(p[0]) * cy - float(p[1]) * sy + float(bp[0])
    py = float(p[0]) * sy + float(p[1]) * cy + float(bp[1])
    if p.shape[0] == 3:
        return np.array([px, py, float(p[2])], dtype=np.float32)
    return np.array([px, py], dtype=np.float32)


# ---------------------------------------------------------------------------
# Base ↔ object collision (used by the leader env)
# ---------------------------------------------------------------------------

def enforce_base_object_collision(
    next_pos: np.ndarray,
    object_pos: np.ndarray,
    config: MobileArmConfig,
) -> np.ndarray:
    """Clamp ``next_pos`` so the base never overlaps the object.

    Uses the inscribed-circle approximation for the square base. If the
    proposed position would put the base centre within
    ``base_radius + object_radius`` of the object, push the centre out along
    the line from the object back to the proposed position.
    """
    p = np.asarray(next_pos, dtype=np.float32).copy()
    o = np.asarray(object_pos, dtype=np.float32)
    contact = base_inscribed_radius(config) + float(config.object_radius)
    diff = p - o
    dist = float(np.linalg.norm(diff))
    if dist >= contact:
        return p
    if dist < 1e-8:
        # Degenerate — push along +x.
        diff = np.array([1.0, 0.0], dtype=np.float32)
        dist = 1.0
    return (o + diff / dist * contact).astype(np.float32)


def enforce_base_object_collision_batch(
    next_pos: np.ndarray,
    object_pos: np.ndarray,
    config: MobileArmConfig,
) -> np.ndarray:
    """Vectorised :func:`enforce_base_object_collision` for shape ``(B, 2)``."""
    p = np.asarray(next_pos, dtype=np.float32).copy()
    o = np.asarray(object_pos, dtype=np.float32)
    if o.ndim == 1:
        o = np.broadcast_to(o, p.shape)
    contact = base_inscribed_radius(config) + float(config.object_radius)
    diff = p - o
    dist = np.linalg.norm(diff, axis=1)
    mask = dist < contact
    if not np.any(mask):
        return p
    safe_dist = np.where(dist > 1e-8, dist, 1.0)
    direction = diff / safe_dist[:, None]
    # Replace zero-norm rows with +x direction.
    direction = np.where(
        (dist > 1e-8)[:, None],
        direction,
        np.array([1.0, 0.0], dtype=np.float32),
    )
    p[mask] = (o[mask] + direction[mask] * contact).astype(np.float32)
    return p


# ---------------------------------------------------------------------------
# Object spawn helpers (follower env)
# ---------------------------------------------------------------------------

def sample_object_in_base_frame(
    rng: np.random.Generator,
    config: MobileArmConfig,
    r_min: float,
    r_max: float,
    phi_low: float = -np.pi,
    phi_high: float = np.pi,
) -> np.ndarray:
    """Sample object position in base-frame polar coords ``(r, φ, 0)``.

    ``phi`` is sampled uniformly in ``[phi_low, phi_high]`` — the spawn
    cone in front of the base, with ``φ = 0`` along base +x toward the
    tray. Defaults to the full circle for back-compat; the follower env
    passes ``config.object_theta_low/high`` so the cone is configurable
    per-training run.
    """
    r = float(rng.uniform(r_min, r_max))
    phi = float(rng.uniform(float(phi_low), float(phi_high)))
    return np.array(
        [r * np.cos(phi), r * np.sin(phi), 0.0],
        dtype=np.float32,
    )


__all__ = [
    "MobileArmConfig",
    "arm_collides",
    "base_inscribed_radius",
    "base_to_world",
    "base_world_bounds",
    "enforce_base_object_collision",
    "enforce_base_object_collision_batch",
    "forward_kinematics",
    "forward_kinematics_batch",
    "joint_endpoints",
    "sample_object_in_base_frame",
    "tray_pos_base",
    "world_to_base",
    "world_to_base_batch",
]
