"""Matplotlib rendering helpers for the mobile_arm scenario.

Two projections of the 3D scene:

* **Top-down** (xy-plane): used by the leader env. World bounds, base
  square + heading arrow, object circle, optional desired-distance ring.
* **Side view** (rz-plane): used to render the arm in its work plane
  (the plane defined by ``θ₁``). Shoulder + upper arm + lower arm + EE,
  tray cross-section, object dot at ``z = 0``.

The follower env uses two-panel rendering: top-down + side view.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from critic_sharing.config.schema import _MobileArmBaseEnvConfig
from critic_sharing.scenarios.mobile_arm.common import (
    tray_pos_base,
)


COLOR_BASE = "#37474F"
COLOR_BASE_FILL = "#90A4AE"
COLOR_OBJECT = "#6A1B9A"
COLOR_OBJECT_OUTLINE = "#311B92"
COLOR_TRAY = "#FB8C00"
COLOR_ARM_UPPER = "#1565C0"
COLOR_ARM_LOWER = "#0D47A1"
COLOR_EE = "#43A047"
COLOR_DESIRED_RING = "#43A047"
COLOR_LEADER_TRAIL = "#FFB74D"


# ---------------------------------------------------------------------------
# Leader top-down (world frame)
# ---------------------------------------------------------------------------

def draw_leader_arena(
    ax,
    config: _MobileArmBaseEnvConfig,
    desired_distance: Optional[float] = None,
    sigma: Optional[float] = None,
) -> None:
    """Draw world bounds + (optional) desired-distance ring around origin."""
    import matplotlib.patches as mpatches

    ax.set_xlim(config.world_min, config.world_max)
    ax.set_ylim(config.world_min, config.world_max)
    ax.set_aspect("equal")
    ax.add_patch(
        mpatches.Rectangle(
            (config.world_min, config.world_min),
            config.world_max - config.world_min,
            config.world_max - config.world_min,
            fill=False,
            edgecolor=COLOR_BASE,
            linewidth=1.2,
            zorder=1,
        )
    )
    if desired_distance is not None and sigma is not None:
        # Lightly shaded annulus at d* ± σ around the origin (for the leader
        # arena, the actual object position varies — caller decides where to
        # centre the ring; we draw a placeholder at origin as a legend cue).
        pass


def draw_base_topdown(
    ax,
    base_pos: np.ndarray,
    base_yaw: float,
    config: _MobileArmBaseEnvConfig,
    color: str = COLOR_BASE_FILL,
) -> None:
    """Render the base as a rotated square + a forward-pointing heading arrow."""
    import matplotlib.patches as mpatches
    import matplotlib.transforms as mtransforms

    side = float(config.base_size)
    sq = mpatches.Rectangle(
        (-side / 2, -side / 2), side, side,
        facecolor=color, edgecolor=COLOR_BASE, linewidth=1.2, alpha=0.9, zorder=4,
    )
    tf = (
        mtransforms.Affine2D()
        .rotate(float(base_yaw))
        .translate(float(base_pos[0]), float(base_pos[1]))
    )
    sq.set_transform(tf + ax.transData)
    ax.add_patch(sq)

    # Heading arrow.
    arrow_len = side * 0.7
    x0 = float(base_pos[0])
    y0 = float(base_pos[1])
    x1 = x0 + arrow_len * float(np.cos(base_yaw))
    y1 = y0 + arrow_len * float(np.sin(base_yaw))
    ax.plot([x0, x1], [y0, y1], color=COLOR_BASE, linewidth=1.5, zorder=5)


def draw_object_topdown(
    ax,
    object_pos: np.ndarray,
    config: _MobileArmBaseEnvConfig,
) -> None:
    """Render the object at its xy position."""
    import matplotlib.patches as mpatches

    ax.add_patch(
        mpatches.Circle(
            (float(object_pos[0]), float(object_pos[1])),
            float(config.object_radius),
            facecolor=COLOR_OBJECT,
            edgecolor=COLOR_OBJECT_OUTLINE,
            linewidth=1.0,
            zorder=6,
        )
    )


def draw_desired_distance_ring(
    ax,
    object_pos: np.ndarray,
    desired_distance: float,
    sigma: float,
    color: str = COLOR_DESIRED_RING,
) -> None:
    """Annulus at ``d* ± σ`` around the object — the leader's "good park" zone."""
    import matplotlib.patches as mpatches

    ox = float(object_pos[0])
    oy = float(object_pos[1])
    # Inner faint disk + outer ring for the σ-band.
    ax.add_patch(
        mpatches.Circle(
            (ox, oy), desired_distance,
            fill=False, edgecolor=color, linewidth=1.5, linestyle="--", alpha=0.85, zorder=3,
        )
    )
    ax.add_patch(
        mpatches.Circle(
            (ox, oy), desired_distance + sigma,
            fill=False, edgecolor=color, linewidth=0.8, linestyle=":", alpha=0.6, zorder=3,
        )
    )
    ax.add_patch(
        mpatches.Circle(
            (ox, oy), max(0.01, desired_distance - sigma),
            fill=False, edgecolor=color, linewidth=0.8, linestyle=":", alpha=0.6, zorder=3,
        )
    )


# ---------------------------------------------------------------------------
# Follower top-down (base frame; arm projected to xy)
# ---------------------------------------------------------------------------

def draw_follower_topdown(
    ax,
    theta: np.ndarray,
    object_pos_base: np.ndarray,
    carry: bool,
    config: _MobileArmBaseEnvConfig,
) -> None:
    """Top-down view of the arm in base frame + object/tray.

    Draws all 3 segments (upper arm, L2a, L2b) projected onto the xy plane.
    Joint sign convention: positive bend forward/down; cumulative angles
    from horizontal compound subtractively.
    """
    import matplotlib.patches as mpatches

    L1 = float(config.L1)
    L2a = float(config.L2a)
    L2b = float(config.L2b)
    reach = L1 + L2a + L2b
    ax.set_xlim(-reach * 1.2, reach * 1.2)
    ax.set_ylim(-reach * 1.2, reach * 1.2)
    ax.set_aspect("equal")

    # Base footprint at origin (yaw=0 in base frame).
    side = float(config.base_size)
    ax.add_patch(
        mpatches.Rectangle(
            (-side / 2, -side / 2), side, side,
            facecolor=COLOR_BASE_FILL, edgecolor=COLOR_BASE,
            linewidth=1.0, alpha=0.5, zorder=2,
        )
    )

    # Tray (top-down circle)
    tray = tray_pos_base(config)
    ax.add_patch(
        mpatches.Circle(
            (float(tray[0]), float(tray[1])), 0.06,
            facecolor=COLOR_TRAY, edgecolor="#E65100", linewidth=1.0, zorder=4,
        )
    )

    # Arm projected to xy. Cumulative angles from horizontal.
    theta1 = float(theta[0])
    theta2 = float(theta[1])
    theta3 = float(theta[2])
    theta4 = float(theta[3])
    phi_upper = theta2
    phi_L2a = theta2 - theta3
    phi_L2b = theta2 - theta3 - theta4

    elbow_r = L1 * np.cos(phi_upper)
    wrist_r = elbow_r + L2a * np.cos(phi_L2a)
    ee_r = wrist_r + L2b * np.cos(phi_L2b)
    elbow_x = elbow_r * np.cos(theta1); elbow_y = elbow_r * np.sin(theta1)
    wrist_x = wrist_r * np.cos(theta1); wrist_y = wrist_r * np.sin(theta1)
    ee_x = ee_r * np.cos(theta1);       ee_y = ee_r * np.sin(theta1)

    ax.plot([0, elbow_x], [0, elbow_y], color=COLOR_ARM_UPPER, linewidth=3.5, zorder=5)
    ax.plot([elbow_x, wrist_x], [elbow_y, wrist_y], color=COLOR_ARM_LOWER, linewidth=3.5, zorder=5)
    ax.plot([wrist_x, ee_x], [wrist_y, ee_y], color=COLOR_ARM_LOWER, linewidth=3.5, alpha=0.85, zorder=5)
    ax.plot(0, 0, marker="o", markersize=8, color=COLOR_ARM_UPPER, zorder=6)
    ax.plot(elbow_x, elbow_y, marker="o", markersize=6, color=COLOR_ARM_LOWER, zorder=6)
    ax.plot(wrist_x, wrist_y, marker="o", markersize=5, color=COLOR_ARM_LOWER, zorder=6)
    ax.plot(ee_x, ee_y, marker="*", markersize=12, color=COLOR_EE,
            markeredgecolor="#1B5E20", zorder=7)

    # Object (xy projection).
    obj_color = COLOR_EE if carry else COLOR_OBJECT
    ax.add_patch(
        mpatches.Circle(
            (float(object_pos_base[0]), float(object_pos_base[1])),
            float(config.object_radius),
            facecolor=obj_color, edgecolor="#311B92", linewidth=1.0, zorder=6,
        )
    )

    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x (base frame)")
    ax.set_ylabel("y (base frame)")


# ---------------------------------------------------------------------------
# Follower side view (rz; arm in its work plane)
# ---------------------------------------------------------------------------

def draw_follower_side(
    ax,
    theta: np.ndarray,
    object_pos_base: np.ndarray,
    carry: bool,
    config: _MobileArmBaseEnvConfig,
) -> None:
    """Side view (rz) of the arm in its work plane.

    Draws all 3 segments (upper arm, L2a, L2b) in the radial-z half-plane
    defined by ``θ_1``. Subtractive joint convention.
    """
    import matplotlib.patches as mpatches

    L1 = float(config.L1)
    L2a = float(config.L2a)
    L2b = float(config.L2b)
    reach = L1 + L2a + L2b
    base_height = float(config.base_height)
    ax.set_xlim(-reach * 0.25, reach * 1.2)
    ax.set_ylim(-0.2, base_height + reach * 1.1)
    ax.set_aspect("equal")

    # Ground line.
    ax.axhline(0.0, color="#37474F", linewidth=1.5, zorder=1)
    # Base profile (rectangle from x=-half to half, z=0..base_height).
    side = float(config.base_size)
    ax.add_patch(
        mpatches.Rectangle(
            (-side / 2, 0), side, base_height,
            facecolor=COLOR_BASE_FILL, edgecolor=COLOR_BASE,
            linewidth=1.0, alpha=0.6, zorder=2,
        )
    )

    # Tray (side view).
    tray = tray_pos_base(config)
    tray_r = float(np.hypot(tray[0], tray[1]))
    tray_z = float(tray[2])
    ax.plot([tray_r - 0.08, tray_r + 0.08], [tray_z, tray_z],
            color=COLOR_TRAY, linewidth=4.0, solid_capstyle="butt", zorder=4)

    # Arm in work plane. Subtractive cumulative angles.
    theta2 = float(theta[1])
    theta3 = float(theta[2])
    theta4 = float(theta[3])
    phi_upper = theta2
    phi_L2a = theta2 - theta3
    phi_L2b = theta2 - theta3 - theta4
    elbow_r = L1 * np.cos(phi_upper)
    elbow_z = base_height + L1 * np.sin(phi_upper)
    wrist_r = elbow_r + L2a * np.cos(phi_L2a)
    wrist_z = elbow_z + L2a * np.sin(phi_L2a)
    ee_r = wrist_r + L2b * np.cos(phi_L2b)
    ee_z = wrist_z + L2b * np.sin(phi_L2b)

    ax.plot([0, elbow_r], [base_height, elbow_z], color=COLOR_ARM_UPPER, linewidth=3.5, zorder=5)
    ax.plot([elbow_r, wrist_r], [elbow_z, wrist_z], color=COLOR_ARM_LOWER, linewidth=3.5, zorder=5)
    ax.plot([wrist_r, ee_r], [wrist_z, ee_z], color=COLOR_ARM_LOWER, linewidth=3.5, alpha=0.85, zorder=5)
    ax.plot(0, base_height, marker="o", markersize=8, color=COLOR_ARM_UPPER, zorder=6)
    ax.plot(elbow_r, elbow_z, marker="o", markersize=6, color=COLOR_ARM_LOWER, zorder=6)
    ax.plot(wrist_r, wrist_z, marker="o", markersize=5, color=COLOR_ARM_LOWER, zorder=6)
    ax.plot(ee_r, ee_z, marker="*", markersize=12, color=COLOR_EE,
            markeredgecolor="#1B5E20", zorder=7)

    # Object in side view: at (radial dist from base in arm plane, z).
    obj_r = float(np.hypot(float(object_pos_base[0]), float(object_pos_base[1])))
    obj_z = float(object_pos_base[2])
    obj_color = COLOR_EE if carry else COLOR_OBJECT
    ax.add_patch(
        mpatches.Circle(
            (obj_r, obj_z), float(config.object_radius),
            facecolor=obj_color, edgecolor="#311B92", linewidth=1.0, zorder=6,
        )
    )

    ax.grid(True, alpha=0.3)
    ax.set_xlabel("r (radial in arm plane)")
    ax.set_ylabel("z")


__all__ = [
    "COLOR_BASE",
    "COLOR_BASE_FILL",
    "COLOR_EE",
    "COLOR_LEADER_TRAIL",
    "COLOR_OBJECT",
    "COLOR_TRAY",
    "draw_base_topdown",
    "draw_desired_distance_ring",
    "draw_follower_side",
    "draw_follower_topdown",
    "draw_leader_arena",
    "draw_object_topdown",
]
