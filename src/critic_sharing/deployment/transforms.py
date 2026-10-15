"""Static SE(2) transform between the real-world parent frame and the sim frame.

Pure NumPy — no ROS dependency, so this module is unit-testable without rclpy.

Convention
----------
``StaticSE2(tx, ty, theta)`` is the pose of the **simulation arena origin
expressed in the real-world parent frame**. Both frames are assumed
right-handed with z pointing up (so positive yaw is counter-clockwise
when viewed from above). If your parent frame is left-handed (e.g.
top-down image coordinates) or NED (z pointing down), apply your own
sign flip before passing positions/yaws into this transform.

Forward (sim → world)::

    p_world = R(theta) @ p_sim + (tx, ty)
    v_world = R(theta) @ v_sim
    yaw_world = yaw_sim + theta

Inverse (world → sim) — what the deployment node uses each tick::

    p_sim  = R(-theta) @ (p_world - (tx, ty))
    yaw_sim = yaw_world - theta
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _wrap_to_pi(angle: float) -> float:
    return float(((angle + np.pi) % (2.0 * np.pi)) - np.pi)


@dataclass(frozen=True)
class StaticSE2:
    """Static planar transform: sim arena origin expressed in real-world frame."""

    tx: float = 0.0
    ty: float = 0.0
    theta: float = 0.0

    def _rotation(self, angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s], [s, c]], dtype=np.float64)

    def world_to_sim_position(self, p_xy: np.ndarray) -> np.ndarray:
        """Map a real-world (x, y) point into sim-frame coordinates."""
        p = np.asarray(p_xy, dtype=np.float64).reshape(2)
        translated = p - np.array([self.tx, self.ty], dtype=np.float64)
        return self._rotation(-self.theta) @ translated

    def sim_to_world_velocity(self, v_xy: np.ndarray) -> np.ndarray:
        """Rotate a sim-frame velocity into the real-world parent frame."""
        v = np.asarray(v_xy, dtype=np.float64).reshape(2)
        return self._rotation(self.theta) @ v

    def world_to_sim_yaw(self, yaw_world: float) -> float:
        """Convert a yaw from the world (parent) frame to the sim frame."""
        return _wrap_to_pi(float(yaw_world) - self.theta)

    def sim_to_world_yaw(self, yaw_sim: float) -> float:
        """Convert a yaw from the sim frame back to the world (parent) frame."""
        return _wrap_to_pi(float(yaw_sim) + self.theta)


__all__ = ["StaticSE2"]
