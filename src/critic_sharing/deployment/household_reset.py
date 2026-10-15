"""Auto-reset controller: drive both robots to their initial-condition poses.

Used between methods within a pair in the comparison harness. The
operator manually re-introduces the cleaner (it was removed for the
follower rollout), then this controller fine-tunes both bodies back
to the alpha-ghost positions captured at the start of the pair using
a simple per-robot P controller publishing ``freyja/ReferenceState``.

Convergence: both robots within tolerance (5 cm / 5° by default) for
``hold_ticks`` consecutive ticks. Timeout: 30 s by default, after which
the orchestrator prompts the operator to redo / skip / abort.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener

from critic_sharing.config.schema import HouseholdDeploymentConfig
from critic_sharing.deployment.household_node import _shutdown_tf_listener
from critic_sharing.deployment.transforms import StaticSE2


def _yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    return float(np.arctan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    ))


def _wrap_to_pi(a: float) -> float:
    return float(((a + np.pi) % (2.0 * np.pi)) - np.pi)


# ---------------------------------------------------------------------------
# Per-robot tracking state
# ---------------------------------------------------------------------------

@dataclass
class _RobotState:
    """Mutable scratch for one of the two robots."""

    name: str           # "cleaner" or "follower"
    frame: str
    topic: str
    target_xy: np.ndarray   # sim-frame target position
    target_yaw: float       # sim-frame target yaw
    current_xy: Optional[np.ndarray] = None
    current_yaw: Optional[float] = None
    last_cmd_sim: np.ndarray = None  # (vx, vy, omega) in sim frame, for arrows
    yaw_setpoint: Optional[float] = None  # integrated world-frame yaw
    in_tolerance_ticks: int = 0


# ---------------------------------------------------------------------------
# Reset node
# ---------------------------------------------------------------------------

class PositionResetNode(Node):
    """Two-robot P controller publishing ``freyja/ReferenceState``."""

    def __init__(
        self,
        config: HouseholdDeploymentConfig,
        ghosts: dict,
        k_xy: float = 1.0,
        k_yaw: float = 1.5,
        pos_tol: float = 0.05,
        yaw_tol: float = float(np.deg2rad(5.0)),
        hold_ticks: int = 10,
    ):
        super().__init__("household_position_reset")
        self._config = config
        self.k_xy = float(k_xy)
        self.k_yaw = float(k_yaw)
        self.pos_tol = float(pos_tol)
        self.yaw_tol = float(yaw_tol)
        self.hold_ticks = int(hold_ticks)

        tx, ty, yaw_parent = config.sim_origin_in_world
        yaw_zup = -float(yaw_parent) if config.parent_z_down else float(yaw_parent)
        self._transform = StaticSE2(tx=float(tx), ty=float(ty), theta=yaw_zup)
        self._parent_z_down = bool(config.parent_z_down)
        self._dt = 1.0 / float(config.rate_hz)
        self._max_speed = float(config.max_speed)
        self._max_yaw_rate = float(config.max_yaw_rate)

        self._tf_buffer = Buffer()
        # Dedicated thread for TF so matplotlib in the orchestrator can't
        # starve the listener.
        self._tf_listener = TransformListener(
            self._tf_buffer, self, spin_thread=True,
        )
        self._tf_timeout = Duration(seconds=float(config.tf_timeout_sec))

        from freyja_msgs.msg import ReferenceState
        self._ReferenceState = ReferenceState
        self._cleaner_pub = self.create_publisher(
            ReferenceState, f"/{config.cleaner_frame}/reference_state", 10,
        )
        self._follower_pub = self.create_publisher(
            ReferenceState, f"/{config.follower_frame}/reference_state", 10,
        )

        self.cleaner = _RobotState(
            name="cleaner",
            frame=config.cleaner_frame,
            topic=f"/{config.cleaner_frame}/reference_state",
            target_xy=np.asarray(ghosts["cleaner_xy_sim"], dtype=np.float64),
            target_yaw=float(ghosts["cleaner_yaw_sim"]),
            last_cmd_sim=np.zeros(3, dtype=np.float32),
        )
        self.follower = _RobotState(
            name="follower",
            frame=config.follower_frame,
            topic=f"/{config.follower_frame}/reference_state",
            target_xy=np.asarray(ghosts["follower_xy_sim"], dtype=np.float64),
            target_yaw=float(ghosts["follower_yaw_sim"]),
            last_cmd_sim=np.zeros(3, dtype=np.float32),
        )

        self._timer = self.create_timer(self._dt, self._tick)
        self._converged = False

    def destroy_node(self) -> None:
        # Drain the TF listener's dedicated spin thread before the global
        # context is torn down — avoids ExternalShutdownException.
        _shutdown_tf_listener(getattr(self, "_tf_listener", None))
        super().destroy_node()

    # ----- public ----------------------------------------------------------

    @property
    def converged(self) -> bool:
        return self._converged

    @property
    def robots(self) -> tuple:
        return (self.cleaner, self.follower)

    def stop_publish_zero_vel(self, n: int = 5) -> None:
        """End-of-reset zero-vel flurry on both topics."""
        for _ in range(n):
            self._publish_zero(self.cleaner)
            self._publish_zero(self.follower)
            time.sleep(self._dt)

    # ----- per-tick --------------------------------------------------------

    def _tick(self) -> None:
        self._update_robot(self.cleaner, self._cleaner_pub)
        self._update_robot(self.follower, self._follower_pub)
        self._converged = (
            self.cleaner.in_tolerance_ticks >= self.hold_ticks
            and self.follower.in_tolerance_ticks >= self.hold_ticks
        )

    def _update_robot(self, robot: _RobotState, publisher) -> None:
        pose = self._lookup_sim_pose(robot.frame)
        if pose is None:
            # No TF → can't drive this robot. Reset its hold counter.
            robot.in_tolerance_ticks = 0
            return
        robot.current_xy, robot.current_yaw = pose

        err_xy = robot.target_xy - robot.current_xy
        err_yaw = _wrap_to_pi(robot.target_yaw - float(robot.current_yaw))
        err_norm = float(np.linalg.norm(err_xy))

        if err_norm <= self.pos_tol and abs(err_yaw) <= self.yaw_tol:
            robot.in_tolerance_ticks += 1
            self._publish_zero(robot, publisher)
            return
        robot.in_tolerance_ticks = 0

        v_xy_sim = np.clip(
            self.k_xy * err_xy.astype(np.float32),
            -self._max_speed, self._max_speed,
        )
        omega_sim = float(np.clip(
            self.k_yaw * err_yaw, -self._max_yaw_rate, self._max_yaw_rate,
        ))
        robot.last_cmd_sim = np.array(
            [float(v_xy_sim[0]), float(v_xy_sim[1]), omega_sim], dtype=np.float32,
        )

        self._publish_velocity(robot, publisher, v_xy_sim, omega_sim)

    # ----- helpers ---------------------------------------------------------

    def _lookup_sim_pose(self, frame: str):
        try:
            tf = self._tf_buffer.lookup_transform(
                self._config.parent_frame, frame, Time(), self._tf_timeout,
            )
        except TransformException:
            return None
        t = tf.transform.translation
        r = tf.transform.rotation
        xy_world = np.array([t.x, t.y], dtype=np.float64)
        yaw_world = _yaw_from_quat(r.x, r.y, r.z, r.w)
        xy_sim = self._transform.world_to_sim_position(xy_world)
        yaw_zup = -yaw_world if self._parent_z_down else yaw_world
        yaw_sim = self._transform.world_to_sim_yaw(yaw_zup)
        return xy_sim, yaw_sim

    def _publish_velocity(self, robot: _RobotState, publisher, v_xy_sim, omega_sim) -> None:
        v_world = self._transform.sim_to_world_velocity(
            np.asarray(v_xy_sim, dtype=np.float32)
        )
        omega_world = -float(omega_sim) if self._parent_z_down else float(omega_sim)
        if robot.yaw_setpoint is None:
            # Seed the yaw setpoint from the current measured yaw the
            # first time we publish to this robot so the freyja layer
            # has a smooth reference to track. ``current_yaw`` is in
            # sim frame; convert back to the parent (world) frame —
            # apply the z-down sign flip *after* the sim→world rotation
            # (mirror of the inbound conversion in _lookup_sim_pose).
            if robot.current_yaw is None:
                return
            yaw_zup_world = self._transform.sim_to_world_yaw(float(robot.current_yaw))
            yaw_world_now = -yaw_zup_world if self._parent_z_down else yaw_zup_world
            robot.yaw_setpoint = float(yaw_world_now)
        robot.yaw_setpoint = _wrap_to_pi(
            robot.yaw_setpoint + omega_world * self._dt
        )
        msg = self._ReferenceState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._config.parent_frame
        msg.vn = float(v_world[1])
        msg.ve = float(v_world[0])
        msg.yaw = float(robot.yaw_setpoint)
        publisher.publish(msg)

    def _publish_zero(self, robot: _RobotState, publisher=None) -> None:
        if publisher is None:
            publisher = (self._cleaner_pub if robot.name == "cleaner"
                         else self._follower_pub)
        msg = self._ReferenceState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._config.parent_frame
        msg.vn = 0.0
        msg.ve = 0.0
        msg.yaw = float(robot.yaw_setpoint if robot.yaw_setpoint is not None else 0.0)
        publisher.publish(msg)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def reset_to_ghosts(
    config: HouseholdDeploymentConfig,
    ghosts: dict,
    *,
    pos_tol: float = 0.05,
    yaw_tol_deg: float = 5.0,
    hold_ticks: int = 10,
    timeout: float = 30.0,
    show_viz: bool = True,
    redraw_fps: float = 5.0,
) -> str:
    """Drive both robots to ``ghosts`` poses. Returns one of:
    ``"converged"``, ``"timeout"``, ``"keyboard_interrupt"``.

    The matplotlib viz (ghosts + live TF + commanded-velocity arrows) is
    shown by default. Pass ``show_viz=False`` for headless runs.
    """
    owns_rclpy = not rclpy.ok()
    if owns_rclpy:
        rclpy.init()

    node = PositionResetNode(
        config=config, ghosts=ghosts,
        pos_tol=pos_tol, yaw_tol=float(np.deg2rad(yaw_tol_deg)),
        hold_ticks=hold_ticks,
    )

    fig = None
    ax = None
    if show_viz:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        plt.ion()
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 5.5))
        fig.canvas.manager.set_window_title("household — auto-reset")

    print(
        f"[reset] Driving cleaner+follower to ghosts "
        f"(tol={pos_tol*100:.0f}cm/{yaw_tol_deg:.0f}°, "
        f"hold={hold_ticks} ticks, timeout={timeout:.0f}s)...",
        flush=True,
    )
    started = time.monotonic()
    last_log = started
    spin_timeout = 1.0 / float(redraw_fps)
    result = "timeout"
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=spin_timeout)
            if show_viz and fig is not None:
                _draw_reset(ax, config, ghosts, node)
                import matplotlib.pyplot as plt
                fig.canvas.draw_idle()
                plt.pause(0.001)
                if not plt.fignum_exists(fig.number):
                    # Operator closed the window — treat as keyboard interrupt.
                    raise KeyboardInterrupt
            now = time.monotonic()
            if now - last_log >= 1.0:
                _log_status(node)
                last_log = now
            if node.converged:
                result = "converged"
                break
            if now - started >= timeout:
                result = "timeout"
                break
    except KeyboardInterrupt:
        result = "keyboard_interrupt"
    finally:
        # Stop motion regardless of outcome.
        try:
            node.stop_publish_zero_vel(n=5)
        except Exception:
            pass
        if fig is not None:
            try:
                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception:
                pass
        node.destroy_node()
        if owns_rclpy:
            try:
                rclpy.shutdown()
            except Exception:
                pass
    print(f"[reset] Result: {result}", flush=True)
    return result


def _log_status(node: PositionResetNode) -> None:
    parts = []
    for r in node.robots:
        if r.current_xy is None:
            parts.append(f"{r.name}: no TF")
            continue
        err = float(np.linalg.norm(r.target_xy - r.current_xy))
        err_yaw = float(np.degrees(_wrap_to_pi(
            r.target_yaw - float(r.current_yaw)
        )))
        parts.append(
            f"{r.name}: Δ={err*100:5.1f}cm/Δθ={err_yaw:+5.1f}° "
            f"hold={r.in_tolerance_ticks}"
        )
    print("[reset]  " + " | ".join(parts), flush=True)


def _draw_reset(ax, config, ghosts, node: PositionResetNode) -> None:
    """Render ghosts + live TF + command-velocity arrows."""
    from critic_sharing.config.resolver import load_env_config
    from critic_sharing.deployment.household_viz_node import _draw_ghost
    from critic_sharing.scenarios.household.render import (
        draw_arena, draw_cleaner, draw_follower, draw_goal_line,
    )

    clean_cfg = load_env_config("household", "leader")
    ax.clear()
    draw_arena(ax, clean_cfg)
    draw_goal_line(ax, clean_cfg)

    cleaner_he = tuple(clean_cfg.cleaner_half_extents)
    follower_he = tuple(clean_cfg.follower_half_extents)
    table_he = tuple(clean_cfg.table_half_extents)

    # Ghosts (low alpha).
    _draw_ghost(ax, "table",
                ghosts["table_xy_sim"], ghosts["table_yaw_sim"], table_he)
    _draw_ghost(ax, "cleaner",
                ghosts["cleaner_xy_sim"], ghosts["cleaner_yaw_sim"], cleaner_he)
    _draw_ghost(ax, "follower",
                ghosts["follower_xy_sim"], ghosts["follower_yaw_sim"], follower_he)

    # Live TF + command arrows.
    for r, draw_fn, he in (
        (node.cleaner, draw_cleaner, cleaner_he),
        (node.follower, draw_follower, follower_he),
    ):
        if r.current_xy is None:
            continue
        draw_fn(ax, r.current_xy, float(r.current_yaw), he)
        cmd = r.last_cmd_sim
        if cmd is not None and float(np.linalg.norm(cmd[:2])) > 1e-3:
            ax.annotate(
                "",
                xy=(float(r.current_xy[0] + cmd[0] * 0.5),
                    float(r.current_xy[1] + cmd[1] * 0.5)),
                xytext=(float(r.current_xy[0]), float(r.current_xy[1])),
                arrowprops={"arrowstyle": "->", "color": "#1976D2", "lw": 1.5},
                zorder=6,
            )

    deltas = []
    for r in node.robots:
        if r.current_xy is None:
            deltas.append(f"{r.name}=—")
            continue
        d = float(np.linalg.norm(r.target_xy - r.current_xy))
        dy = float(np.degrees(_wrap_to_pi(r.target_yaw - float(r.current_yaw))))
        deltas.append(
            f"{r.name} Δ={d*100:.1f}cm/{dy:+.0f}° hold={r.in_tolerance_ticks}"
        )
    ax.set_title("AUTO-RESET (ghosts = target)\n" + "  ".join(deltas),
                 fontsize=8)


__all__ = ["PositionResetNode", "reset_to_ghosts"]
