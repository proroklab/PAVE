"""Live TF visualizer for the household scenario.

Subscribes to the ROS TF tree, looks up the cleaner, follower, and
table frames each tick (default 10 Hz), transforms them into sim
coordinates via :class:`StaticSE2` (with optional z-down chirality
flip), and renders them in a matplotlib window using the standard
``scenarios.household.render`` helpers.

The window auto-updates as you move the robots/obstacle in the lab —
useful for confirming that ``sim_origin_in_world`` / ``parent_z_down``
in the YAML actually align the lab geometry with the simulator's
arena before running the closed-loop policy.

Run via::

    critic-sharing deploy visualize-frames \\
        --config /path/to/household_overrides.yaml

Override individual frame names with ``--cleaner-frame``,
``--follower-frame``, ``--table-frame``.
"""

from __future__ import annotations


import numpy as np

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener

from critic_sharing.config.schema import HouseholdDeploymentConfig
from critic_sharing.config.resolver import load_env_config
from critic_sharing.deployment.transforms import StaticSE2


def _yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    return float(np.arctan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    ))


def _wrap_to_pi(angle: float) -> float:
    return float(((angle + np.pi) % (2.0 * np.pi)) - np.pi)


class HouseholdTFTracker(Node):
    """Background TF poller — stores the latest sim-frame pose per body."""

    def __init__(self, config: HouseholdDeploymentConfig):
        super().__init__("household_tf_tracker")
        self._config = config

        tx, ty, yaw_parent = config.sim_origin_in_world
        yaw_zup = -float(yaw_parent) if config.parent_z_down else float(yaw_parent)
        self._transform = StaticSE2(tx=float(tx), ty=float(ty), theta=yaw_zup)
        self._parent_z_down = bool(config.parent_z_down)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._tf_timeout = Duration(seconds=float(config.tf_timeout_sec))

        # latest sim-frame poses; None until first valid TF lookup.
        self._poses_sim: dict[str, tuple[np.ndarray, float] | None] = {
            "cleaner": None,
            "follower": None,
            "table": None,
        }
        self._dirt_sim: np.ndarray | None = None

        period = 1.0 / float(config.rate_hz)
        self._timer = self.create_timer(period, self._poll)

    @property
    def poses(self) -> dict[str, tuple[np.ndarray, float] | None]:
        return self._poses_sim

    @property
    def dirt_sim(self) -> np.ndarray | None:
        return self._dirt_sim

    def _lookup_pose(self, child_frame: str) -> tuple[np.ndarray, float] | None:
        try:
            tf = self._tf_buffer.lookup_transform(
                self._config.parent_frame, child_frame, Time(), self._tf_timeout,
            )
        except TransformException as exc:
            self.get_logger().warn(
                "TF lookup {p}->{c} failed: {e}".format(
                    p=self._config.parent_frame, c=child_frame, e=exc,
                ),
                throttle_duration_sec=2.0,
            )
            return None
        t = tf.transform.translation
        r = tf.transform.rotation
        xy_world = np.array([t.x, t.y], dtype=np.float64)
        yaw_world = _yaw_from_quat(r.x, r.y, r.z, r.w)
        xy_sim = self._transform.world_to_sim_position(xy_world)
        yaw_zup = -yaw_world if self._parent_z_down else yaw_world
        yaw_sim = self._transform.world_to_sim_yaw(yaw_zup)
        return xy_sim, yaw_sim

    def _poll(self) -> None:
        cleaner = self._lookup_pose(self._config.cleaner_frame)
        follower = self._lookup_pose(self._config.follower_frame)
        table = self._lookup_pose(self._config.table_frame)
        if cleaner is not None:
            self._poses_sim["cleaner"] = cleaner
        if follower is not None:
            self._poses_sim["follower"] = follower
        if table is not None:
            self._poses_sim["table"] = table
            # Latch dirt at the table's centroid on first valid table lookup
            # (matches the sim mechanic of dirt = table spawn centroid).
            if self._dirt_sim is None and not self._config.dirt_frame:
                self._dirt_sim = table[0].copy()
                self.get_logger().info(
                    "Dirt latched at sim ({x:+.3f}, {y:+.3f})".format(
                        x=float(self._dirt_sim[0]), y=float(self._dirt_sim[1]),
                    )
                )
        if self._config.dirt_frame:
            dirt = self._lookup_pose(self._config.dirt_frame)
            if dirt is not None:
                self._dirt_sim = dirt[0].copy()


def run_visualizer(config: HouseholdDeploymentConfig, fps: float = 10.0) -> None:
    """Initialise rclpy + matplotlib and render until Ctrl-C."""

    import matplotlib
    matplotlib.use("TkAgg")  # interactive backend
    import matplotlib.pyplot as plt

    from critic_sharing.scenarios.household.render import (
        draw_arena, draw_cleaner, draw_dirt, draw_follower, draw_goal_line,
        draw_table,
    )

    clean_cfg = load_env_config("household", "leader")

    rclpy.init()
    node = HouseholdTFTracker(config)

    plt.ion()
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 5.5))
    fig.canvas.manager.set_window_title("household — live TF")

    def _redraw():
        ax.clear()
        draw_arena(ax, clean_cfg)
        draw_goal_line(ax, clean_cfg)

        # Table first (so other agents draw on top).
        table_pose = node.poses["table"]
        if table_pose is not None:
            draw_table(ax, table_pose[0], table_pose[1],
                       tuple(clean_cfg.table_half_extents))
        if node.dirt_sim is not None:
            draw_dirt(ax, node.dirt_sim)

        cleaner_pose = node.poses["cleaner"]
        if cleaner_pose is not None:
            draw_cleaner(ax, cleaner_pose[0], cleaner_pose[1],
                         tuple(clean_cfg.cleaner_half_extents))
        follower_pose = node.poses["follower"]
        if follower_pose is not None:
            draw_follower(ax, follower_pose[0], follower_pose[1],
                          tuple(clean_cfg.follower_half_extents))

        title_parts = ["live TF"]
        for name in ("cleaner", "follower", "table"):
            p = node.poses[name]
            title_parts.append(
                f"{name}=" + (
                    "({x:+.2f},{y:+.2f},{th:+.0f}°)".format(
                        x=float(p[0][0]), y=float(p[0][1]),
                        th=float(np.degrees(p[1])),
                    )
                    if p is not None else "—"
                )
            )
        ax.set_title("\n".join(title_parts), fontsize=8)
        fig.canvas.draw_idle()

    interval_s = 1.0 / float(fps)
    try:
        while rclpy.ok() and plt.fignum_exists(fig.number):
            rclpy.spin_once(node, timeout_sec=interval_s)
            _redraw()
            plt.pause(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        plt.close(fig)
        node.destroy_node()
        rclpy.shutdown()


def capture_initial_conditions(
    config: HouseholdDeploymentConfig,
    timeout: float = 5.0,
) -> dict:
    """One-shot TF read of cleaner+follower+table in sim coords.

    Spins a transient :class:`HouseholdTFTracker` until all three frames
    have produced a valid lookup or ``timeout`` elapses. Returns a dict
    with ``cleaner_xy_sim``, ``cleaner_yaw_sim``, ``follower_xy_sim``,
    ``follower_yaw_sim``, ``table_xy_sim``, ``table_yaw_sim``. Raises
    :class:`TimeoutError` if any frame fails to publish in time.

    If rclpy is already initialized externally (e.g. by a parent
    harness), this function reuses that context and does not touch
    ``rclpy.init`` / ``rclpy.shutdown``. Otherwise it owns both.
    """
    import time as _time

    owns_rclpy = not rclpy.ok()
    if owns_rclpy:
        rclpy.init()
    try:
        node = HouseholdTFTracker(config)
        try:
            deadline = _time.monotonic() + float(timeout)
            while _time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.05)
                p = node.poses
                if (p["cleaner"] is not None and p["follower"] is not None
                        and p["table"] is not None):
                    cleaner_xy, cleaner_yaw = p["cleaner"]
                    follower_xy, follower_yaw = p["follower"]
                    table_xy, table_yaw = p["table"]
                    return {
                        "cleaner_xy_sim": np.asarray(cleaner_xy, dtype=np.float64),
                        "cleaner_yaw_sim": float(cleaner_yaw),
                        "follower_xy_sim": np.asarray(follower_xy, dtype=np.float64),
                        "follower_yaw_sim": float(follower_yaw),
                        "table_xy_sim": np.asarray(table_xy, dtype=np.float64),
                        "table_yaw_sim": float(table_yaw),
                    }
            raise TimeoutError(
                "capture_initial_conditions: TF for cleaner+follower+table "
                "did not become available within {to:.1f}s.".format(to=timeout)
            )
        finally:
            node.destroy_node()
    finally:
        if owns_rclpy:
            try:
                rclpy.shutdown()
            except Exception:
                pass


def _draw_ghost(ax, kind: str, center, theta: float, half_extents) -> None:
    """Low-alpha overlay of a body's initial pose."""
    from critic_sharing.scenarios.household.common import draw_rect

    palette = {
        "table":    ("#FFB74D", "#E65100"),
        "cleaner":  ("#90CAF9", "#1565C0"),
        "follower": ("#A5D6A7", "#1B5E20"),
    }
    face, edge = palette[kind]
    draw_rect(ax, center, float(theta), tuple(half_extents),
              face=face, edge=edge, alpha=0.25, linewidth=1.0)


def run_reset_view(
    config: HouseholdDeploymentConfig,
    ghosts: dict,
    fps: float = 10.0,
) -> None:
    """Reset-helper visualization: ghost overlay + live TF, press Enter to exit.

    ``ghosts`` is the dict returned by :func:`capture_initial_conditions`
    — the captured cleaner+follower+table sim poses drawn at low alpha
    so the operator can reposition the bodies back to where they were
    at the start of the pair.

    Exits when the user presses Enter in the terminal (stdin-driven so
    the matplotlib window doesn't need focus).
    """
    import threading

    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    from critic_sharing.scenarios.household.render import (
        draw_arena, draw_cleaner, draw_dirt, draw_follower, draw_goal_line,
        draw_table,
    )

    clean_cfg = load_env_config("household", "leader")

    owns_rclpy = not rclpy.ok()
    if owns_rclpy:
        rclpy.init()
    node = HouseholdTFTracker(config)

    plt.ion()
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 5.5))
    fig.canvas.manager.set_window_title("household — RESET (ghosts = target)")

    # Set up a stdin-watcher in a background thread.
    enter_event = threading.Event()

    def _await_enter() -> None:
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            pass
        enter_event.set()

    print(">>> RESET VIEW: align bodies with the alpha ghosts, "
          "then press Enter to continue. (Ctrl-C aborts the session.)",
          flush=True)
    stdin_thread = threading.Thread(target=_await_enter, daemon=True)
    stdin_thread.start()

    cleaner_he = tuple(clean_cfg.cleaner_half_extents)
    follower_he = tuple(clean_cfg.follower_half_extents)
    table_he = tuple(clean_cfg.table_half_extents)

    def _redraw():
        ax.clear()
        draw_arena(ax, clean_cfg)
        draw_goal_line(ax, clean_cfg)

        # Ghosts (low alpha) — drawn first so live TF is on top.
        _draw_ghost(ax, "table",
                    ghosts["table_xy_sim"], ghosts["table_yaw_sim"], table_he)
        _draw_ghost(ax, "cleaner",
                    ghosts["cleaner_xy_sim"], ghosts["cleaner_yaw_sim"],
                    cleaner_he)
        _draw_ghost(ax, "follower",
                    ghosts["follower_xy_sim"], ghosts["follower_yaw_sim"],
                    follower_he)

        # Live TF
        table_pose = node.poses["table"]
        if table_pose is not None:
            draw_table(ax, table_pose[0], table_pose[1], table_he)
        if node.dirt_sim is not None:
            draw_dirt(ax, node.dirt_sim)
        cleaner_pose = node.poses["cleaner"]
        if cleaner_pose is not None:
            draw_cleaner(ax, cleaner_pose[0], cleaner_pose[1], cleaner_he)
        follower_pose = node.poses["follower"]
        if follower_pose is not None:
            draw_follower(ax, follower_pose[0], follower_pose[1], follower_he)

        # Per-body alignment delta (live − ghost), to give the operator
        # a numeric sense of how close the reset is.
        deltas = []
        for name, key in (("cleaner", "cleaner"),
                          ("follower", "follower"),
                          ("table", "table")):
            live = node.poses[name]
            if live is None:
                deltas.append(f"{name}=—")
                continue
            gx, gy = ghosts[f"{key}_xy_sim"]
            dx = float(live[0][0]) - float(gx)
            dy = float(live[0][1]) - float(gy)
            d = float(np.sqrt(dx * dx + dy * dy))
            deltas.append(f"{name} Δ={d:.02f}m")
        ax.set_title("RESET: align with ghosts, press Enter\n" + "  ".join(deltas),
                     fontsize=8)
        fig.canvas.draw_idle()

    interval_s = 1.0 / float(fps)
    try:
        while (rclpy.ok() and plt.fignum_exists(fig.number)
               and not enter_event.is_set()):
            rclpy.spin_once(node, timeout_sec=interval_s)
            _redraw()
            plt.pause(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        plt.close(fig)
        node.destroy_node()
        if owns_rclpy:
            try:
                rclpy.shutdown()
            except Exception:
                pass


__all__ = [
    "HouseholdTFTracker",
    "run_visualizer",
    "run_reset_view",
    "capture_initial_conditions",
]
