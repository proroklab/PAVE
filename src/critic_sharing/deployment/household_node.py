"""ROS2 node that drives a Robomaster from a trained household policy.

Pipeline per tick (default 10 Hz, matching the sim dt of 0.1 s):

1. Look up ``parent_frame -> {self|table}_frame`` on the TF tree to get
   each rigid body's planar pose (x, y, yaw).
2. Map positions into the simulation frame via the static SE(2)
   transform from ``sim_origin_in_world``; rotate yaw by ``-yaw_offset``.
3. Build the role-specific obs and call
   ``model.predict(obs, deterministic=True)`` to get a sim-frame
   ``(vx_sim, vy_sim, omega)``.
4. Clamp linear speed to ``max_speed`` and yaw rate to
   ``max_yaw_rate``.
5. Rotate the linear velocity back into ``parent_frame`` and publish a
   ``freyja_msgs/ReferenceState`` setpoint:
     * ``vn``/``ve`` from the rotated linear velocity (parent frame is
       assumed ENU: x=east → ve, y=north → vn).
     * ``yaw`` from the integrated yaw setpoint (current yaw + ω · dt).

The dirt is frozen at the table's pose captured at node startup
(matches the sim mechanic of dirt = table spawn centroid). If
``dirt_frame`` is set, that frame's pose is used instead and treated
as fixed.

ROS2 dependencies (``rclpy``, ``tf2_ros``, ``freyja_msgs``) come from
the system ROS2 install — source ``/opt/ros/<distro>/setup.bash`` and
the Freyja workspace's ``install/setup.bash`` before running. They are
intentionally not pinned in ``pyproject.toml``.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

import rclpy
from freyja_msgs.msg import ReferenceState
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener

from critic_sharing.config.schema import HouseholdDeploymentConfig
from critic_sharing.deployment._profile import PhaseTimer
from critic_sharing.deployment.controllers import ControllerAdapter
from critic_sharing.deployment.transforms import StaticSE2


def _yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    """Extract yaw from a quaternion (planar deployment, x-forward)."""

    return float(np.arctan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    ))


def _wrap_to_pi(angle: float) -> float:
    return float(((angle + np.pi) % (2.0 * np.pi)) - np.pi)


def _shutdown_tf_listener(listener) -> None:
    """Stop a ``TransformListener(spin_thread=True)``'s dedicated executor.

    The listener runs its own ``SingleThreadedExecutor`` in a daemon
    thread and only tears it down in ``__del__`` (GC-timed). If the
    global ``rclpy.shutdown()`` lands first, that thread's ``spin()``
    raises ``ExternalShutdownException`` and dumps a traceback. Shutting
    the executor down here — before ``destroy_node()`` / ``rclpy.shutdown()``
    — drains the thread cleanly. Idempotent: ``__del__`` may call it
    again harmlessly.
    """
    if listener is None:
        return
    executor = getattr(listener, "executor", None)
    thread = getattr(listener, "dedicated_listener_thread", None)
    if executor is not None:
        try:
            executor.shutdown()
        except Exception:
            pass
    if thread is not None and thread.is_alive():
        try:
            thread.join(timeout=5.0)
        except Exception:
            pass


class HouseholdPolicyNode(Node):
    """Closed-loop ROS2 driver for a household PPO policy.

    One node handles either the cleaner or the follower role,
    selected via ``config.role`` (``"cleaner"`` or ``"follower"``).
    """

    def __init__(
        self,
        config: HouseholdDeploymentConfig,
        controller: Optional[ControllerAdapter],
        termination_predicate: Optional[Callable[[dict], Optional[str]]] = None,
        start_timer: bool = True,
    ):
        super().__init__("household_policy")
        self._config = config
        # ``controller`` is the adapter (PPO/CEM/…). May be None when
        # the node is in a debug mode (goto_origin / goto_dirt) — those
        # branches synthesize the action directly without invoking the
        # controller.
        self._controller = controller
        # Injected by the episode supervisor. Evaluated each tick after
        # pose lookup, before publishing — returns a termination reason
        # string ("pickup", "out_of_arena", "max_steps", "goal_reached")
        # or None to continue. When None, the node runs unbounded.
        self._termination_predicate = termination_predicate

        # Tick counter — incremented only on ticks with valid TF lookups
        # (failed lookups don't count, matching the design contract).
        self._tick_count: int = 0
        # Set by _tick when the predicate fires (or externally by the
        # supervisor on KeyboardInterrupt). None while the episode runs.
        self.terminated_reason: Optional[str] = None
        # Snapshot of per-tick state for the supervisor (viz/predicate).
        # Empty dict until the first successful tick populates it.
        self._latest_state: dict = {}
        # Last action commanded in sim coords (vx, vy, ω). Used for the
        # velocity-arrow overlay in the recording.
        self._last_action_sim: np.ndarray = np.zeros(3, dtype=np.float64)
        # Profiler (no-op when disabled). Supervisor enables it via
        # ``self.profiler.enabled = True`` when --profile is set.
        self.profiler: PhaseTimer = PhaseTimer(enabled=False)
        # Wall-clock timestamp of the previous tick, used to measure the
        # actual inter-tick period (i.e., observed control rate).
        self._last_tick_at: Optional[float] = None
        # First valid (self, table, dirt) sim pose snapshot — the real
        # initial condition this rollout actually started from. Used by
        # the sim-replay tooling to reset a sim env to the same start.
        self._initial_state: Optional[dict] = None

        tx, ty, yaw_parent = config.sim_origin_in_world
        # Internally we work in the sim's z-up (CCW-positive) convention.
        # If the parent frame is z-down, the yaw stored in the YAML is
        # in CW-positive convention — negate it so StaticSE2's
        # yaw math sees the equivalent CCW-positive value.
        yaw_zup = -float(yaw_parent) if config.parent_z_down else float(yaw_parent)
        self._transform = StaticSE2(tx=float(tx), ty=float(ty), theta=yaw_zup)
        self._parent_z_down = bool(config.parent_z_down)

        self._tf_buffer = Buffer()
        # spin_thread=True dedicates a thread to draining /tf, so it
        # doesn't compete with the control timer (or matplotlib) for
        # the main executor — without this, a busy main loop can let
        # the TF subscription queue back up to hundreds of ms of lag
        # while the buffer's "latest" data goes stale.
        self._tf_listener = TransformListener(
            self._tf_buffer, self, spin_thread=True,
        )
        self._tf_timeout = Duration(seconds=float(config.tf_timeout_sec))

        self._publisher = self.create_publisher(ReferenceState, config.cmd_topic, 10)

        # Frozen dirt position (latched on first valid table lookup, unless
        # an explicit dirt_frame is configured).
        self._dirt_sim: np.ndarray | None = None

        # Integrated yaw setpoint (rad in parent_frame). Initialised on the
        # first successful pose lookup.
        self._yaw_setpoint: float | None = None
        self._dt = 1.0 / float(config.rate_hz)
        # In episode mode, the supervisor starts the timer explicitly
        # after Phase 1 so policy ticks don't fire during TF warm-up.
        self._timer = None
        if start_timer:
            self.start_control_timer()

        self._role = config.role
        if self._role not in ("cleaner", "follower"):
            raise ValueError(
                f"HouseholdDeploymentConfig.role must be 'cleaner' or "
                f"'follower' (got {self._role!r})"
            )

        self.get_logger().info(
            "Household {role} deployment: model={model}, parent={parent}, "
            "rate={hz:.1f} Hz, cmd={topic}".format(
                role=self._role,
                model=("(debug)" if (config.goto_origin or config.goto_dirt)
                       else config.model_path),
                parent=config.parent_frame,
                hz=float(config.rate_hz),
                topic=config.cmd_topic,
            )
        )

    def destroy_node(self) -> None:
        # Drain the TF listener's dedicated spin thread before the node
        # (and the global context) go away — avoids ExternalShutdownException.
        _shutdown_tf_listener(getattr(self, "_tf_listener", None))
        super().destroy_node()

    # ------------------------------------------------------------------
    # TF lookup helpers
    # ------------------------------------------------------------------

    def _lookup_pose(self, child_frame: str) -> tuple[np.ndarray, float] | None:
        """Return ``(xy_world (2,), yaw_world)`` for ``child_frame``."""

        try:
            tf = self._tf_buffer.lookup_transform(
                self._config.parent_frame,
                child_frame,
                Time(),
                self._tf_timeout,
            )
        except TransformException as exc:
            self.get_logger().warn(
                "TF lookup {parent}->{child} failed: {exc}".format(
                    parent=self._config.parent_frame,
                    child=child_frame,
                    exc=exc,
                ),
                throttle_duration_sec=2.0,
            )
            return None

        # Measure the AGE of the looked-up TF data: how stale is the pose
        # at the moment we consume it? `Time()` (above) asks for the
        # latest available sample regardless of timestamp, so the lookup
        # itself is fast — but the data may be 100s of ms old if the
        # mocap → ROS bridge has pipeline lag.
        if self.profiler.enabled:
            try:
                now_msg = self.get_clock().now().to_msg()
                stamp = tf.header.stamp
                age_ms = (
                    (now_msg.sec - stamp.sec) * 1000.0
                    + (now_msg.nanosec - stamp.nanosec) / 1e6
                )
                self.profiler.record(
                    "tf_age_" + ("self" if child_frame == self._self_frame()
                                 else "table" if child_frame == self._config.table_frame
                                 else "other"),
                    age_ms,
                )
            except Exception:
                pass
        # Fall through to the existing parsing logic via the original
        # return path below.
        return self._parse_tf_message(tf)

    def _parse_tf_message(self, tf) -> tuple[np.ndarray, float]:
        t = tf.transform.translation
        r = tf.transform.rotation
        xy = np.array([t.x, t.y], dtype=np.float64)
        yaw = _yaw_from_quat(r.x, r.y, r.z, r.w)
        return xy, yaw

    def _self_frame(self) -> str:
        return (self._config.cleaner_frame if self._role == "cleaner"
                else self._config.follower_frame)

    # ------------------------------------------------------------------
    # Sim-frame pose / dirt latching
    # ------------------------------------------------------------------

    def _world_to_sim_pose(
        self, xy_world: np.ndarray, yaw_world: float,
    ) -> tuple[np.ndarray, float]:
        xy_sim = self._transform.world_to_sim_position(xy_world)
        # In z-down parent frames yaw is CW-positive; negate to z-up CCW.
        yaw_zup = -float(yaw_world) if self._parent_z_down else float(yaw_world)
        yaw_sim = self._transform.world_to_sim_yaw(yaw_zup)
        return xy_sim, yaw_sim

    def _sim_yaw_to_world(self, yaw_sim: float) -> float:
        yaw_zup_world = self._transform.sim_to_world_yaw(yaw_sim)
        return -yaw_zup_world if self._parent_z_down else yaw_zup_world

    def _latch_dirt(self, table_xy_sim: np.ndarray) -> None:
        if self._dirt_sim is None:
            self._dirt_sim = table_xy_sim.copy()
            self.get_logger().info(
                "Dirt latched at sim ({x:+.3f}, {y:+.3f})".format(
                    x=float(self._dirt_sim[0]), y=float(self._dirt_sim[1]),
                )
            )

    # ------------------------------------------------------------------
    # Per-tick control loop
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        # Once terminated, the timer keeps firing until the supervisor
        # breaks out of its main loop. Make those ticks no-ops so we
        # don't accidentally publish another policy command — the
        # supervisor will issue the 5× zero-vel flurry on exit.
        if self.terminated_reason is not None:
            return

        import time as _time
        tick_t0 = _time.perf_counter()
        # Inter-tick period (observed control rate). NaN on the very
        # first tick; recorded into the "tick_period" phase.
        if self._last_tick_at is not None:
            self.profiler.record("tick_period", (tick_t0 - self._last_tick_at) * 1000.0)
        self._last_tick_at = tick_t0

        with self.profiler.measure("lookup_self"):
            self_lookup = self._lookup_pose(self._self_frame())
        with self.profiler.measure("lookup_table"):
            table_lookup = self._lookup_pose(self._config.table_frame)
        if self_lookup is None or table_lookup is None:
            return

        self_xy_world, self_yaw_world = self_lookup
        self_xy_sim, self_yaw_sim = self._world_to_sim_pose(self_xy_world, self_yaw_world)

        table_xy_world, table_yaw_world = table_lookup
        table_xy_sim, table_yaw_sim = self._world_to_sim_pose(table_xy_world, table_yaw_world)

        # Latch dirt at the first valid table observation (unless a
        # dedicated dirt_frame is configured or the supervisor already
        # latched it during Phase 1).
        if self._config.dirt_frame:
            dirt_lookup = self._lookup_pose(self._config.dirt_frame)
            if dirt_lookup is None:
                return
            dirt_xy_sim = self._transform.world_to_sim_position(dirt_lookup[0])
        else:
            self._latch_dirt(table_xy_sim)
            dirt_xy_sim = self._dirt_sim

        # Initialise the integrated yaw setpoint from the current pose.
        if self._yaw_setpoint is None:
            self._yaw_setpoint = float(self_yaw_world)

        # This tick has valid poses — count it toward the episode length.
        self._tick_count += 1

        # Latch the real starting pose on the first valid tick (self pose
        # under "cleaner_*" keys regardless of role, as elsewhere).
        if self._initial_state is None:
            self._initial_state = {
                "cleaner_xy_sim": np.asarray(self_xy_sim, dtype=np.float64).copy(),
                "cleaner_yaw_sim": float(self_yaw_sim),
                "table_xy_sim": np.asarray(table_xy_sim, dtype=np.float64).copy(),
                "table_yaw_sim": float(table_yaw_sim),
                "dirt_xy_sim": np.asarray(dirt_xy_sim, dtype=np.float64).copy(),
            }

        # Snapshot for the supervisor (viz + predicate). last_action_sim
        # carries the previous tick's command, used for the velocity
        # arrow overlay.
        self._latest_state = {
            "cleaner_xy_sim": self_xy_sim,
            "cleaner_yaw_sim": self_yaw_sim,
            "table_xy_sim": table_xy_sim,
            "table_yaw_sim": table_yaw_sim,
            "dirt_xy_sim": np.asarray(dirt_xy_sim, dtype=np.float64).copy(),
            "tick_count": self._tick_count,
            "last_action_sim": self._last_action_sim.copy(),
            "role": self._role,
        }

        # Termination predicate (injected by the supervisor) — evaluated
        # synchronously with the publish so the last commanded velocity
        # is unambiguously zero on termination.
        if self._termination_predicate is not None:
            reason = self._termination_predicate(self._latest_state)
            if reason is not None:
                self.terminated_reason = str(reason)
                self.publish_zero_velocity()
                self.get_logger().info(
                    "Episode terminated: reason={r}, tick={t}".format(
                        r=self.terminated_reason, t=self._tick_count,
                    )
                )
                return

        # Build the role-specific obs and query the policy (unless we're
        # in a debug mode).
        if self._config.goto_origin:
            action = np.array([-self_xy_sim[0], -self_xy_sim[1], 0.0], dtype=np.float64)
        elif self._config.goto_dirt:
            target = dirt_xy_sim - self_xy_sim
            action = np.array([target[0], target[1], 0.0], dtype=np.float64)
        else:
            if self._controller is None:
                raise RuntimeError(
                    "HouseholdPolicyNode is in policy mode but no "
                    "controller adapter was provided."
                )
            with self.profiler.measure("build_obs"):
                obs = self._build_obs(self_xy_sim, self_yaw_sim,
                                      table_xy_sim, table_yaw_sim,
                                      dirt_xy_sim)
            with self.profiler.measure("policy_predict"):
                action = self._controller.act(obs)
            action = np.asarray(action, dtype=np.float64).reshape(3)

        # Clamp linear speed and yaw rate.
        linear = action[:2]
        if self._config.clamp_action:
            speed = float(np.linalg.norm(linear))
            if speed > self._config.max_speed:
                linear = linear / (speed + 1e-8) * self._config.max_speed
            action[2] = float(np.clip(
                action[2],
                -self._config.max_yaw_rate,
                self._config.max_yaw_rate,
            ))

        # Cache the sim-frame action for the viz overlay (velocity arrow).
        self._last_action_sim = np.array(
            [float(linear[0]), float(linear[1]), float(action[2])],
            dtype=np.float64,
        )

        # Rotate linear velocity back into parent_frame.
        v_world = self._transform.sim_to_world_velocity(linear)

        # Integrate yaw rate into the yaw setpoint. ``action[2]`` is the
        # sim's CCW-positive ω; in a z-down parent it maps to CW-positive,
        # i.e., negate before integrating.
        omega_world = -float(action[2]) if self._parent_z_down else float(action[2])
        self._yaw_setpoint = _wrap_to_pi(self._yaw_setpoint + omega_world * self._dt)

        with self.profiler.measure("publish"):
            msg = ReferenceState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self._config.parent_frame
            # parent_frame ENU: x=east → ve, y=north → vn.
            msg.vn = float(v_world[1])
            msg.ve = float(v_world[0])
            msg.yaw = float(self._yaw_setpoint)
            self._publisher.publish(msg)

        # Total time inside _tick (excludes the timer's wait, which is
        # what spin_once is doing — that's in the supervisor's "spin" phase).
        self.profiler.record("tick_total",
                             (_time.perf_counter() - tick_t0) * 1000.0)

    # ------------------------------------------------------------------
    # Supervisor-facing helpers (episode mode)
    # ------------------------------------------------------------------

    @property
    def latest_state(self) -> dict:
        """Snapshot of the most recent successful tick. Empty before first."""
        return self._latest_state

    @property
    def initial_state(self) -> Optional[dict]:
        """The real starting pose latched on the first valid tick (or None)."""
        return self._initial_state

    @property
    def tick_count(self) -> int:
        return self._tick_count

    def start_control_timer(self) -> None:
        """Create the per-tick timer if it hasn't been created yet."""
        if self._timer is None:
            self._timer = self.create_timer(self._dt, self._tick)

    def set_controller(self, controller: Optional[ControllerAdapter]) -> None:
        """Inject (or replace) the controller before the timer starts.

        Used by the CEM cleaner deployment path: the controller can only
        be built once the follower's spawn pose is measured from TF, so
        the supervisor constructs the node with ``controller=None``, runs
        Phase 1 to lock in the pose, then calls this setter and finally
        :meth:`start_control_timer`.
        """
        self._controller = controller

    def publish_zero_velocity(self) -> None:
        """Publish a ReferenceState commanding zero velocity, holding yaw."""
        msg = ReferenceState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._config.parent_frame
        msg.vn = 0.0
        msg.ve = 0.0
        msg.vd = 0.0
        msg.an = 0.0
        msg.ae = 0.0
        msg.ad = 0.0
        msg.yaw = float(
            self._yaw_setpoint if self._yaw_setpoint is not None else 0.0
        )
        self._publisher.publish(msg)

    def latch_dirt_now(self) -> Optional[np.ndarray]:
        """Eagerly latch the dirt position from the current table TF.

        Used by the supervisor's Phase 1 to set the dirt before the
        control timer starts. Returns the latched sim position, or None
        if the table TF isn't available yet.
        """
        if self._dirt_sim is not None:
            return self._dirt_sim.copy()
        table_lookup = self._lookup_pose(self._config.table_frame)
        if table_lookup is None:
            return None
        table_xy_sim = self._transform.world_to_sim_position(table_lookup[0])
        self._latch_dirt(table_xy_sim)
        return self._dirt_sim.copy() if self._dirt_sim is not None else None

    # ------------------------------------------------------------------
    # Role-specific obs builder
    # ------------------------------------------------------------------

    def _build_obs(self,
                   self_xy_sim: np.ndarray, self_yaw_sim: float,
                   table_xy_sim: np.ndarray, table_yaw_sim: float,
                   dirt_xy_sim: np.ndarray) -> np.ndarray:
        if self._role == "cleaner":
            # 10-D: [c_x, c_y, sin θ_c, cos θ_c,
            #        t_x, t_y, sin θ_t, cos θ_t,
            #        dirt_x, dirt_y]
            return np.array(
                [
                    self_xy_sim[0], self_xy_sim[1],
                    float(np.sin(self_yaw_sim)), float(np.cos(self_yaw_sim)),
                    table_xy_sim[0], table_xy_sim[1],
                    float(np.sin(table_yaw_sim)), float(np.cos(table_yaw_sim)),
                    dirt_xy_sim[0], dirt_xy_sim[1],
                ],
                dtype=np.float32,
            )
        # 8-D: [f_x, f_y, sin θ_f, cos θ_f,
        #       t_x, t_y, sin θ_t, cos θ_t]
        return np.array(
            [
                self_xy_sim[0], self_xy_sim[1],
                float(np.sin(self_yaw_sim)), float(np.cos(self_yaw_sim)),
                table_xy_sim[0], table_xy_sim[1],
                float(np.sin(table_yaw_sim)), float(np.cos(table_yaw_sim)),
            ],
            dtype=np.float32,
        )


def run(config: HouseholdDeploymentConfig) -> None:
    """Initialise rclpy, spin the policy node, shut down cleanly."""

    from critic_sharing.deployment.controllers import PPOPolicyAdapter

    if config.goto_origin or config.goto_dirt:
        controller = None
    else:
        if not config.model_path:
            raise ValueError("HouseholdDeploymentConfig.model_path must be set.")
        from stable_baselines3 import PPO

        controller = PPOPolicyAdapter(PPO.load(config.model_path))

    rclpy.init()
    node = HouseholdPolicyNode(config=config, controller=controller)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


__all__ = ["HouseholdPolicyNode", "run"]
