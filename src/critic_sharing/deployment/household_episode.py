"""Bounded-episode supervisor for hardware household rollouts.

This module wraps :class:`HouseholdPolicyNode` with a defined
termination contract — pickup (cleaner) or goal_reached (follower),
out-of-arena, max_steps, or KeyboardInterrupt — plus an explicit
ready phase that latches the virtual dirt to the table TF before the
control timer starts, an optional matplotlib viz, an MP4 recorder,
and a 5× zero-velocity flurry on exit.

The supervisor owns the ROS spin loop. The node exposes ``latest_state``
and a ``terminated_reason`` flag; the supervisor injects a termination
predicate closure into the node so the predicate is evaluated
synchronously with the policy publish (last-message-is-zero-vel
guarantee).

Termination predicate contract::

    def predicate(state: dict) -> Optional[str]:
        # state keys: cleaner_xy_sim, cleaner_yaw_sim, table_xy_sim,
        #             table_yaw_sim, dirt_xy_sim, tick_count, role,
        #             last_action_sim
        # returns: termination reason string, or None to continue

The reason strings are part of the metadata.json contract:
``"pickup"``, ``"goal_reached"``, ``"out_of_arena"``, ``"max_steps"``,
``"keyboard_interrupt"``. ``"debug"`` predicates (used by goto_origin /
goto_dirt modes) always return None — only Ctrl-C terminates them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from critic_sharing.config.schema import (
    HouseholdCleanerEnvConfig,
    HouseholdDeploymentConfig,
    HouseholdNavEnvConfig,
)
from critic_sharing.deployment._profile import PhaseTimer
from critic_sharing.scenarios.household.common import (
    point_in_rect,
    rect_circumscribed_radius,
)


TerminationPredicate = Callable[[dict], Optional[str]]


# ---------------------------------------------------------------------------
# Default per-role max_steps (matches the sim's HouseholdCleaner/Nav configs).
# ---------------------------------------------------------------------------

def default_max_steps(role: str) -> int:
    if role == "cleaner":
        return HouseholdCleanerEnvConfig().max_steps
    if role == "follower":
        return HouseholdNavEnvConfig().max_steps
    raise ValueError(f"unknown role: {role!r}")


# ---------------------------------------------------------------------------
# Predicate factories
# ---------------------------------------------------------------------------

def make_cleaner_predicate(
    cleaner_half_extents: tuple,
    arena_bounds: tuple,
    arena_margin: float,
    max_steps: int,
    post_pickup_steps: int = 0,
) -> TerminationPredicate:
    """Build the cleaner-role termination predicate.

    Mirrors the sim cleaner env: pickup does **not** end the episode; it
    starts a ``post_pickup_steps`` aid window during which the cleaner keeps
    running (CEM/V_F-driven) to reposition the table for the follower. Fires
    on (in priority order):
      * ``aid_complete`` — ``post_pickup_steps`` ticks after the first pickup
        (immediately on pickup when ``post_pickup_steps == 0``)
      * ``out_of_arena`` — cleaner center outside ``[x_min+m, x_max-m] ×
        [y_min+m, y_max-m]``
      * ``max_steps`` — tick_count ≥ max_steps

    ``arena_bounds`` is ``(x_min, x_max, y_min, y_max)``.
    """
    x_min, x_max, y_min, y_max = arena_bounds
    m = float(arena_margin)
    x_lo, x_hi = x_min + m, x_max - m
    y_lo, y_hi = y_min + m, y_max - m
    he = (float(cleaner_half_extents[0]), float(cleaner_half_extents[1]))
    n_aid = int(post_pickup_steps)
    pickup_tick: dict = {"t": None}

    def predicate(state: dict) -> Optional[str]:
        cleaner_xy = state["cleaner_xy_sim"]
        cleaner_yaw = float(state["cleaner_yaw_sim"])
        dirt_xy = state["dirt_xy_sim"]
        tick = int(state["tick_count"])

        if pickup_tick["t"] is None and point_in_rect(
            np.asarray(dirt_xy), np.asarray(cleaner_xy), cleaner_yaw, he
        ):
            pickup_tick["t"] = tick
        if pickup_tick["t"] is not None and (tick - pickup_tick["t"]) >= n_aid:
            return "aid_complete"

        cx, cy = float(cleaner_xy[0]), float(cleaner_xy[1])
        if not (x_lo <= cx <= x_hi and y_lo <= cy <= y_hi):
            return "out_of_arena"

        if tick >= max_steps:
            return "max_steps"

        return None

    return predicate


def make_follower_predicate(
    follower_half_extents: tuple,
    arena_bounds: tuple,
    arena_margin: float,
    max_steps: int,
    env_config=None,
    goal_x_half_extent: Optional[float] = None,
) -> TerminationPredicate:
    """Build the follower-role termination predicate.

    Mirrors the sim nav env (centered goal), with one hardware-only addition
    (table-collision termination). Fires on (in priority order):
      * ``goal_reached`` — ``pos.y ≤ y_min + r`` AND ``|pos.x - mid_x| ≤ hx``
        (the centered half of the lower boundary)
      * ``collision`` — SAT overlap with the table only (walls are not
        terminating). Only checked when ``env_config`` is provided (the
        table half-extents come from it).
      * ``out_of_arena`` — center outside the margin-shrunk box
      * ``max_steps`` — tick_count ≥ max_steps

    ``goal_x_half_extent`` (hx) defaults to ``(x_max - x_min) / 4`` (the
    centered half of the boundary).

    Note: ``nav_env.py`` (sim) treats *all* collisions as a penalty,
    not a termination, because PPO needs recovery experience. On
    hardware we want to stop the robot the moment it touches the
    table (the only object that physically moves and would mess up
    the experiment). Wall scrapes are still legal — the resolver
    keeps the robot inside.
    """
    from critic_sharing.scenarios.household.common import (
        rect_corners, sat_overlap,
    )

    x_min, x_max, y_min, y_max = arena_bounds
    m = float(arena_margin)
    x_lo, x_hi = x_min + m, x_max - m
    y_lo, y_hi = y_min + m, y_max - m
    r = rect_circumscribed_radius(
        (float(follower_half_extents[0]), float(follower_half_extents[1]))
    )
    goal_y = y_min + r
    he = (float(follower_half_extents[0]), float(follower_half_extents[1]))
    table_he = (
        (float(env_config.table_half_extents[0]),
         float(env_config.table_half_extents[1]))
        if env_config is not None else None
    )
    mid_x = 0.5 * (x_min + x_max)
    hx = (0.25 * (x_max - x_min) if goal_x_half_extent is None
          else float(goal_x_half_extent))

    def predicate(state: dict) -> Optional[str]:
        xy = state["cleaner_xy_sim"]  # supervisor reuses the same key for self pose
        theta = float(state["cleaner_yaw_sim"])
        tick = int(state["tick_count"])
        py = float(xy[1])
        px = float(xy[0])

        if py <= goal_y and abs(px - mid_x) <= hx:
            return "goal_reached"

        if table_he is not None:
            self_corners = rect_corners(
                np.asarray(xy, dtype=np.float32), theta, he,
            )
            table_xy = np.asarray(state["table_xy_sim"], dtype=np.float32)
            table_theta = float(state["table_yaw_sim"])
            tcorners = rect_corners(table_xy, table_theta, table_he)
            ov, _, _ = sat_overlap(self_corners, tcorners)
            if ov:
                return "collision"

        if not (x_lo <= px <= x_hi and y_lo <= py <= y_hi):
            return "out_of_arena"

        if tick >= max_steps:
            return "max_steps"

        return None

    return predicate


def make_no_op_predicate() -> TerminationPredicate:
    """Predicate for debug modes (goto_origin / goto_dirt). Never fires."""
    def predicate(state: dict) -> Optional[str]:
        return None
    return predicate


def build_predicate_for_config(
    deploy_config: HouseholdDeploymentConfig,
    env_config,
) -> TerminationPredicate:
    """Choose the appropriate predicate based on deploy_config.role and debug flags.

    ``env_config`` is the role-matched sim env config
    (HouseholdCleanerEnvConfig or HouseholdNavEnvConfig) — used for
    half-extents and world bounds.
    """
    if deploy_config.goto_origin or deploy_config.goto_dirt:
        return make_no_op_predicate()

    max_steps = (
        int(deploy_config.max_steps_override)
        if deploy_config.max_steps_override is not None
        else default_max_steps(deploy_config.role)
    )
    arena_bounds = (
        float(env_config.world_x_min), float(env_config.world_x_max),
        float(env_config.world_y_min), float(env_config.world_y_max),
    )
    margin = float(deploy_config.arena_margin)

    if deploy_config.role == "cleaner":
        return make_cleaner_predicate(
            cleaner_half_extents=env_config.cleaner_half_extents,
            arena_bounds=arena_bounds,
            arena_margin=margin,
            max_steps=max_steps,
            post_pickup_steps=int(getattr(env_config, "post_pickup_steps", 0)),
        )
    if deploy_config.role == "follower":
        return make_follower_predicate(
            follower_half_extents=env_config.follower_half_extents,
            arena_bounds=arena_bounds,
            arena_margin=margin,
            max_steps=max_steps,
            env_config=env_config,
            goal_x_half_extent=getattr(env_config, "goal_x_half_extent", None),
        )
    raise ValueError(f"unknown role: {deploy_config.role!r}")


__all__ = [
    "TerminationPredicate",
    "HouseholdEpisodeSupervisor",
    "default_max_steps",
    "make_cleaner_predicate",
    "make_follower_predicate",
    "make_no_op_predicate",
    "build_predicate_for_config",
    "run_episode",
]


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------

import json
import time
from datetime import datetime, timezone
from pathlib import Path


_BADGE_COLORS = {
    "pickup": "#2E7D32",         # green
    "aid_complete": "#2E7D32",   # green (cleaner finished pickup + aid window)
    "goal_reached": "#2E7D32",
    "collision": "#C62828",      # red
    "out_of_arena": "#C62828",   # red
    "max_steps": "#F9A825",      # yellow
    "keyboard_interrupt": "#616161",  # gray
}


def _utc_iso_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _default_record_dir(role: str) -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return (repo_root / "experiments" / "deployments"
            / f"household_{role}" / _utc_iso_compact())


class HouseholdEpisodeSupervisor:
    """Owns the spin loop, viz, recording, and termination contract for one rollout.

    The node is the ROS-side controller (it publishes ReferenceState
    messages on a timer); this class owns the lifecycle around it.
    """

    def __init__(
        self,
        config: HouseholdDeploymentConfig,
        controller,  # ControllerAdapter | None (None in debug modes / factory mode)
        env_config,
        controller_factory: Optional[
            Callable[[np.ndarray, float], Any]
        ] = None,
        on_motion_start: Optional[Callable[[], None]] = None,
    ):
        # XOR: either a pre-built controller OR a factory that builds
        # one once the follower TF pose is known. The CEM cleaner path
        # uses the factory; PPO / debug paths use the pre-built one.
        if controller is not None and controller_factory is not None:
            raise ValueError(
                "Pass either controller or controller_factory, not both.",
            )
        self.config = config
        self.controller = controller
        self.controller_factory = controller_factory
        self.env_config = env_config
        # Fired once, right before the control timer starts (i.e. the
        # instant the robot begins to move) — used by the comparison
        # harness to start recording only the motion, not the CEM
        # calibration / TF-wait that precedes it.
        self.on_motion_start = on_motion_start

        # Resolve max_steps once so the metadata + predicate agree.
        self.max_steps = (
            int(config.max_steps_override)
            if config.max_steps_override is not None
            else default_max_steps(config.role)
        )

        # Recording reuses the matplotlib figure; if viz is off, force
        # record off and warn (printed at run time).
        self._record = bool(config.record) and bool(config.viz)
        self._viz = bool(config.viz)

        # Resolve output dir: write metadata.json + (optional) MP4 here.
        # Output dir is still useful for metadata + profile.json when
        # not recording video — only require it if either record or
        # profile is set.
        if self._record or config.profile:
            self.output_dir: Path | None = (
                Path(config.record_dir) if config.record_dir
                else _default_record_dir(config.role)
            )
        else:
            self.output_dir = None

        self._started_at: str | None = None
        self._ended_at: str | None = None
        # Per-phase latency accumulator for the main loop. The node has
        # its own PhaseTimer (instrumenting _tick internals); the
        # supervisor enables both when --profile is set.
        self.profiler: PhaseTimer = PhaseTimer(enabled=bool(config.profile))

    # ----- lifecycle -------------------------------------------------------

    def run(self) -> str:
        """Run the episode end-to-end. Returns the termination reason string."""
        import rclpy

        from critic_sharing.deployment.household_node import HouseholdPolicyNode

        predicate = build_predicate_for_config(self.config, self.env_config)

        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init()
        node = HouseholdPolicyNode(
            config=self.config,
            controller=self.controller,
            termination_predicate=predicate,
            start_timer=False,
        )
        if self.config.profile:
            node.profiler.enabled = True

        fig = None
        writer = None
        try:
            # Phase 1 — wait for valid TF, latch dirt.
            self._phase_ready(rclpy, node)

            # Confirm gate.
            if not self.config.no_confirm:
                print(">>> Press Enter to begin episode (Ctrl-C to abort)...",
                      flush=True)
                try:
                    input()
                except (EOFError, KeyboardInterrupt):
                    node.terminated_reason = "keyboard_interrupt"
                    return self._finalize(node, fig, writer)

            # Phase 2 — open figure if --viz; otherwise stay headless.
            if self._viz:
                fig, ax, writer = self._open_viz_and_recorder()
                if self.config.record and not self._record:
                    print("[episode] --record requested without --viz; "
                          "skipping MP4 (recorder needs the figure).",
                          flush=True)
            else:
                if self.config.record:
                    print("[episode] --record disabled because --no-viz was set.",
                          flush=True)
                fig, ax, writer = None, None, None

            # Signal "about to move" so external recorders capture only
            # the motion. Failures here must not abort the episode.
            if self.on_motion_start is not None:
                try:
                    self.on_motion_start()
                except Exception:
                    import traceback
                    print("[episode] on_motion_start hook failed:\n"
                          + traceback.format_exc(), flush=True)

            self._started_at = _utc_iso_compact()
            node.start_control_timer()
            if self._viz:
                self._main_loop(rclpy, node, fig, ax, writer)
            else:
                self._main_loop_headless(rclpy, node)

            return self._finalize(node, fig, writer)
        except KeyboardInterrupt:
            if node.terminated_reason is None:
                node.terminated_reason = "keyboard_interrupt"
            return self._finalize(node, fig, writer)
        except Exception:
            if node.terminated_reason is None:
                node.terminated_reason = "error"
            self._finalize(node, fig, writer)
            raise

    # ----- phase 1 ---------------------------------------------------------

    def _phase_ready(self, rclpy_mod, node) -> None:
        """Block until first valid self + table TF. For the cleaner, also
        latch the virtual dirt to the table's sim position. For the
        follower, dirt is unused — skip the latch.

        Emits a pre-flight warning if (a) the follower's spawn is already
        past the goal line, or (b) the table TF lands outside the sim
        arena bounds — both indicate a setup mistake the user wants to
        see *before* pressing Enter rather than after the robot moves.
        """
        is_cleaner = (self.config.role == "cleaner")
        self_frame = (self.config.cleaner_frame if is_cleaner
                      else self.config.follower_frame)
        # Factory mode (CEM cleaner) also needs the follower's pose so
        # the per-episode V_F can bake it in. PPO / debug modes don't.
        need_follower_for_factory = (
            self.controller_factory is not None
            and is_cleaner
            and self.config.follower_frame != self_frame
        )
        frames_msg = "parent={p}, self={s}, table={t}".format(
            p=self.config.parent_frame, s=self_frame, t=self.config.table_frame,
        )
        if need_follower_for_factory:
            frames_msg += ", follower={f}".format(f=self.config.follower_frame)
        print(
            "[episode] Waiting for TF: {f} (timeout={to:.1f}s)".format(
                f=frames_msg, to=float(self.config.ready_timeout),
            ),
            flush=True,
        )

        deadline = time.monotonic() + float(self.config.ready_timeout)
        while time.monotonic() < deadline:
            rclpy_mod.spin_once(node, timeout_sec=0.05)
            self_lookup = node._lookup_pose(self_frame)
            table_lookup = node._lookup_pose(self.config.table_frame)
            if self_lookup is None or table_lookup is None:
                continue
            follower_lookup = None
            if need_follower_for_factory:
                follower_lookup = node._lookup_pose(self.config.follower_frame)
                if follower_lookup is None:
                    continue

            self_xy_sim, self_yaw_sim = node._world_to_sim_pose(
                self_lookup[0], self_lookup[1]
            )
            table_xy_sim, table_yaw_sim = node._world_to_sim_pose(
                table_lookup[0], table_lookup[1]
            )
            follower_xy_sim = None
            follower_yaw_sim = None
            if follower_lookup is not None:
                follower_xy_sim, follower_yaw_sim = node._world_to_sim_pose(
                    follower_lookup[0], follower_lookup[1]
                )

            if is_cleaner:
                dirt_sim = node.latch_dirt_now()
                if dirt_sim is None:
                    continue  # shouldn't happen — table_lookup was valid
                print(
                    "[episode] Ready: dirt @ sim({dx:+.3f}, {dy:+.3f}), "
                    "self @ sim({sx:+.3f}, {sy:+.3f}, yaw={yaw:+.0f}°)".format(
                        dx=float(dirt_sim[0]), dy=float(dirt_sim[1]),
                        sx=float(self_xy_sim[0]), sy=float(self_xy_sim[1]),
                        yaw=float(np.degrees(self_yaw_sim)),
                    ),
                    flush=True,
                )
            else:
                print(
                    "[episode] Ready: table @ sim({tx:+.3f}, {ty:+.3f}, "
                    "yaw={tyaw:+.0f}°), self @ sim({sx:+.3f}, {sy:+.3f}, "
                    "yaw={yaw:+.0f}°)".format(
                        tx=float(table_xy_sim[0]), ty=float(table_xy_sim[1]),
                        tyaw=float(np.degrees(table_yaw_sim)),
                        sx=float(self_xy_sim[0]), sy=float(self_xy_sim[1]),
                        yaw=float(np.degrees(self_yaw_sim)),
                    ),
                    flush=True,
                )
            if follower_xy_sim is not None:
                print(
                    "[episode] Ready: follower @ sim({fx:+.3f}, {fy:+.3f}, "
                    "yaw={fyaw:+.0f}°)".format(
                        fx=float(follower_xy_sim[0]),
                        fy=float(follower_xy_sim[1]),
                        fyaw=float(np.degrees(follower_yaw_sim)),
                    ),
                    flush=True,
                )

            self._preflight_warnings(self_xy_sim, table_xy_sim)

            if self.controller_factory is not None:
                # Build the CEM adapter now that we know the follower's
                # spawn pose; inject it into the node before the
                # control timer is started by run().
                adapter = self.controller_factory(
                    follower_xy_sim
                    if follower_xy_sim is not None
                    else np.zeros(2, dtype=np.float32),
                    float(follower_yaw_sim)
                    if follower_yaw_sim is not None else 0.0,
                )
                node.set_controller(adapter)
                self.controller = adapter

            return

        raise TimeoutError(
            "Ready phase exceeded {to:.1f}s — TF tree never published both "
            "{self_frame} and {table_frame}.".format(
                to=float(self.config.ready_timeout),
                self_frame=self_frame, table_frame=self.config.table_frame,
            )
        )

    def _preflight_warnings(self, self_xy_sim, table_xy_sim) -> None:
        """Print warnings for setup mistakes that would otherwise turn
        into a confusingly-fast termination once the episode starts.
        Non-fatal: the user can override by pressing Enter anyway.
        """
        x_min = float(self.env_config.world_x_min)
        x_max = float(self.env_config.world_x_max)
        y_min = float(self.env_config.world_y_min)
        y_max = float(self.env_config.world_y_max)

        # Follower already past the goal line?
        if self.config.role == "follower":
            from critic_sharing.scenarios.household.common import (
                rect_circumscribed_radius,
            )
            he = self.env_config.follower_half_extents
            r = rect_circumscribed_radius((float(he[0]), float(he[1])))
            goal_y = y_min + r
            mid_x = 0.5 * (x_min + x_max)
            hx = getattr(self.env_config, "goal_x_half_extent", None)
            if hx is None:
                hx = 0.25 * (x_max - x_min)
            hx = float(hx)
            if float(self_xy_sim[1]) <= goal_y and abs(float(self_xy_sim[0]) - mid_x) <= hx:
                print(
                    "[episode] ⚠️  Follower is already in the centered goal region "
                    "(y_sim={y:+.3f} ≤ goal_y={gy:+.3f}, |x-{mx:+.2f}|={dx:.3f} ≤ "
                    "{hx:.3f}). Episode will terminate immediately. Move the robot "
                    "upward and re-run, or press Enter to override.".format(
                        y=float(self_xy_sim[1]), gy=float(goal_y), mx=mid_x,
                        dx=abs(float(self_xy_sim[0]) - mid_x), hx=hx,
                    ),
                    flush=True,
                )

        # Self already outside arena bounds?
        sx, sy = float(self_xy_sim[0]), float(self_xy_sim[1])
        if not (x_min <= sx <= x_max and y_min <= sy <= y_max):
            print(
                "[episode] ⚠️  Self pose is outside the sim arena "
                "(sim({x:+.3f}, {y:+.3f}) ∉ "
                "[{xmin:+.2f},{xmax:+.2f}]×[{ymin:+.2f},{ymax:+.2f}]). "
                "Check sim_origin_in_world or robot placement.".format(
                    x=sx, y=sy, xmin=x_min, xmax=x_max, ymin=y_min, ymax=y_max,
                ),
                flush=True,
            )

        # Table outside arena bounds?
        tx, ty = float(table_xy_sim[0]), float(table_xy_sim[1])
        if not (x_min <= tx <= x_max and y_min <= ty <= y_max):
            print(
                "[episode] ⚠️  Table pose is outside the sim arena "
                "(sim({x:+.3f}, {y:+.3f}) ∉ "
                "[{xmin:+.2f},{xmax:+.2f}]×[{ymin:+.2f},{ymax:+.2f}]). "
                "Check sim_origin_in_world or table calibration — the "
                "policy was not trained on out-of-bounds table poses.".format(
                    x=tx, y=ty, xmin=x_min, xmax=x_max, ymin=y_min, ymax=y_max,
                ),
                flush=True,
            )

    # ----- phase 2 ---------------------------------------------------------

    def _open_viz_and_recorder(self):
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FFMpegWriter, PillowWriter

        plt.ion()
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 5.5))
        fig.canvas.manager.set_window_title(
            f"household {self.config.role} — episode"
        )

        writer = None
        if self.config.record:
            assert self.output_dir is not None
            self.output_dir.mkdir(parents=True, exist_ok=True)
            try:
                writer = FFMpegWriter(fps=int(round(self.config.rate_hz)),
                                       bitrate=2000)
                writer.setup(fig, str(self.output_dir / "episode.mp4"), dpi=120)
                print(f"[episode] Recording to {self.output_dir / 'episode.mp4'}",
                      flush=True)
            except (FileNotFoundError, RuntimeError) as exc:
                print(f"[episode] ffmpeg unavailable ({exc}); "
                      "falling back to GIF.", flush=True)
                writer = PillowWriter(fps=int(round(self.config.rate_hz)))
                writer.setup(fig, str(self.output_dir / "episode.gif"), dpi=120)
        return fig, ax, writer

    def _main_loop(self, rclpy_mod, node, fig, ax, writer) -> None:
        import matplotlib.pyplot as plt
        from critic_sharing.scenarios.household.render import (
            draw_arena, draw_cleaner, draw_dirt, draw_follower,
            draw_goal_line, draw_table,
        )

        cleaner_he = tuple(self.env_config.cleaner_half_extents)
        follower_he = tuple(self.env_config.follower_half_extents)
        table_he = tuple(self.env_config.table_half_extents)

        def _redraw():
            ax.clear()
            draw_arena(ax, self.env_config)
            draw_goal_line(ax, self.env_config)

            state = node.latest_state
            if not state:
                ax.set_title("waiting for first tick…", fontsize=9)
                fig.canvas.draw_idle()
                return

            draw_table(ax, state["table_xy_sim"], state["table_yaw_sim"], table_he)
            draw_dirt(ax, state["dirt_xy_sim"])

            self_xy = state["cleaner_xy_sim"]
            self_yaw = state["cleaner_yaw_sim"]
            if self.config.role == "cleaner":
                draw_cleaner(ax, self_xy, self_yaw, cleaner_he)
            else:
                draw_follower(ax, self_xy, self_yaw, follower_he)

            # Policy command velocity arrow (sim frame).
            act = state["last_action_sim"]
            if float(np.linalg.norm(act[:2])) > 1e-4:
                ax.annotate(
                    "", xy=(float(self_xy[0] + act[0] * 0.5),
                           float(self_xy[1] + act[1] * 0.5)),
                    xytext=(float(self_xy[0]), float(self_xy[1])),
                    arrowprops={"arrowstyle": "->", "color": "#1976D2",
                                "lw": 1.5}, zorder=6,
                )

            tick = int(state["tick_count"])
            if self.config.role == "cleaner":
                dist = float(np.linalg.norm(state["dirt_xy_sim"] - state["cleaner_xy_sim"]))
                metric = f"dist→dirt={dist:.2f}"
            else:
                y_min = float(self.env_config.world_y_min)
                from critic_sharing.scenarios.household.common import (
                    rect_circumscribed_radius,
                )
                r = rect_circumscribed_radius(
                    (float(follower_he[0]), float(follower_he[1]))
                )
                metric = f"dist→goal={float(self_xy[1]) - (y_min + r):+.2f}"
            ax.set_title(f"t={tick:03d}/{self.max_steps}  {metric}", fontsize=9)

            reason = node.terminated_reason
            if reason is not None:
                ax.text(0.97, 0.03, reason.upper(),
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=11, fontweight="bold",
                        color="white",
                        bbox={"facecolor": _BADGE_COLORS.get(reason, "#212121"),
                              "edgecolor": "none", "pad": 4})

            fig.canvas.draw_idle()

        # Number of post-termination frames to keep recording so the badge
        # stays on screen at the end of the saved video.
        TAIL_FRAMES = 10
        tail = 0
        spin_timeout = 1.0 / float(self.config.rate_hz)
        try:
            while rclpy_mod.ok() and plt.fignum_exists(fig.number):
                loop_t0 = time.perf_counter()
                with self.profiler.measure("spin"):
                    rclpy_mod.spin_once(node, timeout_sec=spin_timeout)
                with self.profiler.measure("redraw"):
                    _redraw()
                if writer is not None:
                    with self.profiler.measure("grab_frame"):
                        writer.grab_frame()
                with self.profiler.measure("plt_pause"):
                    plt.pause(0.001)
                self.profiler.record(
                    "loop_total", (time.perf_counter() - loop_t0) * 1000.0
                )
                if node.terminated_reason is not None:
                    tail += 1
                    if tail >= TAIL_FRAMES:
                        break
        except KeyboardInterrupt:
            if node.terminated_reason is None:
                node.terminated_reason = "keyboard_interrupt"

    def _main_loop_headless(self, rclpy_mod, node) -> None:
        """No matplotlib. Just spin the executor and watch for termination.

        Used when ``--no-viz`` is set, to free the executor for the
        TransformListener so it can keep the TF buffer fresh. spin_once
        is given a short timeout so we check the termination flag and
        Ctrl-C condition promptly.
        """
        TAIL_FRAMES = 5
        tail = 0
        spin_timeout = max(1.0 / float(self.config.rate_hz) / 5.0, 0.005)
        try:
            while rclpy_mod.ok():
                loop_t0 = time.perf_counter()
                with self.profiler.measure("spin"):
                    rclpy_mod.spin_once(node, timeout_sec=spin_timeout)
                self.profiler.record(
                    "loop_total", (time.perf_counter() - loop_t0) * 1000.0
                )
                if node.terminated_reason is not None:
                    tail += 1
                    if tail >= TAIL_FRAMES:
                        break
        except KeyboardInterrupt:
            if node.terminated_reason is None:
                node.terminated_reason = "keyboard_interrupt"

    # ----- shutdown --------------------------------------------------------

    def _finalize(self, node, fig, writer) -> str:
        import rclpy

        # 5× zero-vel flurry at the control rate — cheap insurance.
        try:
            dt = 1.0 / float(self.config.rate_hz)
            for _ in range(5):
                node.publish_zero_velocity()
                time.sleep(dt)
        except Exception:
            pass

        # Finalize writer + metadata.
        if writer is not None:
            try:
                writer.finish()
            except Exception:
                pass
        if fig is not None:
            try:
                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception:
                pass

        self._ended_at = _utc_iso_compact()
        reason = node.terminated_reason or "unknown"
        if self.output_dir is not None:
            self._write_metadata(node, reason)

        if self.config.profile:
            target_ms = 1000.0 / float(self.config.rate_hz)
            print("[episode] Supervisor main-loop profile:", flush=True)
            print(self.profiler.report(target_period_ms=target_ms), flush=True)
            print("[episode] Node _tick profile:", flush=True)
            print(node.profiler.report(target_period_ms=target_ms), flush=True)
            if self.output_dir is not None:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                prof_path = self.output_dir / "profile.json"
                prof_path.write_text(json.dumps({
                    "rate_hz": float(self.config.rate_hz),
                    "supervisor": self.profiler.to_dict(),
                    "node": node.profiler.to_dict(),
                }, indent=2))
                print(f"[episode] Profile JSON: {prof_path}", flush=True)

        print(f"[episode] Done: reason={reason}, ticks={node.tick_count}",
              flush=True)

        try:
            node.destroy_node()
        finally:
            if getattr(self, "_owns_rclpy", True):
                try:
                    rclpy.shutdown()
                except Exception:
                    pass
        return reason

    def _write_metadata(self, node, reason: str) -> None:
        assert self.output_dir is not None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        state = dict(node.latest_state)
        out = {
            "role": self.config.role,
            "controller": str(self.config.controller),
            "model_path": str(self.config.model_path),
            "sim_origin_in_world": list(self.config.sim_origin_in_world),
            "parent_z_down": bool(self.config.parent_z_down),
            "terminated_reason": reason,
            "tick_count": int(node.tick_count),
            "max_steps": int(self.max_steps),
            "arena_margin": float(self.config.arena_margin),
            "rate_hz": float(self.config.rate_hz),
            "started_at": self._started_at,
            "ended_at": self._ended_at,
            "goto_origin": bool(self.config.goto_origin),
            "goto_dirt": bool(self.config.goto_dirt),
        }
        init = getattr(node, "initial_state", None)
        if init:
            out["initial_state"] = {
                "self_xy_sim": [float(v) for v in init["cleaner_xy_sim"]],
                "self_yaw_sim": float(init["cleaner_yaw_sim"]),
                "table_xy_sim": [float(v) for v in init["table_xy_sim"]],
                "table_yaw_sim": float(init["table_yaw_sim"]),
                "dirt_xy_sim": [float(v) for v in init["dirt_xy_sim"]],
            }
        if state:
            out.update({
                "final_cleaner_xy_sim": [float(v) for v in state["cleaner_xy_sim"]],
                "final_cleaner_yaw_sim": float(state["cleaner_yaw_sim"]),
                "final_table_xy_sim": [float(v) for v in state["table_xy_sim"]],
                "final_table_yaw_sim": float(state["table_yaw_sim"]),
                "dirt_xy_sim": [float(v) for v in state["dirt_xy_sim"]],
            })
        meta_path = self.output_dir / "metadata.json"
        meta_path.write_text(json.dumps(out, indent=2))
        print(f"[episode] Metadata saved to {meta_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _CemResources:
    """Pre-loaded artifacts + calibration outputs for the cleaner CEM path.

    Built once at deploy startup (PPO loads + OOD load + unguided
    calibration mini-eval). Everything that doesn't depend on the
    follower's measured spawn pose lives here. The final CEMAdapter is
    constructed from these + the TF-measured follower pose inside
    :func:`_build_cem_adapter_from_pose`, called from the supervisor's
    Phase 1.
    """
    cleaner_model: Any
    follower_model: Any
    cleaner_env_config: Any
    follower_env_config: Any
    cem_config: Any
    leader_gamma: float
    utility_type: str
    ood_metric: Any  # build_household_ood_metric(...) | None
    effective_alpha: float
    leader_range: Optional[tuple[float, float]]
    follower_range: Optional[tuple[float, float]]


# Calibrated normalization ranges depend only on the two checkpoints and
# the calibration parameters — not on anything that changes between
# rollouts. Cache them so a multi-rollout session (e.g. the comparison
# harness) calibrates once and reuses, instead of paying ~30 sim eps × 2
# per CEM rollout. Process-local; cleared when the process exits.
_RANGE_CACHE: dict = {}


def _load_cem_resources(
    config: HouseholdDeploymentConfig,
    *,
    cleaner_env_config,
    follower_env_config,
) -> _CemResources:
    """Heavy upfront work for ``--controller cem``: PPOs, OOD, calibration.

    Runs **before** the rclpy node starts so the calibration pre-pass
    (a few tens of seconds of sim) doesn't compete with TF for executor
    time. The leader / follower ranges put L_norm and F_norm in [0, 1]
    inside :class:`JointUtilityCost` — same contract as the eval
    pipeline's :func:`calibrate_ranges`.

    Raises :class:`ValueError` early if the cleaner / follower
    checkpoints aren't available.
    """
    from stable_baselines3 import PPO

    from critic_sharing.config.resolver import (
        load_cost_config, load_planner_config,
    )

    if config.role != "cleaner":
        raise ValueError(
            "--controller cem is cleaner-only (got role={!r}).".format(config.role)
        )
    if not config.model_path:
        raise ValueError("--controller cem requires --model-path (cleaner PPO).")
    if not config.follower_model_path:
        raise ValueError("--controller cem requires --follower-model-path.")

    print(f"[episode] Loading cleaner PPO from {config.model_path}", flush=True)
    cleaner_model = PPO.load(config.model_path)
    print(f"[episode] Loading follower PPO from {config.follower_model_path}",
          flush=True)
    follower_model = PPO.load(config.follower_model_path)

    cost_config = load_cost_config("household")
    # Deployment-scoped CEM overrides (shrink per-replan cost so a replan
    # fits the control period; e.g. replan_interval=1 to replan every
    # tick). Only the deployment planner is affected — eval reads
    # planner.yaml directly.
    cem_overrides = {
        k: v for k, v in {
            "num_samples": config.cem_num_samples,
            "horizon": config.cem_horizon,
            "num_iters": config.cem_num_iters,
            "replan_interval": config.cem_replan_interval,
        }.items() if v is not None
    }
    cem_config = load_planner_config("household", overrides=cem_overrides or None)
    if cem_overrides:
        print(f"[episode] CEM planner overrides: {cem_overrides}", flush=True)
    leader_gamma = float(getattr(cost_config, "leader_gamma", 0.99))
    utility_type = getattr(cost_config, "utility_type", "sum")
    effective_alpha = float(config.alpha) if config.alpha is not None else 1.0

    ood_metric = None
    if config.vae_path:
        from critic_sharing.scenarios.household.ood_query import (
            build_household_ood_metric,
        )
        print(f"[episode] Loading VAE from {config.vae_path} "
              f"(α={effective_alpha:.2f})", flush=True)
        ood_metric = build_household_ood_metric(str(config.vae_path))

    leader_range: Optional[tuple[float, float]] = None
    follower_range: Optional[tuple[float, float]] = None
    if getattr(cost_config, "normalize_returns", True):
        n_eps = int(getattr(cost_config, "calibration_episodes", 30))
        success_only = bool(getattr(cost_config, "calibrate_on_success_only", True))
        pool = "successful" if success_only else "all"

        cache_key = (
            str(config.model_path), str(config.follower_model_path),
            n_eps, success_only, round(float(leader_gamma), 6),
        )
        cached = _RANGE_CACHE.get(cache_key)
        if cached is not None:
            leader_range, follower_range = cached
            print(f"[episode] Reusing cached calibration ranges: "
                  f"leader=({leader_range[0]:+.3f}, {leader_range[1]:+.3f}), "
                  f"follower=({follower_range[0]:+.3f}, {follower_range[1]:+.3f}).",
                  flush=True)
        else:
            from critic_sharing.scenarios.household.nav_env import (
                HouseholdNavigatorEnv,
            )
            from critic_sharing.scenarios.household.push_env import (
                HouseholdCleanerEnv,
            )
            from critic_sharing.utils.calibration import calibrate_range

            print(f"[episode] Calibrating leader_range ({pool}) over {n_eps} "
                  f"naive sim episodes...", flush=True)
            leader_env = HouseholdCleanerEnv(cleaner_env_config)
            try:
                leader_range = calibrate_range(
                    leader_env, cleaner_model, n_episodes=n_eps,
                    gamma=leader_gamma, success_only=success_only,
                )
            finally:
                leader_env.close()
            print(f"[episode]   leader_range = "
                  f"({leader_range[0]:+.3f}, {leader_range[1]:+.3f})", flush=True)

            print(f"[episode] Calibrating follower_range ({pool}) over {n_eps} "
                  f"naive sim episodes...", flush=True)
            follower_env = HouseholdNavigatorEnv(follower_env_config)
            try:
                follower_range = calibrate_range(
                    follower_env, follower_model, n_episodes=n_eps,
                    gamma=leader_gamma, success_only=success_only,
                )
            finally:
                follower_env.close()
            print(f"[episode]   follower_range = "
                  f"({follower_range[0]:+.3f}, {follower_range[1]:+.3f})",
                  flush=True)

            _RANGE_CACHE[cache_key] = (leader_range, follower_range)
    else:
        print("[episode] normalize_returns=false; skipping calibration.",
              flush=True)

    return _CemResources(
        cleaner_model=cleaner_model,
        follower_model=follower_model,
        cleaner_env_config=cleaner_env_config,
        follower_env_config=follower_env_config,
        cem_config=cem_config,
        leader_gamma=leader_gamma,
        utility_type=utility_type,
        ood_metric=ood_metric,
        effective_alpha=effective_alpha,
        leader_range=leader_range,
        follower_range=follower_range,
    )


def _build_cem_adapter_from_pose(
    resources: _CemResources,
    *,
    follower_spawn_pos: np.ndarray,
    follower_spawn_theta: float,
    planner_seed: Optional[int] = None,
):
    """Finalize a :class:`CEMAdapter` with per-episode V_F.

    The follower's measured spawn pose ``(x, y, θ)`` is baked into the
    V_F closure via :func:`build_household_follower_value_fn` — same
    semantics as the eval pipeline (warehouse / mobile_arm / OA /
    household are all per-episode now). One PPO forward per CEM
    candidate; no distilled aggregator.
    """
    import gymnasium as gym
    import torch as th

    from critic_sharing.deployment.controllers import CEMAdapter
    from critic_sharing.planning.cem import CEMController
    from critic_sharing.planning.costs import JointUtilityCost
    from critic_sharing.planning.distributions import make_distribution
    from critic_sharing.planning.prior import PolicyPrior
    from critic_sharing.scenarios.household import (
        build_household_follower_value_fn,
    )
    from critic_sharing.scenarios.household.push_env import cleaner_dynamics_batch

    clean_cfg = resources.cleaner_env_config
    nav_cfg = resources.follower_env_config
    cem_cfg = resources.cem_config
    cleaner_model = resources.cleaner_model

    value_fn = build_household_follower_value_fn(
        follower_model=resources.follower_model,
        follower_config=nav_cfg,
        follower_spawn_pos=np.asarray(follower_spawn_pos, dtype=np.float32),
        follower_spawn_theta=float(follower_spawn_theta),
    )

    def follower_value_fn(obs_batch):
        return np.asarray(value_fn(obs_batch), dtype=np.float32).reshape(-1)

    def leader_value_fn(obs_batch):
        obs_t, _ = cleaner_model.policy.obs_to_tensor(
            np.asarray(obs_batch, dtype=np.float32),
        )
        with th.no_grad():
            values = cleaner_model.policy.predict_values(obs_t)
        return values.detach().cpu().numpy().reshape(-1)

    cost = JointUtilityCost(
        follower_value_fn=follower_value_fn,
        leader_gamma=resources.leader_gamma,
        utility_type=resources.utility_type,
        leader_range=resources.leader_range,
        follower_range=resources.follower_range,
        leader_value_fn=leader_value_fn,
        ood_metric=resources.ood_metric,
        alpha=resources.effective_alpha,
    )

    def dynamics_fn(obs_batch, act_batch):
        return cleaner_dynamics_batch(obs_batch, act_batch, clean_cfg)

    action_space = gym.spaces.Box(
        low=np.array(
            [-clean_cfg.max_speed, -clean_cfg.max_speed,
             -clean_cfg.max_yaw_rate], dtype=np.float32,
        ),
        high=np.array(
            [clean_cfg.max_speed, clean_cfg.max_speed,
             clean_cfg.max_yaw_rate], dtype=np.float32,
        ),
        dtype=np.float32,
    )
    prior = PolicyPrior(
        predict_fn=lambda obs: cleaner_model.predict(obs, deterministic=True)[0],
        predict_batch_fn=lambda obs_batch: cleaner_model.predict(
            np.asarray(obs_batch, dtype=np.float32), deterministic=True,
        )[0],
    )

    controller = CEMController(
        action_space=action_space,
        dynamics_fn=dynamics_fn,
        cost=cost,
        distribution=make_distribution(
            cem_cfg.distribution,
            noise_sigma=cem_cfg.noise_sigma,
            min_std=cem_cfg.min_std,
        ),
        horizon=cem_cfg.horizon,
        num_samples=cem_cfg.num_samples,
        num_iters=cem_cfg.num_iters,
        elite_frac=cem_cfg.elite_frac,
        replan_interval=cem_cfg.replan_interval,
        prior=prior,
        seed=planner_seed,
    )
    mode_label = "value_and_ood" if resources.ood_metric is not None else "value"
    fx = float(follower_spawn_pos[0])
    fy = float(follower_spawn_pos[1])
    print(f"[episode] CEM ready: mode={mode_label}, "
          f"horizon={cem_cfg.horizon}, num_samples={cem_cfg.num_samples}, "
          f"replan_interval={cem_cfg.replan_interval}, "
          f"follower_spawn=({fx:+.3f}, {fy:+.3f}, "
          f"θ={float(follower_spawn_theta):+.2f})", flush=True)
    return CEMAdapter(
        controller=controller,
        cost=cost,
        cleaner_env_config=clean_cfg,
        leader_gamma=resources.leader_gamma,
    )


def run_episode(
    config: HouseholdDeploymentConfig,
    *,
    on_motion_start: Optional[Callable[[], None]] = None,
) -> str:
    """CLI-facing helper: build supervisor, load controller, run.

    ``on_motion_start`` (optional) fires once, right before the robot
    starts moving — after model load / CEM calibration / TF-wait.
    """
    from critic_sharing.config.resolver import load_env_config
    from critic_sharing.deployment.controllers import PPOPolicyAdapter

    # The env config provides arena bounds + half-extents; we read it
    # for the role being deployed.
    scenario_role = "leader" if config.role == "cleaner" else "follower"
    env_config = load_env_config("household", scenario_role)

    # Validate controller × debug-mode mutex (grill Q7a).
    if config.controller == "cem" and (config.goto_origin or config.goto_dirt):
        raise ValueError(
            "--controller cem is mutually exclusive with --goto-origin / "
            "--goto-dirt. Use --controller policy for debug modes."
        )

    controller = None
    controller_factory: Optional[Callable[[np.ndarray, float], Any]] = None

    if config.goto_origin or config.goto_dirt:
        controller = None
    elif config.controller == "cem":
        # Heavy work (PPO loads + OOD load + calibration mini-eval)
        # happens here, before the rclpy node starts. The per-episode
        # V_F closure is deferred until Phase 1 measures the follower's
        # spawn pose from TF — captured by `controller_factory`.
        resources = _load_cem_resources(
            config,
            cleaner_env_config=load_env_config("household", "leader"),
            follower_env_config=load_env_config("household", "follower"),
        )

        def controller_factory(
            follower_pos: np.ndarray, follower_theta: float,
        ):
            return _build_cem_adapter_from_pose(
                resources,
                follower_spawn_pos=follower_pos,
                follower_spawn_theta=follower_theta,
            )
    elif config.controller == "policy":
        if not config.model_path:
            raise ValueError("HouseholdDeploymentConfig.model_path must be set.")
        from stable_baselines3 import PPO
        controller = PPOPolicyAdapter(PPO.load(config.model_path))
    else:
        raise ValueError(
            f"Unknown controller: {config.controller!r} "
            "(expected 'policy' or 'cem')."
        )

    supervisor = HouseholdEpisodeSupervisor(
        config=config,
        controller=controller,
        controller_factory=controller_factory,
        env_config=env_config,
        on_motion_start=on_motion_start,
    )
    return supervisor.run()
