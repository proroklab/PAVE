"""Hardware comparison harness: naive vs CEM, side-by-side.

Per session: ``--repeats N`` pairs. Per pair: 4 rollouts (method-A
cleaner, method-A follower, [reset view], method-B cleaner, method-B
follower). Each rollout is bracketed by a rosbag (``ros2 bag record
-a``) and a USB webcam recording (OpenCV).

Run via the ``critic-sharing deploy household-compare`` CLI subcommand;
this module only owns the orchestration.

Output layout::

    experiments/comparisons/<ts>/
      manifest.json
      pair_001/
        initial_conditions.json
        naive_cleaner/{bag/, webcam.mp4, metadata.json, profile.json}
        naive_follower/{...}
        ours_cleaner/{...}
        ours_follower/{...}
      pair_002/
        ...

Aborted/redone rollouts are renamed ``<dir>.discarded.<idx>`` to keep
the canonical directory clean on retry.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from critic_sharing.config.schema import HouseholdDeploymentConfig


# ---------------------------------------------------------------------------
# Method spec — what controller drives each role for each method label.
# ---------------------------------------------------------------------------

METHODS: dict = {
    "naive": {"cleaner": "policy", "follower": "policy"},
    "ours":  {"cleaner": "cem",    "follower": "policy"},
}


def _utc_iso_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@dataclass(frozen=True)
class HouseholdComparisonConfig:
    """Comparison-level config — layered on top of HouseholdDeploymentConfig."""

    repeats: int = 1
    order: Tuple[str, ...] = ("naive", "ours")
    output_dir: Optional[Path] = None
    # Camera spec: name substring (stable across reconnect), index, or
    # /dev path. Resolved via recorders.resolve_camera at open time.
    webcam_device: object = "OsmoAction5pro"
    webcam_fps: int = 30
    webcam_preview: bool = True
    bag_enabled: bool = True
    webcam_enabled: bool = True

    def __post_init__(self):
        for m in self.order:
            if m not in METHODS:
                raise ValueError(
                    f"unknown method label: {m!r} (choose from {sorted(METHODS)})"
                )
        if self.repeats < 1:
            raise ValueError("repeats must be >= 1")
        if len(self.order) != len(set(self.order)):
            raise ValueError(f"order has duplicates: {self.order}")


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------

class HouseholdComparisonSupervisor:
    """Drives the (pair × method × role) matrix with recording brackets."""

    def __init__(
        self,
        base_config: HouseholdDeploymentConfig,
        compare_config: HouseholdComparisonConfig,
    ):
        self.base = base_config
        self.cmp = compare_config
        # One webcam for the whole session (opened in run()); reopening
        # per rollout races the V4L2 release and only the first rollout
        # ends up with a video.
        self._webcam = None
        self.session_dir = (
            Path(compare_config.output_dir)
            if compare_config.output_dir is not None
            else (Path("experiments/comparisons") / _utc_iso_compact())
        )

    # ----- public entry point ----------------------------------------------

    def run(self) -> Path:
        import rclpy

        self.session_dir.mkdir(parents=True, exist_ok=True)
        manifest: dict = {
            "started_at": _utc_iso_compact(),
            "order": list(self.cmp.order),
            "repeats": self.cmp.repeats,
            "webcam_device": self.cmp.webcam_device,
            "base_config": self._base_summary(),
            "pairs": [],
        }
        print(f"[compare] Session dir: {self.session_dir}", flush=True)
        # The harness owns rclpy for the full session — sub-functions
        # (capture_initial_conditions, run_reset_view, the per-rollout
        # episode supervisor) detect this via rclpy.ok() and skip their
        # own init/shutdown. One init per session, one shutdown.
        owned_rclpy = not rclpy.ok()
        if owned_rclpy:
            rclpy.init()
        self._open_webcam()
        try:
            for pair_idx in range(self.cmp.repeats):
                pair_num = pair_idx + 1
                pair_dir = self.session_dir / f"pair_{pair_num:03d}"
                pair_dir.mkdir(parents=True, exist_ok=True)
                if pair_idx > 0:
                    self._between_pair_prompt(pair_num)
                init = self._capture_initial_conditions(pair_num)
                (pair_dir / "initial_conditions.json").write_text(
                    json.dumps(_init_to_json(init), indent=2)
                )
                pair_info: dict = {
                    "pair": pair_num,
                    "initial_conditions": _init_to_json(init),
                    "rollouts": [],
                }
                for m_idx, method in enumerate(self.cmp.order):
                    if m_idx > 0:
                        self._reset_view(init)
                    for role in ("cleaner", "follower"):
                        if role == "follower":
                            self._between_roles_prompt(method)
                        else:
                            self._before_leader_prompt(method)
                        rollout = self._run_rollout(
                            method=method, role=role, pair_dir=pair_dir,
                        )
                        pair_info["rollouts"].append(rollout)
                manifest["pairs"].append(pair_info)
                # Flush manifest after each pair so partial sessions are
                # still useful if Ctrl-C lands between rollouts.
                self._write_manifest(manifest)
        except KeyboardInterrupt:
            print("\n[compare] Session aborted by KeyboardInterrupt.", flush=True)
            manifest["aborted"] = True
        finally:
            manifest["ended_at"] = _utc_iso_compact()
            self._write_manifest(manifest)
            print(f"[compare] Session manifest: {self.session_dir / 'manifest.json'}",
                  flush=True)
            if self._webcam is not None:
                self._webcam.close()
                self._webcam = None
            if owned_rclpy:
                try:
                    rclpy.shutdown()
                except Exception:
                    pass
        return self.session_dir

    def _open_webcam(self) -> None:
        """Open the session-wide webcam once. Failure is non-fatal — the
        operator can run without video."""
        if not self.cmp.webcam_enabled:
            return
        from critic_sharing.deployment.recorders import (
            RecorderError, SessionWebcam,
        )
        cam = SessionWebcam(
            device=self.cmp.webcam_device,
            fps=self.cmp.webcam_fps,
            preview=self.cmp.webcam_preview,
        )
        try:
            cam.open()
        except RecorderError as e:
            print(f"[compare] Webcam unavailable ({e}); continuing without "
                  "video.", flush=True)
            return
        print(f"[compare] Webcam open (device {self.cmp.webcam_device}).",
              flush=True)
        self._webcam = cam

    # ----- prompts ---------------------------------------------------------

    def _capture_initial_conditions(self, pair_num: int) -> dict:
        from critic_sharing.deployment.household_viz_node import (
            capture_initial_conditions,
        )
        prompt = (f"\n=== Pair {pair_num}/{self.cmp.repeats} ===\n"
                  ">>> Place objects (cleaner, follower, table) in their "
                  "starting configuration.\n>>> Press Enter to capture "
                  "initial conditions...")
        print(prompt, flush=True)
        input()
        print("[compare] Reading TF...", flush=True)
        init = capture_initial_conditions(
            self.base, timeout=float(self.base.ready_timeout),
        )
        print(
            "[compare] Captured initial conditions:\n"
            "  cleaner  sim=({cx:+.3f}, {cy:+.3f}, yaw={cyaw:+.0f}°)\n"
            "  follower sim=({fx:+.3f}, {fy:+.3f}, yaw={fyaw:+.0f}°)\n"
            "  table    sim=({tx:+.3f}, {ty:+.3f}, yaw={tyaw:+.0f}°)".format(
                cx=init["cleaner_xy_sim"][0], cy=init["cleaner_xy_sim"][1],
                cyaw=float(np.degrees(init["cleaner_yaw_sim"])),
                fx=init["follower_xy_sim"][0], fy=init["follower_xy_sim"][1],
                fyaw=float(np.degrees(init["follower_yaw_sim"])),
                tx=init["table_xy_sim"][0], ty=init["table_xy_sim"][1],
                tyaw=float(np.degrees(init["table_yaw_sim"])),
            ),
            flush=True,
        )
        return init

    def _before_leader_prompt(self, method: str) -> None:
        print(f"\n--- {method.upper()} LEADER ---\n"
              f">>> Press Enter to roll out the {method} leader...",
              flush=True)
        input()

    def _between_roles_prompt(self, method: str) -> None:
        print(f"\n--- {method.upper()} FOLLOWER ---\n"
              f">>> {method} leader done. Remove the leader from the arena, "
              f"then press Enter to roll out the {method} follower...",
              flush=True)
        input()

    def _between_pair_prompt(self, pair_num: int) -> None:
        print(f"\n>>> Pair {pair_num}: reset the arena (initial conditions "
              "may differ from the last pair). Press Enter to continue...",
              flush=True)
        input()

    def _reset_view(self, init: dict) -> None:
        """Auto-drive both robots back to the captured initial poses.

        The operator re-introduces the cleaner manually, then this
        helper runs a per-robot P controller (12 cm / 12° tolerance, 1 s
        hold, 30 s timeout) until convergence. On timeout we prompt
        redo/skip/abort.
        """
        from critic_sharing.deployment.household_reset import reset_to_ghosts

        print(
            "\n>>> Re-introduce the cleaner robot, then press Enter to "
            "start auto-reset...",
            flush=True,
        )
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            raise

        while True:
            result = reset_to_ghosts(
                self.base, init,
                pos_tol=0.12, yaw_tol_deg=12.0, hold_ticks=10,
                timeout=30.0, show_viz=bool(self.base.viz),
            )
            if result == "converged":
                return
            if result == "keyboard_interrupt":
                raise KeyboardInterrupt("Auto-reset aborted by user")
            # timeout — ask the operator.
            choice = self._reset_timeout_prompt()
            if choice == "redo":
                continue
            if choice == "skip":
                # Fall back to manual confirmation with the static
                # ghost view; lets the operator nudge by hand.
                from critic_sharing.deployment.household_viz_node import (
                    run_reset_view,
                )
                run_reset_view(self.base, init)
                return
            raise KeyboardInterrupt("Auto-reset aborted by user")

    def _reset_timeout_prompt(self) -> str:
        prompt = (
            "\n[compare] Auto-reset timed out.\n"
            ">>> [r]edo / [s]kip to manual reset / [a]bort? "
        )
        while True:
            try:
                ans = input(prompt).strip().lower()
            except (EOFError, KeyboardInterrupt):
                ans = "a"
            if ans in ("r", "redo"):
                return "redo"
            if ans in ("s", "skip"):
                return "skip"
            if ans in ("a", "abort"):
                return "abort"
            print("  Please answer r / s / a.", flush=True)

    # ----- per-rollout orchestration ---------------------------------------

    def _run_rollout(
        self, method: str, role: str, pair_dir: Path,
    ) -> dict:
        """One bracketed episode: rosbag + webcam + run_episode."""
        from critic_sharing.deployment.household_episode import run_episode
        from critic_sharing.deployment.recorders import (
            RecorderError, RosbagRecorder,
        )

        controller = METHODS[method][role]
        attempt = 0
        while True:
            attempt += 1
            rollout_dir = pair_dir / f"{method}_{role}"
            if rollout_dir.exists():
                # Rename the previous (failed) attempt out of the way.
                self._move_to_discarded(rollout_dir, attempt - 1)
            rollout_dir.mkdir(parents=True, exist_ok=True)

            cfg = self._make_rollout_config(
                method=method, role=role, controller=controller,
                rollout_dir=rollout_dir,
            )

            bag = None
            if self.cmp.bag_enabled:
                bag = RosbagRecorder(output_dir=rollout_dir)

            started_at = _utc_iso_compact()
            recorder_errors: list = []
            webcam_active = False
            recording_started = False

            def _begin_recording() -> None:
                # Fired by run_episode the instant the robot starts moving,
                # so the recording captures only the motion — not the CEM
                # calibration / TF-wait that precedes it.
                nonlocal bag, webcam_active, recording_started
                if recording_started:
                    return
                recording_started = True
                if bag is not None:
                    try:
                        bag.start()
                    except RecorderError as e:
                        recorder_errors.append(("bag", str(e)))
                        print(f"[compare] rosbag start failed: {e}", flush=True)
                        bag = None
                if self._webcam is not None:
                    try:
                        self._webcam.begin_segment(rollout_dir)
                        webcam_active = True
                    except RecorderError as e:
                        recorder_errors.append(("webcam", str(e)))
                        print(f"[compare] webcam segment failed: {e}", flush=True)

            reason = None
            episode_error: Optional[BaseException] = None
            try:
                reason = run_episode(cfg, on_motion_start=_begin_recording)
            except KeyboardInterrupt:
                # run_episode usually catches KI itself, but defend in
                # depth — if we get here, the rollout is aborted.
                reason = "keyboard_interrupt"
            except Exception as e:
                import traceback
                episode_error = e
                reason = "error"
                print(
                    f"\n[compare] {method}/{role} raised:\n"
                    + traceback.format_exc(),
                    flush=True,
                )
            finally:
                if bag is not None:
                    bag.stop()
                fw, fd = (0, 0)
                if webcam_active:
                    fw, fd = self._webcam.end_segment()
                ended_at = _utc_iso_compact()

            result = {
                "method": method,
                "role": role,
                "controller": controller,
                "rollout_dir": str(rollout_dir.relative_to(self.session_dir)),
                "terminated_reason": reason,
                "started_at": started_at,
                "ended_at": ended_at,
                "webcam_frames_written": int(fw),
                "webcam_frames_dropped": int(fd),
                "recorder_errors": recorder_errors,
                "attempts": attempt,
            }
            if episode_error is not None:
                result["error"] = repr(episode_error)

            choice = self._post_rollout_prompt(method, role, reason)
            if choice == "redo":
                if bag is not None:
                    bag.discard()
                if webcam_active:
                    self._webcam.discard_segment()
                self._move_to_discarded(rollout_dir, attempt)
                continue
            if choice == "abort":
                # Surface as a KeyboardInterrupt so the session loop
                # exits cleanly through its except clause.
                raise KeyboardInterrupt(
                    f"User aborted at {method}/{role} (attempt {attempt})"
                )
            return result

    # ----- prompts: post-rollout -------------------------------------------

    def _post_rollout_prompt(
        self, method: str, role: str, reason: Optional[str],
    ) -> str:
        """After a rollout, decide whether to redo / continue / abort.

        For clean terminations (pickup, goal_reached, max_steps,
        out_of_arena) we auto-continue. For ``keyboard_interrupt`` or
        ``error`` we ask the operator.
        """
        if reason not in ("keyboard_interrupt", "error", None):
            print(
                f"[compare] {method}/{role} terminated: {reason}",
                flush=True,
            )
            return "continue"
        prompt = (
            f"\n[compare] {method}/{role} ended with reason='{reason}'.\n"
            ">>> [r]edo / [c]ontinue / [a]bort? "
        )
        while True:
            try:
                ans = input(prompt).strip().lower()
            except (EOFError, KeyboardInterrupt):
                ans = "a"
            if ans in ("r", "redo"):
                return "redo"
            if ans in ("c", "continue", ""):
                return "continue"
            if ans in ("a", "abort"):
                return "abort"
            print("  Please answer r / c / a.", flush=True)

    # ----- helpers ---------------------------------------------------------

    def _make_rollout_config(
        self, method: str, role: str, controller: str, rollout_dir: Path,
    ) -> HouseholdDeploymentConfig:
        """Per-rollout HouseholdDeploymentConfig (no_confirm + role + ...)."""
        # The comparison harness owns its own prompts; the episode
        # supervisor's "press Enter" gate should NOT fire.
        #
        # ``model_path`` in the base config is the cleaner PPO checkpoint;
        # the follower PPO lives at ``follower_model_path``. Swap them
        # in based on the role being deployed so the episode supervisor
        # loads the correct policy.
        if role == "cleaner":
            model_path = self.base.model_path
            frame = self.base.cleaner_frame
        else:
            model_path = self.base.follower_model_path
            frame = self.base.follower_frame
        # Each robot listens on /<frame>/reference_state — the
        # single-shot deploy uses a hardcoded cmd_topic, but the
        # comparison harness has to switch per-role.
        cmd_topic = f"/{frame}/reference_state"
        return replace(
            self.base,
            role=role,
            episode=True,
            controller=controller,
            model_path=model_path,
            cmd_topic=cmd_topic,
            record_dir=str(rollout_dir),
            no_confirm=True,
            record=False,         # we record via webcam, not matplotlib
            # follower_model_path is preserved (CEM needs it for V_F)
        )

    def _move_to_discarded(self, rollout_dir: Path, attempt_idx: int) -> None:
        if not rollout_dir.exists():
            return
        suffix = f".discarded.{attempt_idx}" if attempt_idx > 0 else ".discarded"
        target = rollout_dir.with_name(rollout_dir.name + suffix)
        i = 1
        while target.exists():
            target = rollout_dir.with_name(rollout_dir.name + f"{suffix}.{i}")
            i += 1
        os.rename(rollout_dir, target)
        print(f"[compare] Discarded rollout dir → {target}", flush=True)

    def _base_summary(self) -> dict:
        return {
            "rate_hz": float(self.base.rate_hz),
            "parent_frame": self.base.parent_frame,
            "cleaner_frame": self.base.cleaner_frame,
            "follower_frame": self.base.follower_frame,
            "table_frame": self.base.table_frame,
            "sim_origin_in_world": list(self.base.sim_origin_in_world),
            "parent_z_down": bool(self.base.parent_z_down),
            "cleaner_model_path": str(self.base.model_path or ""),
            "follower_model_path": str(self.base.follower_model_path or ""),
            "alpha": float(self.base.alpha),
            "vae_path": str(self.base.vae_path or ""),
            "cem_overrides": {
                k: v for k, v in {
                    "num_samples": self.base.cem_num_samples,
                    "horizon": self.base.cem_horizon,
                    "num_iters": self.base.cem_num_iters,
                    "replan_interval": self.base.cem_replan_interval,
                }.items() if v is not None
            },
        }

    def _write_manifest(self, manifest: dict) -> None:
        (self.session_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_to_json(init: dict) -> dict:
    """numpy → plain Python so json.dumps works."""
    return {
        "cleaner_xy_sim": [float(v) for v in init["cleaner_xy_sim"]],
        "cleaner_yaw_sim": float(init["cleaner_yaw_sim"]),
        "follower_xy_sim": [float(v) for v in init["follower_xy_sim"]],
        "follower_yaw_sim": float(init["follower_yaw_sim"]),
        "table_xy_sim": [float(v) for v in init["table_xy_sim"]],
        "table_yaw_sim": float(init["table_yaw_sim"]),
    }


# ---------------------------------------------------------------------------
# CLI entrypoint helper
# ---------------------------------------------------------------------------

def run_comparison(
    base_config: HouseholdDeploymentConfig,
    compare_config: HouseholdComparisonConfig,
) -> Path:
    """CLI-facing: build the supervisor, run, return the session dir."""
    supervisor = HouseholdComparisonSupervisor(
        base_config=base_config, compare_config=compare_config,
    )
    return supervisor.run()


__all__ = [
    "METHODS",
    "HouseholdComparisonConfig",
    "HouseholdComparisonSupervisor",
    "run_comparison",
]
