"""Out-of-band recording primitives for hardware experiment phases.

These wrap ``ros2 bag record -a`` (rosbag) and OpenCV ``VideoCapture``
(USB webcam) as standalone start/stop objects so they can bracket a
:func:`run_episode` call without coupling to the episode's matplotlib
or rclpy lifecycle.

Both recorders are silent-on-failure friendly: if a recorder fails to
start, the error is surfaced but the rollout still proceeds (the
operator can decide whether to redo the phase).
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


class RecorderError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Stable camera resolution
# ---------------------------------------------------------------------------
#
# A USB camera's ``/dev/videoN`` number is NOT stable — it changes when the
# device is replugged or the machine reboots. ``/dev/v4l/by-id`` symlinks,
# by contrast, are derived from the device's USB descriptor and are stable.
# Resolving a camera by a name substring (e.g. "OsmoAction5pro") therefore
# survives reconnection where a hardcoded index does not.

_V4L_BY_ID = "/dev/v4l/by-id"


def list_cameras() -> list:
    """Return ``[(stable_name, dev_index)]`` for capture nodes.

    Only ``*-video-index0`` links are reported — for UVC cameras index0 is
    the capture node (index1+ are metadata/duplicate nodes that open but
    yield no usable frames).
    """
    out: list = []
    for link in sorted(glob.glob(os.path.join(_V4L_BY_ID, "*-video-index0"))):
        name = os.path.basename(link)
        try:
            real = os.path.realpath(link)
        except OSError:
            continue
        m = re.search(r"/dev/video(\d+)$", real)
        if m:
            out.append((name, int(m.group(1))))
    return out


def resolve_camera(spec) -> int:
    """Resolve a camera ``spec`` to a V4L2 device index.

    ``spec`` may be:

    * an ``int`` or digit string → used as the index directly;
    * a ``/dev/...`` path → resolved (through symlinks) to its index;
    * any other string → matched case-insensitively as a substring against
      the stable ``/dev/v4l/by-id`` capture-node names, so the camera is
      found even after its ``/dev/videoN`` number changes on replug.

    Raises :class:`RecorderError` if a name/path doesn't resolve.
    """
    if isinstance(spec, int):
        return spec
    s = str(spec).strip()
    if s.isdigit():
        return int(s)
    if s.startswith("/dev/"):
        real = os.path.realpath(s)
        m = re.search(r"/dev/video(\d+)$", real)
        if m:
            return int(m.group(1))
        raise RecorderError(f"Not a V4L2 video device path: {s!r}")
    cams = list_cameras()
    matches = [(n, i) for (n, i) in cams if s.lower() in n.lower()]
    if not matches:
        avail = ", ".join(n for n, _ in cams) or "(none under /dev/v4l/by-id)"
        raise RecorderError(f"No camera matching {s!r}. Available: {avail}")
    matches.sort(key=lambda t: t[1])  # deterministic: lowest index wins
    return matches[0][1]


# ---------------------------------------------------------------------------
# Rosbag
# ---------------------------------------------------------------------------

@dataclass
class RosbagRecorder:
    """``ros2 bag record -a`` as a subprocess.

    The bag lands at ``<output_dir>/bag/`` (ros2 bag creates the
    subdirectory itself — it errors if the directory already exists).
    """

    output_dir: Path
    extra_args: tuple = ()
    _proc: Optional[subprocess.Popen] = field(default=None, init=False, repr=False)

    def start(self) -> None:
        if self._proc is not None:
            raise RecorderError("RosbagRecorder already started.")
        self.output_dir = Path(self.output_dir)
        bag_path = self.output_dir / "bag"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if bag_path.exists():
            raise RecorderError(
                f"Rosbag output directory already exists: {bag_path!s}"
            )
        cmd = [
            "ros2", "bag", "record", "-a", "-o", str(bag_path), *self.extra_args,
        ]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
        except FileNotFoundError as exc:
            raise RecorderError(
                "`ros2` not found on PATH. Is the ROS2 environment sourced?"
            ) from exc

        # Give it a moment to spin up so we catch immediate failures
        # (no ROS daemon, etc.).
        time.sleep(0.5)
        if self._proc.poll() is not None:
            rc = self._proc.returncode
            self._proc = None
            raise RecorderError(
                f"`ros2 bag record` exited immediately (rc={rc})."
            )

    def stop(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.send_signal(signal.SIGINT)
            try:
                self._proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                self._proc.terminate()
                self._proc.wait(timeout=2.0)
        finally:
            self._proc = None

    def discard(self) -> None:
        """Stop and delete the bag directory (for aborted/redone phases)."""
        self.stop()
        bag_path = self.output_dir / "bag"
        if bag_path.exists():
            shutil.rmtree(bag_path, ignore_errors=True)


# ---------------------------------------------------------------------------
# Webcam
# ---------------------------------------------------------------------------

class SessionWebcam:
    """A USB webcam opened **once** for a whole session.

    USB / V4L2 cameras don't release instantly: opening a fresh
    ``cv2.VideoCapture`` immediately after the previous one released
    races the kernel and the second open hangs or fails (only the first
    rollout ends up with a video). So instead of one capture per
    rollout, we keep a single capture + preview window alive for the
    entire session and only swap the :class:`cv2.VideoWriter` per
    rollout via :meth:`begin_segment` / :meth:`end_segment`.

    The capture loop runs in a daemon thread; ``cv2.imshow`` /
    ``waitKey`` live there too so they don't compete with the main
    thread's matplotlib / stdin.
    """

    def __init__(
        self,
        device=9,
        fps: int = 30,
        codec: str = "mp4v",
        preview: bool = True,
        warm_up_frames: int = 5,
        reopen_after: int = 30,
    ):
        # ``device`` is a spec (index, /dev path, or name substring), kept
        # raw so it can be re-resolved on reconnect — see resolve_camera.
        self.device = device
        self.fps = int(fps)
        self.codec = str(codec)
        self.preview = bool(preview)
        self.warm_up_frames = int(warm_up_frames)
        # Consecutive failed reads before we release + reopen the capture
        # (a USB hiccup can wedge the fd; reopening also picks up a new
        # /dev/videoN if the camera re-enumerated).
        self.reopen_after = int(reopen_after)

        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[Exception] = None
        self._frame_size: Optional[tuple] = None  # (w, h)

        # Active segment state, guarded by ``_lock`` (touched by both the
        # capture thread and the caller's begin/end_segment).
        self._lock = threading.Lock()
        self._writer = None
        self._writer_path: Optional[Path] = None
        self._seg_written: int = 0
        self._seg_dropped: int = 0

    @property
    def error(self) -> Optional[Exception]:
        return self._error

    @property
    def is_open(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ----- lifecycle -------------------------------------------------------

    def open(self) -> None:
        """Open the camera and start the capture loop (blocks until the
        first frame is in hand, so failures surface immediately)."""
        if self._thread is not None:
            raise RecorderError("SessionWebcam already open.")
        self._stop_event.clear()
        self._ready_event.clear()
        self._error = None
        self._thread = threading.Thread(
            target=self._capture_loop, name="session-webcam", daemon=True,
        )
        self._thread.start()
        # Wait for either a frame (ready) or a failure. Opening a USB cam
        # + warm-up can take a couple of seconds.
        self._ready_event.wait(timeout=8.0)
        if self._error is not None:
            err = self._error
            self.close()
            raise RecorderError(f"Webcam device {self.device} failed: {err}")
        if not self._ready_event.is_set():
            self.close()
            raise RecorderError(
                f"Webcam device {self.device} produced no frame within 8s."
            )

    def close(self) -> None:
        """Stop the capture loop and release the camera + preview window."""
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=10.0)
        self._thread = None
        if self._error is not None:
            print(f"[webcam] session error: {self._error}",
                  file=sys.stderr, flush=True)

    # ----- per-rollout segments --------------------------------------------

    def begin_segment(self, output_dir: Path) -> None:
        """Start writing frames to ``<output_dir>/webcam.mp4``."""
        import cv2

        if not self.is_open:
            raise RecorderError("SessionWebcam is not open.")
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "webcam.mp4"
        with self._lock:
            if self._writer is not None:
                raise RecorderError("A webcam segment is already active.")
            if self._frame_size is None:
                raise RecorderError("SessionWebcam has no frame size yet.")
            w, h = self._frame_size
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            writer = cv2.VideoWriter(str(path), fourcc, float(self.fps), (w, h))
            if not writer.isOpened():
                raise RecorderError(
                    f"cv2.VideoWriter({path!s}, codec={self.codec!r}) "
                    "failed to open."
                )
            self._writer = writer
            self._writer_path = path
            self._seg_written = 0
            self._seg_dropped = 0

    def end_segment(self) -> tuple:
        """Close the active segment; return ``(frames_written, dropped)``."""
        writer, written, dropped = self._take_writer()
        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass
        return written, dropped

    def discard_segment(self) -> None:
        """Close the active segment and delete its file (redo/abort)."""
        writer, _, _ = self._take_writer()
        path = self._writer_path
        self._writer_path = None
        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass
        if path is not None and path.exists():
            try:
                path.unlink()
            except OSError:
                pass

    def _take_writer(self) -> tuple:
        with self._lock:
            writer = self._writer
            written, dropped = self._seg_written, self._seg_dropped
            self._writer = None
        return writer, written, dropped

    # ----- capture thread --------------------------------------------------

    def _open_capture(self, cv2, *, first: bool):
        """Resolve + open the camera, returning a primed ``VideoCapture``.

        On failure returns ``None`` (and sets ``self._error`` only when
        ``first`` — a transient mid-session reopen failure shouldn't be
        treated as a fatal session error).
        """
        try:
            index = resolve_camera(self.device)
        except RecorderError as e:
            if first:
                self._error = e
            return None

        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            if first:
                self._error = RecorderError(
                    f"cv2.VideoCapture({index}) failed to open."
                )
            return None

        # Prefer MJPG (USB cams expose higher resolutions there).
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FPS, self.fps)
        except Exception:
            pass

        first_frame = None
        for _ in range(self.warm_up_frames + 1):
            ok, frame = cap.read()
            if ok and frame is not None:
                first_frame = frame
                break
        if first_frame is None:
            try:
                cap.release()
            except Exception:
                pass
            if first:
                self._error = RecorderError(
                    f"Webcam {self.device!r} opened but yielded no frames."
                )
            return None

        # Lock the writer geometry to the first capture; a reopened camera
        # must keep the same size or the active writer would reject frames.
        if self._frame_size is None:
            h, w = first_frame.shape[:2]
            self._frame_size = (w, h)
        return cap

    def _capture_loop(self) -> None:
        try:
            import cv2
        except ImportError as e:
            self._error = e
            self._ready_event.set()
            return

        cap = self._open_capture(cv2, first=True)
        if cap is None:
            self._ready_event.set()
            return

        win_name = f"webcam {self.device}"
        if self.preview:
            try:
                cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(win_name, 640, 360)
            except Exception:
                pass

        # Ready: a frame is in hand and the frame size is known, so
        # begin_segment() can construct a writer.
        self._ready_event.set()
        consecutive_fail = 0
        try:
            while not self._stop_event.is_set():
                if cap is None:
                    # Lost the device; back off, then try to re-acquire.
                    time.sleep(0.3)
                    cap = self._open_capture(cv2, first=False)
                    continue

                ok, frame = cap.read()
                if not ok or frame is None:
                    consecutive_fail += 1
                    with self._lock:
                        if self._writer is not None:
                            self._seg_dropped += 1
                    if consecutive_fail >= self.reopen_after:
                        print(
                            f"[webcam] {consecutive_fail} consecutive read "
                            f"failures on {self.device!r}; reopening...",
                            file=sys.stderr, flush=True,
                        )
                        try:
                            cap.release()
                        except Exception:
                            pass
                        cap = self._open_capture(cv2, first=False)
                        consecutive_fail = 0
                    else:
                        time.sleep(0.01)
                    continue

                consecutive_fail = 0
                self._handle_frame(cv2, frame, win_name)
        except Exception as e:
            self._error = e
        finally:
            with self._lock:
                if self._writer is not None:
                    try:
                        self._writer.release()
                    except Exception:
                        pass
                    self._writer = None
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            if self.preview:
                try:
                    cv2.destroyWindow(win_name)
                except Exception:
                    pass

    def _handle_frame(self, cv2, frame, win_name: str) -> None:
        with self._lock:
            if self._writer is not None:
                self._writer.write(frame)
                self._seg_written += 1
        if self.preview:
            try:
                cv2.imshow(win_name, frame)
                # waitKey(1) drives the GUI; 'q' stops the whole session.
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    self._stop_event.set()
            except Exception:
                pass


__all__ = [
    "RecorderError",
    "RosbagRecorder",
    "SessionWebcam",
    "list_cameras",
    "resolve_camera",
]
