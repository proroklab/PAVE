"""Unit tests for the rosbag and webcam recorders.

The webcam recorder uses cv2; we mock it. The rosbag recorder spawns
``ros2 bag record`` via subprocess.Popen; we patch Popen so the tests
don't depend on a ROS2 install.
"""

from __future__ import annotations

import signal
import subprocess
from pathlib import Path

import pytest

from critic_sharing.deployment.recorders import (
    RecorderError, RosbagRecorder, SessionWebcam,
)


# ---------------------------------------------------------------------------
# RosbagRecorder
# ---------------------------------------------------------------------------

class _FakeProc:
    def __init__(self, *, alive: bool = True, exit_code: int = 0):
        self._alive = alive
        self._exit_code = exit_code
        self.signals: list = []
        self.terminated = False
        self.returncode = 0

    def poll(self):
        return None if self._alive else self._exit_code

    def send_signal(self, sig):
        self.signals.append(sig)
        self._alive = False
        self.returncode = 0

    def wait(self, timeout=None):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self._alive = False
        self.returncode = -15


def test_rosbag_start_spawns_subprocess(tmp_path: Path, monkeypatch):
    bag = RosbagRecorder(output_dir=tmp_path / "rollout")
    fake = _FakeProc()
    captured = {}

    def _popen(cmd, **kw):
        captured["cmd"] = cmd
        return fake

    monkeypatch.setattr(subprocess, "Popen", _popen)
    bag.start()

    assert captured["cmd"][:5] == ["ros2", "bag", "record", "-a", "-o"]
    assert captured["cmd"][5].endswith("/bag")
    # Output dir was created (but the bag/ subdir was not pre-created).
    assert (tmp_path / "rollout").exists()
    assert not (tmp_path / "rollout" / "bag").exists()
    bag.stop()
    assert signal.SIGINT in fake.signals


def test_rosbag_start_errors_when_bag_exists(tmp_path: Path, monkeypatch):
    (tmp_path / "rollout" / "bag").mkdir(parents=True)
    bag = RosbagRecorder(output_dir=tmp_path / "rollout")
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _FakeProc())
    with pytest.raises(RecorderError, match="already exists"):
        bag.start()


def test_rosbag_start_surfaces_missing_ros2(tmp_path: Path, monkeypatch):
    def _raise(*a, **k):
        raise FileNotFoundError("ros2")
    monkeypatch.setattr(subprocess, "Popen", _raise)
    bag = RosbagRecorder(output_dir=tmp_path / "rollout")
    with pytest.raises(RecorderError, match="not found on PATH"):
        bag.start()


def test_rosbag_start_surfaces_immediate_exit(tmp_path: Path, monkeypatch):
    fake = _FakeProc(alive=False, exit_code=1)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: fake)
    bag = RosbagRecorder(output_dir=tmp_path / "rollout")
    with pytest.raises(RecorderError, match="exited immediately"):
        bag.start()


def test_rosbag_discard_removes_bag_dir(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _FakeProc())
    bag = RosbagRecorder(output_dir=tmp_path / "rollout")
    bag.start()
    # Simulate ros2 bag having written some content.
    bag_path = tmp_path / "rollout" / "bag"
    bag_path.mkdir()
    (bag_path / "metadata.yaml").write_text("dummy")
    bag.discard()
    assert not bag_path.exists()


def test_rosbag_stop_is_idempotent(tmp_path: Path):
    bag = RosbagRecorder(output_dir=tmp_path / "rollout")
    # Never started — stop() must not raise.
    bag.stop()
    bag.stop()


# ---------------------------------------------------------------------------
# SessionWebcam — mock cv2 by injecting into sys.modules.
# ---------------------------------------------------------------------------

class _FakeFrame:
    shape = (480, 640, 3)


class _FakeCap:
    def __init__(self, *, opens: bool = True, frame_seq=None, fail_after=None):
        self._opens = opens
        self._frames = list(frame_seq) if frame_seq is not None else None
        self._fail_after = fail_after
        self._reads = 0
        self.released = 0

    def isOpened(self):
        return self._opens

    def set(self, *a, **k):
        return True

    def read(self):
        self._reads += 1
        if self._fail_after is not None and self._reads > self._fail_after:
            return False, None
        if self._frames is not None:
            return True, self._frames.pop(0) if self._frames else (False, None)
        return True, _FakeFrame()

    def release(self):
        self.released += 1


class _FakeWriter:
    """Records to a real file on disk so file-existence assertions work."""

    def __init__(self, *a, **k):
        self.frames = 0
        self.opened = True
        self.path = a[0] if a else None
        if self.path:
            p = Path(self.path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"")

    def isOpened(self):
        return self.opened

    def write(self, frame):
        self.frames += 1

    def release(self):
        pass


def _install_fake_cv2(monkeypatch, *, cap=None, writer_cls=None):
    import sys
    import types
    cv2 = types.ModuleType("cv2")
    counts = {"captures": 0}
    cv2.CAP_PROP_FOURCC = 0
    cv2.CAP_PROP_FPS = 1
    cv2.CAP_V4L2 = 200
    cv2.WINDOW_NORMAL = 0
    cv2.VideoWriter_fourcc = lambda *a: 0

    def _open_capture(idx, *backend):
        counts["captures"] += 1
        return cap or _FakeCap()

    cv2.VideoCapture = _open_capture
    cv2.VideoWriter = writer_cls or _FakeWriter
    cv2.namedWindow = lambda *a, **k: None
    cv2.resizeWindow = lambda *a, **k: None
    cv2.imshow = lambda *a, **k: None
    cv2.waitKey = lambda *a, **k: 0
    cv2.destroyWindow = lambda *a, **k: None
    cv2._counts = counts
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    return cv2


def test_webcam_open_errors_when_capture_does_not_open(monkeypatch):
    _install_fake_cv2(monkeypatch, cap=_FakeCap(opens=False))
    cam = SessionWebcam(device=99, preview=False)
    with pytest.raises(RecorderError, match="failed to open"):
        cam.open()
    cam.close()


def test_webcam_open_errors_when_no_frames(monkeypatch):
    _install_fake_cv2(monkeypatch, cap=_FakeCap(fail_after=0))
    cam = SessionWebcam(device=99, preview=False)
    with pytest.raises(RecorderError, match="yielded no frames"):
        cam.open()
    cam.close()


def test_webcam_segment_before_open_raises(tmp_path, monkeypatch):
    _install_fake_cv2(monkeypatch)
    cam = SessionWebcam(device=99, preview=False)
    with pytest.raises(RecorderError, match="not open"):
        cam.begin_segment(tmp_path / "rollout")


def test_webcam_records_segment_and_reports_counts(tmp_path, monkeypatch):
    import time
    _install_fake_cv2(monkeypatch)
    cam = SessionWebcam(device=99, preview=False, fps=30)
    cam.open()
    cam.begin_segment(tmp_path / "rollout")
    time.sleep(0.1)  # let the capture loop spin
    fw, fd = cam.end_segment()
    cam.close()
    assert fw >= 1
    assert fd == 0
    assert (tmp_path / "rollout" / "webcam.mp4").exists()


def test_webcam_two_segments_reuse_one_capture(tmp_path, monkeypatch):
    """The regression: two rollouts must both record from a single open."""
    import time
    cv2 = _install_fake_cv2(monkeypatch)
    cam = SessionWebcam(device=99, preview=False, fps=30)
    cam.open()

    cam.begin_segment(tmp_path / "r1")
    time.sleep(0.05)
    fw1, _ = cam.end_segment()

    cam.begin_segment(tmp_path / "r2")
    time.sleep(0.05)
    fw2, _ = cam.end_segment()
    cam.close()

    assert fw1 >= 1
    assert fw2 >= 1
    assert (tmp_path / "r1" / "webcam.mp4").exists()
    assert (tmp_path / "r2" / "webcam.mp4").exists()
    # The camera was opened exactly once for the whole session.
    assert cv2._counts["captures"] == 1


def test_webcam_no_frames_written_between_segments(tmp_path, monkeypatch):
    """Frames read while no segment is active are neither written nor
    counted as drops."""
    import time
    _install_fake_cv2(monkeypatch)
    cam = SessionWebcam(device=99, preview=False, fps=30)
    cam.open()
    time.sleep(0.05)  # idle, no active segment
    cam.begin_segment(tmp_path / "rollout")
    time.sleep(0.05)
    fw, fd = cam.end_segment()
    cam.close()
    assert fw >= 1


def test_webcam_discard_segment_removes_file(tmp_path, monkeypatch):
    _install_fake_cv2(monkeypatch)
    cam = SessionWebcam(device=99, preview=False)
    cam.open()
    cam.begin_segment(tmp_path / "rollout")
    assert (tmp_path / "rollout" / "webcam.mp4").exists()
    cam.discard_segment()
    cam.close()
    assert not (tmp_path / "rollout" / "webcam.mp4").exists()


# ---------------------------------------------------------------------------
# resolve_camera — stable name → index resolution
# ---------------------------------------------------------------------------

def test_resolve_camera_passes_through_int():
    from critic_sharing.deployment.recorders import resolve_camera
    assert resolve_camera(9) == 9


def test_resolve_camera_parses_digit_string():
    from critic_sharing.deployment.recorders import resolve_camera
    assert resolve_camera("7") == 7


def test_resolve_camera_matches_name_substring(monkeypatch):
    from critic_sharing.deployment import recorders
    monkeypatch.setattr(recorders, "list_cameras", lambda: [
        ("usb-Generic_Integrated_Camera_x-video-index0", 0),
        ("usb-DJI_OsmoAction5pro_SN-video-index0", 4),
    ])
    assert recorders.resolve_camera("OsmoAction5pro") == 4
    # Case-insensitive.
    assert recorders.resolve_camera("osmoaction5pro") == 4


def test_resolve_camera_raises_on_no_match(monkeypatch):
    from critic_sharing.deployment import recorders
    monkeypatch.setattr(recorders, "list_cameras", lambda: [
        ("usb-Generic_Integrated_Camera_x-video-index0", 0),
    ])
    with pytest.raises(RecorderError, match="No camera matching"):
        recorders.resolve_camera("Osmo")
