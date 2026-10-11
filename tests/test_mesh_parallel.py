"""Concurrent native mesh inference preserves every frame and retry semantics."""
from contextlib import contextmanager
from pathlib import Path
from threading import Barrier, Event, get_ident

import cv2
import numpy as np
import pytest

from lerobot_robot_ufactory.datasets import stream_recorder as persistence


def recorder_with_frames(tmp_path, names=("left", "right"), count=8):
    recorder = persistence.TactileStreamRecorder.__new__(persistence.TactileStreamRecorder)
    recorder._staging_root = tmp_path
    recorder._base = "episode_000000"
    recorder._close_writer = lambda: None
    recorder.stream_fps = dict.fromkeys(names, 60)
    recorder._rows = {}
    for name in names:
        frames = tmp_path / name / "frames"
        frames.mkdir(parents=True)
        recorder._rows[name] = [{"mesh_path": None} for _ in range(count)]
        for index in range(count):
            assert cv2.imwrite(str(frames / f"frame_{index:06d}.png"),
                               np.full((4, 4, 3), index, dtype=np.uint8))
    return recorder


class OrderedCamera:
    deferred_feature_shapes = {"mesh_motion_3d": (2, 2, 3)}

    def __init__(self, barrier=None, fail=False):
        self.barrier = barrier
        self.fail = fail
        self.calls = []
        self.released = False

    @contextmanager
    def deferred_session(self, runtime_dir):
        self.owner = get_ident()
        try:
            yield
        finally:
            assert get_ident() == self.owner
            self.released = True

    def compute_deferred_features(self, image, runtime_dir):
        assert get_ident() == self.owner
        index = int(image[0, 0, 0])
        if not self.calls and self.barrier is not None:
            self.barrier.wait(timeout=5)
        self.calls.append(index)
        if self.fail:
            raise RuntimeError("injected solver failure")
        return {"mesh_motion_3d": np.full((2, 2, 3), index + 0.123, dtype=np.float64)}


def test_two_solvers_overlap_and_preserve_bit_exact_frames(tmp_path):
    recorder = recorder_with_frames(tmp_path)
    barrier = Barrier(2)
    cameras = {name: OrderedCamera(barrier) for name in recorder._rows}
    progress = []
    result = recorder.compute_mesh(cameras, tmp_path, progress=progress.append)
    for name, camera in cameras.items():
        assert camera.calls == list(range(8))
        assert camera.released
        expected = np.stack([np.full((2, 2, 3), i + 0.123, np.float64) for i in range(8)])
        assert result[name].dtype == expected.dtype
        assert result[name].tobytes() == expected.tobytes()
        assert [r["mesh_frame_index"] for r in recorder._rows[name]] == list(range(8))
        assert len(list((tmp_path / name / "frames").glob("*.png"))) == 8
    assert cameras["left"].owner != cameras["right"].owner
    for name in cameras:
        updates = [p for p in progress if p["camera"] == name]
        assert updates[0]["completed_frames"] == 0
        assert updates[-1] == {"camera": name, "stage": "mesh",
                               "completed_frames": 8, "total_frames": 8}


def test_failure_releases_workers_and_retry_reuses_completed_stream(tmp_path):
    recorder = recorder_with_frames(tmp_path)
    cameras = {"left": OrderedCamera(), "right": OrderedCamera(fail=True)}
    with pytest.raises(RuntimeError, match="injected solver failure"):
        recorder.compute_mesh(cameras, tmp_path)
    assert all(camera.released for camera in cameras.values())
    assert all(row["mesh_path"] is None for row in recorder._rows["right"])
    cameras["right"] = OrderedCamera()
    result = recorder.compute_mesh(cameras, tmp_path)
    assert cameras["left"].calls == list(range(8))  # cached, not inferred twice
    assert cameras["right"].calls == list(range(8))
    assert result["right"].shape == (8, 2, 2, 3)
    assert len(list(tmp_path.rglob("*.png"))) == 16


def test_png_prefetch_is_bounded_and_stops_when_consumer_exits(tmp_path, monkeypatch):
    reads = []
    filled = Event()

    def read(path, flags):
        reads.append(Path(path).name)
        if len(reads) == 5:
            filled.set()
        return np.zeros((4, 4, 3), np.uint8)

    monkeypatch.setattr(persistence.cv2, "imread", read)
    with persistence._prefetched_tactile_images(tmp_path, 1000) as images:
        assert next(images)[0] == 0
        assert filled.wait(timeout=5)
        assert len(reads) == 5  # one consumed frame and four prefetched frames
    assert len(reads) == 5


def test_missing_png_reports_failure_without_marking_stream_complete(tmp_path):
    recorder = recorder_with_frames(tmp_path, names=("left",))
    (tmp_path / "left/frames/frame_000003.png").unlink()
    camera = OrderedCamera()
    with pytest.raises(persistence.TactilePersistenceError, match="frame_000003.png"):
        recorder.compute_mesh({"left": camera}, tmp_path)
    assert camera.released
    assert all(row["mesh_path"] is None for row in recorder._rows["left"])
