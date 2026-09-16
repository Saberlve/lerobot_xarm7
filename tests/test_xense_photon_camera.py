"""Exercise the real camera/factory interface with an in-memory Xense SDK."""

import sys
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import draccus
import numpy as np
import pytest
import yaml
from lerobot.cameras.configs import CameraConfig, ColorMode
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from lerobot_robot_ufactory.cameras.utils import make_cameras_from_configs
from lerobot_robot_ufactory.cameras.xense_photon_camera import (
    XensePhotonCamera,
    XensePhotonCameraConfig,
    camera_xense_photon,
)
from lerobot_robot_ufactory.cameras.xense_photon_camera.camera_xense_photon import (
    XensePhotonSample,
)


@pytest.fixture
def sdk(monkeypatch):
    instances = []

    # The test verifies the camera/SDK boundary, not the OpenCV wheel. This
    # keeps it independent from whether the environment has the GUI or
    # headless OpenCV distribution installed.
    class Cv2:
        COLOR_BGR2RGB = 1

        @staticmethod
        def resize(frame, size):
            width, height = size
            return np.resize(frame, (height, width, frame.shape[2]))

        @staticmethod
        def cvtColor(frame, _code):
            return frame[..., ::-1].copy()

    monkeypatch.setattr(camera_xense_photon, "cv2", Cv2)

    class Sensor:
        OutputType = SimpleNamespace(
            Rectify=1,
            Raw=2,
            Difference=3,
            Marker3DFlow=4,
            TimeStamp=5,
            Mesh3DFlow=6,
        )

        @staticmethod
        def scanSerialNumber():
            return {"LEFT": 0, "RIGHT": 1}

        @classmethod
        def create(cls, serial, **kwargs):
            obj = cls()
            obj.serial, obj.kwargs = serial, kwargs
            obj.frame = np.full((6, 4, 3), [10, 20, 30 if serial == "LEFT" else 90], np.uint8)
            obj.marker_motion_3d = np.full((35, 20, 3), 1.25, np.float32)
            obj.timestamp = np.array(1_700_000_042.5, dtype=np.float64)
            obj.released = False
            obj.error = None
            obj.gate = None
            obj.entered = Event()
            instances.append(obj)
            return obj

        def selectSensorInfo(self, *outputs):
            self.output = outputs[0] if len(outputs) == 1 else outputs
            self.entered.set()
            if self.gate:
                self.gate.wait(2)
            if self.error:
                raise self.error
            if len(outputs) == 1:
                return self.frame
            if len(outputs) == 2:
                return self.frame, self.timestamp
            return self.frame, self.marker_motion_3d, self.timestamp

        def release(self):
            self.released = True

    monkeypatch.setitem(sys.modules, "xensesdk", SimpleNamespace(Sensor=Sensor))
    return Sensor, instances


def config(serial="LEFT", **kwargs):
    return XensePhotonCameraConfig(serial_number=serial, width=8, height=12, fps=100, **kwargs)


def test_registration_and_yaml():
    path = (
        Path(__file__).resolve().parents[1]
        / "config/gello/xarm7_gello_record_xense_photon_config.yaml"
    )
    raw = yaml.safe_load(path.read_text())
    configs = {
        key: draccus.decode(CameraConfig, value)
        for key, value in raw["robot"]["cameras"].items()
        if value["type"] == "photon"
    }
    cameras = make_cameras_from_configs(configs)
    assert set(cameras) == {"photon_right", "photon_left"}
    assert all(isinstance(cam, XensePhotonCamera) for cam in cameras.values())
    assert cameras["photon_right"].config.color_mode == ColorMode.RGB
    assert cameras["photon_right"].saves_marker_motion_3d
    assert not cameras["photon_right"].config.disable_infer


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(width=0),
        dict(fps=-1),
        dict(serial_number=""),
        dict(output_type="Depth"),
        dict(color_mode="gray"),
        dict(save_marker_motion_3d=True, disable_infer=True),
    ],
)
def test_bad_config(kwargs):
    with pytest.raises(ValueError):
        XensePhotonCameraConfig(**({"serial_number": "LEFT"} | kwargs))


def test_duplicate_serial_rejected():
    with pytest.raises(ValueError, match="distinct"):
        make_cameras_from_configs({"left": config(), "right": config()})


def test_two_sensors_and_lifecycle(sdk):
    _, instances = sdk
    assert len(XensePhotonCamera.find_cameras()) == 2
    left = XensePhotonCamera(config(config_path="/calibration", use_gpu=False,
                                    disable_infer=True, save_marker_motion_3d=False))
    right = XensePhotonCamera(config("RIGHT", output_type="Raw",
                                     disable_infer=True, save_marker_motion_3d=False))
    with pytest.raises(DeviceNotConnectedError):
        left.read()
    left.connect()
    right.connect()
    try:
        with pytest.raises(DeviceAlreadyConnectedError):
            left.connect()
        frame = left.async_read()
        assert frame.shape == (12, 8, 3)
        np.testing.assert_array_equal(frame[0, 0], [30, 20, 10])
        np.testing.assert_array_equal(right.read()[0, 0], [90, 20, 10])
        np.testing.assert_array_equal(left.read(ColorMode.BGR)[0, 0], [10, 20, 30])
        frame[:] = 0
        assert left.read()[0, 0, 0] == 30
        assert instances[0].kwargs == dict(
            use_gpu=False, disable_infer=True, config_path="/calibration"
        )
        assert instances[1].output == (2, 5)
    finally:
        left.disconnect()
        right.disconnect()
    assert all(obj.released for obj in instances)
    left.connect()
    left.disconnect()


def test_marker_motion_3d_and_rgb_share_one_sdk_sample(sdk):
    _, instances = sdk
    camera = XensePhotonCamera(
        config(disable_infer=False, save_marker_motion_3d=True)
    )
    camera.connect()
    try:
        frame, tactile = camera.async_read_with_marker_motion_3d()
        assert instances[0].output == (1, 4, 5)
        assert frame.shape == (12, 8, 3)
        np.testing.assert_array_equal(frame[0, 0], [30, 20, 10])
        assert tactile["marker_motion_3d"].shape == (35, 20, 3)
        assert tactile["marker_motion_3d"].dtype == np.float32
        assert tactile["sensor_timestamp_s"] == pytest.approx(1_700_000_042.5)
        assert tactile["capture_monotonic_s"] > 0
        tactile["marker_motion_3d"][:] = 0
        assert camera.async_read_with_marker_motion_3d()[1]["marker_motion_3d"][0, 0, 0] == 1.25
    finally:
        camera.disconnect()


def test_nearest_marker_sample_enforces_the_time_budget(sdk):
    camera = XensePhotonCamera(config(disable_infer=False, save_marker_motion_3d=True))
    # Exercise the time-pairing queue directly; no SDK thread is needed.
    camera._sensor = object()
    target = camera_xense_photon.perf_counter()
    sample = XensePhotonSample(
        frame_bgr=np.zeros((12, 8, 3), dtype=np.uint8),
        marker_motion_3d=np.ones((35, 20, 3), dtype=np.float32),
        sensor_timestamp_s=1.0,
        capture_monotonic_s=target + 0.001,
    )
    with camera._condition:
        camera._sample_history.append(sample)

    frame, tactile = camera.async_read_with_marker_motion_3d_nearest(
        target_monotonic_s=target,
        max_skew_ms=2,
        wait_ms=0,
    )
    assert frame.shape == (12, 8, 3)
    assert tactile["sync_offset_ms"] == pytest.approx(1.0, abs=0.5)

    with pytest.raises(TimeoutError, match="Nearest Xense frame"):
        camera.async_read_with_marker_motion_3d_nearest(
            target_monotonic_s=target,
            max_skew_ms=0.1,
            wait_ms=0,
        )


def test_empty_warmup_releases(sdk, monkeypatch):
    sensor, instances = sdk
    monkeypatch.setattr(sensor, "selectSensorInfo", lambda self, *outputs: (None, None, None))
    camera = XensePhotonCamera(config(timeout_ms=30))
    with pytest.raises(TimeoutError):
        camera.connect()
    assert instances[0].released
    assert not camera.is_connected


def test_worker_failure_propagates_and_releases(sdk, monkeypatch):
    sensor, instances = sdk

    def fail(self, *outputs):
        raise OSError("USB disconnected")

    monkeypatch.setattr(sensor, "selectSensorInfo", fail)
    camera = XensePhotonCamera(config())
    with pytest.raises(RuntimeError, match="capture failed") as exc:
        camera.connect()
    assert isinstance(exc.value.__cause__, OSError)
    assert instances[0].released


def test_stale_cache_and_blocked_disconnect(sdk):
    _, instances = sdk
    camera = XensePhotonCamera(config(timeout_ms=30, max_frame_age_ms=10))
    camera.connect()
    obj = instances[0]
    obj.gate = Event()
    obj.entered.clear()
    assert obj.entered.wait(1)
    try:
        with camera._condition:
            camera._frame_time = 0
        with pytest.raises(TimeoutError):
            camera.async_read(timeout_ms=20)
        with pytest.raises(TimeoutError, match="blocked"):
            camera.disconnect()
        assert not obj.released
    finally:
        obj.gate.set()
        camera._thread.join(1)
        camera.disconnect()
    assert obj.released


def test_missing_sdk_is_optional(monkeypatch):
    monkeypatch.setitem(sys.modules, "xensesdk", None)
    camera = XensePhotonCamera(config())
    with pytest.raises(ImportError, match="pip install"):
        camera.connect()
    assert not camera.is_connected


def test_sdk_21_create_without_use_gpu(sdk, monkeypatch):
    sensor, instances = sdk
    original = sensor.create

    def modern_create(serial, *, disable_infer, config_path=None):
        return original(serial, disable_infer=disable_infer, config_path=config_path)

    monkeypatch.setattr(sensor, "create", modern_create)
    camera = XensePhotonCamera(config())
    camera.connect()
    try:
        assert "use_gpu" not in instances[0].kwargs
        assert camera.sync_samples()
    finally:
        camera.disconnect()


def test_repeated_sdk_timestamp_does_not_refresh_history(sdk):
    from time import sleep
    camera = XensePhotonCamera(config())
    camera.connect()
    try:
        first = camera.sync_samples()[0]
        sleep(0.05)
        samples = camera.sync_samples()
        assert len(samples) == 1
        assert samples[0].capture_monotonic_s == first.capture_monotonic_s
    finally:
        camera.disconnect()


def test_mesh_displacement_has_distinct_dataset_key(sdk):
    _, instances = sdk
    camera = XensePhotonCamera(config(motion_3d_output="Mesh3DFlow"))
    camera.connect()
    try:
        frame, timing = camera.export_sync_sample(camera.sync_samples()[0])
        assert instances[0].output == (1, 6, 5)
        assert "mesh_motion_3d" in timing
        assert "marker_motion_3d" not in timing
        assert frame.shape == (12, 8, 3)
    finally:
        camera.disconnect()
