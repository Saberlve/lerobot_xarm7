"""Tests for the GelSight Mini camera plugin.

No real sensor is required: OpenCVCamera accepts a video file as
`index_or_path`, so a synthetic clip with a known column gradient stands in
for the hardware.
"""

import cv2
import draccus
import numpy as np
import pytest
from lerobot.cameras.configs import CameraConfig

from lerobot_robot_ufactory.cameras.gsmini_camera import GsminiCamera, GsminiCameraConfig

WIDTH, HEIGHT = 320, 240
BORDER_FRACTION = 0.15


def _make_gradient_video(path, n_frames=60):
    """Write a clip encoding the column index in 8-pixel blocks.

    mp4 is lossy, so a per-pixel gradient gets blurred; 8-pixel blocks with a
    step of 6 survive with a couple of counts of error.
    """
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (WIDTH, HEIGHT))
    assert writer.isOpened()
    column_gradient = np.tile((np.arange(WIDTH) // 8 * 6).astype(np.uint8), (HEIGHT, 1))
    frame = cv2.merge([column_gradient] * 3)  # BGR, all channels identical
    for _ in range(n_frames):
        writer.write(frame)
    writer.release()


@pytest.fixture()
def gradient_video(tmp_path):
    video_path = tmp_path / "fake_gsmini.mp4"
    _make_gradient_video(video_path)
    return video_path


def test_config_registered_and_parses():
    assert CameraConfig.get_choice_class("uf::gsmini_camera") is GsminiCameraConfig
    cfg = draccus.decode(
        GsminiCameraConfig,
        {
            "index_or_path": "/dev/v4l/by-id/usb-GelSight_Mini-video-index0",
            "width": WIDTH,
            "height": HEIGHT,
            "fps": 25,
            "border_fraction": 0.2,
        },
    )
    assert str(cfg.index_or_path) == "/dev/v4l/by-id/usb-GelSight_Mini-video-index0"
    assert cfg.border_fraction == pytest.approx(0.2)
    assert cfg.type == "uf::gsmini_camera"


def test_border_fraction_clamped_like_sdk():
    cfg = GsminiCameraConfig(index_or_path=0, border_fraction=0.9)
    assert cfg.border_fraction == pytest.approx(0.49)
    cfg = GsminiCameraConfig(index_or_path=0, border_fraction=-1.0)
    assert cfg.border_fraction == pytest.approx(0.0)


def test_read_applies_border_crop(gradient_video):
    # No width/height/fps in config so connect() skips property validation on
    # the file backend; self.width/height fall back to the clip's dimensions.
    config = GsminiCameraConfig(index_or_path=gradient_video, border_fraction=BORDER_FRACTION)
    camera = GsminiCamera(config)
    camera.connect(warmup=False)
    try:
        frame = camera.read()
    finally:
        camera.disconnect()

    assert frame.shape == (HEIGHT, WIDTH, 3)

    border = int(WIDTH * BORDER_FRACTION)  # 48 columns cropped from each side
    kept_width = WIDTH - 2 * border
    # After resize back to WIDTH, output column x samples source column
    # border + (x + 0.5) * kept_width / WIDTH. The clip encodes columns in
    # 8-pixel blocks of value step 6; mp4 blurs block edges, allow slack.
    def encoded(col):
        return (int(col) // 8) * 6

    left_col = border + 0.5 * kept_width / WIDTH
    right_col = border + (WIDTH - 0.5) * kept_width / WIDTH
    assert frame[HEIGHT // 2, 0, 0] == pytest.approx(encoded(left_col), abs=8)
    assert frame[HEIGHT // 2, WIDTH - 1, 0] == pytest.approx(encoded(right_col), abs=8)


def test_zero_border_fraction_keeps_image(gradient_video):
    camera = GsminiCamera(GsminiCameraConfig(index_or_path=gradient_video, border_fraction=0.0))
    camera.connect(warmup=False)
    try:
        frame = camera.read()
    finally:
        camera.disconnect()

    assert frame.shape == (HEIGHT, WIDTH, 3)
    # Sample mid-block columns: value = (column // 8) * 6.
    assert frame[HEIGHT // 2, 100, 0] == pytest.approx((100 // 8) * 6, abs=6)
    assert frame[HEIGHT // 2, 252, 0] == pytest.approx((252 // 8) * 6, abs=6)


def test_async_read_returns_latest_frame(gradient_video):
    config = GsminiCameraConfig(index_or_path=gradient_video, border_fraction=BORDER_FRACTION)
    camera = GsminiCamera(config)
    camera.connect(warmup=False)
    try:
        frame = camera.async_read(timeout_ms=2000)
    finally:
        camera.disconnect()

    assert frame.shape == (HEIGHT, WIDTH, 3)
