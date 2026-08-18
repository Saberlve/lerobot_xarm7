import socket
import threading
import time
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pytest
import yaml

from lerobot_robot_ufactory.utils.web_preview import (
    RecordingWebPreview,
    WebPreviewConfig,
    _WEB_PAGE,
)


def _preview(**overrides):
    values = {
        "enabled": True,
        "host": "127.0.0.1",
        "port": 8765,
        "fps": 30,
        "width": 32,
        "jpeg_quality": 60,
    }
    values.update(overrides)
    return RecordingWebPreview(WebPreviewConfig(**values))


def _start_encoder(preview):
    preview._encoder_thread = threading.Thread(target=preview._encode_loop, daemon=True)
    preview._encoder_thread.start()


def test_config_rejects_invalid_resource_limits():
    with pytest.raises(ValueError, match="fps"):
        WebPreviewConfig(fps=0).validate()
    with pytest.raises(ValueError, match="jpeg_quality"):
        WebPreviewConfig(jpeg_quality=101).validate()


def test_embedded_page_has_parseable_camera_signature_expression():
    page = _WEB_PAGE.decode()

    assert "const next=p.cameras.join('|')" in page
    assert "join('\n')" not in page


def test_gello_record_config_enables_low_rate_web_preview():
    config_path = Path("config/gello/xarm7_gello_record_config.yaml")
    config = yaml.safe_load(config_path.read_text())

    assert config["web_preview"]["enabled"] is True
    assert config["web_preview"]["fps"] == 8
    assert config["web_preview"]["width"] == 480


def test_publish_keeps_only_latest_frame_references():
    preview = _preview()
    first = np.zeros((24, 32, 3), dtype=np.uint8)
    second = np.ones((24, 32, 3), dtype=np.uint8)

    preview.publish({"J1.pos": 0.0, "camera": first})
    preview.publish({"camera": second})

    assert preview.camera_names() == ["camera"]
    assert preview._latest_frames["camera"] is second
    assert preview._source_generation == 2


def test_encoder_is_idle_without_a_browser_client(monkeypatch):
    preview = _preview()
    calls = []
    monkeypatch.setattr(preview, "_encode_frame", lambda frame: calls.append(frame) or b"jpeg")
    _start_encoder(preview)
    try:
        preview.publish({"camera": np.zeros((24, 32, 3), dtype=np.uint8)})
        time.sleep(0.08)
        assert calls == []
    finally:
        preview.stop()


def test_client_receives_background_encoded_latest_frame():
    preview = _preview()
    _start_encoder(preview)
    try:
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        frame[:, :, 0] = 255
        preview.publish({"camera": frame})
        feed = preview.feed("camera")
        feed.add_client()
        with preview._condition:
            preview._condition.notify()
        jpeg, frame_id, stopped = feed.wait_for_jpeg(0)
        feed.remove_client()

        assert jpeg is not None and jpeg.startswith(b"\xff\xd8")
        assert frame_id == 1
        assert not stopped
    finally:
        preview.stop()


def test_status_endpoint_lists_published_cameras():
    try:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
    except PermissionError:
        pytest.skip("local sockets are disabled by the test sandbox")
    preview = _preview(port=port)
    preview.start()
    try:
        preview.publish({"camera2": np.zeros((24, 32, 3), dtype=np.uint8)})
        with urlopen(preview.url + "api/status", timeout=2) as response:
            body = response.read()
        assert response.status == 200
        assert b'"camera2"' in body
    finally:
        preview.stop()
