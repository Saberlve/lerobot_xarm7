#!/usr/bin/env python3
"""Web viewer for one or more Intel RealSense color streams."""

from __future__ import annotations

import argparse
import http.server
import json
import threading
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlsplit

import cv2
import pyrealsense2 as rs

from lerobot.cameras import ColorMode
from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig


DEFAULT_SERIAL = None
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 30
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8765


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview one or more Intel RealSense color streams in a browser"
    )
    parser.add_argument(
        "-s",
        "--serial",
        action="append",
        default=DEFAULT_SERIAL,
        help="camera serial number; repeat to limit the device list to selected cameras",
    )
    parser.add_argument("-W", "--width", type=int, default=DEFAULT_WIDTH, help="color width")
    parser.add_argument("-H", "--height", type=int, default=DEFAULT_HEIGHT, help="color height")
    parser.add_argument(
        "-F", "--fps", type=int, default=DEFAULT_FPS, help="capture and preview FPS (default: 30)"
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "opencv", "web"),
        default="web",
        help="preview backend (default: web; web provides device selection)",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="web server bind address (default: 0.0.0.0; use 127.0.0.1 for local-only access)",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="web server port")
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="capture frames without opening a preview",
    )
    return parser.parse_args(argv)


def opencv_has_gui() -> bool:
    return next(
        ("NONE" not in line for line in cv2.getBuildInformation().splitlines() if "GUI:" in line),
        False,
    )


@dataclass(frozen=True)
class RealSenseDevice:
    serial: str
    name: str
    product_line: str


def discover_devices() -> list[RealSenseDevice]:
    """Discover connected RealSense devices without opening their streams."""
    devices: list[RealSenseDevice] = []
    context = rs.context()
    for device in context.query_devices():
        devices.append(
            RealSenseDevice(
                serial=device.get_info(rs.camera_info.serial_number),
                name=device.get_info(rs.camera_info.name),
                product_line=device.get_info(rs.camera_info.product_line),
            )
        )
    return devices


class _CameraFeed:
    """Own one camera connection and publish its latest JPEG frame."""

    def __init__(self, device: RealSenseDevice, width: int, height: int, fps: int) -> None:
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.camera = RealSenseCamera(
            RealSenseCameraConfig(
                serial_number_or_name=device.serial,
                width=width,
                height=height,
                fps=fps,
                color_mode=ColorMode.BGR,
            )
        )
        self.condition = threading.Condition()
        self.jpeg: bytes | None = None
        self.frame: Any | None = None
        self.frame_id = 0
        self.frame_count = 0
        self.error: str | None = None
        self.connected = False
        self.stopped = False
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def connect(self) -> None:
        self.camera.connect()
        self.connected = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"realsense-{self.device.serial}",
            daemon=True,
        )
        self._thread.start()

    def _set_error(self, error: Exception) -> None:
        with self.condition:
            self.error = f"{type(error).__name__}: {error}"
            self.condition.notify_all()

    def _capture_loop(self) -> None:
        period = 1.0 / self.fps
        next_deadline = time.monotonic()
        while not self._stop_event.is_set():
            try:
                frame = self.camera.read()
                encoded, buffer = cv2.imencode(".jpg", frame)
                if not encoded:
                    raise RuntimeError("JPEG encoding failed")
                with self.condition:
                    self.frame = frame
                    self.jpeg = buffer.tobytes()
                    self.frame_id += 1
                    self.frame_count += 1
                    self.error = None
                    self.condition.notify_all()
            except Exception as exc:
                self._set_error(exc)
                if self._stop_event.wait(0.1):
                    break

            next_deadline += period
            delay = next_deadline - time.monotonic()
            if delay > 0:
                self._stop_event.wait(delay)
            elif -delay > period * 2:
                next_deadline = time.monotonic()

    def wait_for_frame(self, last_frame_id: int) -> tuple[bytes | None, int, bool]:
        with self.condition:
            self.condition.wait_for(
                lambda: self.frame_id > last_frame_id or self.stopped,
                timeout=1.0,
            )
            return self.jpeg, self.frame_id, self.stopped

    def latest_frame(self) -> Any | None:
        with self.condition:
            return None if self.frame is None else self.frame.copy()

    def status(self, selected: bool) -> dict[str, Any]:
        with self.condition:
            return {
                "serial": self.device.serial,
                "name": self.device.name,
                "product_line": self.device.product_line,
                "width": self.width,
                "height": self.height,
                "fps": self.fps,
                "connected": self.connected,
                "selected": selected,
                "frame_count": self.frame_count,
                "error": self.error,
            }

    def stop(self) -> None:
        self._stop_event.set()
        with self.condition:
            self.stopped = True
            self.condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.connected:
            try:
                self.camera.disconnect()
            except Exception as exc:
                self._set_error(exc)
            self.connected = False


class _PreviewState:
    def __init__(self, feeds: list[_CameraFeed], selected: list[str], fps: int) -> None:
        self.feeds = {feed.device.serial: feed for feed in feeds}
        self.selected = set(selected)
        self.fps = fps

    def get_feed(self, serial: str | None) -> _CameraFeed | None:
        return self.feeds.get(serial) if serial else None

    def device_status(self) -> list[dict[str, Any]]:
        return [
            feed.status(feed.device.serial in self.selected)
            for feed in self.feeds.values()
        ]

    def stop(self) -> None:
        for feed in self.feeds.values():
            feed.stop()


WEB_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RealSense Viewer</title>
<style>
:root { color-scheme: dark; font-family: system-ui, sans-serif; }
* { box-sizing: border-box; }
body { margin: 0; background: #101214; color: #e8eaed; }
header { position: sticky; top: 0; z-index: 2; padding: 14px 18px; background: #191c20; border-bottom: 1px solid #30343a; }
h1 { margin: 0 0 10px; font-size: 20px; font-weight: 600; }
#status { color: #9da5af; font-size: 13px; }
.toolbar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-top: 12px; }
button { border: 1px solid #4a515b; border-radius: 5px; padding: 7px 11px; color: #e8eaed; background: #262b31; cursor: pointer; }
button:hover { background: #323941; }
#devices { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.device { display: flex; align-items: center; gap: 7px; padding: 7px 9px; border: 1px solid #3b4149; border-radius: 5px; background: #20242a; font-size: 13px; }
.device input { width: 16px; height: 16px; accent-color: #4da3ff; }
.device small { color: #9da5af; }
.device.error { border-color: #a44d4d; }
#grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 12px; padding: 14px; }
.tile { min-width: 0; overflow: hidden; background: #191c20; border: 1px solid #30343a; border-radius: 6px; }
.tile h2 { margin: 0; padding: 9px 11px; font-size: 14px; font-weight: 500; }
.tile img { display: block; width: 100%; height: auto; background: #08090a; }
.empty { padding: 36px 18px; color: #9da5af; text-align: center; }
</style>
</head>
<body>
<header>
  <h1>RealSense Viewer</h1>
  <div id="status">Loading devices...</div>
  <div class="toolbar">
    <button id="all" type="button">Select all</button>
    <button id="none" type="button">Clear</button>
  </div>
  <div id="devices"></div>
</header>
<main id="grid"><div class="empty">Loading...</div></main>
<script>
const devicesEl = document.getElementById('devices');
const gridEl = document.getElementById('grid');
const statusEl = document.getElementById('status');
let devices = [];
let selected = new Set();

function deviceLabel(device) {
  return `${device.name} (${device.serial})`;
}

function streamUrl(serial) {
  return `/stream?serial=${encodeURIComponent(serial)}`;
}

function renderGrid() {
  gridEl.replaceChildren();
  const visible = devices.filter(device => selected.has(device.serial) && device.connected);
  if (!visible.length) {
    const empty = document.createElement('div');
    empty.className = 'empty';
    empty.textContent = 'Select a connected device to display.';
    gridEl.appendChild(empty);
    return;
  }
  for (const device of visible) {
    const tile = document.createElement('section');
    tile.className = 'tile';
    const title = document.createElement('h2');
    title.textContent = deviceLabel(device);
    const image = document.createElement('img');
    image.alt = deviceLabel(device);
    image.src = streamUrl(device.serial);
    tile.append(title, image);
    gridEl.appendChild(tile);
  }
}

function renderDevices() {
  devicesEl.replaceChildren();
  for (const device of devices) {
    const label = document.createElement('label');
    label.className = `device${device.error ? ' error' : ''}`;
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.checked = selected.has(device.serial);
    checkbox.disabled = !device.connected;
    checkbox.addEventListener('change', () => {
      if (checkbox.checked) selected.add(device.serial);
      else selected.delete(device.serial);
      renderGrid();
    });
    const text = document.createElement('span');
    text.textContent = deviceLabel(device);
    const details = document.createElement('small');
    details.textContent = device.connected ? `${device.width}x${device.height} @ ${device.fps} Hz` : (device.error || 'offline');
    label.append(checkbox, text, details);
    devicesEl.appendChild(label);
  }
}

async function refresh() {
  try {
    const response = await fetch('/api/devices', { cache: 'no-store' });
    const payload = await response.json();
    devices = payload.devices;
    if (!selected.size) {
      for (const device of devices.filter(device => device.selected && device.connected)) selected.add(device.serial);
    }
    statusEl.textContent = `${devices.filter(device => device.connected).length}/${devices.length} devices connected · ${payload.fps} Hz`;
    renderDevices();
    renderGrid();
  } catch (error) {
    statusEl.textContent = `Device service unavailable: ${error}`;
  }
}

document.getElementById('all').addEventListener('click', () => {
  for (const device of devices) if (device.connected) selected.add(device.serial);
  renderDevices(); renderGrid();
});
document.getElementById('none').addEventListener('click', () => {
  selected.clear(); renderDevices(); renderGrid();
});
refresh();
</script>
</body>
</html>
""".encode("utf-8")


class _PreviewHandler(http.server.BaseHTTPRequestHandler):
    state: _PreviewState

    def _send_bytes(self, status: int, content_type: str, body: bytes) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        path = parsed.path

        if path == "/":
            self._send_bytes(200, "text/html; charset=utf-8", WEB_PAGE)
            return
        if path == "/favicon.ico":
            self._send_bytes(204, "text/plain; charset=utf-8", b"")
            return
        if path == "/health":
            self._send_bytes(200, "application/json; charset=utf-8", self._json(self.state.device_status()))
            return
        if path == "/api/devices":
            self._send_bytes(
                200,
                "application/json; charset=utf-8",
                self._json(self.state.device_status(), include_fps=True),
            )
            return
        if path == "/stream":
            serial = parse_qs(parsed.query).get("serial", [None])[0]
            self._stream(serial)
            return
        try:
            self.send_error(404)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, devices: list[dict[str, Any]], include_fps: bool = False) -> bytes:
        payload: dict[str, Any] = {"ok": True, "devices": devices}
        if include_fps:
            payload["fps"] = self.state.fps
        return json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def _stream(self, serial: str | None) -> None:
        feed = self.state.get_feed(serial)
        if feed is None:
            self._send_bytes(404, "text/plain; charset=utf-8", b"Unknown camera serial\n")
            return
        if not feed.connected:
            self._send_bytes(503, "text/plain; charset=utf-8", b"Camera is not connected\n")
            return

        try:
            self.send_response(200)
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            last_frame_id = 0
            while True:
                jpeg, frame_id, stopped = feed.wait_for_frame(last_frame_id)
                if stopped:
                    return
                if jpeg is None or frame_id == last_frame_id:
                    continue
                last_frame_id = frame_id
                self.wfile.write(
                    b"--frame\r\nContent-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                    + jpeg
                    + b"\r\n"
                )
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format: str, *args: Any) -> None:
        pass


def start_web_server(
    state: _PreviewState, host: str, port: int
) -> http.server.ThreadingHTTPServer:
    handler = type("RealSensePreviewHandler", (_PreviewHandler,), {"state": state})
    server = http.server.ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    if host == "0.0.0.0":
        print(f"Open http://127.0.0.1:{port}/ on this machine to view the streams.")
        print(f"For another machine, use http://<server-ip>:{port}/")
    else:
        print(f"Open http://{host}:{port}/ in a browser to view the streams.")
    print("Press Ctrl+C to exit.")
    return server


def _make_mosaic(feeds: list[_CameraFeed]) -> Any | None:
    tiles = []
    for feed in feeds:
        frame = feed.latest_frame()
        if frame is None:
            continue
        cv2.putText(
            frame,
            f"{feed.device.name} {feed.device.serial}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        tiles.append(frame)
    if not tiles:
        return None
    height = max(tile.shape[0] for tile in tiles)
    aligned = [
        cv2.resize(tile, (int(tile.shape[1] * height / tile.shape[0]), height))
        if tile.shape[0] != height
        else tile
        for tile in tiles
    ]
    return cv2.hconcat(aligned)


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise SystemExit("FPS must be positive")

    if args.no_display:
        backend = "none"
    elif args.backend == "auto":
        backend = "opencv" if opencv_has_gui() else "web"
    else:
        backend = args.backend

    devices = discover_devices()
    requested = list(dict.fromkeys(args.serial or []))
    if requested:
        requested_set = set(requested)
        devices = [device for device in devices if device.serial in requested_set]
        missing = sorted(requested_set - {device.serial for device in devices})
        if missing:
            print(f"Requested cameras not found: {', '.join(missing)}")
    if not devices:
        raise SystemExit("No RealSense cameras found.")

    feeds: list[_CameraFeed] = []
    for device in devices:
        feed = _CameraFeed(device, args.width, args.height, args.fps)
        try:
            print(f"Opening RealSense {device.name} {device.serial}: {args.width}x{args.height} @ {args.fps} FPS")
            feed.connect()
            print(f"Connected {device.serial}")
        except Exception as exc:
            feed.error = f"{type(exc).__name__}: {exc}"
            print(f"Failed to connect {device.serial}: {feed.error}")
        feeds.append(feed)

    initially_selected = requested or ([feeds[0].device.serial] if feeds else [])
    state = _PreviewState(feeds, initially_selected, args.fps)
    web_server = None

    if backend == "web":
        web_server = start_web_server(state, args.host, args.port)
    elif backend == "opencv":
        print("Press q or Esc to exit.")

    try:
        if backend == "web":
            while True:
                time.sleep(1.0)
        elif backend == "none":
            while True:
                time.sleep(1.0)
                statuses = [
                    f"{feed.device.serial}:{'OK' if feed.connected and feed.error is None else 'ERROR'}"
                    for feed in feeds
                ]
                print(" | ".join(statuses), flush=True)
        else:
            while True:
                mosaic = _make_mosaic(feeds)
                if mosaic is not None:
                    cv2.imshow("RealSense Color", mosaic)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        state.stop()
        if web_server is not None:
            web_server.shutdown()
            web_server.server_close()
        if backend == "opencv":
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        print("RealSense preview stopped.")


if __name__ == "__main__":
    main()
