"""Best-effort web preview for frames already captured by a recording loop.

The recorder only publishes references to its latest frames.  Resizing, JPEG
encoding, and network writes happen on background threads, and preview frames
are deliberately dropped whenever those threads cannot keep up.
"""

from __future__ import annotations

import http.server
import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class WebPreviewConfig:
    """Configuration for the optional recording-time camera preview."""

    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 8765
    fps: float = 8.0
    width: int = 480
    jpeg_quality: int = 65

    def validate(self) -> None:
        if not self.host:
            raise ValueError("web_preview.host must not be empty")
        if not 1 <= self.port <= 65535:
            raise ValueError("web_preview.port must be between 1 and 65535")
        if self.fps <= 0:
            raise ValueError("web_preview.fps must be positive")
        if self.width <= 0:
            raise ValueError("web_preview.width must be positive")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("web_preview.jpeg_quality must be between 1 and 100")


class _PreviewFeed:
    def __init__(self, name: str) -> None:
        self.name = name
        self.condition = threading.Condition()
        self.jpeg: bytes | None = None
        self.frame_id = 0
        self.clients = 0
        self.stopped = False

    def add_client(self) -> None:
        with self.condition:
            self.clients += 1

    def remove_client(self) -> None:
        with self.condition:
            self.clients = max(0, self.clients - 1)

    def publish_jpeg(self, jpeg: bytes) -> None:
        with self.condition:
            self.jpeg = jpeg
            self.frame_id += 1
            self.condition.notify_all()

    def wait_for_jpeg(self, previous_id: int) -> tuple[bytes | None, int, bool]:
        with self.condition:
            self.condition.wait_for(
                lambda: self.frame_id > previous_id or self.stopped,
                timeout=1.0,
            )
            return self.jpeg, self.frame_id, self.stopped

    def stop(self) -> None:
        with self.condition:
            self.stopped = True
            self.condition.notify_all()


class RecordingWebPreview:
    """Serve a lossy, asynchronous preview of recording observations."""

    def __init__(self, config: WebPreviewConfig) -> None:
        config.validate()
        self.config = config
        self._condition = threading.Condition()
        self._latest_frames: dict[str, np.ndarray] = {}
        self._source_generation = 0
        self._encoded_frames = 0
        self._last_encode_ms = 0.0
        self._max_encode_ms = 0.0
        self._stop_event = threading.Event()
        self._feeds: dict[str, _PreviewFeed] = {}
        self._encoder_thread: threading.Thread | None = None
        self._server: http.server.ThreadingHTTPServer | None = None
        self._server_thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        host = "127.0.0.1" if self.config.host == "0.0.0.0" else self.config.host
        return f"http://{host}:{self.config.port}/"

    def start(self) -> None:
        if self._server is not None:
            return
        handler = type("RecordingPreviewHandler", (_PreviewHandler,), {"preview": self})
        self._server = http.server.ThreadingHTTPServer(
            (self.config.host, self.config.port), handler
        )
        self._server.daemon_threads = True
        self._encoder_thread = threading.Thread(
            target=self._encode_loop,
            name="uf-recording-preview-encoder",
            daemon=True,
        )
        self._server_thread = threading.Thread(
            target=self._server.serve_forever,
            name="uf-recording-preview-http",
            daemon=True,
        )
        self._encoder_thread.start()
        self._server_thread.start()

    def publish(self, observation: dict[str, Any]) -> None:
        """Non-blockingly replace the latest previewable image references."""
        frames = {
            key: value
            for key, value in observation.items()
            if isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[2] in (3, 4)
        }
        if not frames:
            return
        with self._condition:
            self._latest_frames = frames
            for name in frames:
                self._feeds.setdefault(name, _PreviewFeed(name))
            self._source_generation += 1
            self._condition.notify()

    def camera_names(self) -> list[str]:
        with self._condition:
            return sorted(self._feeds)

    def feed(self, name: str | None) -> _PreviewFeed | None:
        if name is None:
            return None
        with self._condition:
            return self._feeds.get(name)

    def timing_stats(self) -> dict[str, float | int]:
        """Return a cheap snapshot for recording synchronization diagnostics."""
        with self._condition:
            return {
                "preview_clients": sum(feed.clients for feed in self._feeds.values()),
                "preview_source_generation": self._source_generation,
                "preview_encoded_frames": self._encoded_frames,
                "preview_last_encode_ms": self._last_encode_ms,
                "preview_max_encode_ms": self._max_encode_ms,
            }

    def _has_clients(self) -> bool:
        return any(feed.clients > 0 for feed in self._feeds.values())

    def _encode_loop(self) -> None:
        last_generation = 0
        next_encode_at = 0.0
        period = 1.0 / self.config.fps
        while not self._stop_event.is_set():
            with self._condition:
                self._condition.wait_for(
                    lambda: self._stop_event.is_set()
                    or (
                        self._source_generation > last_generation
                        and self._has_clients()
                    ),
                    timeout=1.0,
                )
                if self._stop_event.is_set():
                    return
                generation = self._source_generation
                frames = dict(self._latest_frames)

            delay = next_encode_at - time.monotonic()
            if delay > 0 and self._stop_event.wait(delay):
                return
            next_encode_at = time.monotonic() + period
            last_generation = generation

            for name, frame in frames.items():
                feed = self.feed(name)
                if feed is None or feed.clients == 0:
                    continue
                try:
                    encode_started = time.perf_counter()
                    jpeg = self._encode_frame(frame)
                    encode_ms = (time.perf_counter() - encode_started) * 1000
                except Exception:
                    logger.exception("Failed to encode web preview frame for %s", name)
                    continue
                feed.publish_jpeg(jpeg)
                with self._condition:
                    self._encoded_frames += 1
                    self._last_encode_ms = encode_ms
                    self._max_encode_ms = max(self._max_encode_ms, encode_ms)

    def _encode_frame(self, frame: np.ndarray) -> bytes:
        height, width = frame.shape[:2]
        if width != self.config.width:
            target_height = max(1, round(height * self.config.width / width))
            frame = cv2.resize(
                frame,
                (self.config.width, target_height),
                interpolation=cv2.INTER_AREA,
            )
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ok, buffer = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality]
        )
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        return buffer.tobytes()

    def stop(self) -> None:
        if self._stop_event.is_set():
            return
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        for feed in list(self._feeds.values()):
            feed.stop()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._server_thread is not None:
            self._server_thread.join(timeout=2.0)
        if self._encoder_thread is not None:
            self._encoder_thread.join(timeout=2.0)


_WEB_PAGE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LeRobot 录制预览</title>
<style>
:root{color-scheme:dark;font-family:system-ui,sans-serif}body{margin:0;background:#101214;color:#eee}
header{padding:14px 18px;background:#191c20;border-bottom:1px solid #333}h1{margin:0;font-size:20px}
#status{margin-top:7px;color:#aeb5bd;font-size:13px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px;padding:14px}
.tile{overflow:hidden;background:#191c20;border:1px solid #333;border-radius:7px}.tile h2{margin:0;padding:9px 11px;font-size:14px}.tile img{display:block;width:100%;background:#08090a}
</style></head><body><header><h1>LeRobot 录制相机预览</h1><div id="status">正在等待相机帧…</div></header>
<main id="grid" class="grid"></main><script>
let signature='';async function refresh(){try{const r=await fetch('/api/status',{cache:'no-store'});const p=await r.json();
document.getElementById('status').textContent=p.cameras.length?`${p.cameras.length} 路相机 · 预览 ${p.fps} FPS · 采集优先，预览允许丢帧`:'正在等待相机帧…';
const next=p.cameras.join('|');if(next!==signature){signature=next;const grid=document.getElementById('grid');grid.replaceChildren();
for(const name of p.cameras){const tile=document.createElement('section');tile.className='tile';const h=document.createElement('h2');h.textContent=name;
const img=document.createElement('img');img.alt=name;img.src='/stream?camera='+encodeURIComponent(name);tile.append(h,img);grid.appendChild(tile);}}}catch(e){document.getElementById('status').textContent='预览服务不可用：'+e}}
refresh();setInterval(refresh,2000);</script></body></html>""".encode()


class _PreviewHandler(http.server.BaseHTTPRequestHandler):
    preview: RecordingWebPreview

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        if parsed.path == "/":
            self._send(200, "text/html; charset=utf-8", _WEB_PAGE)
        elif parsed.path == "/api/status":
            body = json.dumps(
                {"ok": True, "cameras": self.preview.camera_names(), "fps": self.preview.config.fps}
            ).encode()
            self._send(200, "application/json; charset=utf-8", body)
        elif parsed.path == "/stream":
            self._stream(parse_qs(parsed.query).get("camera", [None])[0])
        elif parsed.path == "/favicon.ico":
            self._send(204, "text/plain", b"")
        else:
            self.send_error(404)

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _stream(self, name: str | None) -> None:
        feed = self.preview.feed(name)
        if feed is None:
            self._send(404, "text/plain; charset=utf-8", b"Unknown camera\n")
            return
        feed.add_client()
        with self.preview._condition:
            self.preview._condition.notify()
        try:
            self.send_response(200)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            frame_id = 0
            while True:
                jpeg, next_id, stopped = feed.wait_for_jpeg(frame_id)
                if stopped:
                    return
                if jpeg is None or next_id == frame_id:
                    continue
                frame_id = next_id
                self.wfile.write(
                    b"--frame\r\nContent-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                    + jpeg
                    + b"\r\n"
                )
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            feed.remove_client()

    def log_message(self, format: str, *args: Any) -> None:
        pass
