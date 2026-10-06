"""Loopback HTTP controls for the unified GELLO model and tuning page."""

import argparse
import hmac
import json
import secrets
import signal
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ..config import DeviceProfile, MAX_TUNING_SLEW_A_S, MIN_TUNING_SLEW_A_S
from ..control.tuning import INITIAL_SLEW_A_S, TuningSession
from .model_web import render_html, viewer_data

def tuning_page(profile, token):
    data = viewer_data(profile)
    data["tuning"] = {"constant_damping_a": profile.constant_damping_a.tolist(), "constant_current_a": profile.constant_current_a, "token": token,
                      "initial_slew_a_s": INITIAL_SLEW_A_S,
                      "min_slew_a_s": MIN_TUNING_SLEW_A_S, "max_slew_a_s": MAX_TUNING_SLEW_A_S}
    return render_html(data).encode("utf-8")


def make_tuning_server(port, page, session, token):
    class Handler(BaseHTTPRequestHandler):
        def allowed_host(self):
            return self.headers.get("Host") in (
                f"127.0.0.1:{self.server.server_port}",
                f"localhost:{self.server.server_port}",
            )

        def reply(self, code, body, content_type="application/json; charset=utf-8"):
            if not isinstance(body, bytes):
                body = json.dumps(body, allow_nan=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if not self.allowed_host():
                self.reply(403, {"error": "仅允许本机访问"})
                return
            path = self.path.split("?", 1)[0]
            if path == "/":
                self.reply(200, page, "text/html; charset=utf-8")
            elif path == "/api/state":
                self.reply(200, session.snapshot())
            else:
                self.reply(404, {"error": "Unknown route"})

        def do_POST(self):
            expected_origin = f"http://{self.headers.get('Host')}"
            if (
                not self.allowed_host()
                or self.headers.get("Origin", expected_origin) != expected_origin
                or not hmac.compare_digest(self.headers.get("X-Gello-Token", ""), token)
            ):
                self.reply(403, {"error": "请求来源或控制凭据无效"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("Invalid request length")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("JSON object required")
                if self.path == "/api/heartbeat":
                    session.heartbeat()
                    self.reply(200, {"ok": True})
                elif self.path == "/api/start":
                    session.start()
                    self.reply(200, session.snapshot())
                elif self.path == "/api/stop":
                    session.stop()
                    self.reply(200, session.snapshot())
                elif self.path == "/api/view":
                    session.set_view_mode(body.get("mode"))
                    self.reply(200, session.snapshot())
                elif self.path == "/api/slew":
                    session.set_current_slew(body.get("current_slew_a_s"))
                    self.reply(200, session.snapshot())
                else:
                    self.reply(404, {"error": "Unknown route"})
            except (ValueError, TypeError) as exc:
                self.reply(400, {"error": str(exc)})
            except Exception as exc:
                self.reply(503, {"error": str(exc)})

        def log_message(self, format, *args):
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--log-dir", default="logs/current_tuning")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535")
    profile = DeviceProfile(args.profile)
    session = TuningSession(profile, args.log_dir)
    token = secrets.token_urlsafe(32)
    server = make_tuning_server(args.port, tuning_page(profile, token), session, token)
    stop = threading.Event()
    old_handlers = {
        sig: signal.signal(sig, lambda *_: stop.set())
        for sig in (signal.SIGINT, signal.SIGTERM)
    }
    server.timeout = 0.2
    print(f"GELLO 恒流与阻尼调参: http://127.0.0.1:{server.server_port}", flush=True)
    print("网页按钮控制启停；当前未启用电机扭矩。Ctrl+C 卸力并退出。", flush=True)
    try:
        while not stop.is_set():
            server.handle_request()
    finally:
        try:
            session.stop()
        finally:
            server.server_close()
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    main()
