"""Single-port GELLO configuration editor, control console and camera preview."""
from __future__ import annotations

import argparse
import asyncio
from collections import deque
import hmac
import json
import multiprocessing
from pathlib import Path
import secrets
import struct
import time
from uuid import uuid4

from aiohttp import web, WSMsgType
import yaml

from lerobot_robot_ufactory.utils.webapp.recording_web_config import (
    ConfigStore, ConfigError, Conflict, validate_text, dataset_status, dataset_stamp,
)


class SessionManager:
    def __init__(self, project, simulate=False):
        self.project = Path(project).resolve()
        self.store = ConfigStore(project)
        self.simulate = simulate
        self.token = secrets.token_urlsafe(32)
        self.state = {"phase": "idle", "session_id": None, "version": 0,
                      "stage": "Select a configuration", "saved": 0, "frames": 0,
                      "elapsed": 0, "joint_mode": "all", "has_unsaved": False}
        self.process = None
        self.connection = None
        self.image_connection = None
        self.folder = None
        self.owner = None
        self.clients = {}
        self.log_cursors = {}
        self.preview_clients = {}
        self.images = {}
        self.logs = deque(maxlen=500)
        self.log_offset = 0
        self.log_sequence = 0
        self.confirmations = {}
        self.lock = asyncio.Lock()
        self.closing = False
        self.effective = None

    def send(self, message):
        if self.connection is not None and self.process is not None and self.process.is_alive():
            try:
                self.connection.send(message)
            except (OSError, EOFError):
                pass

    def subscriptions(self):
        names = set()
        for chosen in self.preview_clients.values():
            names.update(chosen)
        self.send({"action": "preview", "cameras": sorted(names)})

    async def inspect(self, payload):
        item = await asyncio.to_thread(self.store.read, payload["path"])
        if payload.get("revision") != item["revision"]:
            raise Conflict("Configuration changed; reload before launch")
        raw, _ = await asyncio.to_thread(validate_text, item["text"])
        status = await asyncio.to_thread(dataset_status, self.project, raw)
        stamp = None if self.simulate else await asyncio.to_thread(dataset_stamp, status["root"])
        if self.simulate:
            status = {"root": "SIMULATION: session-local files only", "exists": False,
                      "episodes": 0, "resumable": False, "reason": None}
        ticket = secrets.token_urlsafe(24)
        self.confirmations = {k: v for k, v in self.confirmations.items() if v["expires"] > time.monotonic()}
        self.confirmations[ticket] = {"path": item["path"], "revision": item["revision"],
                                     "status": status, "stamp": stamp,
                                     "expires": time.monotonic() + 300}
        return {"ticket": ticket, "dataset": status, "config": raw}

    async def launch(self, payload, client_id):
        async with self.lock:
            if self.owner != client_id:
                raise Conflict("This page does not hold control")
            if self.process is not None and self.process.is_alive():
                raise Conflict("A session is already running")
            ticket = self.confirmations.pop(payload.get("ticket"), None)
            if ticket is None or ticket["expires"] < time.monotonic():
                raise Conflict("Launch confirmation expired")
            item = self.store.read(ticket["path"])
            if item["revision"] != ticket["revision"]:
                raise Conflict("Configuration changed since confirmation")
            raw, _ = await asyncio.to_thread(validate_text, item["text"])
            mode = payload.get("dataset_mode", "new")
            status = ticket["status"]
            if status["exists"]:
                if mode == "resume" and not status["resumable"]:
                    raise ConfigError(status["reason"])
                if mode not in ("resume", "rebuild"):
                    raise ConfigError("Choose resume or rebuild for the existing dataset")
                if mode == "rebuild" and payload.get("confirm_root") != status["root"]:
                    raise ConfigError("Rebuild requires the exact dataset path confirmation")
            elif mode != "new":
                raise ConfigError("Dataset does not exist; choose new")
            gripper_mode = payload.get("gripper_mode", raw["teleop"].get("gripper_control_mode", "gello"))
            if gripper_mode not in ("gello", "keyboard"):
                raise ConfigError("Invalid gripper control source")
            j7 = payload.get("j7_enabled", raw["teleop"].get("joint7_only_mode_enabled", False))
            if type(j7) is not bool:
                raise ConfigError("j7_enabled must be boolean")
            raw["teleop"]["gripper_control_mode"] = gripper_mode
            raw["teleop"]["joint7_only_mode_enabled"] = j7
            text = yaml.safe_dump(raw, sort_keys=False, allow_unicode=True)
            await asyncio.to_thread(validate_text, text)
            cameras = payload.get("cameras", [])
            valid_cameras = raw.get("robot", {}).get("cameras", {})
            if not isinstance(cameras, list) or any(name not in valid_cameras for name in cameras):
                raise ConfigError("Unknown preview camera")
            sid = uuid4().hex
            folder = self.project / ".web-record" / "sessions" / sid
            folder.mkdir(parents=True)
            (folder / "config.yaml").write_text(item["text"], encoding="utf-8")
            (folder / "effective.yaml").write_text(text, encoding="utf-8")
            (folder / "launch.json").write_text(json.dumps({"path": item["path"], "dataset_mode": mode,
                                                          "simulate": self.simulate}), encoding="utf-8")
            if self.connection is not None:
                self.connection.close()
                self.image_connection.close()
                self.process.join(timeout=0)
            context = multiprocessing.get_context("spawn")
            parent, child = context.Pipe()
            images, sender = context.Pipe(duplex=False)
            from lerobot_robot_ufactory.utils.webapp.recording_web_worker import worker_main
            options = {"dataset_mode": mode, "dataset_stamp": ticket["stamp"],
                       "cameras": cameras, "simulate": self.simulate}
            process = context.Process(target=worker_main,
                                      args=(str(self.project), str(folder), sid, child, sender, options),
                                      name="uf-recording-worker")
            process.start()
            child.close()
            sender.close()
            self.process, self.connection, self.image_connection = process, parent, images
            self.folder, self.effective = folder, {"path": item["path"], "config": raw,
                                                 "dataset_mode": mode, "dataset": status}
            self.images.clear()
            self.logs.clear()
            self.log_offset = 0
            self.state = {"phase": "initializing", "session_id": sid, "version": 0,
                          "saved": status["episodes"] if mode == "resume" else 0,
                          "frames": 0, "elapsed": 0, "stage": "Starting worker", "has_unsaved": False,
                          "gripper_mode": gripper_mode, "j7_enabled": j7, "joint_mode": "all"}
            return self.snapshot(client_id)

    def snapshot(self, client_id=None):
        return {"type": "snapshot", "state": self.state, "effective": self.effective,
                "simulate": self.simulate, "controller": self.owner == client_id,
                "can_claim": self.owner is None, "client_id": client_id}

    async def monitor(self):
        tick = 0
        while not self.closing:
            if self.connection is not None:
                try:
                    for _ in range(50):
                        if not self.connection.poll():
                            break
                        message = self.connection.recv()
                        if message["type"] == "state":
                            self.state = message["state"]
                            if self.state["phase"] == "ready":
                                self.subscriptions()
                        elif message["type"] == "ack" and self.owner in self.clients:
                            try:
                                await asyncio.wait_for(self.clients[self.owner].send_json(message), 0.5)
                            except (OSError, asyncio.TimeoutError):
                                self.send({"action": "disconnect"})
                except (OSError, EOFError):
                    pass
                try:
                    for _ in range(16):
                        if not self.image_connection.poll():
                            break
                        packet = self.image_connection.recv_bytes(8 * 1024 * 1024)
                        size = struct.unpack("!I", packet[:4])[0]
                        header = json.loads(packet[4:4 + size])
                        self.images[header["camera"]] = (packet, time.monotonic())
                except (OSError, EOFError):
                    pass
                if not self.process.is_alive() and self.state["phase"] not in ("finished", "error"):
                    self.state = {**self.state, "phase": "error", "version": self.state["version"] + 1,
                                  "error": f"Worker exited ({self.process.exitcode}); inspect logs"}
            tick += 1
            if tick % 4 == 0:
                await self.broadcast()
            await asyncio.sleep(0.05)

    async def broadcast(self):
        if self.folder and (self.folder / "console.log").exists():
            with (self.folder / "console.log").open(encoding="utf-8", errors="replace") as stream:
                stream.seek(self.log_offset)
                chunk = stream.read(65536)
                self.log_offset = stream.tell()
            if chunk:
                self.log_sequence += 1
                self.logs.append({"id": self.log_sequence, "text": chunk})
        for client_id, socket in list(self.clients.items()):
            try:
                cursor = self.log_cursors.get(client_id, 0)
                await asyncio.wait_for(socket.send_json({**self.snapshot(client_id),
                    "logs": [chunk for chunk in self.logs if chunk["id"] > cursor]}), 0.5)
                self.log_cursors[client_id] = self.log_sequence
            except (OSError, ConnectionResetError, asyncio.TimeoutError):
                await socket.close()

    async def shutdown(self):
        self.closing = True
        self.send({"action": "disconnect"})
        if self.connection is not None:
            # Closing the pipe asks the worker to stop and clean up without a hard kill.
            self.connection.close()
        if self.process is not None:
            await asyncio.to_thread(self.process.join, 5)
        if self.image_connection is not None:
            self.image_connection.close()


MANAGER_KEY = web.AppKey("recording_session_manager", SessionManager)


@web.middleware
async def errors(request, handler):
    try:
        return await handler(request)
    except Conflict as exc:
        return web.json_response({"error": str(exc)}, status=409)
    except (ConfigError, ValueError, KeyError, TypeError) as exc:
        return web.json_response({"error": str(exc)}, status=400)
    except FileNotFoundError as exc:
        return web.json_response({"error": str(exc)}, status=404)


def authorize(request):
    manager = request.app[MANAGER_KEY]
    origin = f"{request.scheme}://{request.host}"
    if request.headers.get("Origin") != origin:
        raise web.HTTPForbidden(text="Same-origin request required")
    token = request.headers.get("X-Record-Token", request.query.get("token", ""))
    if not hmac.compare_digest(token, manager.token):
        raise web.HTTPForbidden(text="Invalid session token")
    return manager


async def index(request):
    page = Path(__file__).with_name("recording_web_assets") / "index.html"
    body = page.read_text(encoding="utf-8").replace("__RECORD_TOKEN__", request.app[MANAGER_KEY].token)
    return web.Response(text=body, content_type="text/html", headers={"Cache-Control": "no-store"})


async def api(request):
    manager = request.app[MANAGER_KEY]
    if request.method == "GET":
        if request.path == "/api/configs":
            return web.json_response({"configs": await asyncio.to_thread(manager.store.listing)})
        if request.path == "/api/config":
            return web.json_response(await asyncio.to_thread(manager.store.read, request.query["path"]))
        if request.path == "/api/trash":
            return web.json_response({"items": manager.store.trash()})
        if request.path == "/api/status":
            return web.json_response(manager.snapshot())
        if request.path == "/api/log":
            if manager.folder is None or not (manager.folder / "console.log").exists():
                return web.Response(text="")
            return web.FileResponse(manager.folder / "console.log", headers={"Content-Disposition": "attachment; filename=recording.log"})
    authorize(request)
    data = await request.json()
    if request.path == "/api/config":
        if request.method == "DELETE":
            result = await asyncio.to_thread(manager.store.delete, data["path"], data["revision"])
        else:
            result = await asyncio.to_thread(manager.store.save, data["path"], data["text"], data.get("revision"))
    elif request.path == "/api/validate":
        raw, _ = await asyncio.to_thread(validate_text, data["text"])
        result = {"ok": True, "config": raw}
    elif request.path == "/api/restore":
        result = await asyncio.to_thread(manager.store.restore, data["trash_id"])
    elif request.path == "/api/preflight":
        result = await manager.inspect(data)
    elif request.path == "/api/start":
        result = await manager.launch(data, data["client_id"])
    else:
        raise web.HTTPNotFound()
    return web.json_response(result)


async def control_socket(request):
    manager = authorize(request)
    socket = web.WebSocketResponse(max_msg_size=16384, heartbeat=10)
    await socket.prepare(request)
    client_id = uuid4().hex
    manager.clients[client_id] = socket
    if manager.owner is None:
        manager.owner = client_id
    await socket.send_json(manager.snapshot(client_id))
    try:
        async for message in socket:
            if message.type != WSMsgType.TEXT:
                continue
            try:
                data = json.loads(message.data)
                if data.get("type") == "claim":
                    if manager.owner is None:
                        manager.owner = client_id
                    await socket.send_json(manager.snapshot(client_id))
                elif manager.owner != client_id:
                    await socket.send_json({"type": "ack", "ok": False, "error": "This page is read-only"})
                else:
                    manager.send(data)
            except (ValueError, TypeError):
                await socket.send_json({"type": "ack", "ok": False, "error": "Invalid command"})
    finally:
        manager.clients.pop(client_id, None)
        manager.log_cursors.pop(client_id, None)
        if manager.owner == client_id:
            manager.owner = None
            manager.send({"action": "disconnect"})
    return socket


async def preview_socket(request):
    manager = authorize(request)
    socket = web.WebSocketResponse(max_msg_size=16384, heartbeat=10)
    await socket.prepare(request)
    manager.preview_clients[socket] = set()

    async def transmit():
        seen = {}
        while not socket.closed:
            for camera in list(manager.preview_clients.get(socket, [])):
                frame = manager.images.get(camera)
                if frame is not None and seen.get(camera) != frame[1]:
                    await asyncio.wait_for(socket.send_bytes(frame[0]), 1)
                    seen[camera] = frame[1]
            await asyncio.sleep(0.05)

    task = asyncio.create_task(transmit())
    try:
        async for message in socket:
            if message.type == WSMsgType.TEXT:
                try:
                    data = json.loads(message.data)
                    names = data.get("cameras", [])
                    valid = (manager.effective or {}).get("config", {}).get("robot", {}).get("cameras", {})
                    if not isinstance(names, list) or any(name not in valid for name in names):
                        raise ValueError("Unknown cameras")
                    manager.preview_clients[socket] = set(names)
                    manager.subscriptions()
                except (ValueError, TypeError):
                    await socket.send_json({"error": "Invalid camera subscription"})
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        manager.preview_clients.pop(socket, None)
        manager.subscriptions()
    return socket


def create_app(project, simulate=False):
    app = web.Application(middlewares=[errors], client_max_size=1024 * 1024)
    manager = SessionManager(project, simulate)
    app[MANAGER_KEY] = manager
    app.router.add_get("/", index)
    app.router.add_get("/ws/control", control_socket)
    app.router.add_get("/ws/preview", preview_socket)
    for path in ("configs", "config", "status", "log", "trash"):
        app.router.add_get(f"/api/{path}", api)
    for path in ("config", "validate", "preflight", "start", "restore"):
        app.router.add_post(f"/api/{path}", api)
    app.router.add_delete("/api/config", api)

    async def lifecycle(app):
        task = asyncio.create_task(manager.monitor())
        yield
        await manager.shutdown()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    app.cleanup_ctx.append(lifecycle)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8769)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--simulate", action="store_true", help="Synthetic preview/control; never connect hardware")
    args = parser.parse_args()
    web.run_app(create_app(args.project, args.simulate), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
