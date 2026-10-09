"""Web/control integration tests: no physical devices are opened."""
import asyncio
import json
from pathlib import Path
import struct
import threading
import time
from types import SimpleNamespace

from aiohttp.test_utils import TestClient, TestServer
import pytest
import yaml

from lerobot_robot_ufactory.utils.webapp.uf_lerobot_record_web import create_app, MANAGER_KEY
from lerobot_robot_ufactory.utils.webapp.recording_control import RecordingControl
from lerobot_robot_ufactory.utils.webapp.recording_web_config import ConfigStore, ConfigError, Conflict, validate_text
from lerobot_robot_ufactory.utils.realtime_teleop import RealtimeTeleopController
from lerobot_robot_ufactory.utils.webapp.web_preview import RecordingWebPreview, WebPreviewConfig
from lerobot_robot_ufactory.utils.webapp.recording_web_worker import prepare_web_dataset
from lerobot_robot_ufactory.utils.webapp.recording_web_config import dataset_stamp


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    configs = root / "config/gello/tasks/example"
    configs.mkdir(parents=True)
    raw = yaml.safe_load(Path("config/gello/xarm7_gello_base.yaml").read_text())
    raw["dataset"]["root"] = "datasets/test-web"
    raw["dataset"]["episode_time_s"] = 30
    raw["teleop"]["joint7_only_mode_enabled"] = True
    (configs / "example.yaml").write_text("# preserved comment\n" + yaml.safe_dump(raw, sort_keys=False))
    return root


def test_config_crud_versions_comments_and_restore(project):
    store = ConfigStore(project)
    name = "tasks/example/example.yaml"
    original = store.read(name)
    assert [item["path"] for item in store.listing()] == [name]
    text = original["text"].replace("datasets/test-web", "datasets/changed")
    updated = store.save(name, text, original["revision"])
    assert updated["text"].startswith("# preserved comment")
    with pytest.raises(Conflict):
        store.save(name, text, original["revision"])
    clone = store.save("tasks/new/clone.yml", text, None)
    with pytest.raises(Conflict):
        store.save("tasks/new/clone.yml", text, None)
    deleted = store.delete(clone["path"], clone["revision"])
    assert len(store.listing()) == 1
    assert store.trash()[0]["trash_id"] == deleted["trash_id"]
    assert store.restore(deleted["trash_id"])["text"] == text


@pytest.mark.parametrize("name", ["../outside.yaml", "/tmp/outside.yaml", ".web-trash/evil.yaml", "bad.txt", "..\\evil.yaml"])
def test_config_paths_cannot_escape(project, name):
    with pytest.raises(ConfigError):
        ConfigStore(project).path(name)


def test_symlink_is_not_followed(project, tmp_path):
    external = tmp_path / "external.yaml"
    external.write_text("secret: true")
    (project / "config/gello/link.yaml").symlink_to(external)
    with pytest.raises(ConfigError):
        ConfigStore(project).read("link.yaml")


@pytest.mark.parametrize("mutation", ["unknown", "nested", "duplicate", "wrong_type"])
def test_strict_validation_rejects_mistakes(project, mutation):
    text = ConfigStore(project).read("tasks/example/example.yaml")["text"]
    if mutation == "duplicate":
        text += "\nsynchronize: false\nsynchronize: true\n"
    else:
        raw = yaml.safe_load(text)
        if mutation == "unknown":
            raw["unknown_setting"] = 1
        elif mutation == "nested":
            raw["teleop"]["griper_mode"] = "keyboard"
        else:
            raw["dataset"]["fps"] = "not-a-number"
        text = yaml.safe_dump(raw)
    with pytest.raises(ConfigError):
        validate_text(text)


@pytest.mark.parametrize("mode", ["new", "rebuild", "resume"])
def test_dataset_selection_and_recoverable_rebuild(project, mode):
    root = project / "datasets/existing"
    root.mkdir(parents=True)
    (root / "sentinel.txt").write_text("previous data")
    cfg = SimpleNamespace(dataset=SimpleNamespace(root=root))
    raw = {"dataset": {"root": str(root)}}
    options = {"dataset_mode": mode, "dataset_stamp": dataset_stamp(root)}
    if mode == "rebuild":
        prepare_web_dataset(project, cfg, raw, options)
        assert not root.exists()
        archives = list((root.parent / ".web-dataset-trash").glob("*/sentinel.txt"))
        assert len(archives) == 1
        assert archives[0].read_text() == "previous data"
    else:
        with pytest.raises(RuntimeError):
            prepare_web_dataset(project, cfg, raw, options)
        assert (root / "sentinel.txt").read_text() == "previous data"


def test_rebuild_rejects_stale_confirmation_and_outside_directory(project, tmp_path):
    root = project / "datasets/existing"
    root.mkdir(parents=True)
    options = {"dataset_mode": "rebuild", "dataset_stamp": dataset_stamp(root)}
    (root / "new-episode.txt").touch()
    cfg = SimpleNamespace(dataset=SimpleNamespace(root=root))
    with pytest.raises(RuntimeError, match="changed"):
        prepare_web_dataset(project, cfg, {"dataset": {"root": str(root)}}, options)
    outside = tmp_path / "outside"
    outside.mkdir()
    cfg.dataset.root = outside
    options["dataset_stamp"] = dataset_stamp(outside)
    with pytest.raises(RuntimeError, match="restricted"):
        prepare_web_dataset(project, cfg, {"dataset": {"root": str(outside)}}, options)
    assert outside.exists()


def mailbox():
    messages = []
    now = [0.0]
    control = RecordingControl("session", send=messages.append, clock=lambda: now[0])
    teleop = SimpleNamespace(config=SimpleNamespace(gripper_control_mode="keyboard", joint7_only_mode_enabled=True),
                             _joint7_only_active=False, keys={}, switches=0)
    teleop.set_gripper_keyboard_state = lambda **keys: teleop.keys.update(keys)
    def joint(pressed):
        if pressed:
            teleop.switches += 1
    teleop.set_joint7_mode_key = joint
    control.bind(teleop)
    control.transition("ready")
    return control, teleop, messages, now


def command(control, action, **values):
    message = {"session_id": control.session_id, "version": control.state["version"],
               "request_id": f"request-{time.time_ns()}", "action": action, **values}
    control.command(message)
    return message


def test_mailbox_rejects_duplicate_stale_and_cross_session_commands():
    control, teleop, messages, now = mailbox()
    message = command(control, "start")
    control.command(message)
    assert messages[-1]["ok"] is False
    assert control.actions.qsize() == 1
    control.transition("recording")
    command(control, "joint_mode", session_id="other")
    assert teleop.switches == 0
    command(control, "joint_mode", version=-1)
    assert teleop.switches == 0
    command(control, "joint_mode")
    assert teleop.switches == 1


def test_gripper_release_and_disconnect_watchdog():
    control, teleop, messages, now = mailbox()
    control.transition("recording")
    command(control, "gripper", key="close", pressed=True)
    assert teleop.keys["close"] is True
    command(control, "gripper", key="close", pressed=False, version=-1)
    assert teleop.keys["close"] is False
    command(control, "gripper", key="open", pressed=True)
    now[0] = 2.01
    control.watchdog()
    assert control.events["pause_recording"] and control.events["exit_early"]
    assert not any(teleop.keys.values())
    assert not control.events["rerecord_episode"]
    assert not control.events["stop_recording"]
    control.transition("paused", has_unsaved=True)
    command(control, "start")
    assert messages[-1]["ok"] is False
    command(control, "save")
    assert control.actions.get_nowait() == "save"


@pytest.mark.parametrize("phase", ["ready", "saving", "resetting", "paused"])
def test_motion_commands_disabled_outside_recording(phase):
    control, teleop, messages, now = mailbox()
    control.transition(phase)
    command(control, "gripper", key="close", pressed=True)
    assert not teleop.keys
    assert messages[-1]["ok"] is False
    command(control, "joint_mode")
    assert teleop.switches == 0


def test_gello_gripper_source_disables_web_gripper():
    control, teleop, messages, now = mailbox()
    control.state["gripper_mode"] = "gello"
    control.transition("recording")
    command(control, "gripper", key="close", pressed=True)
    assert messages[-1]["ok"] is False


def test_physical_pause_executes_on_realtime_io_owner():
    calls = []
    robot = SimpleNamespace(send_action=lambda action: action,
                            pause_motion=lambda: calls.append(threading.get_ident()))
    teleop = SimpleNamespace(config=SimpleNamespace(), get_action=lambda: {"J1.pos": 0.0})
    controller = RealtimeTeleopController(robot, teleop, lambda pair: pair[0], lambda pair: pair[0],
                                         100, {"J1.pos": 0.0})
    controller.start()
    controller.request_pause()
    controller.stop()
    assert calls == [controller._thread.ident]


def test_preview_only_encodes_subscribed_cameras(monkeypatch):
    import numpy as np
    encoded = []
    preview = RecordingWebPreview(WebPreviewConfig(), camera_types={"rgb": "intelrealsense", "photon": "photon"})
    preview.jpeg_sink = lambda name, jpeg, timestamp: encoded.append(name)
    preview.set_subscriptions(["rgb"])
    preview.start_encoder()
    try:
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        preview.publish({"rgb": frame, "photon": frame})
        deadline = time.monotonic() + 2
        while not encoded and time.monotonic() < deadline:
            time.sleep(0.01)
        assert encoded == ["rgb"]
        assert set(preview._latest_frames) == {"rgb", "photon"}
        preview.set_subscriptions([])
        preview.publish({"rgb": frame, "photon": frame})
        time.sleep(0.15)
        assert encoded == ["rgb"]
    finally:
        preview.stop()


def test_slow_state_receiver_cannot_block_recorder():
    entered = threading.Event()
    release = threading.Event()
    class Connection:
        def send(self, value):
            entered.set()
            release.wait(2)
        def poll(self, timeout):
            time.sleep(0.01)
            return False
    control = RecordingControl("backpressure", connection=Connection(), clock=lambda: 0)
    try:
        control.transition("ready")
        assert entered.wait(1)
        started = time.monotonic()
        for _ in range(1000):
            control.transition("recording")
        assert time.monotonic() - started < 1
        assert control._outgoing.qsize() <= 64
    finally:
        release.set()
        control.close()


def test_preview_does_not_retimestamp_frozen_frame(monkeypatch):
    import numpy as np
    from lerobot_robot_ufactory.utils.webapp import web_preview
    ready = threading.Event()
    packets = []
    preview = RecordingWebPreview(WebPreviewConfig())
    def sink(name, jpeg, timestamp):
        packets.append(timestamp)
        ready.set()
    preview.jpeg_sink = sink
    preview.set_subscriptions(["rgb"])
    monkeypatch.setattr(web_preview.time, "time", lambda: 1234.0)
    preview.start_encoder()
    try:
        preview.publish({"rgb": np.zeros((24, 32, 3), dtype=np.uint8)})
        assert ready.wait(2)
        # Cross the encoder's one-second condition timeout without new frames.
        time.sleep(1.15)
        assert packets == [1234.0]
    finally:
        preview.stop()


async def wait_phase(manager, phase, timeout=12):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if manager.state["phase"] == phase:
            return manager.state.copy()
        if manager.state["phase"] == "error":
            raise AssertionError(manager.state)
        await asyncio.sleep(0.03)
    raise AssertionError(f"Expected {phase}: {manager.state}")


def test_http_auth_and_configuration(project):
    async def scenario():
        async with TestClient(TestServer(create_app(project, simulate=True))) as client:
            manager = client.app[MANAGER_KEY]
            origin = str(client.make_url("")).rstrip("/")
            headers = {"Origin": origin, "X-Record-Token": manager.token}
            response = await client.get("/")
            assert response.status == 200
            assert "仅 J7" in await response.text()
            item = await (await client.get("/api/config?path=tasks/example/example.yaml")).json()
            response = await client.post("/api/validate", json={"text": item["text"]})
            assert response.status == 403
            response = await client.post("/api/validate", json={"text": item["text"]},
                                         headers={**headers, "Origin": "http://evil.example"})
            assert response.status == 403
            response = await client.post("/api/config", json={"path": item["path"], "text": item["text"],
                                                               "revision": "stale"}, headers=headers)
            assert response.status == 409
            response = await client.post("/api/validate", json={"text": item["text"]}, headers=headers)
            assert response.status == 200
    asyncio.run(scenario())


def test_simulated_session_preview_lease_pause_save_and_exit(project):
    async def scenario():
        async with TestClient(TestServer(create_app(project, simulate=True))) as client:
            manager = client.app[MANAGER_KEY]
            origin = str(client.make_url("")).rstrip("/")
            headers = {"Origin": origin, "X-Record-Token": manager.token}
            url = "/ws/control?token=" + manager.token
            ws = await client.ws_connect(url, headers={"Origin": origin})
            initial = await ws.receive_json()
            assert initial["controller"] is True
            observer = await client.ws_connect(url, headers={"Origin": origin})
            assert (await observer.receive_json())["controller"] is False
            stop = asyncio.Event()
            async def heartbeat():
                while not stop.is_set():
                    await ws.send_json({"action": "heartbeat"})
                    await asyncio.sleep(0.2)
            task = asyncio.create_task(heartbeat())
            item = manager.store.read("tasks/example/example.yaml")
            preflight = await (await client.post("/api/preflight", json={"path": item["path"],
                "revision": item["revision"]}, headers=headers)).json()
            launch = {"ticket": preflight["ticket"], "client_id": initial["client_id"],
                      "dataset_mode": "new", "gripper_mode": "keyboard", "j7_enabled": True,
                      "cameras": ["wrist_camera"]}
            response = await client.post("/api/start", json=launch, headers=headers)
            assert response.status == 200, await response.text()
            await wait_phase(manager, "ready")
            assert not (project / "datasets/test-web").exists()
            preview = await client.ws_connect("/ws/preview?token=" + manager.token, headers={"Origin": origin})
            await preview.send_json({"cameras": ["wrist_camera"]})
            frame = await preview.receive(timeout=5)
            size = struct.unpack("!I", frame.data[:4])[0]
            meta = json.loads(frame.data[4:4 + size])
            assert meta["camera"] == "wrist_camera"
            assert frame.data[4 + size:].startswith(b"\xff\xd8")
            assert not any(name.startswith("photon") for name in manager.images)

            async def act(action, **extra):
                await ws.send_json({"action": action, "session_id": manager.state["session_id"],
                    "version": manager.state["version"], "request_id": str(time.time_ns()), **extra})
            await act("start")
            await wait_phase(manager, "recording")
            await act("joint_mode")
            deadline = time.monotonic() + 3
            while manager.state["joint_mode"] != "j7" and time.monotonic() < deadline:
                await asyncio.sleep(0.03)
            assert manager.state["joint_mode"] == "j7"
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await ws.close()
            await wait_phase(manager, "paused")
            assert manager.state["saved"] == 0
            assert manager.state["has_unsaved"]
            await observer.send_json({"type": "claim"})
            await asyncio.sleep(0.1)
            assert manager.owner is not None
            await observer.send_json({"action": "save", "session_id": manager.state["session_id"],
                "version": manager.state["version"], "request_id": str(time.time_ns())})
            await wait_phase(manager, "ready")
            assert manager.state["saved"] == 1
            assert manager.state["joint_mode"] == "all"
            await observer.send_json({"action": "exit", "session_id": manager.state["session_id"],
                "version": manager.state["version"], "request_id": str(time.time_ns())})
            await wait_phase(manager, "finished")
            assert (manager.folder / "simulated_episode_000001.json").exists()
            await preview.close()
            await observer.close()
    asyncio.run(scenario())
