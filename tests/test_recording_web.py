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
from lerobot_robot_ufactory.utils.webapp.recording_web_config import dataset_stamp, dataset_status


@pytest.mark.parametrize("deferred", [True, False])
def test_raw_only_dataset_counts_committed_episodes_and_can_resume(project, deferred):
    root = project / "datasets/raw-only"
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text(json.dumps({"total_episodes": 0, "fps": 15}))
    for index in range(2):
        episode = root / "raw_episodes" / f"episode_{index:06d}"
        episode.mkdir(parents=True)
        (episode / "manifest.json").write_text(json.dumps(
            {"version": 1, "episode_index": index, "size": 3, "fps": 15}))
        (episode / "frames.parquet").touch()
    raw = {"dataset": {"root": str(root), "fps": 15}, "defer_processing": deferred}
    status = dataset_status(project, raw)
    assert status["episodes"] == 2
    assert status["resumable"] is deferred
    cfg = SimpleNamespace(dataset=SimpleNamespace(root=root))
    options = {"dataset_mode": "resume", "dataset_stamp": dataset_stamp(root)}
    if deferred:
        assert status["reason"] is None
        prepare_web_dataset(project, cfg, raw, options)
    else:
        assert "--postprocess-only" in status["reason"]
        with pytest.raises(RuntimeError, match="--postprocess-only"):
            prepare_web_dataset(project, cfg, raw, options)
    assert len(list((root / "raw_episodes").glob("*/manifest.json"))) == 2


@pytest.mark.parametrize("problem", ["gap", "missing_frames", "fps", "published_missing", "no_raw"])
def test_raw_dataset_resume_rejects_incomplete_or_incompatible_data(project, problem):
    root = project / "datasets/raw-only"
    (root / "meta").mkdir(parents=True)
    processed = 1 if problem == "published_missing" else 0
    (root / "meta/info.json").write_text(json.dumps(
        {"total_episodes": processed, "fps": 15}))
    if problem != "no_raw":
        index = 1 if problem in ("gap", "published_missing") else 0
        episode = root / "raw_episodes" / f"episode_{index:06d}"
        episode.mkdir(parents=True)
        (episode / "manifest.json").write_text(json.dumps(
            {"version": 1, "episode_index": index, "size": 3,
             "fps": 60 if problem == "fps" else 15}))
        if problem != "missing_frames":
            (episode / "frames.parquet").touch()
    status = dataset_status(project, {"dataset": {"root": str(root), "fps": 15},
                                      "defer_processing": True})
    assert status["resumable"] is False
    assert status["reason"]


def test_postprocessing_blocks_recording_commands_and_survives_disconnect():
    control, teleop, messages, now = mailbox()
    progress = {"stage": "mesh", "total_episodes": 2, "completed_episodes": 0,
                "episode_index": 0, "streams": {}, "elapsed_s": 1}
    control.postprocess_progress(progress)
    assert control.state["phase"] == "postprocessing"
    for action in ("start", "save", "discard", "exit", "joint_mode"):
        command(control, action)
        assert messages[-1]["ok"] is False
    control.disconnect()
    assert not any(control.events.values())
    assert control.state["postprocess"] == progress
    control.postprocess_progress({**progress, "stage": "complete", "completed_episodes": 2})
    assert messages[-1]["state"]["postprocess"]["completed_episodes"] == 2


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
def test_dataset_selection_and_overwrite_rebuild(project, mode):
    root = project / "datasets/existing"
    root.mkdir(parents=True)
    (root / "sentinel.txt").write_text("previous data")
    sibling = root.parent / "other-dataset"
    sibling.mkdir()
    (sibling / "sentinel.txt").write_text("other data")
    cfg = SimpleNamespace(dataset=SimpleNamespace(root=root))
    raw = {"dataset": {"root": str(root)}}
    options = {"dataset_mode": mode, "dataset_stamp": dataset_stamp(root)}
    if mode == "rebuild":
        prepare_web_dataset(project, cfg, raw, options)
        assert not root.exists()
        assert not (root.parent / ".web-dataset-trash").exists()
    else:
        with pytest.raises(RuntimeError):
            prepare_web_dataset(project, cfg, raw, options)
        assert (root / "sentinel.txt").read_text() == "previous data"
    assert (sibling / "sentinel.txt").read_text() == "other data"


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
                      "dataset_mode": "new", "gripper_mode": "keyboard", "j7_enabled": True, "j7_keyboard_enabled": True,
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
            assert manager.state["j7_keyboard_enabled"]
            assert manager.effective["config"]["teleop"]["joint7_keyboard_mode_enabled"]
            await act("joint7_plus")
            deadline = time.monotonic() + 3
            while manager.state["joint_mode"] != "j7_keyboard" and time.monotonic() < deadline:
                await asyncio.sleep(0.03)
            assert manager.state["joint_mode"] == "j7_keyboard"
            assert manager.state["j7_keyboard_busy"]
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
            # One save command ends capture and saves the episode.
            async def owner_action(action):
                await observer.send_json({"action": action,
                    "session_id": manager.state["session_id"],
                    "version": manager.state["version"], "request_id": str(time.time_ns())})
            await owner_action("start")
            await wait_phase(manager, "recording")
            await owner_action("save")
            await wait_phase(manager, "ready")
            assert manager.state["saved"] == 2
            assert not manager.state["has_unsaved"]
            assert (manager.folder / "simulated_episode_000002.json").exists()
            await owner_action("start")
            await wait_phase(manager, "recording")
            await owner_action("discard")
            await wait_phase(manager, "ready")
            assert manager.state["saved"] == 2
            assert not (manager.folder / "simulated_episode_000003.json").exists()
            await observer.send_json({"action": "exit", "session_id": manager.state["session_id"],
                "version": manager.state["version"], "request_id": str(time.time_ns())})
            await wait_phase(manager, "finished")
            assert (manager.folder / "simulated_episode_000001.json").exists()
            await preview.close()
            await observer.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("action,phase", [("save", "saving"), ("discard", "resetting")])
def test_capture_command_selects_save_or_discard_immediately(action, phase):
    control, teleop, messages, now = mailbox()
    control.transition("recording", has_unsaved=True)
    command(control, action)
    assert messages[-1]["ok"] is True
    assert control.state["phase"] == phase
    assert control.events["exit_early"]
    assert control.events["rerecord_episode"] is (action == "discard")
    assert control.actions.empty()


def test_stage_change_bypasses_progress_throttle():
    control, teleop, messages, now = mailbox()
    now[0] = 1.0
    control.progress(frames=2)
    control.transition("saving", stage="Validating images")
    now[0] += 0.01
    control.progress(stage="Computing Mesh3DFlow")
    assert messages[-1]["state"]["stage"] == "Computing Mesh3DFlow"


def test_standalone_postprocess_web_job_uses_selected_dataset(project, tmp_path, monkeypatch):
    """Exercise the actual spawned offline worker and HTTP detection, without cameras."""
    import numpy as np
    from lerobot_robot_ufactory.datasets.native_dataset import NativeRateLeRobotDataset
    from lerobot_robot_ufactory.datasets.raw_episodes import RawEpisodeStore
    from lerobot_robot_ufactory.scripts import uf_lerobot_record as recording

    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "hf-cache"))
    import datasets.config
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "hf-cache")
    store = ConfigStore(project)
    item = store.read("tasks/example/example.yaml")
    raw = yaml.safe_load(item["text"])
    raw["offline_mesh3dflow"] = False
    raw["robot"]["cameras"] = {}
    raw["robot"]["record_space"] = "joint"
    raw["robot"]["calibration_dir"] = str(tmp_path / "calibration")
    item = store.save(item["path"], yaml.safe_dump(raw), item["revision"])
    _, cfg = validate_text(item["text"])
    robot = recording.make_robot_from_config(cfg.robot)
    processors = recording.make_default_processors()
    features = recording.combine_feature_dicts(
        recording.aggregate_pipeline_dataset_features(
            pipeline=processors[0], initial_features=recording.create_initial_features(action=robot.action_features),
            use_videos=cfg.dataset.video),
        recording.aggregate_pipeline_dataset_features(
            pipeline=processors[2], initial_features=recording.create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video),
    )
    dataset = NativeRateLeRobotDataset.create(
        cfg.dataset.repo_id, fps=cfg.dataset.fps, root=project / cfg.dataset.root,
        features=features, robot_type="xarm7", use_videos=cfg.dataset.video,
    )
    for index in range(3):
        dataset.add_frame({**{key: np.full(feature["shape"], index, dtype=feature["dtype"])
                              for key, feature in features.items()}, "task": "offline web test"})
    RawEpisodeStore(dataset).save(dataset.episode_buffer)
    dataset.finalize()

    async def scenario():
        async with TestClient(TestServer(create_app(project))) as client:
            manager = client.app[MANAGER_KEY]
            origin = str(client.make_url("")).rstrip("/")
            headers = {"Origin": origin, "X-Record-Token": manager.token}
            ws = await client.ws_connect("/ws/control?token=" + manager.token, headers={"Origin": origin})
            owner = (await ws.receive_json())["client_id"]
            status_url = "/api/postprocess-status?path=" + item["path"]
            status = await (await client.get(status_url)).json()
            assert status["revision"] == item["revision"]
            assert status["ready"] and status["pending_episodes"] == 1 and status["pending_frames"] == 3
            response = await client.post("/api/postprocess", json={"path": item["path"],
                "revision": "stale", "client_id": owner}, headers=headers)
            assert response.status == 409
            response = await client.post("/api/postprocess", json={"path": item["path"],
                "revision": item["revision"], "client_id": "observer"}, headers=headers)
            assert response.status == 409
            response = await client.post("/api/postprocess", json={"path": item["path"],
                "revision": item["revision"], "client_id": owner}, headers=headers)
            assert response.status == 200, await response.text()
            assert (await response.json())["state"]["operation"] == "postprocess"
            assert not manager.images
            state = await wait_phase(manager, "finished", timeout=30)
            assert state["saved"] == 1
            assert state["postprocess"]["stage"] == "complete"
            assert state["postprocess"]["completed_episodes"] == 1
            assert not manager.images
            await asyncio.to_thread(manager.process.join, 5)
            assert manager.process.exitcode == 0
            status = await (await client.get(status_url)).json()
            assert status["processed_episodes"] == 1
            assert status["pending_episodes"] == status["pending_frames"] == 0
            assert not status["ready"]
            response = await client.post("/api/postprocess", json={"path": item["path"],
                "revision": item["revision"], "client_id": owner}, headers=headers)
            assert response.status == 400
            empty = yaml.safe_load(item["text"])
            empty["dataset"]["root"] = "datasets/empty"
            store.save("tasks/example/empty.yaml", yaml.safe_dump(empty), None)
            status = await (await client.get("/api/postprocess-status?path=tasks/example/empty.yaml")).json()
            assert status["pending_episodes"] == 0 and not status["ready"]
            await ws.close()
    asyncio.run(scenario())


def test_xy_plane_web_mode_queues_toggle_and_reports_applied_state():
    from lerobot_robot_ufactory.teleoperators.gello_teleop.xy_plane_mode import XYPlaneMode

    control, teleop, messages, now = mailbox()
    teleop.config.xy_plane_only_mode_enabled = True
    teleop._xy_plane_mode = XYPlaneMode()
    teleop.set_xy_plane_mode_key = lambda pressed: teleop._xy_plane_mode.key(pressed, True)
    control.bind(teleop)
    assert control.state['xy_plane_enabled']
    command(control, 'xy_plane_mode')
    assert not messages[-1]['ok']
    assert not teleop._xy_plane_mode.pending
    control.transition('recording')
    command(control, 'xy_plane_mode', version=-1)
    assert not teleop._xy_plane_mode.pending
    command(control, 'xy_plane_mode')
    assert messages[-1]['ok']
    assert teleop._xy_plane_mode.pending
    # Browser command queues the toggle; status follows the control owner's result.
    assert control.state['joint_mode'] == 'all'
    teleop._xy_plane_mode.active = True
    now[0] += 1
    control.progress()
    assert control.state['joint_mode'] == 'xy'
    teleop._xy_plane_mode.reset()
    now[0] += 1
    control.progress()
    assert control.state['joint_mode'] == 'all'
    control.state['xy_plane_enabled'] = False
    command(control, 'xy_plane_mode')
    assert not messages[-1]['ok']
    assert not teleop._xy_plane_mode.pending


def test_precision_web_mode_queues_toggle_and_reports_applied_state():
    control, teleop, messages, now = mailbox()
    teleop.config.joint_precision_mode_enabled = True
    teleop.config.joint_precision_scale = 0.2
    teleop._joint_precision_active = False
    presses = []
    teleop.set_joint_precision_mode_key = lambda pressed: presses.append(pressed)
    control.bind(teleop)
    assert control.state['precision_enabled']
    assert control.state['precision_scale'] == 0.2
    command(control, 'precision_mode')
    assert not messages[-1]['ok']
    assert not presses
    control.transition('recording')
    command(control, 'precision_mode', version=-1)
    assert not presses
    command(control, 'precision_mode')
    assert messages[-1]['ok']
    assert presses == [True, False]
    # Browser command queues the toggle; status follows the control owner's result.
    assert control.state['precision_active'] is False
    teleop._joint_precision_active = True
    now[0] += 1
    control.progress()
    assert control.state['precision_active'] is True
    teleop._joint_precision_active = False
    now[0] += 1
    control.progress()
    assert control.state['precision_active'] is False
    control.state['precision_enabled'] = False
    command(control, 'precision_mode')
    assert not messages[-1]['ok']
    assert presses == [True, False]


@pytest.mark.parametrize('close_first', [False, True])
def test_refresh_recovers_control_without_old_socket_releasing_new_owner(project, close_first):
    async def scenario():
        async with TestClient(TestServer(create_app(project, simulate=True))) as client:
            manager = client.server.app[MANAGER_KEY]
            origin = str(client.make_url('')).rstrip('/')
            headers = {'Origin': origin}
            url = '/ws/control?token=' + manager.token
            commands = []
            manager.send = commands.append
            old = await client.ws_connect(url, headers=headers)
            initial = await old.receive_json()
            assert initial['controller'] and initial['resume_token']
            observer = await client.ws_connect(url, headers=headers)
            observed = await observer.receive_json()
            assert not observed['controller'] and observed['resume_token'] is None
            assert manager.snapshot()['resume_token'] is None
            manager.state['phase'] = 'recording'
            if close_first:
                await old.close()
                for _ in range(100):
                    if manager.owner is None:
                        break
                    await asyncio.sleep(.01)
                assert manager.owner is None
            refreshed = await client.ws_connect(url + '&resume=' + initial['resume_token'], headers=headers)
            resumed = await refreshed.receive_json()
            assert resumed['controller']
            assert resumed['client_id'] != initial['client_id']
            assert resumed['resume_token'] != initial['resume_token']
            assert commands == [{'action': 'disconnect'}]
            # Simulate cleanup arriving after the replacement has claimed control.
            await old.close()
            await asyncio.sleep(.05)
            assert manager.owner == resumed['client_id']
            assert commands == [{'action': 'disconnect'}]
            stale = await client.ws_connect(url + '&resume=' + initial['resume_token'], headers=headers)
            stale_state = await stale.receive_json()
            assert not stale_state['controller'] and stale_state['resume_token'] is None
            await refreshed.send_json({'action': 'heartbeat'})
            for _ in range(100):
                if commands[-1] == {'action': 'heartbeat'}:
                    break
                await asyncio.sleep(.01)
            assert commands[-1] == {'action': 'heartbeat'}
            assert manager.owner == resumed['client_id']
            await stale.close()
            await observer.close()
            await refreshed.close()
    asyncio.run(scenario())


def test_explicit_takeover_releases_old_owner_and_rotates_credential(project):
    async def scenario():
        async with TestClient(TestServer(create_app(project, simulate=True))) as client:
            manager = client.server.app[MANAGER_KEY]
            messages = []
            manager.send = messages.append
            origin = str(client.make_url('')).rstrip('/')
            headers = {'Origin': origin}
            bootstrap = await client.get('/api/bootstrap')
            assert bootstrap.headers['Cache-Control'] == 'no-store'
            assert (await bootstrap.json())['token'] == manager.token
            url = '/ws/control?token=' + manager.token
            old = await client.ws_connect(url, headers=headers)
            old_state = await old.receive_json()
            new = await client.ws_connect(url, headers=headers)
            assert not (await new.receive_json())['controller']
            await new.send_json({'type': 'claim'})
            assert not (await new.receive_json())['controller']
            manager.state['phase'] = 'recording'
            await new.send_json({'type': 'claim', 'takeover': True})
            received = await new.receive_json()
            assert received['controller']
            assert received['resume_token'] != old_state['resume_token']
            assert messages == [{'action': 'disconnect'}]
            await old.send_json({'action': 'start'})
            while True:
                rejection = await old.receive_json()
                if rejection['type'] == 'ack':
                    break
            assert not rejection['ok']
            await old.close()
            await asyncio.sleep(.05)
            assert manager.owner == received['client_id']
            assert messages == [{'action': 'disconnect'}]
            await new.close()
    asyncio.run(scenario())


def test_keyboard_j7_mailbox_gating_status_busy_and_release():
    control, teleop, messages, now = mailbox()
    presses = []
    teleop.config.joint7_keyboard_mode_enabled = True
    teleop._keyboard_j7_active = False
    teleop.keyboard_joint7_busy = lambda: teleop._keyboard_j7_active
    def rotate(direction, pressed):
        presses.append((direction, pressed))
        if pressed:
            if teleop._keyboard_j7_active:
                return False
            teleop._keyboard_j7_active = True
            return True
        return False
    teleop.set_joint7_keyboard_key = rotate
    control.bind(teleop)
    command(control, 'joint7_plus')
    assert messages[-1]['ok'] is False  # ready state cannot move
    control.transition('recording')
    command(control, 'joint7_plus', session_id='old-session')
    command(control, 'joint7_plus', version=-1)
    assert not presses
    command(control, 'joint7_plus')
    assert messages[-1]['ok']
    assert presses == [(1, True), (1, False)]
    control.progress()
    assert control.state['joint_mode'] == 'j7_keyboard'
    assert control.state['j7_keyboard_busy']
    command(control, 'joint7_minus')
    assert messages[-1]['ok'] is False
    control.release_keys()
    assert presses[-2:] == [(1, False), (-1, False)]
    control.transition('paused')
    command(control, 'joint7_minus')
    assert messages[-1]['ok'] is False
    control.state['j7_keyboard_enabled'] = False
    control.transition('recording')
    command(control, 'joint7_plus')
    assert messages[-1]['ok'] is False


def test_keyboard_j7_hold_web_edges_and_stale_release():
    control, teleop, messages, now = mailbox()
    presses = []
    teleop.config.joint7_keyboard_mode_enabled = True
    teleop.config.joint7_keyboard_hold_to_rotate = True
    teleop.set_joint7_keyboard_key = lambda direction, pressed: presses.append((direction, pressed)) or pressed
    control.bind(teleop)
    control.transition('recording')
    command(control, 'joint7_plus')
    assert not messages[-1]['ok']
    assert not presses
    command(control, 'joint7_plus', pressed=True)
    assert messages[-1]['ok']
    assert presses == [(1, True)]
    command(control, 'joint7_plus', pressed=False, version=-1)
    assert messages[-1]['ok']
    assert presses[-1] == (1, False)
    control.release_keys()
    assert presses[-2:] == [(1, False), (-1, False)]


@pytest.mark.parametrize('modifier', ['ctrl', 'shift'])
def test_keyboard_j7_web_forwards_modifier(modifier):
    control, teleop, messages, now = mailbox()
    presses = []
    teleop.config.joint7_keyboard_mode_enabled = True
    teleop.config.joint7_keyboard_angle_deg = 150
    teleop.set_joint7_keyboard_key = lambda direction, pressed, **kwargs: presses.append((direction, pressed, kwargs)) or pressed
    control.bind(teleop)
    assert control.state['j7_keyboard_angle_deg'] == 150
    control.transition('recording')
    command(control, 'joint7_minus', modifier=modifier)
    assert messages[-1]['ok']
    assert presses == [(-1, True, {'modifier': modifier}), (-1, False, {})]
    command(control, 'joint7_plus', modifier='invalid')
    assert not messages[-1]['ok']
    assert len(presses) == 2
