"""Exercise the real capture/save/tactile path with fake hardware."""
from collections import deque
from types import MethodType, SimpleNamespace

import pytest

from test_recording_timeout import session as _session_fixture
from lerobot_robot_ufactory.scripts import uf_lerobot_record as recording
from lerobot_robot_ufactory.robots.uf_robot.uf_robot import UFRobot
from lerobot_robot_ufactory.teleoperators.gello_teleop.gello_teleop import GelloTeleop
from lerobot_robot_ufactory.teleoperators.gello_teleop.gello_teleop_config import GelloTeleopConfig
from lerobot_robot_ufactory.utils.webapp.recording_control import RecordingControl
from lerobot_robot_ufactory.utils.webapp.web_preview import RecordingWebPreview, WebPreviewConfig

session = _session_fixture  # Reuse image-writing/tactile fixture without hardware.


@pytest.fixture(autouse=True)
def isolate_fake_device_lock(monkeypatch):
    # Fake-hardware tests must never contend with a live operator's recorder.
    from lerobot_robot_ufactory.utils import recording_lock
    monkeypatch.setattr(recording_lock, "_depth", 1)


@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("fault", ["emergency_stop", "controller_error", "unexpected",
                                   "connection", "pause_interrupt", "pause_cleanup", "interrupt", "exit"])
@pytest.mark.parametrize("fault_frame", [1, 2])
def test_capture_fault_discards_only_current_episode(session, monkeypatch, deferred, fault, fault_frame):
    robot = session.robot
    original_connect = robot.connect
    robot.connect = lambda *, defer_motion=False: original_connect()
    robot.outcomes = [False, False]
    robot.pause_motion = lambda: None
    if fault == "pause_cleanup":
        def fail_pause():
            raise OSError("pause cleanup failed")
        robot.pause_motion = fail_pause
    robot.real_arm = SimpleNamespace(state=0, error_code=0)
    robot._motion_status = lambda: f"state={robot.real_arm.state}, error_code={robot.real_arm.error_code}"
    robot.check_recording_health = MethodType(UFRobot.check_recording_health, robot)
    original_reset = robot.reset_to_initial
    robot.reset_to_initial = lambda cancel_check=None: original_reset()
    robot.get_gripper_motion_parameters = lambda: (50.0, 84.0)
    teleop = GelloTeleop(GelloTeleopConfig(gripper_control_mode="keyboard"))
    teleop.connect = lambda: None
    teleop.disconnect = lambda: None
    teleop.set_teleop_enabled = lambda *args, **kwargs: None
    teleop.get_action = lambda: {"J1.pos": float(robot.attempt)}
    session.cfg.robot.manual_mode = False
    session.cfg.teleop = teleop.config
    session.cfg.defer_processing = deferred
    monkeypatch.setattr(recording, "make_teleoperator_from_config", lambda cfg: teleop)
    control = RecordingControl("capture-fault-test")
    control.prepare_dataset = lambda: None
    control.preview = RecordingWebPreview(WebPreviewConfig())

    def send(message):
        if message["type"] == "state" and message["state"]["phase"] == "ready":
            control.command({"action": "start", "session_id": control.session_id,
                             "version": control.state["version"],
                             "request_id": str(control.state["version"])})

    control.send = send
    original_loop = recording.record_loop

    def loop(**kwargs):
        dataset = session.dataset
        original_add = dataset.add_frame

        def add_frame(frame):
            original_add(frame)
            if dataset.episode_buffer["episode_index"] == 1 and dataset.episode_buffer["size"] == fault_frame:
                if fault == "emergency_stop":
                    robot.real_arm.state = 4
                elif fault == "controller_error":
                    robot.real_arm.error_code = 23
                elif fault == "unexpected":
                    raise ValueError("unexpected capture failure")
                elif fault == "connection":
                    raise ConnectionError("capture connection lost")
                elif fault in ("pause_interrupt", "pause_cleanup"):
                    control.events["pause_recording"] = True
                    raise InterruptedError("capture interrupted while paused")
                elif fault == "interrupt":
                    raise KeyboardInterrupt("capture interrupted")
                else:
                    raise SystemExit("capture exited")

        dataset.add_frame = add_frame
        try:
            return original_loop(**kwargs)
        finally:
            dataset.add_frame = original_add

    monkeypatch.setattr(recording, "record_loop", loop)
    expected = {"emergency_stop": RuntimeError, "controller_error": RuntimeError,
                "unexpected": ValueError, "connection": ConnectionError,
                "pause_interrupt": InterruptedError, "pause_cleanup": OSError, "interrupt": KeyboardInterrupt,
                "exit": SystemExit}[fault]
    message = {"emergency_stop": "controller stopped or faulted",
               "controller_error": "controller stopped or faulted",
               "unexpected": "unexpected capture failure", "connection": "capture connection lost",
               "pause_interrupt": "capture interrupted while paused", "interrupt": "capture interrupted",
               "pause_cleanup": "pause cleanup failed",
               "exit": "capture exited"}[fault]
    try:
        with pytest.raises(expected, match=message):
            recording.record(session.cfg, recording_control=control)
        dataset = session.dataset
        assert dataset.episode_buffer["size"] == 0
        assert dataset.episode_buffer["episode_index"] == 1
        assert not dataset._get_image_file_dir(1, "observation.images.photon").exists()
        assert not (dataset.root / "timestamps/episode_000001.parquet").exists()
        assert not (dataset.root / "tactile_streams/photon/episode_000001").exists()
        staging = dataset.root / "tactile_streams/.staging"
        assert not staging.exists() or not any(staging.iterdir())
        assert control.state["has_unsaved"] is False
        assert control.realtime_controller is None
        assert dataset.finalized and not robot._is_connected
        assert robot.attempt == 1, "Capture failure must not automatically reset or retry motion"
        assert (dataset.root / "timestamps/episode_000000.parquet").is_file()
        assert (dataset.root / "tactile_streams/photon/episode_000000/samples.parquet").is_file()
        if deferred:
            manifests = recording.RawEpisodeStore.checkpoints(dataset.root)
            assert len(manifests) == 1
            buffer, manifest = recording.RawEpisodeStore(dataset).load(manifests[0])
            assert manifest["episode_index"] == 0 and buffer["size"] == 2
            assert dataset.num_episodes == 0
        else:
            assert dataset.num_episodes == 1 and dataset.saved[0]["size"] == 2
    finally:
        control.close()



@pytest.mark.parametrize("ending,decision", [
    ("disconnect", "save"), ("disconnect", "discard"),
    ("gello", "save"), ("gello", "discard"), ("duration", "save"),
    ("control_fault", "save"), ("control_fault", "discard"),
])
def test_capture_saves_or_discards_without_second_confirmation(session, monkeypatch, decision, ending):
    robot = session.robot
    original_connect = robot.connect
    def connect(*, defer_motion=False):
        assert defer_motion is True
        original_connect()
    robot.connect = connect
    robot.outcomes = [False] * 10
    robot.pause_calls = 0
    def pause_motion():
        robot.pause_calls += 1
    robot.pause_motion = pause_motion
    robot.open_gripper = lambda: None
    original_reset = robot.reset_to_initial
    robot.reset_to_initial = lambda cancel_check=None: original_reset()
    cfg = session.cfg
    cfg.robot.manual_mode = False
    cfg.offline_mesh3dflow = True
    cfg.dataset.video = True
    teleop = GelloTeleop(GelloTeleopConfig(gripper_control_mode="keyboard"))
    teleop.connect = lambda: None
    teleop.disconnect = lambda: None
    teleop.set_teleop_enabled = lambda *args, **kwargs: None
    teleop.get_action = lambda: {"J1.pos": float(robot.attempt)}
    cfg.teleop = teleop.config
    robot.get_gripper_motion_parameters = lambda: (50.0, 84.0)
    monkeypatch.setattr(recording, "make_teleoperator_from_config", lambda cfg: teleop)
    ready_actions = deque(["start", "start", "exit"] if ending == "control_fault" else ["start", "exit"])
    messages = []
    pause_sizes = []
    control = None

    def send(message):
        messages.append(message)
        if message["type"] != "state":
            return
        phase = message["state"]["phase"]
        if phase == "ready" and ready_actions:
            action = ready_actions.popleft()
        elif phase == "paused":
            buffer = recording._get_episode_buffer(session.dataset)
            pause_sizes.append(buffer["size"])
            assert buffer["size"] == 2
            assert session.dataset.num_episodes == 0
            assert not control.events["rerecord_episode"]
            assert not session.mesh_episodes
            action = decision
        else:
            return
        control.command({"action": action, "session_id": "capture-test",
                         "version": control.state["version"], "request_id": str(len(messages))})

    control = RecordingControl("capture-test", send=send)
    control.prepare_dataset = lambda: None
    control.preview = RecordingWebPreview(WebPreviewConfig())
    original_loop = recording.record_loop
    failed_once = False
    def loop(**kwargs):
        nonlocal failed_once
        dataset = session.dataset
        add = dataset.add_frame
        def add_frame(frame):
            nonlocal failed_once
            add(frame)
            if dataset.episode_buffer["size"] == 2:
                if ending == "control_fault" and not failed_once:
                    failed_once = True
                    raise RuntimeError("Realtime joint control thread failed")
                if ending == "disconnect":
                    control.disconnect()
                elif ending == "duration":
                    # Let record_loop return because its duration elapsed.
                    control.events["exit_early"] = False
                else:
                    control.command({"action": decision, "session_id": "capture-test",
                                     "version": control.state["version"],
                                     "request_id": f"end-{robot.attempt}"})
        dataset.add_frame = add_frame
        if ending == "duration":
            kwargs["control_time_s"] = 0.009
        return original_loop(**kwargs)
    monkeypatch.setattr(recording, "record_loop", loop)
    dataset = recording.record(cfg, recording_control=control)
    assert pause_sizes == ([2] if ending == "disconnect" else [])
    assert all(message["state"]["phase"] not in ("review", "finishing")
               for message in messages if message["type"] == "state")
    assert robot.pause_calls == (1 if ending == "disconnect" else 0)
    assert dataset.num_episodes == (1 if decision == "save" else 0)
    assert session.mesh_episodes == ([0] if decision == "save" else [])
    assert dataset.finalized
    assert not robot._is_connected
    if decision == "save":
        assert dataset.saved[0]["size"] == 2
        assert (dataset.root / "timestamps/episode_000000.parquet").exists()
        assert (dataset.root / "tactile_streams/photon/episode_000000/samples.parquet").exists()
    else:
        assert not (dataset.root / "tactile_streams/photon/episode_000000/samples.parquet").exists()
    control.close()


@pytest.mark.parametrize("exit_phase", ["empty", "ready", "recording", "paused"])
def test_exit_discards_current_episode_and_postprocesses_saved_raw(session, monkeypatch, exit_phase):
    robot = session.robot
    original_connect = robot.connect
    robot.connect = lambda *, defer_motion=False: original_connect()
    robot.outcomes = [False] * 5
    robot.pause_motion = lambda: None
    original_reset = robot.reset_to_initial
    robot.reset_to_initial = lambda cancel_check=None: original_reset()
    robot.get_gripper_motion_parameters = lambda: (50.0, 84.0)
    teleop = GelloTeleop(GelloTeleopConfig(gripper_control_mode="keyboard"))
    teleop.connect = lambda: None
    teleop.disconnect = lambda: None
    teleop.set_teleop_enabled = lambda *args, **kwargs: None
    teleop.get_action = lambda: {"J1.pos": float(robot.attempt)}
    session.cfg.robot.manual_mode = False
    session.cfg.teleop = teleop.config
    session.cfg.defer_processing = True
    session.cfg.dataset.num_episodes = 5
    monkeypatch.setattr(recording, "make_teleoperator_from_config", lambda cfg: teleop)
    control = RecordingControl("exit-postprocess-test")
    control.prepare_dataset = lambda: None
    control.preview = RecordingWebPreview(WebPreviewConfig())
    states, processed = [], []

    def command(action):
        control.command({"action": action, "session_id": control.session_id,
                         "version": control.state["version"], "request_id": str(len(states))})

    def send(message):
        if message["type"] != "state":
            return
        state = message["state"]
        states.append(state)
        if state["phase"] == "ready":
            command("exit" if exit_phase == "empty" or exit_phase == "ready" and state["saved"] else "start")
        elif state["phase"] == "paused":
            command("exit")

    control.send = send
    original_loop = recording.record_loop

    def loop(**kwargs):
        dataset = session.dataset
        original_add = dataset.add_frame

        def add_frame(frame):
            original_add(frame)
            if dataset.episode_buffer["episode_index"] == 1 and dataset.episode_buffer["size"] == 1:
                if exit_phase == "paused":
                    control.disconnect()
                else:
                    command("exit")

        dataset.add_frame = add_frame
        try:
            return original_loop(**kwargs)
        finally:
            dataset.add_frame = original_add

    original_postprocess = recording.postprocess_raw_episodes

    def postprocess(dataset, cameras, *, progress):
        assert not robot._is_connected, "Release devices before starting postprocessing"
        assert dataset.finalized
        assert dataset.episode_buffer["size"] == 0
        total = len(recording.RawEpisodeStore.checkpoints(dataset.root))
        processed.append(total)
        progress({"stage": "loading", "total_episodes": total, "completed_episodes": 0,
                  "episode_index": None, "streams": {}, "elapsed_s": 0})
        result = original_postprocess(dataset, cameras)
        progress({"stage": "complete", "total_episodes": total, "completed_episodes": total,
                  "episode_index": None, "streams": {}, "elapsed_s": 1})
        return result

    monkeypatch.setattr(recording, "record_loop", loop)
    monkeypatch.setattr(recording, "postprocess_raw_episodes", postprocess)
    try:
        dataset = recording.record(session.cfg, recording_control=control)
        expected = 0 if exit_phase == "empty" else 1
        assert processed == [expected]
        assert dataset.num_episodes == expected
        assert control.state["postprocess"]["stage"] == "complete"
        assert control.state["postprocess"]["completed_episodes"] == expected
        assert any(state["phase"] == "stopping" for state in states)
        assert any(state["phase"] == "postprocessing" for state in states)
        assert not (dataset.root / "raw_episodes/episode_000001").exists()
        assert not dataset._get_image_file_dir(1, "observation.images.photon").exists()
        if expected:
            assert dataset.saved[0]["episode_index"] == 0 and dataset.saved[0]["size"] == 2
    finally:
        control.close()


@pytest.mark.parametrize("cancelled", [False, True])
def test_web_reset_can_cancel_before_sending_motion(cancelled):
    from lerobot_robot_ufactory.robots.uf_robot.uf_robot import UFRobot
    calls = []
    class Arm:
        def motion_enable(self, **kwargs):
            calls.append("enable")
            return 0
        def clean_error(self):
            return 0
        def set_mode(self, mode):
            return 0
        def set_state(self, state):
            return 0
        def set_servo_angle(self, **kwargs):
            assert kwargs["wait"] is False
            calls.append("move")
            return 0
        def get_servo_angle(self, **kwargs):
            return 0, [0] * 7
    robot = UFRobot.__new__(UFRobot)
    robot._is_connected = True
    robot.real_arm = Arm()
    robot._initial_point = [0] * 7
    robot._dof = 7
    robot._min_tcp_z_mm = None
    robot._check_motion_code = lambda operation, code: None
    robot.configure = lambda: calls.append("configure")
    robot.pause_motion = lambda: calls.append("pause")
    if cancelled:
        with pytest.raises(InterruptedError):
            robot.reset_to_initial(cancel_check=lambda: True)
        assert calls == ["pause"]
    else:
        robot.reset_to_initial(cancel_check=lambda: False)
        assert calls == ["enable", "move", "configure"]


@pytest.mark.parametrize("missed_frames", [1, 3])
def test_idle_preview_timeout_retries_without_ending_session(session, monkeypatch, caplog, missed_frames):
    robot = session.robot
    original_connect = robot.connect
    robot.connect = lambda *, defer_motion=False: original_connect()
    teleop = GelloTeleop(GelloTeleopConfig(gripper_control_mode="keyboard"))
    teleop.connect = lambda: None
    teleop.disconnect = lambda: None
    teleop.set_teleop_enabled = lambda *args, **kwargs: None
    session.cfg.teleop = teleop.config
    robot.get_gripper_motion_parameters = lambda: (50.0, 84.0)
    monkeypatch.setattr(recording, "make_teleoperator_from_config", lambda cfg: teleop)
    control = RecordingControl("idle-preview-test")
    control.prepare_dataset = lambda: None
    preview = RecordingWebPreview(WebPreviewConfig())
    control.preview = preview
    attempts = 0
    original_observation = robot.get_observation

    def observation():
        nonlocal attempts
        attempts += 1
        assert robot._is_connected
        assert control.state["phase"] == "ready"
        if attempts <= missed_frames:
            raise TimeoutError("Latest causal RGB frame for wrist_camera is 79 ms old; limit is 70 ms")
        return original_observation()

    monkeypatch.setattr(robot, "get_observation", observation)
    original_publish = preview.publish
    published = []

    def publish(observation):
        original_publish(observation)
        published.append(observation)
        control.command({"action": "exit", "session_id": control.session_id,
                         "version": control.state["version"], "request_id": "exit-after-preview"})

    monkeypatch.setattr(preview, "publish", publish)
    try:
        dataset = recording.record(session.cfg, recording_control=control)
        assert attempts == missed_frames + 1
        assert len(published) == 1
        assert dataset.num_episodes == 0
        assert robot.attempt == -1  # Preview retry never resets or moves the arm.
        assert not robot._is_connected
        warnings = [record for record in caplog.records if "Skipping idle preview frame" in record.message]
        assert len(warnings) == 1
        assert "wrist_camera" in warnings[0].message
    finally:
        control.close()
