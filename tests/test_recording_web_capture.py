"""Exercise the real capture/save/tactile path with fake hardware."""
from collections import deque

import pytest

from test_recording_timeout import session as _session_fixture
from lerobot_robot_ufactory.scripts import uf_lerobot_record as recording
from lerobot_robot_ufactory.teleoperators.gello_teleop.gello_teleop import GelloTeleop
from lerobot_robot_ufactory.teleoperators.gello_teleop.gello_teleop_config import GelloTeleopConfig
from lerobot_robot_ufactory.utils.webapp.recording_control import RecordingControl
from lerobot_robot_ufactory.utils.webapp.web_preview import RecordingWebPreview, WebPreviewConfig

session = _session_fixture  # Reuse image-writing/tactile fixture without hardware.


@pytest.mark.parametrize("decision", ["save", "discard", "exit"])
def test_disconnected_capture_retains_buffer_until_explicit_decision(session, monkeypatch, decision):
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
    teleop = GelloTeleop(GelloTeleopConfig(gripper_control_mode="keyboard"))
    teleop.connect = lambda: None
    teleop.disconnect = lambda: None
    teleop.set_teleop_enabled = lambda *args, **kwargs: None
    teleop.get_action = lambda: {"J1.pos": float(robot.attempt)}
    cfg.teleop = teleop.config
    robot.get_gripper_motion_parameters = lambda: (50.0, 84.0)
    monkeypatch.setattr(recording, "make_teleoperator_from_config", lambda cfg: teleop)
    ready_actions = deque(["start", "exit"])
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
            action = decision
        else:
            return
        control.command({"action": action, "session_id": "capture-test",
                         "version": control.state["version"], "request_id": str(len(messages))})

    control = RecordingControl("capture-test", send=send)
    control.prepare_dataset = lambda: None
    control.preview = RecordingWebPreview(WebPreviewConfig())
    original_loop = recording.record_loop
    def loop(**kwargs):
        dataset = session.dataset
        add = dataset.add_frame
        def add_frame(frame):
            add(frame)
            if dataset.episode_buffer["size"] == 2:
                control.disconnect()
        dataset.add_frame = add_frame
        return original_loop(**kwargs)
    monkeypatch.setattr(recording, "record_loop", loop)
    dataset = recording.record(cfg, recording_control=control)
    assert pause_sizes == [2]
    assert robot.pause_calls == 1
    assert dataset.num_episodes == (1 if decision == "save" else 0)
    assert dataset.finalized
    assert not robot._is_connected
    if decision == "save":
        assert dataset.saved[0]["size"] == 2
        assert (dataset.root / "timestamps/episode_000000.parquet").exists()
        assert (dataset.root / "tactile_streams/photon/episode_000000/samples.parquet").exists()
    else:
        assert not (dataset.root / "tactile_streams/photon/episode_000000/samples.parquet").exists()
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
