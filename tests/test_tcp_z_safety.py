from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot_robot_ufactory.robots.uf_robot.uf_robot import UFRobot
from lerobot_robot_ufactory.robots.uf_robot.uf_robot_config import UFRobotConfig
from lerobot_robot_ufactory.scripts.uf_read_tcp_z import read_tcp_z


class HeightModel:
    def tcp_position(self, joints):
        return np.asarray([0.0, 0.0, 100.0 + 100.0 * joints[0]])

    def tcp_z_and_jacobian(self, joints):
        return float(self.tcp_position(joints)[2]), np.asarray([100.0, 0, 0, 0, 0, 0, 0])


def make_guard_robot():
    robot = UFRobot.__new__(UFRobot)
    robot._dof = 7
    robot._control_space = "joint"
    robot._min_tcp_z_mm = 95.0
    robot._tcp_z_soft_floor_mm = 100.0
    robot._local_kinematics = HeightModel()
    robot._last_safe_joint_target = np.zeros(7)
    robot._tcp_z_is_clamped = False
    robot._tcp_z_last_log_time = 0.0
    robot._tcp_z_last_error_log_time = 0.0
    robot.real_arm = SimpleNamespace()
    return robot


@pytest.mark.parametrize("failure", ["missing_model", "invalid_target", "singular_jacobian"])
def test_joint_guard_holds_last_safe_target_on_failure(failure):
    robot = make_guard_robot()
    requested = [-0.1] * 7
    if failure == "missing_model":
        robot._local_kinematics = None
    elif failure == "invalid_target":
        requested[0] = float("nan")
    else:
        robot._local_kinematics.tcp_z_and_jacobian = lambda joints: (90.0, np.zeros(7))

    result = robot._guard_joint_target(requested)

    assert result == pytest.approx(np.zeros(7))
    assert robot._last_guard_path == "local_hold"


def test_joint_guard_skips_motion_without_safe_fallback():
    robot = make_guard_robot()
    robot._last_safe_joint_target = None

    assert robot._guard_joint_target([-0.1] * 7) is None


def test_guard_initialization_rejects_current_tcp_below_floor():
    robot = make_guard_robot()
    robot.real_arm.get_joint_states = lambda **kwargs: (0, [[-0.1] * 7])

    with pytest.raises(RuntimeError, match="below the hard floor"):
        robot._initialize_tcp_z_guard()


def test_send_action_sends_and_returns_projected_joint_target():
    robot = make_guard_robot()
    arm = robot.real_arm
    arm.error_code = 0
    arm.mode = 6
    arm.sent_joint_targets = []
    arm.set_servo_angle = lambda **kwargs: arm.sent_joint_targets.append(kwargs["angle"]) or 0
    robot._is_connected = True
    robot._last_logged_controller_error = 0
    robot._cmd_cnt = 20
    robot._max_joint_velocity = 1.0
    robot._gripper_type = 0
    robot.prefix = ""
    robot.logs = {}
    robot.config = SimpleNamespace(
        manual_mode=False,
        no_action=False,
        joint_command_mode=6,
        gripper_error_log_path=None,
    )
    action = {f"J{i + 1}.pos": -0.1 for i in range(7)}

    sent_action = robot.send_action(action)

    sent_joints = [sent_action[f"J{i + 1}.pos"] for i in range(7)]
    assert arm.sent_joint_targets == [pytest.approx(sent_joints)]
    assert robot._local_kinematics.tcp_position(sent_joints)[2] >= 100.0 - 1e-3
    assert sent_joints[1:] == pytest.approx([-0.1] * 6)
    assert robot.logs["safety_guard_dt_s"] >= 0
    assert robot.logs["safety_guard_path"] == "local_projected"
    assert [action[f"J{i + 1}.pos"] for i in range(7)] == pytest.approx([-0.1] * 7)


def test_removed_controller_backend_is_rejected():
    with pytest.raises(ValueError, match="tcp_z_guard_backend must be 'local_projection'"):
        UFRobotConfig(robot_dof=7, min_tcp_z_mm=95.0, tcp_z_guard_backend="controller_rpc")


@pytest.mark.parametrize("state,error_code", [(4, 0), (5, 0), (0, 23)])
@pytest.mark.parametrize("manual_mode", [False, True])
def test_stopped_or_faulted_controller_rejects_actions(state, error_code, manual_mode):
    robot = make_guard_robot()
    robot._is_connected = True
    robot._last_logged_controller_error = 0
    robot.real_arm.state = state
    robot.real_arm.error_code = error_code
    robot.config = SimpleNamespace(manual_mode=manual_mode, gripper_error_log_path=None)
    calls = []
    robot.real_arm.set_state = lambda state: calls.append("state")
    robot.real_arm.set_servo_angle = lambda **kwargs: calls.append("move")
    robot._send_gripper_action = lambda target: calls.append("gripper")
    with pytest.raises(RuntimeError, match="controller stopped or faulted"):
        robot.send_action({})
    assert calls == [], "A stopped controller must not be re-enabled or sent motion"


def test_non_finite_tcp_floor_is_rejected():
    with pytest.raises(ValueError, match="min_tcp_z_mm"):
        UFRobotConfig(robot_dof=7, min_tcp_z_mm=float("nan"))


class FakeMeasurementArm:
    def __init__(self, robot_ip):
        self.robot_ip = robot_ip
        self.connected = True
        self.axis = 7
        self.disconnected = False
        self.forward_calls = []

    def get_joint_states(self, **kwargs):
        return 0, [[0.1] * 7]

    def get_forward_kinematics(self, joints, **kwargs):
        self.forward_calls.append((joints, kwargs))
        return 0, [300.0, 0.0, 87.25, 0.0, 0.0, 0.0]

    def disconnect(self):
        self.disconnected = True


def write_measurement_config(path: Path):
    path.write_text(
        "robot:\n  robot_ip: '192.168.1.245'\n  robot_dof: 7\n",
        encoding="utf-8",
    )


def test_read_tcp_z_adds_margin_and_disconnects(tmp_path):
    config_path = tmp_path / "gello.yaml"
    write_measurement_config(config_path)
    arms = []

    def arm_factory(robot_ip):
        arm = FakeMeasurementArm(robot_ip)
        arms.append(arm)
        return arm

    measured, recommended = read_tcp_z(config_path, margin_mm=5.0, arm_factory=arm_factory)

    assert measured == pytest.approx(87.25)
    assert recommended == pytest.approx(92.25)
    assert arms[0].robot_ip == "192.168.1.245"
    assert arms[0].disconnected is True
    assert arms[0].forward_calls[0][1] == {
        "input_is_radian": True,
        "return_is_radian": True,
    }


def test_read_tcp_z_disconnects_when_fk_fails(tmp_path):
    config_path = tmp_path / "gello.yaml"
    write_measurement_config(config_path)
    arm = FakeMeasurementArm("192.168.1.245")
    arm.get_forward_kinematics = lambda *args, **kwargs: (1, [])

    with pytest.raises(RuntimeError, match="get_forward_kinematics"):
        read_tcp_z(config_path, arm_factory=lambda _: arm)

    assert arm.disconnected is True
