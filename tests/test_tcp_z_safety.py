from pathlib import Path
from types import SimpleNamespace
from threading import Lock

import numpy as np
import pytest

from lerobot_robot_ufactory.robots.uf_robot.uf_robot import UFRobot
from lerobot_robot_ufactory.robots.uf_robot.uf_robot_config import UFRobotConfig
from lerobot_robot_ufactory.scripts.uf_read_tcp_z import read_tcp_z


class FakeKinematicsArm:
    def __init__(self, requested_pose=None, safe_pose=None):
        self.requested_pose = requested_pose or [300.0, 10.0, 90.0, 0.1, 0.2, 0.3]
        self.safe_pose = safe_pose or [300.0, 10.0, 100.0, 0.1, 0.2, 0.3]
        self.inverse_result = [0.3] * 7
        self.inverse_calls = []
        self.fail_forward = False
        self.fail_inverse = False
        self.joint_limit = False

    def get_forward_kinematics(self, angles, **kwargs):
        if self.fail_forward:
            return 1, []
        if np.allclose(angles, self.inverse_result):
            return 0, self.safe_pose.copy()
        return 0, self.requested_pose.copy()

    def get_inverse_kinematics(self, pose, **kwargs):
        self.inverse_calls.append((pose.copy(), kwargs))
        if self.fail_inverse:
            return 2, []
        return 0, self.inverse_result.copy()

    def is_joint_limit(self, target, **kwargs):
        return 0, self.joint_limit


def make_guard_robot(arm, min_tcp_z_mm=100.0, control_space="joint"):
    robot = UFRobot.__new__(UFRobot)
    robot._dof = 7
    robot._control_space = control_space
    robot._min_tcp_z_mm = min_tcp_z_mm
    robot._last_safe_joint_target = np.asarray([0.25] * 7)
    robot._last_safe_cartesian_target = np.asarray([250.0, 0.0, 120.0, 0.0, 0.0, 0.0])
    robot._tcp_z_is_clamped = False
    robot._tcp_z_last_log_time = 0.0
    robot._tcp_z_last_error_log_time = 0.0
    robot.real_arm = arm
    return robot


def test_joint_target_above_floor_is_unchanged():
    arm = FakeKinematicsArm(requested_pose=[300.0, 10.0, 101.0, 0.1, 0.2, 0.3])
    robot = make_guard_robot(arm)
    requested = [0.1] * 7

    result = robot._guard_joint_target(requested)

    assert np.allclose(result, requested)
    assert arm.inverse_calls == []
    assert np.allclose(robot._last_safe_joint_target, requested)


def test_joint_guard_uses_rt_report_fast_path_far_above_floor():
    arm = FakeKinematicsArm()
    robot = make_guard_robot(arm)
    robot._rt_report_normal = True
    robot._update_lock = Lock()
    robot.rt_actual_tcp_pose = [0.0, 0.0, 200.0, 0.0, 0.0, 0.0]
    robot._tcp_z_guard_activation_margin_mm = 50.0

    result = robot._guard_joint_target([0.1] * 7)

    assert np.allclose(result, [0.1] * 7)
    assert arm.inverse_calls == []


def test_joint_target_below_floor_clamps_only_tcp_z_before_inverse_kinematics():
    arm = FakeKinematicsArm()
    robot = make_guard_robot(arm)
    requested = [0.1] * 7

    result = robot._guard_joint_target(requested)

    assert np.allclose(result, arm.inverse_result)
    inverse_pose, inverse_kwargs = arm.inverse_calls[0]
    assert inverse_pose == pytest.approx([300.0, 10.0, 100.0, 0.1, 0.2, 0.3])
    assert inverse_kwargs["limited"] is True
    assert inverse_kwargs["ref_angles"] == pytest.approx([0.25] * 7)
    assert np.allclose(robot._last_safe_joint_target, arm.inverse_result)


def test_successive_clamped_ik_uses_last_accepted_solution_as_reference():
    arm = FakeKinematicsArm()
    robot = make_guard_robot(arm)

    first_result = robot._guard_joint_target([0.1] * 7)
    arm.inverse_result = [0.32] * 7
    second_result = robot._guard_joint_target([0.05] * 7)

    assert first_result == pytest.approx([0.3] * 7)
    assert second_result == pytest.approx([0.32] * 7)
    assert arm.inverse_calls[1][1]["ref_angles"] == pytest.approx([0.3] * 7)


@pytest.mark.parametrize("failed_stage", ["forward", "inverse", "verification"])
def test_joint_guard_holds_last_safe_target_when_kinematics_fails(failed_stage):
    arm = FakeKinematicsArm()
    if failed_stage == "forward":
        arm.fail_forward = True
    elif failed_stage == "inverse":
        arm.fail_inverse = True
    else:
        arm.safe_pose[2] = 99.0
    robot = make_guard_robot(arm)

    result = robot._guard_joint_target([0.1] * 7)

    assert np.allclose(result, [0.25] * 7)


def test_joint_guard_holds_last_safe_target_when_ik_hits_joint_limit():
    arm = FakeKinematicsArm()
    arm.joint_limit = True
    robot = make_guard_robot(arm)

    result = robot._guard_joint_target([0.1] * 7)

    assert np.allclose(result, [0.25] * 7)


def test_joint_guard_rejects_discontinuous_ik_solution():
    arm = FakeKinematicsArm()
    arm.inverse_result = [1.5] * 7
    arm.safe_pose[2] = 100.0
    robot = make_guard_robot(arm)

    result = robot._guard_joint_target([0.1] * 7)

    assert np.allclose(result, [0.25] * 7)


def test_cartesian_guard_preserves_other_axes_and_clamps_z():
    robot = make_guard_robot(object(), control_space="cartesian")

    result = robot._guard_cartesian_target([300.0, 20.0, 90.0, 0.1, 0.2, 0.3])

    assert result == pytest.approx([300.0, 20.0, 100.0, 0.1, 0.2, 0.3])


def test_send_action_sends_and_returns_clamped_joint_target():
    arm = FakeKinematicsArm()
    arm.error_code = 0
    arm.mode = 1
    arm.sent_joint_targets = []
    arm.set_servo_angle_j = lambda target, **kwargs: arm.sent_joint_targets.append(target) or 0
    robot = make_guard_robot(arm)
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
        joint_command_mode=1,
        gripper_error_log_path=None,
    )
    action = {f"J{i + 1}.pos": 0.1 for i in range(7)}

    sent_action = robot.send_action(action)

    assert arm.sent_joint_targets == [pytest.approx(arm.inverse_result)]
    assert [sent_action[f"J{i + 1}.pos"] for i in range(7)] == pytest.approx(
        arm.inverse_result
    )
    assert [action[f"J{i + 1}.pos"] for i in range(7)] == pytest.approx([0.1] * 7)


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
