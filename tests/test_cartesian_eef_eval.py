"""Cartesian EEF control checks with fake feedback and SDK calls only."""
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from lerobot_robot_ufactory.robots.uf_robot.uf_robot import TimestampedStateSample, UFRobot
from lerobot_robot_ufactory.robots.uf_robot.uf_robot_config import UFRobotConfig


@pytest.fixture
def robot(tmp_path):
    cfg = UFRobotConfig(robot_dof=7, control_space="cartesian", min_tcp_z_mm=-2,
                        calibration_dir=tmp_path, max_linear_velocity=100)
    value = UFRobot(cfg)
    value._is_connected = True
    value._rt_report_normal = True
    value.real_arm = SimpleNamespace(
        error_code=0, state=0, mode=7, set_position=Mock(return_value=0),
        set_position_aa=Mock(return_value=0),
    )
    value._send_gripper_action = Mock()
    return value


def command():
    return dict(zip(("pose.x", "pose.y", "pose.z", "pose.rx", "pose.ry", "pose.rz"),
                    (300., -200., 100., .4, -.7, 2.1)),
                _pose_is_rpy=True, **{"gripper.pos": .75})


def test_cartesian_target_uses_absolute_base_frame_mm_and_rpy_radians(robot):
    sent = robot.send_action(command())
    robot.real_arm.set_position.assert_called_once_with(
        x=300., y=-200., z=100., roll=.4, pitch=-.7, yaw=2.1,
        speed=100, is_radian=True, relative=False, wait=False,
    )
    robot.real_arm.set_position_aa.assert_not_called()
    robot._send_gripper_action.assert_called_once_with(.75)
    assert sent == command()


@pytest.mark.parametrize("failure", ["floor", "nonfinite", "mode", "stopped", "sdk"])
def test_invalid_or_rejected_cartesian_target_stops_before_gripper(robot, failure):
    action = command()
    if failure == "floor": action["pose.z"] = -3
    elif failure == "nonfinite": action["pose.rx"] = np.nan
    elif failure == "mode": robot.real_arm.mode = 6
    elif failure == "stopped": robot.real_arm.state = 4
    else: robot.real_arm.set_position.return_value = 23
    with pytest.raises((ValueError, RuntimeError)):
        robot.send_action(action)
    assert robot.real_arm.set_position.call_count == (1 if failure == "sdk" else 0)
    robot._send_gripper_action.assert_not_called()


def test_cartesian_dry_run_sends_no_sdk_or_gripper_commands(robot):
    robot.config.no_action = True
    assert robot.send_action(command()) == command()
    robot.real_arm.set_position.assert_not_called()
    robot._send_gripper_action.assert_not_called()


def test_existing_axis_angle_commands_keep_their_sdk_format(robot):
    action = command()
    action.pop("_pose_is_rpy")
    robot.send_action(action)
    robot.real_arm.set_position.assert_not_called()
    robot.real_arm.set_position_aa.assert_called_once_with(
        axis_angle_pose=[300., -200., 100., .4, -.7, 2.1], speed=100,
        is_radian=True, is_tool_coord=False, relative=False, wait=False,
    )


def test_cartesian_mode_can_read_cached_joints_for_training_fk(robot):
    robot._gripper_type = 0
    robot.get_observation = Mock(side_effect=AssertionError("Must use cached joint feedback"))
    robot._read_synchronized_cameras = lambda anchor: {}
    robot._rt_state_history.append(TimestampedStateSample(
        capture_monotonic_s=perf_counter(), joint_positions=tuple(np.arange(7) * .1),
        joint_velocities=(0.,) * 7, tcp_pose=(0.,) * 6, tcp_velocity=(0.,) * 6,
        gripper_position=None,
    ))
    obs = robot.get_realtime_observation(joint_feedback=True)
    assert [obs[f"J{i + 1}.pos"] for i in range(7)] == pytest.approx(np.arange(7) * .1)
    robot.get_observation.assert_not_called()


def test_cartesian_reset_pose_is_checked_against_floor(robot):
    robot._local_kinematics = SimpleNamespace(tcp_position=lambda q: np.array([300, 0, -3]))
    robot.real_arm.get_joint_states = lambda **kw: (0, [[0.] * 7])
    with pytest.raises(RuntimeError, match="below the hard floor"):
        robot._initialize_tcp_z_guard()


@pytest.mark.parametrize("speed", [0, -1, np.nan, np.inf])
def test_cartesian_speed_must_be_finite_and_positive(speed):
    with pytest.raises(ValueError, match="max_linear_velocity"):
        UFRobotConfig(robot_dof=7, control_space="cartesian", min_tcp_z_mm=-2,
                      max_linear_velocity=speed)
