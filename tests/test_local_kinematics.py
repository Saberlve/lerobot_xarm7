from threading import Lock
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot_robot_ufactory.robots.uf_robot.local_kinematics import XArm7Kinematics
from lerobot_robot_ufactory.robots.uf_robot.uf_robot import UFRobot


NOMINAL_XARM7_ORIGINS = np.asarray(
    [
        [0.0, 0.0, 0.267, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -1.5708, 0.0, 0.0],
        [0.0, -0.293, 0.0, 1.5708, 0.0, 0.0],
        [0.0525, 0.0, 0.0, 1.5708, 0.0, 0.0],
        [0.0775, -0.3425, 0.0, 1.5708, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.5708, 0.0, 0.0],
        [0.076, 0.097, 0.0, -1.5708, 0.0, 0.0],
    ],
    dtype=np.float64,
)


def test_height_jacobian_matches_finite_difference():
    model = XArm7Kinematics(NOMINAL_XARM7_ORIGINS, tcp_offset=[0, 0, 80, 0, 0, 0])
    joints = np.asarray([0.2, -0.4, 0.3, 0.7, -0.2, 0.5, 0.4])

    _, analytic = model.tcp_z_and_jacobian(joints)
    numeric = np.empty(7)
    epsilon = 1e-6
    for index in range(7):
        plus = joints.copy()
        minus = joints.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        numeric[index] = (
            model.tcp_position(plus)[2] - model.tcp_position(minus)[2]
        ) / (2 * epsilon)

    assert analytic == pytest.approx(numeric, abs=1e-5)


def test_tcp_offset_is_applied_in_tool_frame():
    model = XArm7Kinematics(NOMINAL_XARM7_ORIGINS)
    offset_model = XArm7Kinematics(NOMINAL_XARM7_ORIGINS, tcp_offset=[0, 0, 100, 0, 0, 0])
    joints = np.asarray([0.2, -0.4, 0.3, 0.7, -0.2, 0.5, 0.4])

    base_transform = model.forward_matrix(joints)
    expected = base_transform[:3, 3] + base_transform[:3, 2] * 100.0

    assert offset_model.tcp_position(joints) == pytest.approx(expected)
    assert np.linalg.norm(offset_model.tcp_position(joints) - base_transform[:3, 3]) == pytest.approx(
        100.0
    )


class HeightModel:
    """Simple local model with z controlled by J1 and J7 in millimetres."""

    def tcp_position(self, joints):
        joints = np.asarray(joints)
        return np.asarray([0.0, 0.0, 100.0 + 100.0 * joints[0] + 20.0 * joints[6]])

    def tcp_z_and_jacobian(self, joints):
        return float(self.tcp_position(joints)[2]), np.asarray([100.0, 0, 0, 0, 0, 0, 20.0])


def make_local_guard_robot():
    robot = UFRobot.__new__(UFRobot)
    robot._dof = 7
    robot._tcp_z_guard_backend = "local_projection"
    robot._local_kinematics = HeightModel()
    robot._min_tcp_z_mm = 95.0
    robot._tcp_z_soft_floor_mm = 100.0
    robot._last_safe_joint_target = np.zeros(7)
    robot._last_guard_path = "not_run"
    robot._tcp_z_is_clamped = False
    robot._tcp_z_last_log_time = 0.0
    robot._tcp_z_last_error_log_time = 0.0
    robot.real_arm = SimpleNamespace()
    return robot


def test_local_guard_keeps_tangent_motion_and_projects_height():
    robot = make_local_guard_robot()
    desired = np.asarray([-0.1, 0.08, 0.0, 0.0, 0.0, 0.0, 0.1])

    result = robot._guard_joint_target(desired)

    assert robot._last_guard_path == "local_projected"
    assert robot._local_kinematics.tcp_position(result)[2] >= 100.0 - 1e-3
    assert result[1] == pytest.approx(0.08)
    assert result[6] != pytest.approx(0.0)


def test_local_guard_safe_path_does_not_touch_controller():
    class ControllerThatMustNotBeCalled:
        def __getattr__(self, name):
            raise AssertionError(f"unexpected controller call: {name}")

    robot = make_local_guard_robot()
    robot.real_arm = ControllerThatMustNotBeCalled()

    result = robot._guard_joint_target([0.05, 0, 0, 0, 0, 0, 0.1])

    assert result == pytest.approx([0.05, 0, 0, 0, 0, 0, 0.1])
    assert robot._last_guard_path == "local_safe"


def test_local_guard_holds_after_persistent_rt_model_mismatch():
    robot = make_local_guard_robot()
    robot._rt_report_normal = True
    robot._update_lock = Lock()
    robot.rt_actual_joint_pos = np.zeros(7)
    robot.rt_actual_tcp_pose = [0.0, 0.0, 110.0, 0.0, 0.0, 0.0]
    robot._local_model_fault_count = 0
    robot.config = SimpleNamespace(local_kinematics_max_error_mm=2.0)
    robot.logs = {}

    robot._guard_joint_target([0.05, 0, 0, 0, 0, 0, 0])
    robot._guard_joint_target([0.05, 0, 0, 0, 0, 0, 0])
    result = robot._guard_joint_target([0.05, 0, 0, 0, 0, 0, 0])

    assert result == pytest.approx([0.05, 0, 0, 0, 0, 0, 0])
    assert robot._last_guard_path == "model_fault"
    assert robot.logs["local_kinematics_error_mm"] == pytest.approx(10.0)


def test_local_projection_config_requires_joint_xarm7_and_floor():
    from lerobot_robot_ufactory.robots.uf_robot.uf_robot_config import UFRobotConfig

    with pytest.raises(ValueError, match="joint control on an xArm7"):
        UFRobotConfig(
            robot_dof=6,
            control_space="joint",
            min_tcp_z_mm=50.0,
            tcp_z_guard_backend="local_projection",
        )
    with pytest.raises(ValueError, match="requires min_tcp_z_mm"):
        UFRobotConfig(robot_dof=7, tcp_z_guard_backend="local_projection")


def test_controller_boundary_is_configured_once_and_verified():
    class BoundaryArm:
        def __init__(self):
            self.calls = []

        def set_reduced_tcp_boundary(self, boundary):
            self.calls.append(("boundary", boundary))
            return 0

        def set_fence_mode(self, enabled):
            self.calls.append(("fence", enabled))
            return [0]

        def get_reduced_states(self, **kwargs):
            states = [False, [9999, -9999, 9999, -9999, 9999, 50], 0, 0, [], True, False]
            return 0, states

    robot = UFRobot.__new__(UFRobot)
    robot.real_arm = BoundaryArm()
    robot._min_tcp_z_mm = 50.0

    robot._configure_controller_safety_boundary()

    assert robot.real_arm.calls == [
        ("boundary", [9999, -9999, 9999, -9999, 9999, 50]),
        ("fence", True),
    ]


def test_startup_validation_compares_controller_fk_to_flange_not_tcp():
    flange_model = XArm7Kinematics(NOMINAL_XARM7_ORIGINS)
    controller_model = XArm7Kinematics(
        NOMINAL_XARM7_ORIGINS,
        tcp_offset=[0.0, 0.0, 172.0, 0.0, 0.0, 0.0],
    )

    def matrix_to_rpy(rotation):
        pitch = np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0))
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        return np.asarray([roll, pitch, yaw])

    class FlangeFkArm:
        tcp_offset = [0.0, 0.0, 172.0, 0.0, 0.0, 0.0]
        world_offset = [0.0] * 6
        default_is_radian = True

        def get_joint_states(self, **kwargs):
            return 0, [np.zeros(7)]

        def get_forward_kinematics(self, joints, **kwargs):
            transform = controller_model.forward_matrix(joints)
            rpy = matrix_to_rpy(transform[:3, :3])
            return 0, [*transform[:3, 3], *rpy]

    robot = UFRobot.__new__(UFRobot)
    robot.real_arm = FlangeFkArm()
    robot._local_joint_origins = NOMINAL_XARM7_ORIGINS
    robot._min_tcp_z_mm = -100.0
    robot.config = SimpleNamespace(local_kinematics_max_error_mm=2.0)

    robot._initialize_local_kinematics()

    flange_position = flange_model.tcp_position(np.zeros(7))
    tcp_position = robot._local_kinematics.tcp_position(np.zeros(7))
    assert np.linalg.norm(tcp_position - flange_position) == pytest.approx(172.0)
