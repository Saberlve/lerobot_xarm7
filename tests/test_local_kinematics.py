from threading import Lock
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot_robot_ufactory.robots.uf_robot.local_kinematics import (
    XArm7Kinematics,
    axis_angle_continuous,
    rot6d_to_rotation,
    rotation_to_6d,
    rotation_to_axis_angle,
    xarm_rpy_transform,
)
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


def _axis_angle_rotation(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray(
        [
            [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
            [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
            [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
        ],
        dtype=np.float64,
    )


def test_axis_angle_continuous_avoids_pi_flip():
    rotation_a = _axis_angle_rotation([0.0, 0.0, 1.0], np.pi - 1e-3)
    rotation_b = _axis_angle_rotation([0.0, 0.0, 1.0], -(np.pi - 1e-3))

    aa_a = axis_angle_continuous(rotation_a, None)
    aa_b = axis_angle_continuous(rotation_b, aa_a)

    # The raw representatives point along opposite axes (~2*pi apart); the
    # continuous representative stays close to the previous frame instead.
    assert np.linalg.norm(rotation_to_axis_angle(rotation_b) - aa_a) > 5.0
    assert np.linalg.norm(aa_b - aa_a) == pytest.approx(0.0, abs=1e-2)


def test_tcp_record_space_config_validation():
    from lerobot_robot_ufactory.robots.uf_robot.uf_robot_config import UFRobotConfig

    UFRobotConfig(robot_dof=7, control_space="joint", record_space="tcp")

    with pytest.raises(ValueError, match="record_space must be"):
        UFRobotConfig(robot_dof=7, control_space="joint", record_space="bogus")
    with pytest.raises(ValueError, match="requires joint control on an xArm7"):
        UFRobotConfig(robot_dof=7, control_space="cartesian", record_space="tcp")
    with pytest.raises(ValueError, match="requires joint control on an xArm7"):
        UFRobotConfig(robot_dof=6, control_space="joint", record_space="tcp")
    with pytest.raises(ValueError, match="does not support observe_joint_vel"):
        UFRobotConfig(
            robot_dof=7, control_space="joint", record_space="tcp", observe_joint_vel=True
        )
    with pytest.raises(ValueError, match="not supported in manual_mode"):
        UFRobotConfig(robot_dof=7, control_space="joint", record_space="tcp", manual_mode=True)


def make_tcp_record_robot(record_space="tcp"):
    robot = UFRobot.__new__(UFRobot)
    robot.prefix = ""
    robot._dof = 7
    robot._control_space = "joint"
    robot._record_space = record_space
    robot._jnt_obs_has_vel = False
    robot._gripper_type = 2
    robot.cameras = {}
    robot.camera_width = 0
    robot.camera_height = 0
    robot._local_kinematics = XArm7Kinematics(NOMINAL_XARM7_ORIGINS)
    return robot


TCP_RECORD_ROT_KEYS = [f"pose.r{r}{c}" for c in (1, 2) for r in (1, 2, 3)]
TCP_RECORD_POSE_KEYS = ["pose.x", "pose.y", "pose.z", *TCP_RECORD_ROT_KEYS]


def test_tcp_record_space_features():
    robot = make_tcp_record_robot()

    expected_pose = set(TCP_RECORD_POSE_KEYS)
    assert expected_pose | {"gripper.pos"} == set(robot.action_features)
    assert expected_pose | {"gripper.pos"} == set(robot.observation_features)

    robot._record_space = "joint"
    expected_joints = {f"J{i}.pos" for i in range(1, 8)}
    assert expected_joints | {"gripper.pos"} == set(robot.action_features)
    assert expected_joints | {"gripper.pos"} == set(robot.observation_features)


def test_convert_observation_for_recording_uses_fk():
    robot = make_tcp_record_robot()
    joints = np.asarray([0.2, -0.4, 0.3, 0.7, -0.2, 0.5, 0.4])
    obs = {f"J{i + 1}.pos": float(joints[i]) for i in range(7)}
    obs["gripper.pos"] = 0.5
    obs["camera"] = "frame"

    converted = robot.convert_observation_for_recording(obs)

    assert not any(key.startswith("J") for key in converted)
    assert converted["gripper.pos"] == 0.5
    assert converted["camera"] == "frame"
    transform = robot._local_kinematics.forward_matrix(joints)
    assert [converted["pose.x"], converted["pose.y"], converted["pose.z"]] == pytest.approx(
        transform[:3, 3].tolist()
    )
    assert [converted[key] for key in TCP_RECORD_ROT_KEYS] == pytest.approx(
        rotation_to_6d(transform[:3, :3]).tolist()
    )


def test_rot6d_roundtrip_recovers_rotation_matrix():
    rotation = _axis_angle_rotation([0.3, -0.5, 0.8], 2.3)

    recovered = rot6d_to_rotation(rotation_to_6d(rotation))

    assert recovered == pytest.approx(rotation)


def test_rot6d_is_continuous_across_pi_flip():
    rotation_a = _axis_angle_rotation([0.0, 0.0, 1.0], np.pi - 1e-3)
    rotation_b = _axis_angle_rotation([0.0, 0.0, 1.0], -(np.pi - 1e-3))

    # The axis-angle representatives flip by ~2*pi here; the 6D vectors do not.
    assert np.linalg.norm(
        rotation_to_axis_angle(rotation_b) - rotation_to_axis_angle(rotation_a)
    ) > 5.0
    assert np.linalg.norm(rotation_to_6d(rotation_b) - rotation_to_6d(rotation_a)) < 1e-2


def test_convert_action_for_recording_is_deterministic():
    robot = make_tcp_record_robot()
    joints = [0.0] * 7
    action = {f"J{i + 1}.pos": joints[i] for i in range(7)}
    action["gripper.pos"] = 0.1

    first = robot.convert_action_for_recording(action)
    second = robot.convert_action_for_recording(action)

    assert [second[key] for key in TCP_RECORD_ROT_KEYS] == pytest.approx(
        [first[key] for key in TCP_RECORD_ROT_KEYS]
    )
    assert second["gripper.pos"] == 0.1


def test_convert_recording_passthrough_in_joint_space():
    robot = make_tcp_record_robot(record_space="joint")
    obs = {f"J{i + 1}.pos": 0.0 for i in range(7)}
    action = {f"J{i + 1}.pos": 0.0 for i in range(7)}

    assert robot.convert_observation_for_recording(obs) is obs
    assert robot.convert_action_for_recording(action) is action


def test_both_record_space_config_validation():
    from lerobot_robot_ufactory.robots.uf_robot.uf_robot_config import UFRobotConfig

    UFRobotConfig(robot_dof=7, control_space="joint", record_space="both")
    # Joint velocity keys remain valid when joints stay in the dataset.
    UFRobotConfig(
        robot_dof=7, control_space="joint", record_space="both", observe_joint_vel=True
    )

    with pytest.raises(ValueError, match="requires joint control on an xArm7"):
        UFRobotConfig(robot_dof=7, control_space="cartesian", record_space="both")
    with pytest.raises(ValueError, match="not supported in manual_mode"):
        UFRobotConfig(robot_dof=7, control_space="joint", record_space="both", manual_mode=True)


def test_both_record_space_features():
    robot = make_tcp_record_robot(record_space="both")

    expected_joints = {f"J{i}.pos" for i in range(1, 8)}
    expected_pose = set(TCP_RECORD_POSE_KEYS)
    assert expected_joints | expected_pose | {"gripper.pos"} == set(robot.action_features)
    assert expected_joints | expected_pose | {"gripper.pos"} == set(robot.observation_features)

    robot._jnt_obs_has_vel = True
    expected_vel = {f"J{i}.vel" for i in range(1, 8)}
    assert expected_joints | expected_vel | expected_pose | {"gripper.pos"} == set(
        robot.observation_features
    )


def test_both_record_space_keeps_joints_and_adds_tcp_pose():
    robot = make_tcp_record_robot(record_space="both")
    joints = np.asarray([0.2, -0.4, 0.3, 0.7, -0.2, 0.5, 0.4])
    obs = {f"J{i + 1}.pos": float(joints[i]) for i in range(7)}
    obs["gripper.pos"] = 0.5
    obs["camera"] = "frame"

    converted = robot.convert_observation_for_recording(obs)

    for i in range(7):
        assert converted[f"J{i + 1}.pos"] == float(joints[i])
    assert converted["gripper.pos"] == 0.5
    assert converted["camera"] == "frame"
    position = robot._local_kinematics.tcp_position(joints)
    assert [converted["pose.x"], converted["pose.y"], converted["pose.z"]] == pytest.approx(
        position.tolist()
    )

    action = {f"J{i + 1}.pos": float(joints[i]) for i in range(7)}
    action["gripper.pos"] = 0.25
    converted_action = robot.convert_action_for_recording(action)
    assert converted_action["J1.pos"] == float(joints[0])
    assert converted_action["gripper.pos"] == 0.25
    assert "pose.x" in converted_action


def _rotation_angle_error(a: np.ndarray, b: np.ndarray) -> float:
    """Rotation angle (rad) of ``a @ b.T``; 0 when the matrices agree.

    Trace-based angle recovery saturates near 2e-8 rad in float64 (the
    ``acos((tr - 1) / 2)`` noise floor), so tests must not assert below ~1e-7.
    """
    delta = a @ b.T
    cos_angle = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cos_angle))


def test_rot6d_roundtrip_random_rotations():
    rng = np.random.default_rng(0)
    # Mix uniform angles with cases near 0 and near pi, the hard regions for
    # most rotation representations.
    angles = np.concatenate(
        [
            rng.uniform(-np.pi, np.pi, size=150),
            rng.uniform(-1e-6, 1e-6, size=25),
            np.pi - rng.uniform(0.0, 1e-6, size=25),
        ]
    )
    for angle in angles:
        axis = rng.normal(size=3)
        rotation = _axis_angle_rotation(axis, angle)

        recovered = rot6d_to_rotation(rotation_to_6d(rotation))

        assert _rotation_angle_error(recovered, rotation) < 1e-7
        assert recovered.T @ recovered == pytest.approx(np.eye(3), abs=1e-12)
        assert np.linalg.det(recovered) == pytest.approx(1.0, abs=1e-12)


def test_rot6d_roundtrip_edge_poses():
    cases = [np.eye(3)]
    for axis in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]):
        for angle in (np.pi, -np.pi, np.pi / 2, -np.pi / 2):
            cases.append(_axis_angle_rotation(axis, angle))
    # pi about an arbitrary axis: the axis-angle singular point.
    cases.append(_axis_angle_rotation([0.3, -0.5, 0.8], np.pi))

    for rotation in cases:
        recovered = rot6d_to_rotation(rotation_to_6d(rotation))
        assert _rotation_angle_error(recovered, rotation) < 1e-7


def test_rot6d_order_matches_tcp_record_pose_keys():
    """pose.r11/r21/r31/r12/r22/r32 must be the first two matrix columns."""
    rotation = _axis_angle_rotation([0.3, -0.5, 0.8], 1.1)

    vector = rotation_to_6d(rotation)
    expected = [rotation[0, 0], rotation[1, 0], rotation[2, 0],
                rotation[0, 1], rotation[1, 1], rotation[2, 1]]
    assert vector == pytest.approx(expected)

    # The recording pipeline must emit the same convention under these keys.
    robot = make_tcp_record_robot()

    class FixedKinematics:
        def forward_matrix(self, joints):
            transform = np.eye(4)
            transform[:3, :3] = rotation
            return transform

    robot._local_kinematics = FixedKinematics()
    obs = {f"J{i + 1}.pos": 0.0 for i in range(7)}
    obs["gripper.pos"] = 0.0
    converted = robot.convert_observation_for_recording(obs)
    for key, value in zip(TCP_RECORD_ROT_KEYS, expected):
        assert converted[key] == pytest.approx(value)


def test_rot6d_gram_schmidt_tolerates_noise():
    """Noisy 6D vectors (e.g. policy outputs) still recover to a rotation."""
    rng = np.random.default_rng(1)
    rotation = _axis_angle_rotation([0.3, -0.5, 0.8], 1.7)
    clean = rotation_to_6d(rotation)

    for sigma in (1e-4, 1e-2):
        noisy = clean + rng.normal(scale=sigma, size=6)
        recovered = rot6d_to_rotation(noisy)
        assert recovered.T @ recovered == pytest.approx(np.eye(3), abs=1e-9)
        assert np.linalg.det(recovered) == pytest.approx(1.0, abs=1e-9)
        # First-order perturbation bound: angular error stays within a small
        # multiple of the injected noise.
        assert _rotation_angle_error(recovered, rotation) < 10.0 * sigma


def test_rot6d_matches_xarm_rpy_convention():
    """xarm_rpy_transform -> 6D -> matrix must round-trip exactly.

    Pins the chain used to compare local FK rotations against controller
    roll/pitch/yaw output.
    """
    rng = np.random.default_rng(2)
    for _ in range(100):
        pose = np.concatenate(
            [rng.uniform(-500.0, 500.0, size=3), rng.uniform(-np.pi, np.pi, size=3)]
        )
        rotation = xarm_rpy_transform(pose)[:3, :3]
        recovered = rot6d_to_rotation(rotation_to_6d(rotation))
        assert _rotation_angle_error(recovered, rotation) < 1e-7
