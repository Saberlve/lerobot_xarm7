"""Cartesian constraints and transitions without connecting robot hardware."""
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot_robot_ufactory.robots.uf_robot.local_kinematics import xarm_rpy_transform
from lerobot_robot_ufactory.teleoperators.gello_teleop.xy_plane_mode import XYPlaneMode
from lerobot_robot_ufactory.teleoperators.gello_teleop.gello_teleop_config import GelloTeleopConfig
from lerobot_robot_ufactory.utils.realtime_teleop import apply_pending_gello_joint_mode, update_gello_joint_mode_key


def setup_mode():
    mode = XYPlaneMode()
    state = np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.0])
    calls = []

    def fk(q):
        return xarm_rpy_transform(np.r_[np.asarray(q[:3]) * 100, q[3:6]])

    def ik(pose, **kwargs):
        calls.append(pose)
        return 0, np.r_[np.asarray(pose[:3]) / 100, pose[3:6], 0.0]

    robot = SimpleNamespace(
        _control_space='joint',
        _local_kinematics=SimpleNamespace(forward_matrix=fk),
        latest_state_before=lambda _: SimpleNamespace(joint_positions=state.copy()),
        real_arm=SimpleNamespace(
            get_forward_kinematics=lambda q, **kw: (0, np.r_[np.asarray(q[:3]) * 100, q[3:6]]),
            get_inverse_kinematics=ik,
        ),
    )
    teleop = SimpleNamespace(_xy_plane_mode=mode, _teleop_enabled=True)
    return mode, robot, teleop, state, calls


def press(teleop):
    update_gello_joint_mode_key(teleop, SimpleNamespace(char='x'), True)
    update_gello_joint_mode_key(teleop, SimpleNamespace(char='X'), False)


def test_xy_changes_and_gripper_remains_live_then_exit_rebases():
    mode, robot, teleop, state, calls = setup_mode()
    press(teleop)
    apply_pending_gello_joint_mode(robot, teleop, {})
    initial = np.r_[state + 0.05, 0.4]
    assert mode.apply(initial)[:7] == pytest.approx(state)
    moved = initial + 0.01
    result = mode.apply(moved)
    assert calls[-1] == pytest.approx([11, 21, 30, .1, .2, .3])
    assert result[:7] == pytest.approx(state + np.array([.01, .01, 0, 0, 0, 0, 0]))
    assert result[-1] == pytest.approx(.41)
    state[:] = result[:7]
    press(teleop)
    apply_pending_gello_joint_mode(robot, teleop, {})
    assert mode.apply(moved + .5)[:7] == pytest.approx(state)
    assert mode.apply(moved + .51)[:7] == pytest.approx(state + .01)


@pytest.mark.parametrize('failure', ['code', 'nan', 'jump', 'wrong_pose'])
def test_bad_ik_holds_previous_command(failure):
    mode, robot, teleop, state, _ = setup_mode()
    press(teleop)
    apply_pending_gello_joint_mode(robot, teleop, {})
    mode.apply(np.r_[state, .5])
    bad = state.copy()
    if failure == 'nan':
        bad[0] = np.nan
    elif failure == 'jump':
        bad[6] += 2 * np.pi
    elif failure == 'wrong_pose':
        bad[2] += .01
    robot.real_arm.get_inverse_kinematics = lambda *a, **k: (1 if failure == 'code' else 0, bad)
    assert mode.apply(np.r_[state + .01, .7]) == pytest.approx(np.r_[state, .7])


def test_debounce_pause_reset_and_stale_feedback():
    mode, robot, teleop, state, _ = setup_mode()
    key = SimpleNamespace(char='x')
    for _ in range(5):
        update_gello_joint_mode_key(teleop, key, True)
    assert mode.pending
    apply_pending_gello_joint_mode(robot, teleop, {})
    assert mode.active
    mode.reset()
    teleop._teleop_enabled = False
    press(teleop)
    assert not mode.pending and not mode.active
    teleop._teleop_enabled = True
    press(teleop)
    def stale(_):
        raise TimeoutError('stale')
    robot.latest_state_before = stale
    with pytest.raises(TimeoutError):
        apply_pending_gello_joint_mode(robot, teleop, {})
    assert not mode.active


def test_config_validation():
    assert not GelloTeleopConfig().xy_plane_only_mode_enabled
    with pytest.raises(ValueError, match='boolean'):
        GelloTeleopConfig(xy_plane_only_mode_enabled='yes')
    with pytest.raises(ValueError, match='cannot be enabled together'):
        GelloTeleopConfig(xy_plane_only_mode_enabled=True, joint7_only_mode_enabled=True)


def test_gello_action_path_and_pause_clear_constraint():
    from test_gello_joint_mode import make_teleop, joints

    teleop, driver = make_teleop(enabled=False)
    mode, robot, _, state, _ = setup_mode()
    teleop._xy_plane_mode = mode
    press(teleop)
    apply_pending_gello_joint_mode(robot, teleop, {})
    assert joints(teleop.get_action()) == pytest.approx(state)
    driver.positions[:7] += .01
    constrained = joints(teleop.get_action())
    assert constrained[0] > state[0]
    assert constrained[1] < state[1]  # Joint 2 has a negative GELLO mapping.
    assert constrained[2:] == pytest.approx(state[2:])
    teleop.set_teleop_enabled(False)
    assert not mode.active and not mode.pending
    press(teleop)
    assert not mode.pending


def test_large_input_is_bounded_and_does_not_leave_queued_motion():
    mode, robot, teleop, state, _ = setup_mode()
    press(teleop)
    apply_pending_gello_joint_mode(robot, teleop, {})
    mode.apply(np.r_[state, .5])
    moved = np.r_[state, .7]
    moved[0] += .3  # Old implementation rejected this 30 mm move forever.
    result = mode.apply(moved)
    assert result[0] == pytest.approx(state[0] + .02)
    assert result[1:7] == pytest.approx(state[1:])
    assert result[-1] == .7
    assert mode.warning is not None
    for _ in range(5):
        assert mode.apply(moved) == pytest.approx(result)
    moved[0] -= .01
    assert mode.apply(moved)[0] == pytest.approx(state[0] + .01)
    assert mode.warning is None


def test_unreachable_input_does_not_accumulate_and_reverse_recovers():
    mode, robot, teleop, state, calls = setup_mode()
    original_ik = robot.real_arm.get_inverse_kinematics
    robot.real_arm.get_inverse_kinematics = lambda pose, **kw: (
        (1, None) if pose[0] > 10 else original_ik(pose, **kw))
    press(teleop)
    apply_pending_gello_joint_mode(robot, teleop, {})
    mode.apply(np.r_[state, .5])
    for delta in [.1, .2, .3]:
        moved = np.r_[state, .8]
        moved[0] += delta
        assert mode.apply(moved) == pytest.approx(np.r_[state, .8])
        assert '逆解失败' in mode.warning
    moved[0] -= .01
    result = mode.apply(moved)
    assert result[0] == pytest.approx(state[0] - .01)
    assert mode.warning is None


def test_smaller_cartesian_step_recovers_from_branch_jump():
    mode, robot, teleop, state, calls = setup_mode()
    original_ik = robot.real_arm.get_inverse_kinematics
    def ik(pose, **kw):
        code, solution = original_ik(pose, **kw)
        if pose[0] > 11:
            solution[6] = 1.0
        return code, solution
    robot.real_arm.get_inverse_kinematics = ik
    press(teleop)
    apply_pending_gello_joint_mode(robot, teleop, {})
    mode.apply(np.r_[state, .5])
    moved = np.r_[state, .5]
    moved[0] += .02
    result = mode.apply(moved)
    assert [p[0] for p in calls] == pytest.approx([12, 11])
    assert result[:7] == pytest.approx(state + np.array([.01, 0, 0, 0, 0, 0, 0]))


@pytest.mark.parametrize('delta', [(0, 1), (0, -1), (1, 0), (1, 1), (3, 4)])
def test_planar_direction_and_total_step_limit(delta):
    mode, robot, teleop, state, calls = setup_mode()
    press(teleop)
    apply_pending_gello_joint_mode(robot, teleop, {})
    initial = np.r_[state, .5]
    mode.apply(initial)
    moved = initial.copy()
    moved[:2] += np.asarray(delta) / 100
    moved[2:6] += .2  # Height and orientation inputs must be ignored.
    result = mode.apply(moved)
    delta = np.asarray(delta, dtype=float)
    expected = delta * min(1, 2 / np.linalg.norm(delta))
    assert (result[:2] - state[:2]) * 100 == pytest.approx(expected)
    assert result[2:7] == pytest.approx(state[2:])
    assert np.linalg.norm((result[:2] - state[:2]) * 100) <= 2 + 1e-9
    assert mode.apply(moved) == pytest.approx(result)


@pytest.mark.parametrize('mismatch', ['position', 'orientation'])
def test_controller_fk_accepts_valid_ik_despite_local_model_error(mismatch, caplog):
    mode, robot, teleop, state, _ = setup_mode()
    original_local_fk = robot._local_kinematics.forward_matrix
    original_controller_fk = robot.real_arm.get_forward_kinematics
    controller_queries = []

    def approximate_fk(q):
        transform = original_local_fk(q)
        if mismatch == 'position':
            transform[2, 3] += .3  # Within the safety model's 2 mm tolerance.
        else:
            transform[:3, :3] = transform[:3, :3] @ xarm_rpy_transform([0, 0, 0, 0, 0, .002])[:3, :3]
        return transform

    def controller_fk(q, **kwargs):
        controller_queries.append(q)
        return original_controller_fk(q, **kwargs)

    robot._local_kinematics.forward_matrix = approximate_fk
    robot.real_arm.get_forward_kinematics = controller_fk
    press(teleop)
    apply_pending_gello_joint_mode(robot, teleop, {})
    mode.apply(np.r_[state, .5])
    with caplog.at_level('INFO'):
        for delta in [.01, .02, .03]:
            moved = np.r_[state, .7]
            moved[:2] += delta
            result = mode.apply(moved)
            assert result[:2] == pytest.approx(state[:2] + delta)
            assert result[2:7] == pytest.approx(state[2:])
            assert result[-1] == .7
            assert mode.warning is None
    assert len(controller_queries) == 4  # Activation, then one FK per solution.
    assert mode._controller_fk_validation
    assert '控制器正解复核通过' in caplog.text
    assert '位置误差' in caplog.text and '姿态误差' in caplog.text
    mode.reset()
    assert not mode._controller_fk_validation


@pytest.mark.parametrize('failure', ['code', 'invalid', 'position', 'orientation'])
def test_controller_fk_recheck_failure_holds_arm_and_reports_reason(failure):
    mode, robot, teleop, state, _ = setup_mode()
    press(teleop)
    apply_pending_gello_joint_mode(robot, teleop, {})
    mode.apply(np.r_[state, .5])
    # Force controller rechecking, as after detecting model disagreement.
    mode._controller_fk_validation = True
    original_fk = robot.real_arm.get_forward_kinematics

    def controller_fk(q, **kwargs):
        code, pose = original_fk(q, **kwargs)
        if failure == 'code':
            return 1, None
        if failure == 'invalid':
            return 0, [float('nan')] * 6
        pose[2 if failure == 'position' else 5] += 1
        return code, pose

    robot.real_arm.get_forward_kinematics = controller_fk
    moved = np.r_[state + .01, .8]
    assert mode.apply(moved) == pytest.approx(np.r_[state, .8])
    assert '控制器' in mode.warning
    if failure in ('position', 'orientation'):
        assert '位置误差' in mode.warning and '姿态误差' in mode.warning
