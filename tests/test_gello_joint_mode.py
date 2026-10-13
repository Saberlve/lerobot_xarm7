"""Exercise the opt-in joint mask and mode transitions without hardware."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot_robot_ufactory.current_control.control.runtime import RuntimeAgent, RuntimeRobot
from lerobot_robot_ufactory.scripts.uf_lerobot_record import _print_record_controls
from lerobot_robot_ufactory.teleoperators.gello_teleop.gello_teleop import GelloTeleop
from lerobot_robot_ufactory.teleoperators.gello_teleop.gello_teleop_config import GelloTeleopConfig
from lerobot_robot_ufactory.utils.realtime_teleop import (
    RealtimeTeleopController,
    apply_pending_gello_joint_mode,
    update_gello_joint_mode_key,
)


def make_teleop(*, enabled=True, gripper_mode="gello", precision=False, precision_scale=0.2):
    config = GelloTeleopConfig(
        joint7_only_mode_enabled=enabled,
        joint_precision_mode_enabled=precision,
        joint_precision_scale=precision_scale,
        start_joints=(0,) * 7,
        gripper_control_mode=gripper_mode,
        gripper_keyboard_hold_delay_s=0.0,
    )
    teleop = GelloTeleop(config)
    driver = SimpleNamespace(positions=np.zeros(8))
    driver.positions[-1] = np.pi / 4
    driver.get_joints = lambda: driver.positions.copy()
    driver.request = lambda *args: None
    driver.close = lambda: None
    teleop.gello_agent = RuntimeAgent(RuntimeRobot(driver, [1, -1, 1, -1, 1, -1, 1], [8, 0, 90]))
    teleop._is_connected = teleop._teleop_enabled = True
    teleop._needs_alignment = False
    return teleop, driver


def joints(action):
    return np.array([action[f"J{i}.pos"] for i in range(1, 8)])


def observation(positions):
    return {f"J{i + 1}.pos": float(value) for i, value in enumerate(positions)}


def press_s(teleop):
    update_gello_joint_mode_key(teleop, SimpleNamespace(char="s"), True)
    update_gello_joint_mode_key(teleop, SimpleNamespace(char="s"), False)


def test_mode_is_default_off_and_requires_seven_joints():
    assert GelloTeleopConfig().joint7_only_mode_enabled is False
    with pytest.raises(ValueError, match="IDs 1..7"):
        GelloTeleopConfig(joint7_only_mode_enabled=True, joint_ids=(1, 2, 3, 4, 5, 6))


@pytest.mark.parametrize("value", [1, "true", None])
def test_mode_opt_in_requires_a_boolean(value):
    with pytest.raises(ValueError, match="must be a boolean"):
        GelloTeleopConfig(joint7_only_mode_enabled=value)


def test_disabled_mode_ignores_s_and_follows_all_joints():
    teleop, driver = make_teleop(enabled=False)
    press_s(teleop)
    assert not teleop.joint_control_mode_switch_pending()
    # No pose validation or RT reads take place when the feature is disabled.
    apply_pending_gello_joint_mode(object(), teleop, {})
    driver.positions[:7] = 0.2
    assert joints(teleop.get_action()) == pytest.approx([0.2, -0.2, 0.2, -0.2, 0.2, -0.2, 0.2])


@pytest.mark.parametrize("gripper_mode", ["gello", "keyboard"])
def test_j1_to_j6_hold_measured_pose_while_j7_and_gripper_remain_live(gripper_mode):
    teleop, driver = make_teleop(gripper_mode=gripper_mode)
    held = np.arange(6) / 10
    press_s(teleop)
    teleop.apply_pending_joint_control_mode(observation(held))
    first = teleop.get_action()
    driver.positions[:7] += 0.1
    driver.positions[-1] += 0.1
    if gripper_mode == "keyboard":
        teleop.set_gripper_keyboard_state(close=True, open=False)
    second = teleop.get_action()
    assert joints(first)[:6] == pytest.approx(held)
    assert joints(second)[:6] == pytest.approx(held)
    assert second["J7.pos"] > first["J7.pos"]
    assert second["gripper.pos"] > first["gripper.pos"]


def test_resume_realigns_first_six_joints_and_keeps_j7_continuous():
    teleop, driver = make_teleop()
    held = np.arange(6) / 10
    press_s(teleop)
    teleop.apply_pending_joint_control_mode(observation(held))
    teleop.get_action()
    # Move GELLO's blocked joints while continuing to rotate J7.
    driver.positions[:7] += 0.8
    frozen = teleop.get_action()
    resumed_pose = held + 0.02
    press_s(teleop)
    teleop.apply_pending_joint_control_mode(observation(resumed_pose))
    resumed = teleop.get_action()
    assert joints(resumed)[:6] == pytest.approx(resumed_pose)
    # Only the normal encoder smoothing continues for J7; no J7 rebase occurs.
    assert resumed["J7.pos"] >= frozen["J7.pos"]
    previous_mapped = teleop.gello_agent._robot._last_pos.copy()
    driver.positions[:7] += 0.1
    followed = teleop.get_action()
    mapped_delta = teleop.gello_agent._robot._last_pos[:6] - previous_mapped[:6]
    assert joints(followed)[:6] == pytest.approx(resumed_pose + mapped_delta)
    assert mapped_delta[0] > 0 and mapped_delta[1] < 0


def test_s_auto_repeat_is_debounced_and_uppercase_release_works():
    teleop, _ = make_teleop()
    key = SimpleNamespace(char="s")
    for _ in range(5):
        update_gello_joint_mode_key(teleop, key, True)
    assert teleop.joint_control_mode_switch_pending()
    teleop.apply_pending_joint_control_mode(observation(np.zeros(6)))
    update_gello_joint_mode_key(teleop, key, True)
    assert not teleop.joint_control_mode_switch_pending()
    update_gello_joint_mode_key(teleop, SimpleNamespace(char="S"), False)
    update_gello_joint_mode_key(teleop, SimpleNamespace(char="S"), True)
    assert teleop.joint_control_mode_switch_pending()


def test_j7_remains_continuous_across_encoder_wrap_and_mode_resume():
    teleop, driver = make_teleop()
    driver.positions[6] = 2 * np.pi - 0.05
    before_wrap = teleop.get_action()["J7.pos"]
    press_s(teleop)
    teleop.apply_pending_joint_control_mode(observation(np.zeros(6)))
    driver.positions[6] = 0.05
    after_wrap = teleop.get_action()["J7.pos"]
    assert after_wrap - before_wrap == pytest.approx(0.099)
    press_s(teleop)
    teleop.apply_pending_joint_control_mode(observation(np.zeros(6)))
    resumed = teleop.get_action()["J7.pos"]
    assert resumed >= after_wrap
    assert resumed - after_wrap < 0.002


def test_two_quick_presses_cancel_an_unapplied_toggle():
    teleop, _ = make_teleop()
    press_s(teleop)
    press_s(teleop)
    assert not teleop.joint_control_mode_switch_pending()


def test_pause_and_realign_clear_mode_and_ignore_paused_presses():
    teleop, driver = make_teleop()
    press_s(teleop)
    teleop.apply_pending_joint_control_mode(observation(np.ones(6)))
    teleop.get_action()
    teleop.set_teleop_enabled(False)
    press_s(teleop)
    assert not teleop.joint_control_mode_switch_pending()
    driver.positions[:7] += 0.8
    robot_pose = observation(np.arange(7) / 20) | {"gripper.pos": 0.5}
    teleop.set_teleop_enabled(True, robot_pose)
    assert not teleop._joint7_only_active
    assert joints(teleop.get_action()) == pytest.approx(np.arange(7) / 20)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_invalid_robot_pose_does_not_activate_mode(invalid):
    teleop, _ = make_teleop()
    press_s(teleop)
    pose = observation(np.zeros(6))
    pose["J3.pos"] = invalid
    with pytest.raises(ValueError, match="finite robot joint positions"):
        teleop.apply_pending_joint_control_mode(pose)
    assert not teleop._joint7_only_active
    assert teleop.joint_control_mode_switch_pending()


def test_cached_rt_pose_is_used_instead_of_old_targets_or_camera_reads():
    teleop, _ = make_teleop()
    press_s(teleop)
    robot = SimpleNamespace(
        _control_space="joint",
        latest_state_before=lambda _: SimpleNamespace(joint_positions=np.arange(7) / 10),
    )
    apply_pending_gello_joint_mode(robot, teleop, observation(np.full(7, 99)))
    assert joints(teleop.get_action())[:6] == pytest.approx(np.arange(6) / 10)


def test_stale_rt_pose_prevents_switch_without_falling_back_to_old_observation():
    teleop, _ = make_teleop()
    press_s(teleop)

    def stale(_):
        raise TimeoutError("stale RT pose")

    robot = SimpleNamespace(_control_space="joint", latest_state_before=stale)
    with pytest.raises(TimeoutError, match="stale RT pose"):
        apply_pending_gello_joint_mode(robot, teleop, observation(np.zeros(7)))
    assert not teleop._joint7_only_active


def test_generic_joint_control_uses_prefixed_observation():
    teleop, _ = make_teleop()
    press_s(teleop)
    robot = SimpleNamespace(_control_space="joint", prefix="arm_")
    pose = {f"arm_J{i}.pos": i / 10 for i in range(1, 7)}
    apply_pending_gello_joint_mode(robot, teleop, pose)
    assert joints(teleop.get_action())[:6] == pytest.approx(np.arange(1, 7) / 10)


def test_cartesian_control_cannot_activate_joint_mask():
    teleop, _ = make_teleop()
    press_s(teleop)
    with pytest.raises(ValueError, match="joint-space"):
        apply_pending_gello_joint_mode(SimpleNamespace(_control_space="tcp"), teleop, {})


def test_realtime_owner_applies_mask_to_sent_and_recorded_commands():
    teleop, driver = make_teleop()
    press_s(teleop)
    sent_twice = threading.Event()
    actions = []
    owner_threads = []
    held = np.arange(7) / 10

    class Robot:
        _control_space = "joint"

        def latest_state_before(self, _):
            owner_threads.append(threading.current_thread().name)
            return SimpleNamespace(joint_positions=held)

        def send_action(self, action):
            actions.append(dict(action))
            driver.positions[:7] += 0.1
            if len(actions) >= 2:
                sent_twice.set()
            return action

    def identity(value):
        return value[0]

    controller = RealtimeTeleopController(
        Robot(), teleop, identity, identity, fps=100,
        initial_observation=observation(np.full(7, 99)),
    )
    try:
        controller.start()
        assert sent_twice.wait(timeout=1)
    finally:
        controller.stop()
    assert owner_threads == ["uf-servoj-control"]
    assert all(joints(action)[:6] == pytest.approx(held[:6]) for action in actions)
    assert actions[1]["J7.pos"] > actions[0]["J7.pos"]
    assert controller.latest_action() == actions[-1]


def test_s_help_is_displayed_only_when_enabled(capsys):
    enabled, _ = make_teleop()
    disabled, _ = make_teleop(enabled=False)
    _print_record_controls(True, False, enabled)
    assert "[S] J7 only / All joints" in capsys.readouterr().out
    _print_record_controls(True, False, disabled)
    assert "[S]" not in capsys.readouterr().out


def press_z(teleop):
    update_gello_joint_mode_key(teleop, SimpleNamespace(char="z"), True)
    update_gello_joint_mode_key(teleop, SimpleNamespace(char="Z"), False)


@pytest.mark.parametrize("j7_enabled", [False, True])
@pytest.mark.parametrize("precision_scale", [0.1, 0.2, 0.5, 1.0])
def test_precision_scales_deltas_and_rebases_both_transitions(j7_enabled, precision_scale):
    teleop, driver = make_teleop(enabled=j7_enabled, precision=True, precision_scale=precision_scale)
    for scale in (precision_scale, 1.0):
        press_z(teleop)
        pose = np.arange(6) / 10
        apply_pending_gello_joint_mode(SimpleNamespace(_control_space="joint"), teleop, observation(pose))
        first = teleop.get_action()
        assert joints(first)[:6] == pytest.approx(pose)
        before = teleop.gello_agent._robot._last_pos.copy()
        driver.positions += 0.25
        second = teleop.get_action()
        delta = teleop.gello_agent._robot._last_pos - before
        assert joints(second)[:6] == pytest.approx(pose + scale * delta[:6])
        assert second["J7.pos"] - first["J7.pos"] == pytest.approx(delta[6])
        assert second["gripper.pos"] - first["gripper.pos"] == pytest.approx(delta[7])


def test_precision_and_j7_hold_can_be_toggled_independently():
    teleop, driver = make_teleop(precision=True)
    pose = np.arange(6) / 10
    press_z(teleop)
    press_s(teleop)
    teleop.apply_pending_joint_control_mode(observation(pose))
    teleop.get_action()
    driver.positions += 0.5
    assert joints(teleop.get_action())[:6] == pytest.approx(pose)
    # Z changes the future gain but does not move the held joints.
    for _ in range(2):
        press_z(teleop)
        teleop.apply_pending_joint_control_mode(observation(pose + 0.01))
        assert joints(teleop.get_action())[:6] == pytest.approx(pose)
    press_s(teleop)
    teleop.apply_pending_joint_control_mode(observation(pose))
    assert joints(teleop.get_action())[:6] == pytest.approx(pose)
    before = teleop.gello_agent._robot._last_pos.copy()
    driver.positions += 0.2
    after = teleop.get_action()
    delta = teleop.gello_agent._robot._last_pos - before
    assert joints(after)[:6] == pytest.approx(pose + 0.2 * delta[:6])


def test_precision_key_debounce_pause_and_disabled_feature():
    disabled, _ = make_teleop()
    press_z(disabled)
    assert not disabled.joint_control_mode_switch_pending()
    teleop, _ = make_teleop(precision=True)
    key = SimpleNamespace(char="z")
    for _ in range(5):
        update_gello_joint_mode_key(teleop, key, True)
    teleop.apply_pending_joint_control_mode(observation(np.zeros(6)))
    assert teleop._joint_precision_active
    update_gello_joint_mode_key(teleop, key, True)
    assert not teleop.joint_control_mode_switch_pending()
    teleop.set_teleop_enabled(False)
    press_z(teleop)
    assert not teleop._joint_precision_active
    assert not teleop.joint_control_mode_switch_pending()
    teleop.set_teleop_enabled(True, observation(np.zeros(7)) | {"gripper.pos": 0.5})
    press_z(teleop)
    press_z(teleop)
    assert not teleop.joint_control_mode_switch_pending()


def test_precision_invalid_feedback_leaves_switch_pending():
    teleop, _ = make_teleop(precision=True)
    press_z(teleop)
    with pytest.raises(ValueError, match="finite robot joint positions"):
        teleop.apply_pending_joint_control_mode(observation(np.full(6, np.nan)))
    assert not teleop._joint_precision_active
    assert teleop.joint_control_mode_switch_pending()


def test_precision_config_validation_and_help(capsys):
    assert not GelloTeleopConfig().joint_precision_mode_enabled
    with pytest.raises(ValueError, match="boolean"):
        GelloTeleopConfig(joint_precision_mode_enabled=1)
    with pytest.raises(ValueError, match="IDs 1..7"):
        GelloTeleopConfig(joint_precision_mode_enabled=True, joint_ids=(1, 2))
    with pytest.raises(ValueError, match="cannot be enabled together"):
        GelloTeleopConfig(joint_precision_mode_enabled=True, xy_plane_only_mode_enabled=True)
    teleop, _ = make_teleop(precision=True, precision_scale=0.5)
    _print_record_controls(True, False, teleop)
    assert "[Z] J1-J6 precision (x0.5) / Normal" in capsys.readouterr().out


@pytest.mark.parametrize("scale", [0, -0.1, 1.1, float("nan"), float("inf"), True, "0.2", None])
def test_precision_scale_rejects_invalid_values(scale):
    with pytest.raises(ValueError, match="joint_precision_scale"):
        GelloTeleopConfig(joint_precision_scale=scale)


@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("gripper_mode", ["gello", "keyboard"])
def test_keyboard_j7_ramps_exact_relative_turn_and_holds_other_joints(monkeypatch, direction, gripper_mode):
    now = [100.0]
    monkeypatch.setattr("lerobot_robot_ufactory.teleoperators.gello_teleop.gello_teleop.time.monotonic", lambda: now[0])
    teleop, driver = make_teleop(gripper_mode=gripper_mode)
    teleop.config.joint7_keyboard_mode_enabled = True
    pose = np.arange(7) / 10
    key = SimpleNamespace(name="up" if direction > 0 else "down")
    for _ in range(5):
        update_gello_joint_mode_key(teleop, key, True)
    robot = SimpleNamespace(_control_space="joint", latest_state_before=lambda _: SimpleNamespace(joint_positions=pose))
    apply_pending_gello_joint_mode(robot, teleop, observation(np.full(7, 99)))
    assert joints(teleop.get_action()) == pytest.approx(pose)
    assert teleop.keyboard_joint7_busy()
    driver.positions[:7] += 0.8
    driver.positions[-1] += 0.1
    now[0] += 1
    assert joints(teleop.get_action()) == pytest.approx(pose + [0, 0, 0, 0, 0, 0, direction * np.pi / 3])
    assert teleop.get_action()["gripper.pos"] != pytest.approx(1 - 50 / 84)
    assert not teleop.set_joint7_keyboard_key(-direction, True)
    teleop.set_joint7_keyboard_key(-direction, False)
    now[0] += 1
    teleop.update_keyboard_joint7_observation(observation(
        pose + [0, 0, 0, 0, 0, 0, direction * 2 * np.pi / 3]
    ))
    final = teleop.get_action()
    assert joints(final) == pytest.approx(pose + [0, 0, 0, 0, 0, 0, direction * 2 * np.pi / 3])
    assert final["gripper.pos"] == pytest.approx(1 - 50 / 84)
    now[0] += 1
    assert teleop.get_action()["gripper.pos"] == pytest.approx(1 - 50 / 84)
    assert not teleop.keyboard_joint7_busy()
    # Holding the same physical key never queues another turn after completion.
    update_gello_joint_mode_key(teleop, key, True)
    assert not teleop.joint_control_mode_switch_pending()
    update_gello_joint_mode_key(teleop, key, False)
    update_gello_joint_mode_key(teleop, key, True)
    assert teleop.joint_control_mode_switch_pending()


def test_keyboard_j7_resume_rebases_all_seven_joints_and_pause_cancels(monkeypatch):
    now = [10.0]
    monkeypatch.setattr("lerobot_robot_ufactory.teleoperators.gello_teleop.gello_teleop.time.monotonic", lambda: now[0])
    teleop, driver = make_teleop(precision=True)
    teleop.config.joint7_keyboard_mode_enabled = True
    assert teleop.set_joint7_keyboard_key(1, True)
    teleop.set_joint7_keyboard_key(1, False)
    teleop.apply_pending_joint_control_mode(observation(np.arange(7) / 10))
    now[0] += 1
    driver.positions[:7] += 0.9
    teleop.get_action()
    measured = np.arange(7) / 10 + 0.02
    measured[6] += 0.5
    press_s(teleop)
    teleop.apply_pending_joint_control_mode(observation(measured))
    assert not teleop._keyboard_j7_active
    assert not teleop._joint7_only_active
    assert joints(teleop.get_action()) == pytest.approx(measured)
    before = teleop.gello_agent._robot._last_pos[:7].copy()
    driver.positions[:7] += 0.1
    after = teleop.get_action()
    assert joints(after) == pytest.approx(measured + teleop.gello_agent._robot._last_pos[:7] - before)
    assert teleop.set_joint7_keyboard_key(-1, True)
    teleop.set_teleop_enabled(False)
    assert not teleop.joint_control_mode_switch_pending()
    assert not teleop._keyboard_j7_active
    assert not teleop.set_joint7_keyboard_key(1, True)


def test_keyboard_j7_stale_or_invalid_pose_cannot_start():
    teleop, _ = make_teleop()
    teleop.config.joint7_keyboard_mode_enabled = True
    teleop.set_joint7_keyboard_key(1, True)
    pose = observation(np.zeros(7))
    pose['J7.pos'] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        teleop.apply_pending_joint_control_mode(pose)
    assert not teleop._keyboard_j7_active
    def stale(_):
        raise TimeoutError('stale RT pose')
    with pytest.raises(TimeoutError):
        apply_pending_gello_joint_mode(SimpleNamespace(_control_space='joint', latest_state_before=stale), teleop, pose)
    assert not teleop._keyboard_j7_active


def test_keyboard_j7_rejects_out_of_range_full_turn():
    teleop, _ = make_teleop()
    teleop.config.joint7_keyboard_mode_enabled = True
    teleop.set_joint7_keyboard_key(1, True)
    with pytest.raises(ValueError, match='joint limits'):
        teleop.apply_pending_joint_control_mode(observation([0, 0, 0, 0, 0, 0, 5]))
    assert not teleop._keyboard_j7_active
    assert not teleop.joint_control_mode_switch_pending()


@pytest.mark.parametrize('value', [0, -1, 121, float('nan'), float('inf'), True, '30'])
def test_keyboard_j7_config_rejects_invalid_speed(value):
    with pytest.raises(ValueError, match='speed'):
        GelloTeleopConfig(joint7_keyboard_speed_deg_s=value)


def test_keyboard_j7_feature_defaults_off_and_conflicts_with_xy():
    assert not GelloTeleopConfig().joint7_keyboard_mode_enabled
    with pytest.raises(ValueError, match='boolean'):
        GelloTeleopConfig(joint7_keyboard_mode_enabled=1)
    with pytest.raises(ValueError, match='cannot be enabled together'):
        GelloTeleopConfig(joint7_keyboard_mode_enabled=True, xy_plane_only_mode_enabled=True)
    teleop, _ = make_teleop()
    assert not teleop.set_joint7_keyboard_key(1, True)


def test_keyboard_j7_constraint_attached_after_processors_without_changing_features():
    from lerobot_robot_ufactory.utils.realtime_teleop import apply_keyboard_gripper_stop
    teleop, _ = make_teleop()
    teleop._keyboard_j7_active = True
    command = observation(np.zeros(7))
    result = apply_keyboard_gripper_stop(object(), teleop, command)
    assert result['_hold_joints_1_to_6'] is True
    assert '_hold_joints_1_to_6' not in command


def test_keyboard_j7_next_turn_uses_measured_j7_and_keeps_original_j1_j6_anchor(monkeypatch):
    now = [10.0]
    monkeypatch.setattr("lerobot_robot_ufactory.teleoperators.gello_teleop.gello_teleop.time.monotonic", lambda: now[0])
    teleop, _ = make_teleop()
    teleop.config.joint7_keyboard_mode_enabled = True
    teleop.set_joint7_keyboard_key(1, True)
    teleop.set_joint7_keyboard_key(1, False)
    first = np.arange(7) / 10
    teleop.apply_pending_joint_control_mode(observation(first))
    now[0] += 4.1
    second = first + 0.02
    second[6] = first[6] + 2 * np.pi / 3 - 0.005
    teleop.update_keyboard_joint7_observation(observation(second))
    teleop.get_action()
    assert teleop.set_joint7_keyboard_key(-1, True)
    teleop.set_joint7_keyboard_key(-1, False)
    robot = SimpleNamespace(_control_space='joint', prefix='arm_')
    apply_pending_gello_joint_mode(robot, teleop, {f'arm_J{i+1}.pos': float(v) for i,v in enumerate(second)})
    assert joints(teleop.get_action()) == pytest.approx([*first[:6], second[6]])
    now[0] += 5
    assert joints(teleop.get_action()) == pytest.approx([*first[:6], second[6] - 2 * np.pi / 3])


@pytest.mark.parametrize('direction', [1, -1])
@pytest.mark.parametrize('gripper_mode', ['keyboard', 'gello'])
def test_keyboard_j7_hold_stops_at_measured_pose_without_opening(monkeypatch, direction, gripper_mode):
    now = [10.0]
    monkeypatch.setattr('lerobot_robot_ufactory.teleoperators.gello_teleop.gello_teleop.time.monotonic', lambda: now[0])
    teleop, _ = make_teleop(gripper_mode=gripper_mode)
    teleop.config.joint7_keyboard_mode_enabled = True
    teleop.config.joint7_keyboard_hold_to_rotate = True
    grip = teleop.get_action()['gripper.pos']
    teleop.set_joint7_keyboard_key(direction, True)
    teleop.apply_pending_joint_control_mode(observation(np.zeros(7)))
    now[0] += 3
    action = teleop.get_action()
    assert action['J7.pos'] == pytest.approx(direction * np.pi)
    assert action['gripper.pos'] == pytest.approx(grip)
    teleop.set_joint7_keyboard_key(direction, False)
    assert teleop.joint_control_mode_switch_pending()
    measured = [0] * 6 + [direction * 2.9]
    teleop.apply_pending_joint_control_mode(observation(measured))
    now[0] += 5
    assert joints(teleop.get_action()) == pytest.approx(measured)
    assert teleop.get_action()['gripper.pos'] == pytest.approx(grip)
    assert not teleop.keyboard_joint7_busy()
    teleop.set_joint7_keyboard_key(-direction, True)
    teleop.apply_pending_joint_control_mode(observation(measured))
    now[0] += 0.5
    assert teleop.get_action()['J7.pos'] == pytest.approx(direction * (2.9 - np.pi / 6))
    teleop.set_joint7_keyboard_key(direction, True)
    teleop.apply_pending_joint_control_mode(observation(measured))
    now[0] += 1
    assert teleop.get_action()['J7.pos'] == pytest.approx(measured[-1])


def test_keyboard_j7_hold_release_before_control_tick_cancels():
    teleop, _ = make_teleop()
    teleop.config.joint7_keyboard_mode_enabled = True
    teleop.config.joint7_keyboard_hold_to_rotate = True
    teleop.set_joint7_keyboard_key(1, True)
    teleop.set_joint7_keyboard_key(1, False)
    assert not teleop.joint_control_mode_switch_pending()
    assert not teleop.keyboard_joint7_busy()


@pytest.mark.parametrize('direction', [1, -1])
@pytest.mark.parametrize('modifier,angle', [('none', 150), ('ctrl', 10), ('shift', 60)])
@pytest.mark.parametrize('gripper_mode', ['keyboard', 'gello'])
def test_removal_angles_and_opening_are_latched(monkeypatch, direction, modifier, angle, gripper_mode):
    now = [10.0]
    monkeypatch.setattr('lerobot_robot_ufactory.teleoperators.gello_teleop.gello_teleop.time.monotonic', lambda: now[0])
    teleop, _ = make_teleop(gripper_mode=gripper_mode)
    teleop.config.joint7_keyboard_mode_enabled = True
    teleop.config.joint7_keyboard_angle_deg = 150
    teleop.config.joint7_keyboard_speed_deg_s = 75
    grip = teleop.get_action()['gripper.pos']
    if modifier != 'none':
        update_gello_joint_mode_key(teleop, SimpleNamespace(name=modifier + '_l'), True)
    key = SimpleNamespace(name='up' if direction > 0 else 'down')
    update_gello_joint_mode_key(teleop, key, True)
    update_gello_joint_mode_key(teleop, key, False)
    # Releasing the modifier before the control tick must not alter the command.
    if modifier != 'none':
        update_gello_joint_mode_key(teleop, SimpleNamespace(name=modifier + '_l'), False)
    teleop.apply_pending_joint_control_mode(observation(np.zeros(7)))
    now[0] += angle / 75 / 2
    action = teleop.get_action()
    assert action['J7.pos'] == pytest.approx(direction * np.deg2rad(angle / 2))
    assert action['gripper.pos'] == pytest.approx(grip)
    now[0] = 10 + angle / 75 + 1e-6
    teleop.update_keyboard_joint7_observation(
        observation([0] * 6 + [direction * np.deg2rad(angle)])
    )
    action = teleop.get_action()
    assert action['J7.pos'] == pytest.approx(direction * np.deg2rad(angle))
    expected_grip = 1 - 50 / 84 if modifier == 'none' else grip
    assert action['gripper.pos'] == pytest.approx(expected_grip)
    now[0] += 5
    assert teleop.get_action()['gripper.pos'] == pytest.approx(expected_grip)


def test_removal_ctrl_takes_priority_and_tighten_ignores_modifiers(monkeypatch):
    now = [10.0]
    monkeypatch.setattr('lerobot_robot_ufactory.teleoperators.gello_teleop.gello_teleop.time.monotonic', lambda: now[0])
    teleop, _ = make_teleop()
    teleop.config.joint7_keyboard_mode_enabled = True
    for name in ['shift_l', 'ctrl_l', 'ctrl_r']:
        update_gello_joint_mode_key(teleop, SimpleNamespace(name=name), True)
    update_gello_joint_mode_key(teleop, SimpleNamespace(name='ctrl_l'), False)
    teleop.set_joint7_keyboard_key(1, True)
    assert teleop._keyboard_j7_angle_deg == 10
    teleop.set_joint7_keyboard_key(1, False)
    teleop._reset_joint_control_mode()
    teleop.config.joint7_keyboard_hold_to_rotate = True
    teleop.set_joint7_keyboard_key(1, True, modifier='ctrl')
    teleop.apply_pending_joint_control_mode(observation(np.zeros(7)))
    now[0] += 0.5
    assert teleop.get_action()['J7.pos'] == pytest.approx(np.deg2rad(30))
    now[0] += 2.5
    assert teleop.get_action()['J7.pos'] == pytest.approx(np.deg2rad(180))


@pytest.mark.parametrize('direction', [1, -1])
@pytest.mark.parametrize('modifier,angle', [('none', 150), ('ctrl', 10), ('shift', 60)])
@pytest.mark.parametrize('gripper_mode', ['keyboard', 'gello'])
def test_keyboard_j7_waits_for_cached_actual_angle(monkeypatch, direction, modifier, angle, gripper_mode):
    now = [10.0]
    monkeypatch.setattr('lerobot_robot_ufactory.teleoperators.gello_teleop.gello_teleop.time.monotonic', lambda: now[0])
    teleop, _ = make_teleop(gripper_mode=gripper_mode)
    teleop.config.joint7_keyboard_mode_enabled = True
    teleop.config.joint7_keyboard_angle_deg = 150
    teleop.config.joint7_keyboard_speed_deg_s = 75
    pose = np.zeros(7)
    pose[6] = direction * 0.3
    reads = []

    def cached_state(anchor):
        reads.append(anchor)
        return SimpleNamespace(joint_positions=pose.copy())

    robot = SimpleNamespace(_control_space='joint', latest_state_before=cached_state)
    grip = teleop.get_action()['gripper.pos']
    teleop.set_joint7_keyboard_key(direction, True, modifier=modifier)
    teleop.set_joint7_keyboard_key(direction, False)
    apply_pending_gello_joint_mode(robot, teleop, {})
    target = pose[6] + direction * np.deg2rad(angle)
    now[0] += angle / 75 + 1
    # The planned duration elapsed, but the follower has not moved at all.
    apply_pending_gello_joint_mode(robot, teleop, {})
    action = teleop.get_action()
    assert action['J7.pos'] == pytest.approx(target)
    assert action['gripper.pos'] == pytest.approx(grip)
    assert teleop.keyboard_joint7_busy()
    assert not teleop.set_joint7_keyboard_key(-direction, True)
    teleop.set_joint7_keyboard_key(-direction, False)

    # Keep commanding the endpoint while the real arm catches up.
    pose[6] = target - direction * np.deg2rad(1)
    apply_pending_gello_joint_mode(robot, teleop, {})
    assert teleop.get_action()['gripper.pos'] == pytest.approx(grip)
    assert teleop.keyboard_joint7_busy()
    pose[6] = target - direction * np.deg2rad(0.25)
    apply_pending_gello_joint_mode(robot, teleop, {})
    action = teleop.get_action()
    assert action['J7.pos'] == pytest.approx(target)
    assert action['gripper.pos'] == pytest.approx(1 - 50 / 84 if modifier == 'none' else grip)
    assert not teleop.keyboard_joint7_busy()
    assert len(reads) == 4
    # No extra feedback reads once completion has been consumed.
    apply_pending_gello_joint_mode(robot, teleop, {})
    assert len(reads) == 4
    assert teleop.set_joint7_keyboard_key(-direction, True)


@pytest.mark.parametrize('failure', ['stale', 'nan'])
def test_keyboard_j7_invalid_feedback_cannot_finish_turn(monkeypatch, failure):
    now = [10.0]
    monkeypatch.setattr('lerobot_robot_ufactory.teleoperators.gello_teleop.gello_teleop.time.monotonic', lambda: now[0])
    teleop, _ = make_teleop(gripper_mode='keyboard')
    teleop.config.joint7_keyboard_mode_enabled = True
    grip = teleop.get_action()['gripper.pos']
    teleop.set_joint7_keyboard_key(1, True)
    teleop.apply_pending_joint_control_mode(observation(np.zeros(7)))
    now[0] += 10

    def invalid_state(_):
        if failure == 'stale':
            raise TimeoutError('stale RT pose')
        return SimpleNamespace(joint_positions=[0] * 6 + [float('nan')])

    robot = SimpleNamespace(_control_space='joint', latest_state_before=invalid_state)
    with pytest.raises(TimeoutError if failure == 'stale' else ValueError):
        apply_pending_gello_joint_mode(robot, teleop, {})
    assert teleop.keyboard_joint7_busy()
    assert teleop.get_action()['gripper.pos'] == pytest.approx(grip)
    press_s(teleop)
    teleop.apply_pending_joint_control_mode(observation(np.zeros(7)))
    assert not teleop.keyboard_joint7_busy()
    assert not teleop.keyboard_joint7_feedback_pending()


def test_keyboard_j7_full_turn_needs_absolute_arrival_and_pause_cancels(monkeypatch):
    now = [10.0]
    monkeypatch.setattr('lerobot_robot_ufactory.teleoperators.gello_teleop.gello_teleop.time.monotonic', lambda: now[0])
    teleop, _ = make_teleop()
    teleop.config.joint7_keyboard_mode_enabled = True
    teleop.config.joint7_keyboard_angle_deg = 360
    teleop.set_joint7_keyboard_key(1, True)
    teleop.apply_pending_joint_control_mode(observation(np.zeros(7)))
    now[0] += 10
    teleop.update_keyboard_joint7_observation(observation(np.zeros(7)))
    teleop.get_action()
    assert teleop.keyboard_joint7_busy()
    teleop.set_teleop_enabled(False)
    assert not teleop.keyboard_joint7_busy()
    assert not teleop.keyboard_joint7_feedback_pending()
