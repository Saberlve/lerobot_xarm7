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


def make_teleop(*, enabled=True, gripper_mode="gello"):
    config = GelloTeleopConfig(
        joint7_only_mode_enabled=enabled,
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
