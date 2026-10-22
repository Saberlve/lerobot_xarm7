"""Reset ordering for the mock teleoperator's real xArm connection, without hardware."""
from types import SimpleNamespace

import pytest

from lerobot_robot_ufactory.teleoperators.uf_mock_teleop.uf_mock_teleop import UFMockTeleop


@pytest.mark.parametrize('open_code', [0, 23])
def test_mock_reset_waits_for_open_and_aborts_on_gripper_failure(open_code):
    calls = []

    def open_gripper(position, *, wait):
        assert position == 800 and wait is True
        calls.append('open')
        return open_code

    def move(*pose, speed, wait):
        assert pose == (1, 2, 3, 4, 5, 6) and speed == 20 and wait is True
        calls.append('move')
        return 0

    teleop = UFMockTeleop.__new__(UFMockTeleop)
    teleop.config = SimpleNamespace(gripper_type=1, initial_tcp_pose=[1, 2, 3, 4, 5, 6])
    teleop.arm = SimpleNamespace(set_gripper_position=open_gripper, set_position=move)
    if open_code:
        with pytest.raises(RuntimeError, match='open gripper before reset'):
            teleop._reset_arm_to_initial(20)
        assert calls == ['open']
    else:
        assert teleop._reset_arm_to_initial(20) == 0
        assert calls == ['open', 'move']


def test_mock_reset_without_gripper_moves_directly():
    calls = []
    teleop = UFMockTeleop.__new__(UFMockTeleop)
    teleop.config = SimpleNamespace(gripper_type=0, initial_tcp_pose=[0] * 6)
    teleop.arm = SimpleNamespace(set_position=lambda *args, **kwargs: calls.append('move') or 0)
    assert teleop._reset_arm_to_initial(20) == 0
    assert calls == ['move']
