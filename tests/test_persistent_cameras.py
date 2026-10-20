"""Suspend only layout RGB using fake camera objects; no USB or robot calls."""
from threading import Condition, Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lerobot_robot_ufactory.robots.uf_robot.uf_robot import UFRobot


def robot_fixture():
    def camera(kind):
        value = SimpleNamespace(config=SimpleNamespace(type=kind), is_connected=True)
        value.disconnect = Mock(side_effect=lambda: setattr(value, 'is_connected', False))
        value.connect = Mock(side_effect=lambda: setattr(value, 'is_connected', True))
        return value
    thread = Mock()
    thread.is_alive.return_value = False
    return SimpleNamespace(
        cameras={'rgb': camera('intelrealsense'), 'photon': camera('photon')},
        _camera_stop_event=Event(), _camera_cond=Condition(), _camera_thread=thread,
        _stop_rgb_sync_buffers=Mock(), _start_camera_streams=Mock(),
        _latest_camera_obs={'old': 1}, _latest_camera_timing={'old': 1},
        _latest_camera_anchor_s=1., _camera_updated=True,
    )


def test_suspend_and_resume_preserve_photon_and_discard_old_rgb():
    robot = robot_fixture()
    UFRobot.suspend_cameras(robot, ('rgb',))
    assert robot.cameras['photon'].is_connected
    robot.cameras['photon'].disconnect.assert_not_called()
    assert not robot.cameras['rgb'].is_connected
    assert robot._latest_camera_obs == {}
    assert robot._latest_camera_timing == {}
    assert robot._latest_camera_anchor_s is None
    assert not robot._camera_updated
    UFRobot.resume_cameras(robot, ('rgb',))
    robot.cameras['rgb'].connect.assert_called_once()
    robot.cameras['photon'].connect.assert_not_called()
    robot._start_camera_streams.assert_called_once()


def test_unstopped_reader_blocks_device_release():
    robot = robot_fixture()
    robot._camera_thread.is_alive.return_value = True
    with pytest.raises(RuntimeError, match='not released'):
        UFRobot.suspend_cameras(robot, ('rgb',))
    robot.cameras['rgb'].disconnect.assert_not_called()
    robot._stop_rgb_sync_buffers.assert_not_called()


def test_lost_tactile_connection_blocks_resume():
    robot = robot_fixture()
    robot.cameras['photon'].is_connected = False
    with pytest.raises(ConnectionError, match='retained camera'):
        UFRobot.resume_cameras(robot, ('rgb',))
    robot._start_camera_streams.assert_not_called()
