"""Device initialization concurrency using fake cameras only."""
from threading import Event
from types import SimpleNamespace

import pytest

from lerobot_robot_ufactory.cameras.utils import connect_cameras


def camera(kind, connect):
    return SimpleNamespace(config=SimpleNamespace(type=kind), connect=connect, is_connected=True)


def test_rgb_warmups_overlap_each_other_and_serial_tactile_initialization():
    started = [Event(), Event()]
    tactile_ready = Event()
    events = []

    def rgb(index):
        started[index].set()
        assert started[1 - index].wait(2), 'RGB warmups must overlap'
        assert tactile_ready.wait(2), 'Tactile initialization must overlap RGB warmup'
        events.append(f'rgb{index}')

    def left():
        assert all(event.wait(2) for event in started)
        events.append('left')

    def right():
        assert 'left' in events
        events.append('right')
        tactile_ready.set()

    cameras = dict(wrist=camera('intelrealsense', lambda: rgb(0)),
                   external=camera('intelrealsense', lambda: rgb(1)),
                   left=camera('photon', left), right=camera('photon', right))
    connect_cameras(cameras)
    assert list(cameras) == ['wrist', 'external', 'left', 'right']
    assert sorted(events) == ['left', 'rgb0', 'rgb1', 'right']


@pytest.mark.parametrize('failure', ['rgb', 'tactile'])
def test_connection_failure_waits_for_rgb_workers_before_returning(failure):
    released = Event()
    completed = Event()

    def rgb():
        assert released.wait(2)
        completed.set()
        if failure == 'rgb':
            raise RuntimeError('rgb failed')

    def tactile():
        released.set()
        if failure == 'tactile':
            raise RuntimeError('tactile failed')

    with pytest.raises(RuntimeError, match=f'{failure} failed'):
        connect_cameras(dict(rgb=camera('intelrealsense', rgb), tactile=camera('photon', tactile)))
    assert completed.is_set()


def test_disconnected_camera_prevents_success():
    disconnected = camera('photon', lambda: None)
    disconnected.is_connected = False
    with pytest.raises(ConnectionError):
        connect_cameras(dict(tactile=disconnected))
    connect_cameras({})
