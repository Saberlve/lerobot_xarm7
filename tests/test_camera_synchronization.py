from time import sleep

import numpy as np
import pytest

from lerobot_robot_ufactory.cameras.synchronization import TimestampedCameraBuffer


class _Camera:
    def __init__(self, delay_s: float = 0.0):
        self.delay_s = delay_s
        self.index = 0

    def __str__(self):
        return "fake-rgb"

    def async_read(self, timeout_ms: float):
        sleep(self.delay_s)
        self.index += 1
        return np.full((2, 3, 3), self.index % 255 or 1, dtype=np.uint8)


def test_timestamped_rgb_buffer_pairs_a_new_frame_to_anchor():
    buffer = TimestampedCameraBuffer(_Camera(), history_size=8)
    buffer.start()
    try:
        from time import perf_counter

        anchor = perf_counter()
        frame, timing = buffer.nearest(anchor, max_skew_ms=20, wait_ms=200)
        assert frame.shape == (2, 3, 3)
        assert timing["sync_target_monotonic_s"] == anchor
        assert timing["sync_offset_ms"] <= 20
        frame[:] = 0
        later, _ = buffer.nearest(perf_counter(), max_skew_ms=20, wait_ms=200)
        assert later[0, 0, 0] > 0
    finally:
        buffer.stop()


def test_timestamped_rgb_buffer_rejects_a_frame_outside_bound():
    buffer = TimestampedCameraBuffer(_Camera(delay_s=0.03), history_size=4)
    buffer.start()
    try:
        from time import perf_counter

        with pytest.raises(TimeoutError, match="Nearest RGB frame"):
            buffer.nearest(perf_counter(), max_skew_ms=2, wait_ms=150)
    finally:
        buffer.stop()
