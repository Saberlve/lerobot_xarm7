"""Fixed-rate UFACTORY teleoperation isolated from observation and recording work."""

from __future__ import annotations

import threading
import time
from collections import deque

from lerobot.utils.robot_utils import precise_sleep


class RealtimeTeleopController:
    def __init__(
        self,
        robot,
        teleop,
        teleop_action_processor,
        robot_action_processor,
        fps: int,
        initial_observation: dict,
    ) -> None:
        self.robot = robot
        self.teleop = teleop
        self.teleop_action_processor = teleop_action_processor
        self.robot_action_processor = robot_action_processor
        self.period_s = 1.0 / fps
        self._observation = initial_observation
        self._latest_action = None
        self._action_history = deque(maxlen=max(16, fps * 2))
        self._exception = None
        self._heartbeat = time.perf_counter()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._first_action = threading.Event()
        self._thread = threading.Thread(target=self._run, name="uf-servoj-control", daemon=True)

    def start(self) -> None:
        self._thread.start()
        if not self._first_action.wait(timeout=2.0):
            self.raise_if_failed()
            raise RuntimeError("Timed out waiting for the first realtime ServoJ action")
        self.raise_if_failed()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.raise_if_failed()

    def update_observation(self, observation: dict) -> None:
        with self._lock:
            self._observation = observation
            self._heartbeat = time.perf_counter()

    def heartbeat(self) -> None:
        with self._lock:
            self._heartbeat = time.perf_counter()

    def latest_action(self) -> dict:
        self.raise_if_failed()
        with self._lock:
            if self._latest_action is None:
                raise RuntimeError("Realtime controller has not sent an action")
            return dict(self._latest_action)

    def action_at(self, monotonic_s: float) -> dict:
        """Return the command active at a sampled observation time."""
        action, _ = self.action_sample_at(monotonic_s)
        return action

    def action_sample_at(self, monotonic_s: float) -> tuple[dict, float]:
        """Return the command and send timestamp active at observation time."""
        self.raise_if_failed()
        with self._lock:
            if not self._action_history:
                raise RuntimeError("Realtime controller has not sent an action")
            selected_time, selected = self._action_history[0]
            for sent_at_s, action in reversed(self._action_history):
                if sent_at_s <= monotonic_s:
                    selected_time = sent_at_s
                    selected = action
                    break
            return dict(selected), selected_time

    def raise_if_failed(self) -> None:
        if self._exception is not None:
            raise RuntimeError("Realtime ServoJ control thread failed") from self._exception

    def _run(self) -> None:
        next_tick = time.perf_counter()
        try:
            while not self._stop.is_set():
                with self._lock:
                    observation = self._observation
                    heartbeat = self._heartbeat
                if time.perf_counter() - heartbeat > 1.0:
                    raise RuntimeError("Recording/teleop owner heartbeat timed out")
                action = self.teleop.get_action()
                processed = self.teleop_action_processor((action, observation))
                command = self.robot_action_processor((processed, observation))
                sent = self.robot.send_action(command)
                effective = sent if isinstance(sent, dict) else command
                sent_at_s = time.perf_counter()
                with self._lock:
                    self._latest_action = dict(effective)
                    self._action_history.append((sent_at_s, dict(effective)))
                self._first_action.set()

                next_tick += self.period_s
                now = time.perf_counter()
                if next_tick <= now:
                    missed = int((now - next_tick) / self.period_s) + 1
                    next_tick += missed * self.period_s
                precise_sleep(max(next_tick - time.perf_counter(), 0.0))
        except BaseException as exc:
            self._exception = exc
            self._first_action.set()
            self._stop.set()
