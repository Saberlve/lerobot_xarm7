"""Fixed-rate UFACTORY teleoperation isolated from observation and recording work."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque

from lerobot.utils.robot_utils import precise_sleep


logger = logging.getLogger(__name__)

GRIPPER_CURRENT_FEEDBACK_KEY = "gripper.current_ma"


def map_gripper_current_feedback(
    sample,
    *,
    gain: float,
    output_sign: int,
    output_limit_ma: float,
) -> float:
    """Map one cached G2 sample to a safe signed GELLO current target."""
    try:
        gain = float(gain)
        output_limit_ma = float(output_limit_ma)
    except (TypeError, ValueError):
        return 0.0
    if (
        not math.isfinite(gain)
        or gain < 0.0
        or isinstance(output_sign, bool)
        or output_sign not in (-1, 1)
        or not math.isfinite(output_limit_ma)
        or output_limit_ma <= 0.0
    ):
        return 0.0
    if (
        sample is None
        or getattr(sample, "available", False) is not True
        or getattr(sample, "stale", True) is not False
        or getattr(sample, "current_ma", None) is None
        or isinstance(getattr(sample, "current_ma", None), bool)
        or getattr(sample, "error", None) is not None
        or getattr(sample, "reason", None) == "cache_busy"
    ):
        return 0.0
    try:
        current_ma = float(sample.current_ma)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(current_ma):
        return 0.0

    mapped_ma = current_ma * gain * output_sign
    if not math.isfinite(mapped_ma):
        return 0.0
    return max(-output_limit_ma, min(output_limit_ma, mapped_ma))


class RealtimeTeleopController:
    def __init__(
        self,
        robot,
        teleop,
        teleop_action_processor,
        robot_action_processor,
        fps: int,
        initial_observation: dict,
        record_timing: bool = False,
    ) -> None:
        self.robot = robot
        self.teleop = teleop
        self.teleop_action_processor = teleop_action_processor
        self.robot_action_processor = robot_action_processor
        self.period_s = 1.0 / fps
        self._observation = initial_observation
        self._latest_action = None
        self._action_history = deque(maxlen=max(16, fps * 2))
        # Keep episode timing separately from the bounded lookup history.
        self._record_timing = record_timing
        self._action_timings = []
        self._exception = None
        self._heartbeat = time.perf_counter()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._first_action = threading.Event()
        self._thread = threading.Thread(target=self._run, name="uf-servoj-control", daemon=True)

        teleop_config = getattr(teleop, "config", None)
        self._gripper_feedback_enabled = bool(
            getattr(teleop_config, "gripper_force_feedback_enabled", False)
        )
        self._gripper_feedback_gain = getattr(
            teleop_config, "gripper_feedback_gain", None
        )
        self._gripper_feedback_output_sign = getattr(
            teleop_config, "gripper_feedback_output_sign", None
        )
        self._gripper_feedback_output_limit_ma = getattr(
            teleop_config, "gripper_current_limit_ma", None
        )
        self._feedback_error_reported = False

    def start(self) -> None:
        if self._gripper_feedback_enabled:
            start_feedback = getattr(self.teleop, "start_feedback", None)
            if callable(start_feedback):
                try:
                    start_feedback()
                except Exception:
                    # Feedback is optional; current-mode setup failure must not
                    # prevent the existing position teleoperation path.
                    logger.exception(
                        "GELLO gripper feedback could not be started; continuing without it"
                    )
        self._thread.start()
        if not self._first_action.wait(timeout=2.0):
            self.raise_if_failed()
            raise RuntimeError("Timed out waiting for the first realtime joint action")
        self.raise_if_failed()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._safe_stop_feedback_output()
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

    def action_timings(self) -> list[dict[str, int]]:
        """Return a stable copy of the full command timeline for an episode."""
        with self._lock:
            return [dict(item) for item in self._action_timings]

    def raise_if_failed(self) -> None:
        if self._exception is not None:
            raise RuntimeError("Realtime joint control thread failed") from self._exception

    def _update_gripper_feedback(self) -> None:
        if not self._gripper_feedback_enabled:
            return

        feedback_ma = 0.0
        try:
            get_sample = getattr(self.robot, "get_gripper_current_sample", None)
            sample = get_sample() if callable(get_sample) else None
            feedback_ma = map_gripper_current_feedback(
                sample,
                gain=self._gripper_feedback_gain,
                output_sign=self._gripper_feedback_output_sign,
                output_limit_ma=self._gripper_feedback_output_limit_ma,
            )
        except Exception:
            # Cache reads are expected to be non-blocking, but every failure is
            # still converted to this frame's zero-current command.
            feedback_ma = 0.0

        try:
            self.teleop.send_feedback({GRIPPER_CURRENT_FEEDBACK_KEY: feedback_ma})
            self._feedback_error_reported = False
        except Exception:
            # Haptic output is isolated from the arm position-control path.
            if not self._feedback_error_reported:
                logger.exception(
                    "GELLO gripper feedback update failed; normal teleoperation continues"
                )
                self._feedback_error_reported = True
            try:
                self.teleop.send_feedback({GRIPPER_CURRENT_FEEDBACK_KEY: 0.0})
            except Exception:
                pass

    def _safe_stop_feedback_output(self) -> None:
        if not self._gripper_feedback_enabled:
            return
        try:
            self.teleop.send_feedback({GRIPPER_CURRENT_FEEDBACK_KEY: 0.0})
        except Exception:
            pass
        stop_feedback = getattr(self.teleop, "stop_feedback", None)
        if callable(stop_feedback):
            try:
                stop_feedback()
            except Exception:
                logger.exception("Failed to stop GELLO gripper feedback cleanly")

    def _run(self) -> None:
        next_tick = time.perf_counter()
        try:
            while not self._stop.is_set():
                with self._lock:
                    observation = self._observation
                    heartbeat = self._heartbeat
                if time.perf_counter() - heartbeat > 1.0:
                    raise RuntimeError("Recording/teleop owner heartbeat timed out")
                read_start_ns = time.perf_counter_ns()
                action = self.teleop.get_action()
                read_end_ns = time.perf_counter_ns()
                processed = self.teleop_action_processor((action, observation))
                command = self.robot_action_processor((processed, observation))
                send_start_ns = time.perf_counter_ns()
                sent = self.robot.send_action(command)
                send_end_ns = time.perf_counter_ns()
                effective = sent if isinstance(sent, dict) else command
                sent_at_s = send_end_ns / 1_000_000_000
                with self._lock:
                    self._latest_action = dict(effective)
                    self._action_history.append((sent_at_s, dict(effective)))
                    if self._record_timing:
                        self._action_timings.append(
                            {
                                "action_index": len(self._action_timings),
                                "gello_read_start_ns": read_start_ns,
                                "gello_read_end_ns": read_end_ns,
                                "command_send_start_ns": send_start_ns,
                                "command_send_end_ns": send_end_ns,
                            }
                        )
                self._update_gripper_feedback()
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
        finally:
            self._safe_stop_feedback_output()
