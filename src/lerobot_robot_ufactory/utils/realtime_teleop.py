"""Fixed-rate UFACTORY teleoperation isolated from observation and recording work."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass

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


@dataclass(frozen=True)
class GripperFeedbackDiagnostic:
    timestamp_monotonic_s: float | None
    raw_current_ma: float | None
    bias_corrected_ma: float | None
    input_clamped_ma: float | None
    deadzone_output_ma: float | None
    filtered_ma: float | None
    target_ma: float
    command_ma: float
    sample_age_s: float | None
    gripper_state: int | None
    sample_available: bool
    sample_stale: bool
    feedback_enabled: bool
    feedback_active: bool
    reason: str


class GripperFeedbackProcessor:
    """Stateful Phase 4 current conditioning with immediate-zero fail-safe."""

    def __init__(
        self,
        *,
        bias_ma: float,
        deadzone_ma: float,
        input_limit_ma: float,
        ema_beta: float,
        gain: float,
        output_sign: int,
        output_limit_ma: float,
        slew_rate_ma_s: float,
        timeout_s: float,
    ) -> None:
        values = {
            "bias_ma": bias_ma,
            "deadzone_ma": deadzone_ma,
            "input_limit_ma": input_limit_ma,
            "ema_beta": ema_beta,
            "gain": gain,
            "output_limit_ma": output_limit_ma,
            "slew_rate_ma_s": slew_rate_ma_s,
            "timeout_s": timeout_s,
        }
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite number")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if deadzone_ma < 0:
            raise ValueError("deadzone_ma must be non-negative")
        if input_limit_ma <= 0:
            raise ValueError("input_limit_ma must be positive")
        if not 0 <= ema_beta < 1:
            raise ValueError("ema_beta must be in [0, 1)")
        if gain < 0:
            raise ValueError("gain must be non-negative")
        if isinstance(output_sign, bool) or output_sign not in (-1, 1):
            raise ValueError("output_sign must be either -1 or 1")
        if output_limit_ma <= 0:
            raise ValueError("output_limit_ma must be positive")
        if slew_rate_ma_s <= 0:
            raise ValueError("slew_rate_ma_s must be positive")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")

        self.bias_ma = float(bias_ma)
        self.deadzone_ma = float(deadzone_ma)
        self.input_limit_ma = float(input_limit_ma)
        self.ema_beta = float(ema_beta)
        self.gain = float(gain)
        self.output_sign = int(output_sign)
        self.output_limit_ma = float(output_limit_ma)
        self.slew_rate_ma_s = float(slew_rate_ma_s)
        self.timeout_s = float(timeout_s)
        self._lock = threading.Lock()
        self._filtered_ma = 0.0
        self._previous_command_ma = 0.0
        self._last_valid_monotonic_s = None
        self._diagnostic = self._empty_diagnostic("not_started")

    def _empty_diagnostic(
        self,
        reason: str,
        *,
        timestamp_monotonic_s: float | None = None,
        raw_current_ma: float | None = None,
        sample_age_s: float | None = None,
        gripper_state: int | None = None,
        sample_available: bool = False,
        sample_stale: bool = False,
    ) -> GripperFeedbackDiagnostic:
        return GripperFeedbackDiagnostic(
            timestamp_monotonic_s=timestamp_monotonic_s,
            raw_current_ma=raw_current_ma,
            bias_corrected_ma=None,
            input_clamped_ma=None,
            deadzone_output_ma=None,
            filtered_ma=None,
            target_ma=0.0,
            command_ma=0.0,
            sample_age_s=sample_age_s,
            gripper_state=gripper_state,
            sample_available=sample_available,
            sample_stale=sample_stale,
            feedback_enabled=True,
            feedback_active=False,
            reason=reason,
        )

    @staticmethod
    def _finite_optional(value) -> float | None:
        if isinstance(value, bool) or value is None:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    def _reset_filter_locked(self) -> None:
        self._filtered_ma = 0.0
        self._previous_command_ma = 0.0
        self._last_valid_monotonic_s = None

    def reset(self, reason: str = "reset") -> None:
        """Clear EMA and slew state and publish an immediate zero diagnostic."""
        now_s = time.monotonic()
        with self._lock:
            self._reset_filter_locked()
            self._diagnostic = self._empty_diagnostic(
                reason, timestamp_monotonic_s=now_s
            )

    def fail_safe_zero(self, reason: str, sample=None) -> GripperFeedbackDiagnostic:
        """Immediately zero output and discard all state from previous samples."""
        now_s = time.monotonic()
        raw_current_ma = self._finite_optional(
            None if sample is None else getattr(sample, "current_ma", None)
        )
        sample_age_s = self._finite_optional(
            None if sample is None else getattr(sample, "age_s", None)
        )
        with self._lock:
            self._reset_filter_locked()
            self._diagnostic = self._empty_diagnostic(
                reason,
                timestamp_monotonic_s=now_s,
                raw_current_ma=raw_current_ma,
                sample_age_s=sample_age_s,
                gripper_state=(
                    None if sample is None else getattr(sample, "gripper_state", None)
                ),
                sample_available=bool(
                    sample is not None and getattr(sample, "available", False)
                ),
                sample_stale=bool(
                    sample is not None and getattr(sample, "stale", False)
                ),
            )
            return self._diagnostic

    def _invalid_reason(self, sample) -> str | None:
        if sample is None:
            return "no_sample"
        sample_reason = getattr(sample, "reason", None)
        if sample_reason == "cache_busy":
            return "cache_busy"
        if getattr(sample, "error", None) is not None:
            return "monitor_error"
        if getattr(sample, "stale", False) is True:
            return "stale"
        if getattr(sample, "available", False) is not True:
            return str(sample_reason or "unavailable")
        current_ma = getattr(sample, "current_ma", None)
        if current_ma is None:
            return "current_unavailable"
        if self._finite_optional(current_ma) is None:
            return "current_not_finite"
        age_s = getattr(sample, "age_s", None)
        if age_s is None:
            return "sample_age_unavailable"
        finite_age_s = self._finite_optional(age_s)
        if finite_age_s is None or finite_age_s < 0:
            return "sample_age_invalid"
        if finite_age_s > self.timeout_s:
            return "sample_timeout"
        return None

    def process(
        self, sample, *, now_monotonic_s: float | None = None
    ) -> GripperFeedbackDiagnostic:
        now_s = time.monotonic() if now_monotonic_s is None else now_monotonic_s
        if isinstance(now_s, bool) or not isinstance(now_s, (int, float)):
            return self.fail_safe_zero("monotonic_time_invalid", sample)
        now_s = float(now_s)
        if not math.isfinite(now_s):
            return self.fail_safe_zero("monotonic_time_invalid", sample)

        invalid_reason = self._invalid_reason(sample)
        if invalid_reason is not None:
            raw_current_ma = self._finite_optional(
                None if sample is None else getattr(sample, "current_ma", None)
            )
            sample_age_s = self._finite_optional(
                None if sample is None else getattr(sample, "age_s", None)
            )
            with self._lock:
                self._reset_filter_locked()
                self._diagnostic = self._empty_diagnostic(
                    invalid_reason,
                    timestamp_monotonic_s=now_s,
                    raw_current_ma=raw_current_ma,
                    sample_age_s=sample_age_s,
                    gripper_state=(
                        None
                        if sample is None
                        else getattr(sample, "gripper_state", None)
                    ),
                    sample_available=bool(
                        sample is not None and getattr(sample, "available", False)
                    ),
                    sample_stale=bool(
                        sample is not None and getattr(sample, "stale", False)
                    ),
                )
                return self._diagnostic

        raw_current_ma = float(sample.current_ma)
        sample_age_s = float(sample.age_s)
        with self._lock:
            previous_time_s = self._last_valid_monotonic_s
            recovered_after_gap = False
            if previous_time_s is not None:
                elapsed_s = now_s - previous_time_s
                if (
                    not math.isfinite(elapsed_s)
                    or elapsed_s < 0.0
                    or elapsed_s > self.timeout_s
                ):
                    self._reset_filter_locked()
                    previous_time_s = None
                    recovered_after_gap = True

            bias_corrected_ma = raw_current_ma - self.bias_ma
            input_clamped_ma = max(
                -self.input_limit_ma,
                min(self.input_limit_ma, bias_corrected_ma),
            )
            magnitude_ma = abs(input_clamped_ma)
            if magnitude_ma <= self.deadzone_ma:
                deadzone_output_ma = 0.0
            else:
                deadzone_output_ma = math.copysign(
                    magnitude_ma - self.deadzone_ma, input_clamped_ma
                )

            self._filtered_ma = (
                self.ema_beta * self._filtered_ma
                + (1.0 - self.ema_beta) * deadzone_output_ma
            )
            target_ma = self._filtered_ma * self.gain * self.output_sign
            dt_s = 0.0 if previous_time_s is None else now_s - previous_time_s
            delta_max_ma = self.slew_rate_ma_s * dt_s
            command_ma = max(
                self._previous_command_ma - delta_max_ma,
                min(
                    self._previous_command_ma + delta_max_ma,
                    target_ma,
                ),
            )
            command_ma = max(
                -self.output_limit_ma,
                min(self.output_limit_ma, command_ma),
            )
            self._previous_command_ma = command_ma
            self._last_valid_monotonic_s = now_s
            self._diagnostic = GripperFeedbackDiagnostic(
                timestamp_monotonic_s=now_s,
                raw_current_ma=raw_current_ma,
                bias_corrected_ma=bias_corrected_ma,
                input_clamped_ma=input_clamped_ma,
                deadzone_output_ma=deadzone_output_ma,
                filtered_ma=self._filtered_ma,
                target_ma=target_ma,
                command_ma=command_ma,
                sample_age_s=sample_age_s,
                gripper_state=getattr(sample, "gripper_state", None),
                sample_available=True,
                sample_stale=False,
                feedback_enabled=True,
                feedback_active=True,
                reason="valid_after_gap" if recovered_after_gap else "valid",
            )
            return self._diagnostic

    def get_diagnostic(self) -> GripperFeedbackDiagnostic:
        """Return the latest immutable diagnostic snapshot without hardware I/O."""
        with self._lock:
            return self._diagnostic


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
        self._gripper_feedback_processor = None
        if self._gripper_feedback_enabled:
            self._gripper_feedback_processor = GripperFeedbackProcessor(
                bias_ma=teleop_config.gripper_feedback_bias_ma,
                deadzone_ma=teleop_config.gripper_feedback_deadzone_ma,
                input_limit_ma=teleop_config.gripper_feedback_input_limit_ma,
                ema_beta=teleop_config.gripper_feedback_ema_beta,
                gain=teleop_config.gripper_feedback_gain,
                output_sign=teleop_config.gripper_feedback_output_sign,
                output_limit_ma=teleop_config.gripper_feedback_output_limit_ma,
                slew_rate_ma_s=teleop_config.gripper_feedback_slew_rate_ma_s,
                timeout_s=teleop_config.gripper_feedback_timeout_s,
            )
        self._disabled_feedback_diagnostic = GripperFeedbackDiagnostic(
            timestamp_monotonic_s=None,
            raw_current_ma=None,
            bias_corrected_ma=None,
            input_clamped_ma=None,
            deadzone_output_ma=None,
            filtered_ma=None,
            target_ma=0.0,
            command_ma=0.0,
            sample_age_s=None,
            gripper_state=None,
            sample_available=False,
            sample_stale=False,
            feedback_enabled=False,
            feedback_active=False,
            reason="disabled",
        )
        self._gripper_feedback_output_started = False
        self._feedback_error_reported = False

    def start(self) -> None:
        if self._gripper_feedback_enabled:
            start_feedback = getattr(self.teleop, "start_feedback", None)
            if callable(start_feedback):
                try:
                    start_feedback()
                    self._gripper_feedback_output_started = True
                except Exception:
                    # Feedback is optional; current-mode setup failure must not
                    # prevent the existing position teleoperation path.
                    logger.exception(
                        "GELLO gripper feedback could not be started; continuing without it"
                    )
                    self._gripper_feedback_processor.fail_safe_zero(
                        "output_start_failed"
                    )
            else:
                self._gripper_feedback_processor.fail_safe_zero(
                    "output_start_unsupported"
                )
        self._thread.start()
        if not self._first_action.wait(timeout=2.0):
            self._stop.set()
            self._safe_stop_feedback_output()
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

    def get_gripper_feedback_diagnostic(self) -> GripperFeedbackDiagnostic:
        """Return the latest conditioned-feedback state without hardware I/O."""
        if self._gripper_feedback_processor is None:
            return self._disabled_feedback_diagnostic
        return self._gripper_feedback_processor.get_diagnostic()

    def reset_gripper_feedback_state(self, reason: str = "reset") -> None:
        if self._gripper_feedback_processor is not None:
            self._gripper_feedback_processor.reset(reason)

    def _update_gripper_feedback(self) -> None:
        if not self._gripper_feedback_enabled:
            return

        processor = self._gripper_feedback_processor
        if not self._gripper_feedback_output_started:
            diagnostic = processor.fail_safe_zero("output_inactive")
        else:
            try:
                get_sample = getattr(self.robot, "get_gripper_current_sample", None)
                sample = get_sample() if callable(get_sample) else None
                diagnostic = processor.process(sample)
            except Exception:
                # Cache reads are expected to be non-blocking, but every
                # failure is converted to this frame's zero-current command.
                diagnostic = processor.fail_safe_zero("sample_read_exception")

        try:
            self.teleop.send_feedback(
                {GRIPPER_CURRENT_FEEDBACK_KEY: diagnostic.command_ma}
            )
            self._feedback_error_reported = False
        except Exception:
            # Haptic output is isolated from the arm position-control path.
            if not self._feedback_error_reported:
                logger.exception(
                    "GELLO gripper feedback update failed; normal teleoperation continues"
                )
                self._feedback_error_reported = True
            processor.fail_safe_zero("feedback_send_exception")
            try:
                self.teleop.send_feedback({GRIPPER_CURRENT_FEEDBACK_KEY: 0.0})
            except Exception:
                pass
            return

        get_output_status = getattr(self.teleop, "get_feedback_output_status", None)
        if callable(get_output_status):
            try:
                output_active, output_error = get_output_status()
            except Exception:
                output_active, output_error = False, "output_status_exception"
            if not output_active:
                self._gripper_feedback_output_started = False
                processor.fail_safe_zero(
                    "output_error" if output_error else "output_inactive"
                )

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
        self._gripper_feedback_output_started = False
        self.reset_gripper_feedback_state("controller_stopped")

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
