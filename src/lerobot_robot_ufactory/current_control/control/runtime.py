"""One owner per device; GUI, recorder and network never drive the current loop."""

import logging
import queue
import threading
import time
from collections import deque

import numpy as np

from ..config import current_targets, tuning_slew
from .current import CurrentController
from ..hardware.transport import XL330Transport

EXPERIMENT_MAX_DURATION_S = 10.0
EXPERIMENT_MAX_DISPLACEMENT_DEG = 45.0
TUNING_HEARTBEAT_TIMEOUT_S = 3.0
TUNING_TEMPERATURE_LIMIT_C = 45.0


class CurrentRuntime:
    def __init__(self, profile, *, live=False, transport=None, experimental=False, teleop=False, tuning=False, tuning_slew_a_s=None):
        self.profile = profile
        self.live = live
        self.experimental = experimental
        self.teleop = teleop
        self.tuning = tuning
        if tuning and (not live or not experimental or teleop):
            raise ValueError("Web tuning requires an independent experimental live runtime")
        if tuning_slew_a_s is not None and not tuning:
            raise ValueError("Running slew overrides require web tuning mode")
        configured_slew = getattr(profile, "running_current_slew_a_s", None)
        if configured_slew is not None and not (live and (teleop or tuning)):
            raise ValueError("Configured running slew requires continuous live web/teleop mode")
        selected_slew = tuning_slew_a_s if tuning_slew_a_s is not None else configured_slew
        running_slew = None if selected_slew is None else tuning_slew(selected_slew)
        if teleop and not live:
            raise ValueError("Teleop support requires a live runtime")
        if experimental and not live:
            raise ValueError("Experimental permission is only valid for live run")
        if live:
            profile.validate_live(experimental=experimental)
        self.controller = CurrentController(profile, running_slew_a_s=running_slew)
        self.temperature_limit_c = min(profile.temperature_limit_c, TUNING_TEMPERATURE_LIMIT_C) if tuning or running_slew is not None else profile.temperature_limit_c
        self.transport = transport or XL330Transport(profile)
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self._state = None
        self._latest_record = None
        self._error = None
        self._status = "created"
        self._records = deque(maxlen=2048)
        self.dropped_records = 0
        self._commands = queue.Queue(maxsize=16)
        self._gripper_a = 0.0
        self._gripper_stamp = 0.0
        self._browser_stamp = time.monotonic()

    def start(self):
        if self._thread is not None:
            raise RuntimeError("A runtime cannot be restarted; resolve the fault and reconnect")
        self._thread = threading.Thread(
            target=self._run, name=f"gello-{self.profile.serial}", daemon=True
        )
        self._thread.start()
        if not self._ready.wait(10):
            self.stop()
            raise RuntimeError("GELLO initialization timed out")
        self.raise_if_failed()

    def _run(self):
        p = self.profile
        period = 1 / p.rate_hz
        try:
            self.transport.open()
            if self._stop.is_set():
                return
            if self.live:
                if self.experimental:
                    before = self.transport.state()["position"][:7].copy()
                    self.transport.enable(experimental=True)
                    after = self.transport.state()["position"][:7].copy()
                    difference = np.arctan2(np.sin(after - before), np.cos(after - before))
                    if np.max(np.abs(difference)) > np.deg2rad(2):
                        raise RuntimeError("Encoder pose changed during current-mode initialization")
                    initial_position = after
                else:
                    self.transport.enable()
            start = previous = time.monotonic()
            deadline = start
            count = 0
            self._status = "active" if self.live else "observe"
            while not self._stop.is_set():
                if self.tuning:
                    with self._lock:
                        browser_age = time.monotonic() - self._browser_stamp
                    if browser_age > TUNING_HEARTBEAT_TIMEOUT_S:
                        raise RuntimeError("调参页面失联超过 3 秒，已请求卸力")
                if self.experimental and not (self.teleop or self.tuning) and time.monotonic() - start >= EXPERIMENT_MAX_DURATION_S:
                    self._stop.set()
                    break
                tick = time.monotonic()
                self._service_commands()
                state = self.transport.state()
                now = time.monotonic()
                age = now - state["stamp"]
                dt = period if count == 0 else tick - previous
                if age > p.state_timeout_s:
                    raise RuntimeError("GELLO state read exceeded timeout")
                if max(state["temperature_c"]) >= self.temperature_limit_c:
                    raise RuntimeError(f"GELLO temperature exceeded configured limit ({self.temperature_limit_c:g} C)")
                if min(state["voltage_v"]) < 3.5 or max(state["voltage_v"]) > 6.0:
                    raise RuntimeError("GELLO supply voltage outside 3.5..6.0 V")
                if self.experimental:
                    if not (self.teleop or self.tuning) and np.max(np.abs(state["position"][:7] - initial_position)) > np.deg2rad(EXPERIMENT_MAX_DISPLACEMENT_DEG):
                        raise RuntimeError("Experiment stopped: joint moved more than 45 degrees")
                    if np.any(np.abs(state["current_a"][:7]) > p.limits):
                        raise RuntimeError("Experiment stopped: measured current exceeded limit")
                current, record = self.controller.compute(
                    state["position"], state["velocity"], dt, now - start
                )
                if time.monotonic() - state["stamp"] > p.state_timeout_s:
                    raise RuntimeError("Current computation made the encoder state stale")
                if self.experimental and not (self.teleop or self.tuning) and time.monotonic() - start >= EXPERIMENT_MAX_DURATION_S:
                    self._stop.set()
                if self.live and not self._stop.is_set():
                    gripper = self._gripper_a if now - self._gripper_stamp < 0.1 else 0.0
                    self.transport.currents(current, gripper)
                    # Periodic health-register polling is disabled: its serial burst
                    # delays publishing fresh angles. State-based checks remain above.
                end = time.monotonic()
                if end - tick > p.state_timeout_s:
                    raise RuntimeError("GELLO control transaction exceeded timeout")
                record.update(
                    {
                        "device": p.name,
                        "usb_serial": p.serial,
                        "stamp": state["stamp"],
                        "position": state["position"].tolist(),
                        "velocity": state["velocity"].tolist(),
                        "elapsed_s": now - start,
                        "work_s": end - tick,
                        "interval_s": dt,
                        "state_age_s": end - state["stamp"],
                        "missed_deadline": end - tick > period,
                        "live": self.live,
                        "experimental": self.experimental,
                        "teleop": self.teleop,
                        "tuning": self.tuning,
                        "measured_current_a": state["current_a"],
                        "temperature_c": state["temperature_c"],
                        "voltage_v": state["voltage_v"],
                    }
                )
                with self._lock:
                    self._state = state
                    self._latest_record = record
                    if len(self._records) == self._records.maxlen:
                        self.dropped_records += 1
                    self._records.append(record)
                self._ready.set()
                count += 1
                previous = tick
                deadline += period
                if deadline < end:
                    deadline = end
                self._stop.wait(max(0, deadline - time.monotonic()))
        except BaseException as exc:
            self._error = exc
            self._status = "fault"
            logging.exception("GELLO %s compensation stopped", p.name)
        finally:
            try:
                self.transport.disable()
            except BaseException as exc:
                self._error = RuntimeError(f"{self._error or 'Stop'}; {exc}")
            try:
                self.transport.close()
            except BaseException as exc:
                self._error = self._error or exc
            self._status = "fault" if self._error else "stopped"
            while True:
                try:
                    _, _, done, result = self._commands.get_nowait()
                except queue.Empty:
                    break
                result["error"] = self._error or RuntimeError(
                    "GELLO stopped before command execution"
                )
                done.set()
            self._ready.set()

    def raise_if_failed(self):
        if self._error is not None:
            raise RuntimeError(f"GELLO {self.profile.name} fault: {self._error}") from self._error

    def request(self, operation, argument=None):
        self.raise_if_failed()
        if not self.live or self._status != "active" or self._stop.is_set():
            raise RuntimeError("Current mode requires an active live runtime")
        done = threading.Event()
        result = {}
        self._commands.put_nowait((operation, argument, done, result))
        if not done.wait(2):
            self._stop.set()
            raise RuntimeError("GELLO command timed out; stopping the device")
        if "error" in result:
            raise RuntimeError(str(result["error"])) from result["error"]
        return result.get("value")

    def tuning_heartbeat(self):
        if self.tuning:
            with self._lock:
                self._browser_stamp = time.monotonic()


    def set_current_slew(self, values):
        if not self.tuning:
            raise RuntimeError("Online slew changes require web tuning mode")
        return self.request("slew", tuning_slew(values))

    def set_current_targets(self, currents, damping):
        if not self.tuning:
            raise RuntimeError("Online current changes require web tuning mode")
        targets = current_targets(currents, damping, self.profile.limits)
        return self.request("currents", targets)

    def diagnostics(self):
        """Read the last completed control record without consuming the log queue."""
        with self._lock:
            record = self._latest_record
            age = None if record is None else time.monotonic() - record["stamp"]
            return {"record": record, "age_ms": None if age is None else age * 1000}

    def _service_commands(self):
        # At most one management transaction per iteration; arm state remains fresh.
        try:
            operation, argument, done, result = self._commands.get_nowait()
        except queue.Empty:
            return
        try:
            if operation == "slew":
                if not self.tuning:
                    raise RuntimeError("Online slew changes require web tuning mode")
                slew = tuning_slew(argument)
                self.controller.running_slew_a_s = slew
                result["value"] = slew.copy()
            elif operation == "currents":
                if not self.tuning:
                    raise RuntimeError("Online current changes require web tuning mode")
                targets = current_targets(argument["constant_current_a"],
                                          argument["constant_damping_a"], self.profile.limits)
                # Replace both targets on the serial owner, before the next compute.
                # Keep controller.previous so normal output slew still applies.
                self.profile.constant_current_a = targets["constant_current_a"]
                self.profile.constant_damping_a = np.asarray(targets["constant_damping_a"])
                result["value"] = targets
            elif operation == "probe":
                result["value"] = self.transport.gripper_info()
            elif operation == "enable":
                result["value"] = self.transport.enable_gripper(argument)
            elif operation == "disable":
                self.transport.disable_gripper()
                self._gripper_a = 0.0
            elif operation in ("write", "zero"):
                if operation == "zero" and self.transport.gripper_limit is None:
                    return
                if self.transport.gripper_limit is None:
                    raise RuntimeError("ID8 current mode is disabled")
                value = 0.0 if operation == "zero" else argument
                if isinstance(value, bool) or not np.isfinite(value):
                    raise ValueError("ID8 command must be finite")
                self._gripper_a = (
                    int(
                        np.clip(
                            value,
                            -self.transport.gripper_limit * 1000,
                            self.transport.gripper_limit * 1000,
                        )
                    )
                    / 1000
                )
                self._gripper_stamp = time.monotonic()
                # Explicit zeros must reach hardware before a caller proceeds to disable.
                if operation == "zero":
                    self.transport.write(8, 102, 2, 0)
                result["value"] = self._gripper_a * 1000
            else:
                raise ValueError("Unknown runtime operation")
        except BaseException as exc:
            result["error"] = exc
            raise
        finally:
            done.set()

    def state(self):
        self.raise_if_failed()
        if self._stop.is_set() or self._status not in ("active", "observe"):
            raise RuntimeError("GELLO runtime is stopped; reconnect before resuming")
        with self._lock:
            state = self._state
            if state is None or time.monotonic() - state["stamp"] > self.profile.state_timeout_s:
                raise RuntimeError("GELLO state is unavailable or stale")
            return {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in state.items()
            }

    def drain_records(self):
        with self._lock:
            records = list(self._records)
            self._records.clear()
        return records

    def status(self):
        return {
            "device": self.profile.name,
            "state": self._status,
            "live": self.live,
            "experimental": self.experimental,
            "teleop": self.teleop,
            "error": None if self._error is None else str(self._error),
            "dropped_records": self.dropped_records,
        }

    def request_stop(self):
        """Non-blocking shutdown request, also usable from the CLI signal handler."""
        self._stop.set()

    def stop(self, *, raise_on_fault=True):
        self.request_stop()
        if self._thread is not None:
            self._thread.join(timeout=3)
            if self._thread.is_alive():
                raise RuntimeError(
                    "GELLO worker did not stop; hardware watchdog remains the fallback"
                )
        if raise_on_fault:
            self.raise_if_failed()

    # Minimal interface consumed by the existing ContinuousDynamixelRobot.
    def get_joints(self):
        return self.state()["position"]

    def close(self):
        self.stop()


class RuntimeRobot:
    """Position follower mapping independent of the runtime's physical model zero."""

    def __init__(self, runtime, signs, gripper_config):
        self._driver = runtime
        self._joint_signs = np.array(list(signs) + ([1] if gripper_config else []), dtype=float)
        self._joint_offsets = np.zeros(len(self._joint_signs))
        self.gripper_open_close = (
            None if gripper_config is None else tuple(np.deg2rad(gripper_config[1:]))
        )
        self._last_pos = None

    def get_joint_state(self):
        raw = self._driver.get_joints()
        pos = (raw - self._joint_offsets) * self._joint_signs
        if self._last_pos is not None:
            pos[:7] += 2 * np.pi * np.round((self._last_pos[:7] - pos[:7]) / (2 * np.pi))
        if self.gripper_open_close is not None:
            opened, closed = self.gripper_open_close
            pos[-1] = np.clip((raw[-1] - opened) / (closed - opened), 0, 1)
        if self._last_pos is not None:
            pos = self._last_pos * 0.01 + pos * 0.99
        self._last_pos = pos.copy()
        return pos

    def set_torque_mode(self, enabled):
        if enabled:
            raise RuntimeError("Position torque mode is unavailable during current control")
        # Pausing the follower does not alter current support. close() is explicit unload.

    def probe_gripper_dynamixel(self):
        from ...teleoperators.gello_teleop.gello_adapter import GripperDynamixelInfo

        return GripperDynamixelInfo(**self._driver.request("probe"))

    def enable_gripper_current_mode(self, current_limit_ma):
        from ...teleoperators.gello_teleop.gello_adapter import GripperDynamixelInfo

        return GripperDynamixelInfo(**self._driver.request("enable", current_limit_ma))

    def write_gripper_current_ma(self, current_ma):
        return self._driver.request("write", current_ma)

    def zero_gripper_current(self):
        return self._driver.request("zero")

    def disable_gripper_current_mode(self):
        return self._driver.request("disable")


class RuntimeAgent:
    def __init__(self, robot):
        self._robot = robot

    def act(self, observation):
        return self._robot.get_joint_state()
