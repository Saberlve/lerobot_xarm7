"""Supervised tuning session and read-only/compensation ownership handoff."""

import copy
import threading

import numpy as np

from ..config import tuning_slew, vector
from ..monitoring.logging import GravityLog
from ..monitoring.encoder_monitor import EncoderMonitor
from .runtime import GravityRuntime, TUNING_TEMPERATURE_LIMIT_C

# User-approved GELLO A gains from the web tuning session on 2026-10-06.
INITIAL_GAINS = [0.065, 0.15, 0.115, 0.15, 0.06, 0.1, 0.12]
INITIAL_SLEW_A_S = [0.05, 0.12, 0.05, 0.12, 0.05, 0.05, 0.05]


def tuning_gains(values):
    gains = vector(values, "joint_gains", nonnegative=True)
    if np.any(gains > 1):
        raise ValueError("七轴增益必须在 0–1 之间")
    return gains.tolist()


class TuningSession:
    """Manage exclusive reader/control owners independently of the HTTP server."""

    def __init__(self, profile, log_dir, *, gains=None, runtime_factory=GravityRuntime,
                 monitor_factory=EncoderMonitor):
        profile.validate_experiment()
        self.profile = profile
        self.log_dir = log_dir
        self._gains = tuning_gains(INITIAL_GAINS if gains is None else gains)
        self._slew = tuning_slew(INITIAL_SLEW_A_S)
        self._factory = runtime_factory
        self._monitor_factory = monitor_factory
        self._monitor = None
        self._view_mode = "offline"
        self._lock = threading.Lock()
        self._lifecycle = threading.Lock()
        self._cancel = threading.Event()
        self._runtime = None
        self._logger = None
        self._log_path = None
        self._phase = "idle"
        self._error = None

    def heartbeat(self):
        with self._lock:
            runtime = self._runtime
        if runtime is not None:
            runtime.tuning_heartbeat()

    def _stop_log(self):
        if self._logger is not None:
            self._logger.stop()
            self._logger = None

    def _stop_monitor(self):
        if self._monitor is not None:
            self._monitor.stop()
            self._monitor = None

    def set_view_mode(self, mode):
        if mode not in ("offline", "read_only"):
            raise ValueError("查看模式必须为 offline 或 read_only")
        with self._lifecycle:
            with self._lock:
                if self._phase in ("starting", "stopping") or (
                    self._runtime is not None and self._runtime.status()["state"] == "active"
                ):
                    raise RuntimeError("请先立即卸力，再切换查看模式")
                self._cancel.clear()
                self._phase = "starting"
                previous = self._runtime
            try:
                self._stop_monitor()
                if previous is not None:
                    previous.stop(raise_on_fault=False)
                self._stop_log()
                monitor = self._monitor_factory(self.profile) if mode == "read_only" else None
                with self._lock:
                    self._runtime = None
                    self._monitor = monitor
                    self._view_mode = mode
                    self._error = None
                if monitor is not None:
                    monitor.start()
                if self._cancel.is_set():
                    self._stop_monitor()
                    mode = "stopped"
                with self._lock:
                    self._view_mode = mode
                    self._phase = "reading" if mode == "read_only" else "stopped" if mode == "stopped" else "idle"
            except Exception as exc:
                with self._lock:
                    self._phase, self._error = "fault", str(exc)
                raise

    def start(self):
        with self._lifecycle:
            with self._lock:
                if self._phase in ("starting", "stopping") or (
                    self._runtime is not None and self._runtime.status()["state"] == "active"
                ):
                    raise RuntimeError("补偿已启动或正在切换状态")
                self._cancel.clear()
                self._phase = "starting"
                self._error = None
                gains = self._gains.copy()
                slew = self._slew.copy()
                previous = self._runtime
            try:
                # Join and close the read-only owner before opening a torque owner.
                self._stop_monitor()
                if previous is not None:
                    previous.stop(raise_on_fault=False)
                self._stop_log()
                profile = copy.copy(self.profile)
                profile.joint_gains = gains
                # Explicit seven-axis gains replace the legacy J5/J6 overrides.
                profile.j5_gain = profile.j6_gain = None
                runtime = self._factory(profile, live=True, experimental=True, tuning=True, tuning_slew_a_s=slew)
                with self._lock:
                    self._runtime = runtime
                    self._view_mode = "compensation"
                if self._cancel.is_set():
                    runtime.request_stop()
                self._logger = GravityLog(runtime, self.log_dir)
                with self._lock:
                    self._log_path = str(self._logger.path)
                runtime.start()
                if self._cancel.is_set():
                    runtime.stop()
                    self._stop_log()
                    with self._lock:
                        self._phase = "stopped"
                    return
                self._logger.start()
                with self._lock:
                    self._phase = "active"
            except Exception as exc:
                error = str(exc)
                if self._runtime is not None:
                    try:
                        self._runtime.stop()
                    except Exception as cleanup:
                        if str(cleanup) not in error:
                            error += f"; {cleanup}"
                try:
                    self._stop_log()
                except Exception as cleanup:
                    error += f"; {cleanup}"
                with self._lock:
                    self._phase, self._error = "fault", error
                raise RuntimeError(error) from exc

    def stop(self):
        # Cancellation is immediate even while another HTTP request initializes.
        with self._lock:
            self._cancel.set()
            runtime = self._runtime
            self._phase = "stopping"
        if runtime is not None:
            runtime.request_stop()
        with self._lifecycle:
            errors = []
            try:
                self._stop_monitor()
            except Exception as exc:
                errors.append(str(exc))
            with self._lock:
                runtime = self._runtime
            if runtime is not None:
                try:
                    runtime.stop()
                except Exception as exc:
                    errors.append(str(exc))
            try:
                self._stop_log()
            except Exception as exc:
                errors.append(str(exc))
            with self._lock:
                self._phase = "fault" if errors else "stopped"
                self._view_mode = "stopped"
                self._error = "; ".join(errors) if errors else None
            if errors:
                raise RuntimeError("; ".join(errors))

    def set_gains(self, values):
        gains = tuning_gains(values)
        with self._lifecycle:
            with self._lock:
                runtime, phase = self._runtime, self._phase
            if phase in ("starting", "stopping"):
                raise RuntimeError("请等待启动或卸力完成")
            if runtime is not None and runtime.status()["state"] == "active":
                runtime.set_joint_gains(gains)
            with self._lock:
                self._gains = gains
        return gains

    def snapshot(self):
        with self._lock:
            runtime, phase, error = self._runtime, self._phase, self._error
            gains = self._gains.copy()
            slew = self._slew.copy()
            log_path = self._log_path
            monitor, mode = self._monitor, self._view_mode
        diagnostic = {"record": None, "age_ms": None}
        if runtime is not None and mode in ("compensation", "stopped"):
            status = runtime.status()
            diagnostic = runtime.diagnostics()
            if phase == "active" and status["state"] in ("fault", "stopped"):
                phase = status["state"]
            error = error or status["error"]
        record = diagnostic["record"]
        sample = None
        if record is not None:
            sample = {
                "stamp": record["stamp"],
                "encoder_rad": record["position"],
                "model_q_rad": record["q"],
                "temperature_c": record["temperature_c"],
                "voltage_v": record["voltage_v"],
                "read_hz": 1 / record["interval_s"],
            }
        packet = {
            "view_mode": mode,
            "status": "connected" if phase == "active" else "connecting" if phase == "starting" else "error" if phase == "fault" else "stopped",
            "error": error,
            "age_ms": diagnostic["age_ms"],
            "sequence": -1 if record is None else int(record["stamp"] * 1e6),
            "sample": sample,
            "tuning": {
                "state": phase,
                "gains": gains,
                "applied_gains": None if record is None else record["gravity_gains"],
                "current_slew_a_s": slew,
                "applied_slew_a_s": None if record is None else record["current_slew_a_s"],
                "running_slew_a_s": None if record is None else record["running_slew_a_s"],
                "temperature_limit_c": min(self.profile.temperature_limit_c, TUNING_TEMPERATURE_LIMIT_C),
                "current_limit_a": self.profile.limits.tolist(),
                "record": record,
                "log": log_path,
            },
        }
        if monitor is not None and mode == "read_only":
            packet.update(monitor.snapshot())
            if error:
                packet["error"] = error
        return packet

    def set_current_slew(self, values):
        slew = tuning_slew(values)
        with self._lifecycle:
            with self._lock:
                runtime, phase = self._runtime, self._phase
            if phase in ("starting", "stopping"):
                raise RuntimeError("请等待启动或卸力完成")
            if runtime is not None and runtime.status()["state"] == "active":
                runtime.set_current_slew(slew)
            with self._lock:
                self._slew = slew
        return slew
