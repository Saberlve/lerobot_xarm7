"""Supervised tuning session and read-only/compensation ownership handoff."""

import copy
import json
import os
import stat
import tempfile
import threading

import numpy as np
import yaml

from ..config import DeviceProfile, current_targets, tuning_slew
from ..monitoring.logging import CurrentLog
from ..monitoring.encoder_monitor import EncoderMonitor
from .runtime import CurrentRuntime, TUNING_TEMPERATURE_LIMIT_C

INITIAL_SLEW_A_S = [0.17, 0.2, 0.17, 0.17, 0.16, 0.17, 0.17]


class TuningSession:
    """Manage exclusive reader/control owners independently of the HTTP server."""

    def __init__(self, profile, log_dir, *, runtime_factory=CurrentRuntime,
                 monitor_factory=EncoderMonitor):
        profile.validate_experiment()
        self.profile = profile
        self.log_dir = log_dir
        self._slew = tuning_slew(getattr(profile, "running_current_slew_a_s", None)
                                 or profile.default_running_current_slew_a_s or INITIAL_SLEW_A_S)
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
        self._saved_targets = current_targets(profile.constant_current_a,
                                             profile.constant_damping_a.tolist(), profile.limits)
        self._saved_slew = self._slew.copy()

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
                    raise RuntimeError("电流控制已启动或正在切换状态")
                self._cancel.clear()
                self._phase = "starting"
                self._error = None
                slew = self._slew.copy()
                previous = self._runtime
            try:
                # Join and close the read-only owner before opening a torque owner.
                self._stop_monitor()
                if previous is not None:
                    previous.stop(raise_on_fault=False)
                self._stop_log()
                profile = copy.copy(self.profile)
                runtime = self._factory(profile, live=True, experimental=True, tuning=True, tuning_slew_a_s=slew)
                with self._lock:
                    self._runtime = runtime
                    self._view_mode = "compensation"
                if self._cancel.is_set():
                    runtime.request_stop()
                self._logger = CurrentLog(runtime, self.log_dir)
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

    def snapshot(self):
        with self._lock:
            runtime, phase, error = self._runtime, self._phase, self._error
            slew = self._slew.copy()
            log_path = self._log_path
            monitor, mode = self._monitor, self._view_mode
            targets = current_targets(self.profile.constant_current_a,
                                      self.profile.constant_damping_a.tolist(), self.profile.limits)
            unsaved = targets != self._saved_targets
            unsaved_settings = unsaved or slew != self._saved_slew
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
                **targets,
                "unsaved_currents": unsaved,
                "unsaved_settings": unsaved_settings,
                "auto_save": True,
                "profile_path": str(self.profile.path),
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

    def set_current_slew(self, values, *, persist=False):
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
            if persist:
                self._persist_applied_settings()
        return slew

    def set_current_targets(self, currents, damping, *, persist=False):
        targets = current_targets(currents, damping, self.profile.limits)
        with self._lifecycle:
            with self._lock:
                runtime, phase = self._runtime, self._phase
            if phase in ("starting", "stopping"):
                raise RuntimeError("请等待启动或卸力完成")
            if runtime is not None and runtime.status()["state"] == "active":
                runtime.set_current_targets(targets["constant_current_a"],
                                            targets["constant_damping_a"])
            with self._lock:
                self.profile.constant_current_a = targets["constant_current_a"]
                self.profile.constant_damping_a = np.asarray(targets["constant_damping_a"])
            if persist:
                self._persist_applied_settings()
        return targets

    def save_current_targets(self):
        """Persist all three tuning fields; retain the existing API name."""
        with self._lifecycle:
            return self._save_settings()

    def _persist_applied_settings(self):
        try:
            self._save_settings()
        except Exception as exc:
            raise RuntimeError(f"参数已应用，但同步配置失败：{exc}；请点击重新同步参数到配置") from exc

    def _save_settings(self):
        """Caller holds the lifecycle lock; filesystem I/O stays off the serial owner."""
        with self._lock:
            if self._phase in ("starting", "stopping"):
                raise RuntimeError("请等待启动或卸力完成")
            targets = current_targets(self.profile.constant_current_a,
                                      self.profile.constant_damping_a.tolist(), self.profile.limits)
            slew = self._slew.copy()
        path = self.profile.path
        original = path.read_text()
        data = yaml.safe_load(original)
        data.update(targets)
        data["running_current_slew_a_s"] = slew
        data["note"] = "Fixed currents, passive damping and continuous-mode current slew tuned from the web. Geometry is used for visualization only."
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=path.parent,
                                             prefix=".current-tuning-", suffix=".yaml",
                                             delete=False) as stream:
                temporary = stream.name
                stream.write(json.dumps(data, indent=2) + "\n" if original.lstrip().startswith("{")
                             else yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
                stream.flush()
                os.fsync(stream.fileno())
            candidate = DeviceProfile(temporary)
            candidate.validate_experiment()
            if candidate.serial != self.profile.serial or not np.array_equal(candidate.limits, self.profile.limits):
                raise RuntimeError("配置在调参期间发生变化，请重新加载网页服务")
            os.chmod(temporary, stat.S_IMODE(path.stat().st_mode))
            os.replace(temporary, path)
            temporary = None
        finally:
            if temporary is not None:
                os.unlink(temporary)
        with self._lock:
            self.profile.data = data
            self.profile.default_running_current_slew_a_s = slew
            self._saved_targets = targets
            self._saved_slew = slew
        return str(path)
