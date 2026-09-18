"""Latest-value arm sampling, haptic worker and measured CSV timing."""

import csv
import json
import logging
import threading
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .arm_feedback import (
    ArmFeedbackProcessor,
    ArmFeedbackSample,
    ft_wrench_to_joint_torque,
    vector7,
)

logger = logging.getLogger(__name__)


class XArmFeedbackSource:
    """Dedicated read-only SDK connection: never contend with position RPCs.

    Timestamp is REQUEST START, conservatively including RPC latency in age.
    Polling Hz measures responses, not the controller's internal sensor rate.
    Baseline is fixed configuration; no online learning during contact.
    """

    def __init__(self, robot_ip, config, api=None, kinematics=None):
        self.robot_ip, self.config = robot_ip, config
        self.api = api
        self.owns_api = api is None
        self.kinematics = kinematics
        self.latest = None
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        if self.config.source == "disabled":
            raise ValueError("arm feedback source disabled")
        try:
            if self.api is None:
                from xarm.wrapper import XArmAPI

                self.api = XArmAPI(
                    self.robot_ip,
                    is_radian=True,
                    enable_report=False,
                    timeout=self.config.stale_timeout_ms / 1000,
                )
            if self.config.source == "ft_sensor" and self.kinematics is None:
                from ..robots.uf_robot.local_kinematics import (
                    XArm7Kinematics,
                    read_xarm7_kinematics,
                )

                self.kinematics = XArm7Kinematics(read_xarm7_kinematics(self.robot_ip))
            self.thread = threading.Thread(target=self._run, name="xarm-arm-sampler", daemon=True)
            self.thread.start()
        except BaseException:
            self.stop()
            raise

    def read_once(self, sequence=1, previous_ns=None):
        start = time.monotonic_ns()
        code, states = self.api.get_joint_states(is_radian=True, num=3)
        if code != 0 or len(states) != 3:
            raise RuntimeError(f"get_joint_states failed: {code}")
        q = vector7(states[0], "follower_position")
        vector7(states[1], "follower_velocity")
        raw = vector7(states[2], "raw_joint_effort")
        unit = "sdk_effort_unit"
        if self.config.source == "ft_sensor":
            code, wrench = self.api.get_ft_sensor_data(is_raw=False)
            if code != 0:
                raise RuntimeError(f"FT read failed: {code}")
            estimate = ft_wrench_to_joint_torque(
                self.kinematics,
                q,
                wrench,
                self.config.ft_sensor_to_flange,
                self.config.ft_vertical_only,
            )
            unit = "Nm"
        elif self.config.source == "bias_compensated_joint_effort":
            estimate = raw - self.config.baseline
        elif self.config.source == "raw_joint_effort":
            estimate = raw.copy()  # diagnostic only; NOT external torque
        else:
            raise ValueError("source disabled")
        return ArmFeedbackSample(
            start,
            raw,
            estimate,
            unit,
            None,
            (time.monotonic_ns() - start) / 1e6,
            sequence,
            0.0 if previous_ns is None else (start - previous_ns) / 1e6,
        )

    def _run(self):
        sequence, previous_ns = 0, None
        while not self.stop_event.is_set():
            tick = time.monotonic_ns()
            try:
                sequence += 1
                sample = self.read_once(sequence, previous_ns)
                previous_ns = sample.timestamp_ns
                self.latest = sample
            except BaseException as exc:
                self.latest = ArmFeedbackSample(
                    tick, np.zeros(7), np.zeros(7), error=f"{type(exc).__name__}: {exc}"
                )
                return  # latched fault; no silent reconnect/re-enable
            delay = 1 / self.config.sampling_hz - (time.monotonic_ns() - tick) / 1e9
            self.stop_event.wait(max(0, delay))

    def stop(self):
        self.stop_event.set()
        if self.owns_api and self.api is not None:
            self.api.disconnect()  # interrupt pending read before joining
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=2)
            if self.thread.is_alive():
                raise RuntimeError("arm sampler did not stop")


class ArmFeedbackWorker:
    def __init__(self, config, source, adapter, leader_state, motor_signs=(1,) * 7):
        self.config, self.source, self.adapter = config, source, adapter
        self.leader_state = leader_state
        self.motor_signs = vector7(motor_signs)
        self.processor = ArmFeedbackProcessor(config)
        self.stop_event = threading.Event()
        self.thread = None
        self.watchdog = None
        self.heartbeat_ns = time.monotonic_ns()
        self.fault = None
        self.latest = None
        self.counters = dict(stale_count=0, write_error_count=0, loop_overrun_count=0)
        self.log_file = None
        self.log_path = Path(config.log_path)
        self._stop_lock = threading.Lock()

    def start(self):
        if not self.config.enabled:
            return
        path = self.log_path
        path.parent.mkdir(parents=True, exist_ok=True)
        # Avoid silently overwriting earlier experiment evidence.
        if path.exists():
            path = path.with_name(f"{path.stem}_{time.time_ns()}{path.suffix}")
        self.log_path = path
        self.log_file = path.open("x", newline="", encoding="utf-8")
        try:
            self.source.start()
            deadline = time.monotonic() + 3
            while self.source.latest is None or self.leader_state()[0] == 0:
                if time.monotonic() > deadline:
                    raise RuntimeError("timed out waiting for arm/leader sample")
                time.sleep(0.01)
            sample = self.source.latest
            leader = self.leader_state()
            result = self.processor.process(
                sample, leader[2], time.monotonic_ns(), self.motor_signs
            )
            if result.fault or leader[3]:
                raise RuntimeError(result.fault or leader[3])
            if (time.monotonic_ns() - leader[0]) / 1e6 > self.config.stale_timeout_ms:
                raise RuntimeError("leader_stale at startup")
            if not self.config.observe_only:
                self.adapter.enable()
            else:
                self.adapter.discover()  # read-only; no current-mode writes
            self.processor.reset()  # first worker output must still be zero
            metadata = dict(
                config=asdict(self.config),
                motors=self.adapter.infos,
                source_api="get_joint_states(num=3)",
                physical_rates="NOT MEASURED until hardware run",
                effort_unit="SDK unspecified; experimental units",
            )
            path.with_suffix(".metadata.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )
            self.heartbeat_ns = time.monotonic_ns()
            self.thread = threading.Thread(target=self._run, name="gello-arm-haptics", daemon=True)
            self.watchdog = threading.Thread(
                target=self._watchdog, name="gello-arm-watchdog", daemon=True
            )
            self.thread.start()
            self.watchdog.start()
            logger.info("Arm feedback CSV: %s", self.log_path)
        except BaseException as exc:
            self.fault = f"startup_error: {exc}"
            self.stop()
            raise

    def _watchdog(self):
        timeout = self.config.stale_timeout_ms / 1000
        while not self.stop_event.wait(min(timeout / 4, 0.02)):
            if (time.monotonic_ns() - self.heartbeat_ns) / 1e9 > timeout:
                self.fault = "worker_watchdog_timeout"
                self.stop_event.set()
                try:
                    self.adapter.disable()
                except Exception:
                    logger.exception("arm watchdog cleanup failed")
                return

    def tick(self, previous_ns=None):
        start = time.monotonic_ns()
        sample = self.source.latest
        leader_snapshot = self.leader_state()
        leader_ns, position, velocity, leader_error, read_ms = leader_snapshot[:5]
        position = vector7(position, "leader_position") * self.motor_signs
        velocity = vector7(velocity, "leader_velocity") * self.motor_signs
        processing_start = time.monotonic_ns()
        leader_age = (processing_start - leader_ns) / 1e6
        result = self.processor.process(sample, velocity, processing_start, self.motor_signs)
        processing_ms = (time.monotonic_ns() - processing_start) / 1e6
        if leader_error or leader_age < 0 or leader_age > self.config.stale_timeout_ms:
            result.fault = leader_error or "leader_stale"
            result.stale = result.stale or leader_age > self.config.stale_timeout_ms
            result.command_current_ma = np.zeros(7)
        if self.stop_event.is_set():
            result.fault = self.fault or "stopped"
            result.command_current_ma = np.zeros(7)
        actual = np.zeros(7)
        write_start = time.monotonic_ns()
        write_ms = None
        if result.fault:
            self.fault = result.fault
            logger.error("Arm feedback disabled: %s", self.fault)
            self.counters["stale_count"] += int(result.stale)
            self.stop_event.set()
            self.adapter.disable()
        elif not self.config.observe_only:
            try:
                expiry = min(sample.timestamp_ns, leader_ns) + int(
                    self.config.stale_timeout_ms * 1e6
                )
                actual = self.adapter.write(result.command_current_ma, deadline_ns=expiry)
                write_ms = (time.monotonic_ns() - write_start) / 1e6
            except Exception as exc:
                self.counters["write_error_count"] += 1
                self.fault = result.fault = f"write_error: {exc}"
                self.stop_event.set()
        end = time.monotonic_ns()
        dt_ms = 0 if previous_ns is None else (start - previous_ns) / 1e6
        row = dict(
            timestamp_ns=start,
            sample_timestamp_ns=None if sample is None else sample.timestamp_ns,
            sample_sequence=0 if sample is None else sample.sequence,
            sample_period_ms=0 if sample is None else sample.period_ms,
            source_read_latency_ms=0 if sample is None else sample.read_latency_ms,
            unit="unknown" if sample is None else sample.unit,
            sample_age_ms=result.sample_age_ms,
            leader_timestamp_ns=leader_ns,
            leader_sequence=leader_snapshot[5] if len(leader_snapshot) > 5 else 0,
            leader_period_ms=leader_snapshot[6] if len(leader_snapshot) > 6 else 0,
            leader_age_ms=leader_age,
            leader_read_latency_ms=read_ms,
            loop_dt_ms=dt_ms,
            effective_hz=1000 / dt_ms if dt_ms else 0,
            processing_latency_ms=processing_ms,
            write_latency_ms=write_ms,
            serial_transaction_ms=getattr(self.adapter, "last_transaction_ms", None)
            if write_ms is not None
            else None,
            sample_to_command_ms=None if sample is None else (end - sample.timestamp_ns) / 1e6,
            observe_only=self.config.observe_only,
            stale=result.stale,
            clamped=result.clamped,
            fault=result.fault or "",
            **self.counters,
        )
        arrays = dict(
            raw_effort=np.zeros(7) if sample is None else sample.raw_joint_effort,
            baseline=self.config.baseline,
            estimated_contact_torque=np.zeros(7)
            if sample is None
            else sample.estimated_contact_torque,
            leader_position=position,
            leader_velocity=velocity,
            processed_feedback_ma=result.processed_feedback_ma,
            hypothetical_current_ma=result.command_current_ma,
            command_current_ma=actual,
            sign_product=np.zeros(7)
            if sample is None
            else sample.estimated_contact_torque * result.command_current_ma,
        )
        for name, array in arrays.items():
            for j, value in enumerate(array):
                row[f"{name}_{j + 1}"] = float(value)
        self.latest = row
        self.heartbeat_ns = end
        return row

    def _run(self):
        previous_ns = None
        writer = None
        try:
            while not self.stop_event.is_set():
                tick_start = time.monotonic_ns()
                row = self.tick(previous_ns)
                previous_ns = row["timestamp_ns"]
                if writer is None:
                    writer = csv.DictWriter(self.log_file, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                self.log_file.flush()
                delay = 1 / self.config.update_hz - (time.monotonic_ns() - tick_start) / 1e9
                if delay < 0:
                    self.counters["loop_overrun_count"] += 1
                self.stop_event.wait(max(0, delay))
        except BaseException as exc:
            self.fault = f"worker_exception: {exc}"
            logger.exception("arm feedback worker fault")
        finally:
            self.stop_event.set()
            try:
                self.adapter.disable()
            except Exception as exc:
                self.fault = f"{self.fault or ''}; cleanup_error: {exc}"
                logger.exception("arm worker cleanup failed")
            self._write_status()

    def _write_status(self):
        self.log_path.with_suffix(".status.json").write_text(
            json.dumps(dict(fault=self.fault, **self.counters)), encoding="utf-8"
        )

    def stop(self):
        with self._stop_lock:
            self.stop_event.set()
            errors = []
            # Disable immediately, before waiting on a possibly blocked SDK read.
            try:
                self.adapter.disable()
            except Exception as exc:
                errors.append(f"disable: {exc}")
            for worker in (self.thread, self.watchdog):
                if (
                    worker is not None
                    and worker is not threading.current_thread()
                    and worker.is_alive()
                ):
                    worker.join(timeout=2)
                    if worker.is_alive():
                        errors.append(f"{worker.name} did not stop")
            try:
                self.source.stop()
            except Exception as exc:
                errors.append(f"sampler stop: {exc}")
            if self.thread is None or not self.thread.is_alive():
                if self.log_file is not None and not self.log_file.closed:
                    self.log_file.close()
            if errors:
                self.fault = f"{self.fault or ''}; cleanup_error: {'; '.join(errors)}"
            if self.log_file is not None:
                self._write_status()
            if errors:
                raise RuntimeError("; ".join(errors))
