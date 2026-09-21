"""Latest-value arm sampling, haptic worker and measured CSV timing."""

import csv
import json
import logging
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .arm_external_torque import (
    ExternalTorqueEstimate,
    make_external_torque_estimator,
)
from .arm_feedback import (
    ArmFeedbackProcessor,
    ArmFeedbackResult,
    ArmFeedbackSample,
    ft_wrench_to_joint_torque,
    vector7,
)

logger = logging.getLogger(__name__)


def leader_period_budget_ms(config):
    """Accepted steady-state leader read period at startup.

    Free-running (leader_read_hz=0) the SyncRead cadence is ~48 ms, so 100 ms
    bounds the post-enable transient. A throttled reader's steady period IS the
    configured one; accept up to 1.5x it.
    """
    if config.leader_read_hz > 0:
        return max(100.0, 1500.0 / config.leader_read_hz)
    return 100.0


class XArmFeedbackSource:
    """Dedicated read-only SDK connection: never contend with position RPCs.

    The controller only refreshes the effort/velocity half of GET_JOINT_POS
    while a report client is attached (observed on hardware: with no report
    stream the num=3 effort stays bit-frozen for entire sessions while reads
    still return code 0). This connection therefore enables the report stream
    and reads effort from the per-packet report cache (joints_torque) with a
    freshness gate, instead of trusting the RPC effort. Position still comes
    from the get_joint_states RPC (that half is always live).

    Timestamp is REQUEST START, conservatively including RPC latency in age.
    Polling Hz measures responses, not the controller's internal sensor rate.
    Baseline is fixed configuration; no online learning during contact.
    """

    # Report-vs-RPC effort consistency tolerance at startup, in SDK effort
    # units. Guards against a unit/scale mismatch silently breaking the fixed
    # baseline; loose enough for quantization noise at a held pose.
    EFFORT_SCALE_TOLERANCE = 2.0

    def __init__(self, robot_ip, config, api=None, kinematics=None):
        self.robot_ip, self.config = robot_ip, config
        self.api = api
        self.owns_api = api is None
        self.kinematics = kinematics
        self.latest = None
        self.report_age_ms = None
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
                    # Matches the standalone-proven setup: keep the controller
                    # report stream alive so joint effort telemetry is fresh.
                    enable_report=True,
                    report_type="rich",
                    timeout=self.config.stale_timeout_ms / 1000,
                )
            self._wait_report_stream()
            self._check_effort_scale()
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

    def _report_stamp_s(self):
        inner = getattr(self.api, "_arm", None)
        return getattr(inner, "_last_update_cmdnum_time", 0) if inner is not None else 0

    def _wait_report_stream(self, timeout_s=3.0):
        deadline = time.monotonic() + timeout_s
        while not self._report_stamp_s():
            if time.monotonic() > deadline:
                raise RuntimeError(
                    "report stream did not start; enable_report=True is required"
                )
            time.sleep(0.01)

    def _check_effort_scale(self):
        """Report tau and GET_JOINT_POS effort must agree at a held pose.

        Both are the controller's joint torque telemetry in the same SDK
        effort units; a gross mismatch means the fixed baseline (measured via
        get_joint_states) would not apply to the report-stream values. Retry
        briefly to ride out the controller cache catching up right after the
        report stream attaches.
        """
        deadline = time.monotonic() + 3.0
        while True:
            code, states = self.api.get_joint_states(is_radian=True, num=3)
            if code == 0 and len(states) == 3:
                rpc_effort = np.asarray(states[2], dtype=float)
                diff = float(
                    np.abs(rpc_effort - np.asarray(self.api.joints_torque, dtype=float)).max()
                )
                if diff <= self.EFFORT_SCALE_TOLERANCE:
                    return
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"report-stream effort disagrees with get_joint_states effort "
                    f"(max diff {diff:.3f} > {self.EFFORT_SCALE_TOLERANCE:g}); "
                    "re-verify the baseline before enabling feedback"
                )
            time.sleep(0.1)

    def read_once(self, sequence=1, previous_ns=None):
        start = time.monotonic_ns()
        qd = None
        if self.config.dynamic_mode or self.config.estimator.mode == "next":
            # Dynamic mode: q/qd/tau all come from the SAME report packet, so
            # the gravity model in C2 never evaluates q at a different instant
            # than the torque it compensates. No RPC position read here.
            raw, q, qd, report_age_ms = self._report_snapshot()
        else:
            code, states = self.api.get_joint_states(is_radian=True, num=1)
            if code != 0 or len(states) != 1:
                raise RuntimeError(f"get_joint_states failed: {code}")
            q = vector7(states[0], "follower_position")
            raw, report_age_ms = self._report_effort()
        self.report_age_ms = report_age_ms
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
            position=(
                q
                if self.config.dynamic_mode or self.config.estimator.mode == "next"
                else None
            ),
            velocity=qd,
            robot_state=getattr(self.api, "state", None),
            robot_mode=getattr(self.api, "mode", None),
        )

    def _report_effort(self):
        """Live joint effort from the report stream cache, freshness-gated.

        The report parser updates joints_torque and _last_update_cmdnum_time
        on every packet, so a frozen controller cache or dropped stream turns
        into an explicit fault instead of a silently constant zero contact.

        No report-rate assumption is made here: the rich report is only
        ~10 Hz (observed ~100 ms period) on this controller, so the gate is
        purely config.stale_timeout_ms, which must be configured as a
        multiple of the actual report period (500 ms = 5 x 100 ms).
        """
        stamp = self._report_stamp_s()
        if not stamp:
            raise RuntimeError("report stream not started (enable_report=True required)")
        age_ms = (time.monotonic() - stamp) * 1e3
        if age_ms > self.config.stale_timeout_ms:
            raise RuntimeError(
                f"report stream stale: {age_ms:.0f} ms "
                f"> stale_timeout_ms={self.config.stale_timeout_ms:g}"
            )
        return vector7(self.api.joints_torque, "raw_joint_effort"), age_ms

    def _report_snapshot(self):
        """Freshness-gated synchronized (effort, q, qd) from one report packet.

        The rich-report parser updates joints_torque, angles and
        realtime_joint_speeds together on every packet, so reading all three
        after the single stamp/age check in _report_effort yields a
        same-packet snapshot.
        """
        effort, age_ms = self._report_effort()
        q = vector7(self.api.angles, "joint_position")
        qd = vector7(self.api.realtime_joint_speeds, "joint_velocity")
        return effort, q, qd, age_ms

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
    def __init__(
        self,
        config,
        source,
        adapter,
        leader_state,
        motor_signs=(1,) * 7,
        command_state=None,
    ):
        self.config, self.source, self.adapter = config, source, adapter
        self.leader_state = leader_state
        self.motor_signs = vector7(motor_signs)
        self.command_state = command_state
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
        # Damping is the only feedback term that consumes leader velocity.
        # With damping=0 on every enabled joint the leader snapshot only feeds
        # logging: skip the leader age gate and do not let leader staleness
        # shrink the write deadline.
        self.needs_leader_velocity = any(
            damping > 0 and enabled
            for damping, enabled in zip(
                config.damping_ma_per_rad_s, config.enabled_joints, strict=True
            )
        )
        self.leader_period_budget_ms = leader_period_budget_ms(config)
        self.leader_age_budget_ms = max(
            config.stale_timeout_ms, self.leader_period_budget_ms
        )
        # NEXT or Stage-C dynamic mode uses the common estimator interface.
        # Legacy static operation continues to consume the source estimate so
        # Stage A/B behavior and log shape remain unchanged.
        self.estimator = (
            make_external_torque_estimator(config)
            if config.dynamic_mode or config.estimator.mode == "next"
            else None
        )
        # NEXT consumes the command generated from GELLO even when damping is
        # zero, so stale leader telemetry must gate NEXT active output.
        self.needs_leader_freshness = (
            self.needs_leader_velocity or config.estimator.mode == "next"
        )
        self._estimator_ever_ready = False

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
            if (time.monotonic_ns() - leader[0]) / 1e6 > self.leader_age_budget_ms:
                raise RuntimeError("leader_stale at startup")
            if not self.config.observe_only:
                if getattr(self.estimator, "permanently_disabled", False):
                    raise RuntimeError(
                        "NEXT unavailable and next.fallback=disable; arm feedback disabled"
                    )
                self.adapter.enable()
            else:
                self.adapter.discover()  # read-only; no current-mode writes
            # enable()/discover() hold the serial lock and starve the timed
            # leader reader; require a fresh snapshot before the first tick.
            # Freshness alone is not enough: right after enable() a SyncRead
            # cycle can take ~400 ms, and its snapshot carries the pre-read
            # timestamp, so a "fresh-enough" snapshot may still leave almost
            # no write deadline budget while the reader holds the lock for
            # another slow cycle. Also require the last measured read period
            # to be back inside the steady-state budget (the free-running
            # ~48 ms cadence, or 1.5x the configured throttled period).
            deadline = time.monotonic() + 3
            while True:
                leader = self.leader_state()
                if leader[3]:
                    raise RuntimeError(leader[3])
                leader_age_ms = (time.monotonic_ns() - leader[0]) / 1e6
                leader_period_ms = leader[6] if len(leader) > 6 else 0
                if (
                    leader[0]
                    and 0 <= leader_age_ms <= self.leader_age_budget_ms
                    and 0 < leader_period_ms <= self.leader_period_budget_ms
                ):
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError("leader_stale after adapter init")
                time.sleep(0.005)
            self.processor.reset()  # first worker output must still be zero
            metadata = dict(
                config=asdict(self.config),
                motors=self.adapter.infos,
                source_api=(
                    "report_stream(rich) synchronized q/qd/joints_torque"
                    if self.config.dynamic_mode or self.config.estimator.mode == "next"
                    else "report_stream(rich).joints_torque + get_joint_states(num=1)"
                ),
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
        estimate = None
        command = None
        command_ns = 0
        command_error = None
        if sample is not None and sample.position is not None:
            if self.command_state is not None:
                try:
                    command_snapshot = self.command_state()
                    command_ns, command, command_error = command_snapshot[:3]
                except Exception as exc:
                    command_error = f"command_snapshot_error: {exc}"
            elif self.config.estimator.mode == "next":
                command_error = "command_unavailable"
            else:
                command_ns, command = sample.timestamp_ns, sample.position

        command_age_ms = None if not command_ns else (processing_start - command_ns) / 1e6
        command_stale = (
            command_age_ms is not None
            and (command_age_ms < 0 or command_age_ms > self.config.next.command_stale_timeout_ms)
        )
        can_estimate = (
            self.estimator is not None
            and sample is not None
            and sample.error is None
            and sample.position is not None
            and sample.velocity is not None
            and command is not None
            and command_error is None
            and not command_stale
        )
        if can_estimate:
            estimate = self.estimator.update(
                sample.position,
                sample.velocity,
                command,
                sample.raw_joint_effort,
                sample.timestamp_ns,
            )
            if estimate.ready:
                self._estimator_ever_ready = True
                sample = replace(
                    sample,
                    estimated_contact_torque=estimate.external_torque.copy(),
                )

        waiting_for_next = (
            self.config.estimator.mode == "next"
            and (estimate is None or not estimate.ready)
        )
        if waiting_for_next:
            # Startup/history warm-up is expected and must never emit current.
            # Once an estimator has been active, losing q_cmd is a safety fault.
            result = ArmFeedbackResult()
            if estimate is not None and estimate.estimator_mode == "next_disabled":
                result.fault = f"estimator_disabled: {estimate.status}"
            elif self._estimator_ever_ready and (command_error or command_stale):
                result.fault = command_error or "command_stale"
                result.stale = bool(command_stale)
        else:
            result = self.processor.process(sample, velocity, processing_start, self.motor_signs)
        processing_ms = (time.monotonic_ns() - processing_start) / 1e6
        # A negative age is a clock/snapshot anomaly and always faults. An aged
        # leader only faults when damping actually consumes its velocity; with
        # damping=0 on every enabled joint the feedback command must not wait
        # on (or fail because of) the throttled leader reader.
        leader_stale = leader_age < 0 or (
            self.needs_leader_freshness and leader_age > self.leader_age_budget_ms
        )
        if leader_error or leader_stale:
            result.fault = leader_error or "leader_stale"
            result.stale = result.stale or leader_stale
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
                expiry_base = sample.timestamp_ns
                if self.needs_leader_freshness:
                    expiry_base = min(expiry_base, leader_ns)
                expiry = expiry_base + int(self.config.stale_timeout_ms * 1e6)
                actual = self.adapter.write(result.command_current_ma, deadline_ns=expiry)
                write_ms = (time.monotonic_ns() - write_start) / 1e6
            except Exception as exc:
                self.counters["write_error_count"] += 1
                self.fault = result.fault = f"write_error: {exc}"
                self.stop_event.set()
        end = time.monotonic_ns()
        dt_ms = 0 if previous_ns is None else (start - previous_ns) / 1e6
        tau_ext = None if estimate is None or not estimate.ready else estimate.external_torque
        tau_free = None if estimate is None else estimate.predicted_free_torque
        tau_baseline = None if estimate is None else estimate.baseline_external_torque
        if estimate is None:
            estimate = ExternalTorqueEstimate(
                ready=self.estimator is None,
                estimator_mode="source" if self.estimator is None else self.config.estimator.mode,
                status=command_error or ("command_stale" if command_stale else "not_ready"),
            )
        # Read-side lock timing lives on the driver (reader thread); write-side
        # on the adapter. Both are latest values, matching the leader snapshot.
        driver = getattr(self.adapter, "driver", None)
        row = dict(
            timestamp=start / 1e9,
            timestamp_ns=start,
            sample_timestamp_ns=None if sample is None else sample.timestamp_ns,
            sample_sequence=0 if sample is None else sample.sequence,
            sample_period_ms=0 if sample is None else sample.period_ms,
            source_read_latency_ms=0 if sample is None else sample.read_latency_ms,
            unit="unknown" if sample is None else sample.unit,
            sample_age_ms=result.sample_age_ms,
            report_age_ms=getattr(self.source, "report_age_ms", None),
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
            serial_lock_wait_ms_read=getattr(driver, "_arm_reader_lock_wait_ms", None),
            serial_lock_hold_ms_read=getattr(driver, "_arm_reader_lock_hold_ms", None),
            serial_lock_wait_ms_write=getattr(self.adapter, "last_lock_wait_ms", None)
            if write_ms is not None
            else None,
            serial_lock_hold_ms_write=getattr(self.adapter, "last_lock_hold_ms", None)
            if write_ms is not None
            else None,
            sample_to_command_ms=None if sample is None else (end - sample.timestamp_ns) / 1e6,
            command_timestamp_ns=command_ns or None,
            command_age_ms=command_age_ms,
            command_valid=command is not None and command_error is None and not command_stale,
            history_ready=estimate.ready,
            model_valid=estimate.model_valid,
            inference_latency_ms=estimate.inference_latency_ms,
            estimator_mode=estimate.estimator_mode,
            estimator_status=estimate.status,
            feedback_enabled=bool(
                self.config.enabled
                and not self.config.observe_only
                and estimate.ready
                and not result.fault
            ),
            xarm_latency_ms=0 if sample is None else sample.read_latency_ms,
            robot_state=None if sample is None else sample.robot_state,
            robot_mode=None if sample is None else sample.robot_mode,
            validity=bool(
                sample is not None
                and sample.error is None
                and (self.config.estimator.mode != "next" or estimate.ready)
            ),
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
            tau_measured=np.zeros(7) if sample is None else sample.raw_joint_effort,
            tau_free_pred=np.zeros(7) if tau_free is None else tau_free,
            tau_ext_raw=np.zeros(7) if tau_ext is None else tau_ext,
            tau_ext_filtered=result.filtered_external_torque,
            contact=result.contact.astype(float),
            contact_gate=result.contact_gate,
            feedback_target=result.processed_feedback_ma,
            feedback_current_ma=actual,
            feedback_current_raw=getattr(self.adapter, "last_raw", np.zeros(7)),
            q=np.zeros(7) if sample is None or sample.position is None else sample.position,
            qdot=np.zeros(7) if sample is None or sample.velocity is None else sample.velocity,
            qcmd=np.zeros(7) if command is None else command,
            qerror=np.zeros(7)
            if command is None or sample is None or sample.position is None
            else command - sample.position,
            sign_product=np.zeros(7)
            if sample is None
            else sample.estimated_contact_torque * result.command_current_ma,
        )
        if self.config.dynamic_mode or self.config.estimator.mode == "next":
            arrays.update(
                joint_position=np.zeros(7)
                if sample is None or sample.position is None
                else sample.position,
                joint_velocity=np.zeros(7)
                if sample is None or sample.velocity is None
                else sample.velocity,
                estimated_external_torque=np.zeros(7) if tau_ext is None else tau_ext,
            )
        if self.config.estimator.shadow_baseline:
            arrays["tau_ext_baseline"] = (
                np.zeros(7) if tau_baseline is None else tau_baseline
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
