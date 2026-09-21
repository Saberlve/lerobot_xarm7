"""Seven-axis experimental haptics. No hardware I/O in configuration/processing.

Effort units are deliberately NOT labelled Nm: xArm SDK 1.18.4 does not specify
them. gain_ma_per_unit is an experimental calibration, not a torque constant.
For ft_sensor the input unit is Nm and gain is mA/Nm. Damping is mA/(rad/s).
"""

from dataclasses import dataclass, field

import numpy as np


def vector7(value, name="vector"):
    a = np.asarray(value, dtype=float)
    if a.shape != (7,) or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a finite shape=(7,) vector")
    return a.copy()


@dataclass
class ArmEstimatorConfig:
    """Select the source of the external-torque estimate.

    ``baseline`` preserves the validated Stage A/B behavior. ``next`` enables
    the learned free-motion model. ``shadow_baseline`` only adds a parallel
    baseline residual to diagnostics; it never changes the command path.
    """

    mode: str = "baseline"
    shadow_baseline: bool = False

    def __post_init__(self):
        self.mode = str(self.mode).lower()
        if self.mode not in ("baseline", "next"):
            raise ValueError("estimator.mode must be 'baseline' or 'next'")
        if type(self.shadow_baseline) is not bool:
            raise ValueError("estimator.shadow_baseline must be bool")


@dataclass
class NextEstimatorConfig:
    """Paths and runtime policy for the NEXT estimator."""

    enabled: bool = False
    checkpoint: str = ""
    normalization: str = ""
    config: str = ""
    device: str = "cpu"
    fallback: str = "baseline"
    inference_timeout_ms: float = 50.0
    command_stale_timeout_ms: float = 500.0

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("next.enabled must be bool")
        self.fallback = str(self.fallback).lower()
        if self.fallback not in ("baseline", "disable"):
            raise ValueError("next.fallback must be 'baseline' or 'disable'")
        for name in ("inference_timeout_ms", "command_stale_timeout_ms"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"next.{name} must be finite and positive")
        for name in ("checkpoint", "normalization", "config", "device"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"next.{name} must be a string")


@dataclass
class ArmContactConfig:
    """Per-joint contact hysteresis and time-based feedback ramp.

    Thresholds use the estimator output unit. For a NEXT model trained on
    calibrated N-m telemetry they are N-m; the legacy SDK effort source remains
    explicitly unit-unspecified.
    """

    enabled: bool = False
    threshold_nm: tuple[float, ...] = (1.0,) * 7
    release_threshold_nm: tuple[float, ...] = (0.5,) * 7
    debounce_ms: float = 0.0
    ramp_up_ms: float = 100.0
    ramp_down_ms: float = 100.0

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("contact.enabled must be bool")
        self.threshold_nm = tuple(vector7(self.threshold_nm, "contact.threshold_nm"))
        self.release_threshold_nm = tuple(
            vector7(self.release_threshold_nm, "contact.release_threshold_nm")
        )
        if np.any(np.asarray(self.threshold_nm) <= 0):
            raise ValueError("contact.threshold_nm must be positive")
        if np.any(np.asarray(self.release_threshold_nm) < 0):
            raise ValueError("contact.release_threshold_nm must be nonnegative")
        if np.any(np.asarray(self.release_threshold_nm) >= self.threshold_nm):
            raise ValueError("contact release thresholds must be below enter thresholds")
        for name in ("debounce_ms", "ramp_up_ms", "ramp_down_ms"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value < 0:
                raise ValueError(f"contact.{name} must be finite and nonnegative")


@dataclass
class ArmFeedbackConfig:
    enabled: bool = False
    observe_only: bool = True
    source: str = "bias_compensated_joint_effort"
    update_hz: float = 100.0
    sampling_hz: float = 100.0
    stale_timeout_ms: float = 100.0
    baseline: tuple[float, ...] = (0.0,) * 7
    bias: tuple[float, ...] = (0.0,) * 7
    input_limit: tuple[float, ...] = (1.0,) * 7
    spike_limit: tuple[float, ...] | None = None
    deadzone: tuple[float, ...] = (0.0,) * 7
    ema_alpha: tuple[float, ...] = (0.2,) * 7  # weight of NEW sample
    gain_ma_per_unit: tuple[float, ...] = (0.0,) * 7
    sign: tuple[int, ...] = (1,) * 7  # final physical motor direction
    damping_ma_per_rad_s: tuple[float, ...] = (0.0,) * 7
    slew_rate_ma_s: tuple[float, ...] = (10.0,) * 7
    current_limit_ma: tuple[float, ...] = (5.0,) * 7
    enabled_joints: tuple[bool, ...] = (False,) * 7
    # Explicit calibration acknowledgements; never inferred from FACTR.
    baseline_verified: bool = False
    sign_verified: bool = False
    max_temperature_c: float = 55.0
    log_path: str = "logs/arm_feedback.csv"
    # Arm-feedback-session-only leader read rate on the shared 57600 baud bus.
    # 0 keeps the reader free-running (~20 Hz, ~90% bus duty). A positive rate
    # paces the reader so Goal Current writes get free serial-bus windows: one
    # 8-servo SyncRead holds the bus ~47 ms, so the write rate ceiling is
    # roughly update_hz * (1 - 0.047 * leader_read_hz).
    # Damping is the only consumer that needs fresh leader velocity; with
    # damping=0 on every enabled joint the leader snapshot only feeds logging
    # and the leader stale gate is skipped (see ArmFeedbackWorker).
    leader_read_hz: float = 0.0
    # Stage C1: read q/qd from the same report packet as effort (synchronized)
    # and log estimated_external_torque. Math is unchanged: external = effort
    # - baseline. The active write path is untouched.
    dynamic_mode: bool = False
    # FT wrench is expressed at sensor origin in sensor axes. Transform maps
    # sensor -> flange; translation in mm. User must establish this calibration.
    ft_sensor_to_flange: tuple[float, ...] | None = None  # flattened 4x4
    ft_vertical_only: bool = False
    estimator: ArmEstimatorConfig = field(default_factory=ArmEstimatorConfig)
    next: NextEstimatorConfig = field(default_factory=NextEstimatorConfig)
    contact: ArmContactConfig = field(default_factory=ArmContactConfig)

    def __post_init__(self):
        if isinstance(self.estimator, dict):
            self.estimator = ArmEstimatorConfig(**self.estimator)
        if isinstance(self.next, dict):
            self.next = NextEstimatorConfig(**self.next)
        if isinstance(self.contact, dict):
            self.contact = ArmContactConfig(**self.contact)
        if not isinstance(self.estimator, ArmEstimatorConfig):
            raise ValueError("estimator must be an ArmEstimatorConfig")
        if not isinstance(self.next, NextEstimatorConfig):
            raise ValueError("next must be a NextEstimatorConfig")
        if not isinstance(self.contact, ArmContactConfig):
            raise ValueError("contact must be an ArmContactConfig")
        if self.estimator.mode == "next" and not self.next.enabled:
            raise ValueError("estimator.mode=next requires next.enabled=true")
        for name in (
            "enabled",
            "observe_only",
            "baseline_verified",
            "sign_verified",
            "ft_vertical_only",
            "dynamic_mode",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be bool")
        if self.source not in (
            "disabled",
            "raw_joint_effort",
            "bias_compensated_joint_effort",
            "ft_sensor",
        ):
            raise ValueError("unsupported arm feedback source")
        for name in ("update_hz", "sampling_hz", "stale_timeout_ms", "max_temperature_c"):
            v = getattr(self, name)
            if isinstance(v, bool) or not np.isfinite(v) or v <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_temperature_c > 60:
            raise ValueError("experimental arm temperature limit must be <=60 C")
        if isinstance(self.leader_read_hz, bool) or not np.isfinite(self.leader_read_hz):
            raise ValueError("leader_read_hz must be finite")
        if self.leader_read_hz != 0 and not 1 <= self.leader_read_hz <= 100:
            raise ValueError("leader_read_hz must be 0 (free-running) or within 1--100 Hz")
        names = (
            "baseline",
            "bias",
            "input_limit",
            "deadzone",
            "ema_alpha",
            "gain_ma_per_unit",
            "sign",
            "damping_ma_per_rad_s",
            "slew_rate_ma_s",
            "current_limit_ma",
        )
        for name in names:
            raw = getattr(self, name)
            if any(isinstance(x, (bool, np.bool_)) for x in raw):
                raise ValueError(f"{name} must be numeric, not bool")
            a = vector7(raw, name)
            if name not in ("baseline", "bias", "sign") and np.any(a < 0):
                raise ValueError(f"{name} must be nonnegative")
            setattr(self, name, tuple(a))
        if self.spike_limit is not None:
            if any(isinstance(x, (bool, np.bool_)) for x in self.spike_limit):
                raise ValueError("spike_limit must be numeric, not bool")
            spike_limit = vector7(self.spike_limit, "spike_limit")
            if np.any(spike_limit <= 0):
                raise ValueError("spike_limit must be positive")
            self.spike_limit = tuple(spike_limit)
        if not np.all(np.isin(self.sign, [-1, 1])):
            raise ValueError("sign must contain +/-1")
        if np.any(np.asarray(self.ema_alpha) > 1):
            raise ValueError("ema_alpha must be in [0,1]")
        if np.any(np.asarray(self.current_limit_ma) > 100):
            raise ValueError("experimental arm current limit must be <=100 mA")
        if len(self.enabled_joints) != 7 or any(type(x) is not bool for x in self.enabled_joints):
            raise ValueError("enabled_joints must contain seven bools")
        if np.any(np.asarray(self.deadzone) > self.input_limit):
            raise ValueError("deadzone must not exceed input_limit")
        if self.source == "ft_sensor":
            t = np.asarray(self.ft_sensor_to_flange, dtype=float)
            if t.size != 16 or not np.all(np.isfinite(t)):
                raise ValueError("FT requires explicit sensor-to-flange 4x4 calibration")
            t = t.reshape(4, 4)
            if (
                not np.allclose(t[3], [0, 0, 0, 1])
                or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(t[:3, :3]), 1)
            ):
                raise ValueError("invalid FT rigid transform")
        if self.enabled and not self.observe_only:
            if self.source in ("disabled", "raw_joint_effort"):
                raise ValueError("raw/disabled sources are observe-only")
            if not self.sign_verified or not self.baseline_verified:
                raise ValueError("active mode requires baseline_verified and sign_verified")
            if not any(self.enabled_joints):
                raise ValueError("active mode requires enabled_joints")
            mask = np.asarray(self.enabled_joints)
            for name in ("input_limit", "slew_rate_ma_s", "current_limit_ma"):
                if np.any(np.asarray(getattr(self, name))[mask] <= 0):
                    raise ValueError(f"active {name} must be positive")
        if not isinstance(self.log_path, str) or not self.log_path:
            raise ValueError("log_path required")


@dataclass(frozen=True)
class ArmFeedbackSample:
    timestamp_ns: int
    raw_joint_effort: np.ndarray
    estimated_contact_torque: np.ndarray  # effort units OR Nm; see source/unit
    unit: str = "sdk_effort_unit"
    error: str | None = None
    read_latency_ms: float = 0.0
    sequence: int = 0
    period_ms: float = 0.0
    # Stage C1: report-synchronized follower state; only populated in
    # dynamic_mode. The processor never reads these.
    position: np.ndarray | None = None
    velocity: np.ndarray | None = None
    robot_state: int | None = None
    robot_mode: int | None = None


@dataclass
class ArmFeedbackResult:
    command_current_ma: np.ndarray = field(default_factory=lambda: np.zeros(7))
    processed_feedback_ma: np.ndarray = field(default_factory=lambda: np.zeros(7))
    sample_age_ms: float = float("inf")
    stale: bool = False
    clamped: bool = False
    fault: str | None = None
    filtered_external_torque: np.ndarray = field(default_factory=lambda: np.zeros(7))
    contact: np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=bool))
    contact_gate: np.ndarray = field(default_factory=lambda: np.ones(7))


class ContactGate:
    """Per-joint hysteresis/debounce with a continuous time-based gate."""

    def __init__(self, config: ArmContactConfig):
        self.config = config
        self.reset()

    def reset(self):
        self.contact = np.zeros(7, dtype=bool)
        self.gate = np.ones(7) if not self.config.enabled else np.zeros(7)
        self.pending = np.zeros(7, dtype=bool)
        self.pending_since_ns = np.zeros(7, dtype=np.int64)
        self.last_ns = None

    def update(self, values, now_ns):
        values = np.abs(vector7(values, "contact_input"))
        if not self.config.enabled:
            self.contact[:] = False
            self.gate[:] = 1.0
            self.last_ns = now_ns
            return self.contact.copy(), self.gate.copy()

        enter = np.asarray(self.config.threshold_nm)
        release = np.asarray(self.config.release_threshold_nm)
        desired = np.where(self.contact, values > release, values >= enter)
        debounce_ns = int(self.config.debounce_ms * 1e6)
        for joint in range(7):
            if desired[joint] == self.contact[joint]:
                self.pending_since_ns[joint] = 0
                continue
            if self.pending_since_ns[joint] == 0 or self.pending[joint] != desired[joint]:
                self.pending[joint] = desired[joint]
                self.pending_since_ns[joint] = now_ns
            if debounce_ns == 0 or now_ns - self.pending_since_ns[joint] >= debounce_ns:
                self.contact[joint] = desired[joint]
                self.pending_since_ns[joint] = 0

        dt_ms = 0.0 if self.last_ns is None else max(0.0, (now_ns - self.last_ns) / 1e6)
        self.last_ns = now_ns
        for joint, active in enumerate(self.contact):
            duration = self.config.ramp_up_ms if active else self.config.ramp_down_ms
            target = 1.0 if active else 0.0
            if duration == 0:
                self.gate[joint] = target
            else:
                step = dt_ms / duration
                self.gate[joint] += np.clip(target - self.gate[joint], -step, step)
        return self.contact.copy(), self.gate.copy()


class ArmFeedbackProcessor:
    def __init__(self, config: ArmFeedbackConfig):
        self.config = config
        self.reset()

    def reset(self):
        self.filtered = np.zeros(7)
        self.previous = np.zeros(7)
        self.last_ns = None
        self.last_sample_ns = None
        self.contact_gate = ContactGate(self.config.contact)

    def process(self, sample, leader_velocity, now_ns, motor_signs=(1,) * 7):
        result = ArmFeedbackResult()
        try:
            if sample is None:
                raise ValueError("sample_unavailable")
            if sample.error:
                raise ValueError(sample.error)
            if not isinstance(now_ns, int) or not isinstance(sample.timestamp_ns, int):
                raise ValueError("invalid timestamp")
            result.sample_age_ms = (now_ns - sample.timestamp_ns) / 1e6
            if result.sample_age_ms < 0:
                raise ValueError("future sample")
            if result.sample_age_ms > self.config.stale_timeout_ms:
                result.stale = True
                raise ValueError("sample_stale")
            vector7(sample.raw_joint_effort, "raw_effort")
            contact = vector7(sample.estimated_contact_torque, "contact_estimate")
            velocity = vector7(leader_velocity, "leader_velocity")
            signs = vector7(motor_signs, "motor_signs")
            if not np.all(np.isin(signs, [-1, 1])):
                raise ValueError("invalid motor_signs")
            dt = 0 if self.last_ns is None else (now_ns - self.last_ns) / 1e9
            if dt < 0 or dt * 1000 > self.config.stale_timeout_ms:
                raise ValueError("processing_timeout")
            if self.last_sample_ns is not None and sample.timestamp_ns < self.last_sample_ns:
                raise ValueError("out_of_order_sample")
            c = self.config
            corrected = contact - c.bias
            if c.spike_limit is not None and np.any(
                np.abs(corrected) > np.asarray(c.spike_limit)
            ):
                raise ValueError("torque_spike")
            clipped = np.clip(corrected, -np.asarray(c.input_limit), c.input_limit)
            dead = np.sign(clipped) * np.maximum(np.abs(clipped) - c.deadzone, 0)
            # Do not repeatedly filter the same latest-value sample.
            if sample.timestamp_ns != self.last_sample_ns:
                alpha = np.asarray(c.ema_alpha)
                self.filtered += alpha * (dead - self.filtered)
            contact_state, contact_gate = self.contact_gate.update(self.filtered, now_ns)
            # Feedback signs map to physical motor axes. Damping always opposes
            # physical motor velocity, independently of contact feedback signs.
            target = (
                contact_gate * self.filtered * c.gain_ma_per_unit * c.sign
                - np.asarray(c.damping_ma_per_rad_s) * velocity * signs
            )
            delta = np.asarray(c.slew_rate_ma_s) * dt
            limited = np.clip(target, self.previous - delta, self.previous + delta)
            command = np.clip(limited, -np.asarray(c.current_limit_ma), c.current_limit_ma)
            command *= c.enabled_joints
            if not np.all(np.isfinite(command)) or not np.all(np.isfinite(target)):
                raise ValueError("nonfinite processing output")
            result.clamped = bool(np.any(corrected != clipped) or np.any(limited != command))
            result.processed_feedback_ma = target
            result.command_current_ma = command
            result.filtered_external_torque = self.filtered.copy()
            result.contact = contact_state
            result.contact_gate = contact_gate
            self.previous = command.copy()
            self.last_ns = now_ns
            self.last_sample_ns = sample.timestamp_ns
        except (ValueError, TypeError, OverflowError) as exc:
            self.reset()
            result.fault = str(exc)
        return result


def ft_wrench_to_joint_torque(kinematics, q, wrench_sensor, sensor_to_flange, vertical_only=False):
    """Sensor wrench [N,N,N,Nm,Nm,Nm] -> joint Nm, at sensor origin.

    Use a central-difference geometric Jacobian at the sensor origin. Existing
    kinematics is in mm: translation derivatives MUST be divided by 1000.
    No assumption that sensor z is vertical; rotate wrench into model world.
    """
    q = vector7(q)
    wrench = np.asarray(wrench_sensor, dtype=float)
    if wrench.shape != (6,) or not np.all(np.isfinite(wrench)):
        raise ValueError("FT wrench must be finite shape=(6,)")
    transform = np.asarray(sensor_to_flange, dtype=float).reshape(4, 4)
    center = kinematics.forward_matrix(q) @ transform
    rotation = center[:3, :3]
    force = rotation @ wrench[:3]
    moment = rotation @ wrench[3:]
    if vertical_only:
        force = np.array([0.0, 0.0, force[2]])
        moment = np.zeros(3)
    jacobian = np.zeros((6, 7))
    eps = 1e-6
    for j in range(7):
        step = np.eye(7)[j] * eps
        plus = kinematics.forward_matrix(q + step) @ transform
        minus = kinematics.forward_matrix(q - step) @ transform
        jacobian[:3, j] = (plus[:3, 3] - minus[:3, 3]) / (2 * eps * 1000)
        skew = ((plus[:3, :3] - minus[:3, :3]) / (2 * eps)) @ rotation.T
        jacobian[3:, j] = [skew[2, 1], skew[0, 2], skew[1, 0]]
    return jacobian.T @ np.r_[force, moment]
