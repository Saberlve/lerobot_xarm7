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
    # FT wrench is expressed at sensor origin in sensor axes. Transform maps
    # sensor -> flange; translation in mm. User must establish this calibration.
    ft_sensor_to_flange: tuple[float, ...] | None = None  # flattened 4x4
    ft_vertical_only: bool = False

    def __post_init__(self):
        for name in (
            "enabled",
            "observe_only",
            "baseline_verified",
            "sign_verified",
            "ft_vertical_only",
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


@dataclass
class ArmFeedbackResult:
    command_current_ma: np.ndarray = field(default_factory=lambda: np.zeros(7))
    processed_feedback_ma: np.ndarray = field(default_factory=lambda: np.zeros(7))
    sample_age_ms: float = float("inf")
    stale: bool = False
    clamped: bool = False
    fault: str | None = None


class ArmFeedbackProcessor:
    def __init__(self, config: ArmFeedbackConfig):
        self.config = config
        self.reset()

    def reset(self):
        self.filtered = np.zeros(7)
        self.previous = np.zeros(7)
        self.last_ns = None
        self.last_sample_ns = None

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
            clipped = np.clip(corrected, -np.asarray(c.input_limit), c.input_limit)
            dead = np.sign(clipped) * np.maximum(np.abs(clipped) - c.deadzone, 0)
            # Do not repeatedly filter the same latest-value sample.
            if sample.timestamp_ns != self.last_sample_ns:
                alpha = np.asarray(c.ema_alpha)
                self.filtered += alpha * (dead - self.filtered)
            # Feedback signs map to physical motor axes. Damping always opposes
            # physical motor velocity, independently of contact feedback signs.
            target = (
                self.filtered * c.gain_ma_per_unit * c.sign
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
