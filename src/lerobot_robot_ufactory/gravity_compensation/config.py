"""Per-device commissioning data; angles are radians and currents are amperes."""

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

MIN_TUNING_SLEW_A_S = 0.05
MAX_TUNING_SLEW_A_S = 0.20


def tuning_slew(values):
    """Bound continuous-mode running slew; commissioning limits stay intact."""
    slew = vector(values, "current_slew_a_s", positive=True)
    if np.any(slew < MIN_TUNING_SLEW_A_S) or np.any(slew > MAX_TUNING_SLEW_A_S):
        raise ValueError("七轴电流变化率必须在 0.05–0.20 A/s 之间")
    return slew.tolist()


@dataclass
class GravityCompensationConfig:
    enabled: bool = False
    profile_path: str | None = None
    experimental: bool = False
    gain: float | None = None
    j5_gain: float | None = None
    j6_gain: float | None = None
    log_dir: str | None = None
    # Absolute J1..J7 gravity gains. Legacy J5/J6 overrides take precedence.
    joint_gains: list[float] | None = None
    # Applies after startup ramp in continuous web/teleop mode only.
    running_current_slew_a_s: list[float] | None = None

    def __post_init__(self):
        if self.enabled and not self.profile_path:
            raise ValueError("gravity_compensation.profile_path is required when enabled")
        if self.experimental and not self.enabled:
            raise ValueError("Experimental gravity compensation must be enabled explicitly")
        for name in ("gain", "j5_gain", "j6_gain"):
            value = getattr(self, name)
            if value is not None:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise ValueError(f"gravity_compensation.{name} must be finite and nonnegative")
                if not self.enabled:
                    raise ValueError("Gain overrides require enabled gravity compensation")
        if self.joint_gains is not None:
            vector(self.joint_gains, "gravity_compensation.joint_gains", nonnegative=True)
            if not self.enabled:
                raise ValueError("Gain overrides require enabled gravity compensation")
        if self.running_current_slew_a_s is not None:
            self.running_current_slew_a_s = tuning_slew(self.running_current_slew_a_s)
            if not self.enabled:
                raise ValueError("Running slew overrides require enabled gravity compensation")
        if self.log_dir is not None and not self.log_dir.strip():
            raise ValueError("gravity_compensation.log_dir must not be empty")

    def load_profile(self):
        """Apply session overrides without modifying the calibration file."""
        profile = DeviceProfile(self.profile_path)
        if self.gain is not None:
            profile.gain = self.gain
        elif self.experimental:
            profile.gain = min(profile.gain, 0.02)
        profile.j5_gain = self.j5_gain
        profile.j6_gain = self.j6_gain
        profile.joint_gains = self.joint_gains
        profile.running_current_slew_a_s = self.running_current_slew_a_s
        profile.validate_live(experimental=self.experimental)
        return profile


def vector(value, name, length=7, *, positive=False, nonnegative=False):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain {length} numbers")
    if any(isinstance(v, bool) or not isinstance(v, (float, int)) for v in value):
        raise ValueError(f"{name} must contain numbers, not booleans")
    arr = np.asarray(value, dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    if positive and np.any(arr <= 0):
        raise ValueError(f"{name} must be positive")
    if nonnegative and np.any(arr < 0):
        raise ValueError(f"{name} must be non-negative")
    return arr


class DeviceProfile:
    """A profile is tied to a USB serial and the exact commissioned URDF bytes."""

    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        self.data = yaml.safe_load(self.path.read_text())
        d = self.data
        if not isinstance(d, dict) or d.get("version") != 1:
            raise ValueError("Expected a version: 1 device profile")
        self.name = str(d["name"])
        self.serial = str(d["usb_serial"])
        if not self.serial or not self.serial.isalnum():
            raise ValueError("usb_serial must be the FTDI serial number")
        self.port = f"/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_{self.serial}-if00-port0"
        self.baudrate = d.get("baudrate", 57600)
        if type(self.baudrate) is not int or self.baudrate not in (
            57600,
            115200,
            1000000,
            2000000,
            3000000,
            4000000,
        ):
            raise ValueError("Unsupported baudrate")
        self.ids = tuple(d.get("joint_ids", range(1, 8)))
        if self.ids != tuple(range(1, 8)):
            raise ValueError("This xArm7 profile requires arm IDs 1..7")
        self.gripper_id = d.get("gripper_id", 8)
        if self.gripper_id not in (8, -1):
            raise ValueError("gripper_id must be 8 or -1")
        self.all_ids = self.ids + ((8,) if self.gripper_id == 8 else ())
        self.model_numbers = tuple(
            d.get("model_numbers", [1200] * 7 + ([1190] if self.gripper_id == 8 else []))
        )
        if len(self.model_numbers) != len(self.all_ids) or any(
            x not in (1190, 1200) for x in self.model_numbers
        ):
            raise ValueError("Only explicitly identified XL330-M288/M077 models are supported")
        self.joint_names = tuple(d.get("joint_names", [f"joint{i}" for i in self.ids]))
        if len(self.joint_names) != 7 or len(set(self.joint_names)) != 7:
            raise ValueError("Exactly seven distinct joint_names are required")
        self.zeros = vector(d["encoder_zero_rad"], "encoder_zero_rad")
        self.signs = vector(d["model_signs"], "model_signs")
        if not np.isin(self.signs, [-1, 1]).all():
            raise ValueError("model_signs must be +/-1")
        self.nm_per_amp = vector(d["nm_per_amp"], "nm_per_amp", positive=True)
        self.limits = vector(d["current_limit_a"], "current_limit_a", positive=True)
        if np.any(self.limits > 1.75):
            raise ValueError("XL330 current_limit_a exceeds the model maximum")
        # Motor-coordinate amperes; None retains model-based gravity/damping.
        values = d.get("constant_current_a", [None] * 7)
        if not isinstance(values, (list, tuple)) or len(values) != 7:
            raise ValueError("constant_current_a must contain seven numbers or nulls")
        currents = vector([0 if v is None else v for v in values], "constant_current_a")
        if np.any(np.abs(currents) > self.limits):
            raise ValueError("constant_current_a exceeds current_limit_a")
        self.constant_current_a = list(values)
        self.constant_damping_a = vector(d.get("constant_damping_a", [0.0] * 7),
                                         "constant_damping_a", nonnegative=True)
        if np.any(np.abs(currents) + self.constant_damping_a > self.limits):
            raise ValueError("Constant current plus damping exceeds current_limit_a")
        self.damping_deadband_rad_s = d.get("damping_deadband_rad_s", 0.05)
        if (isinstance(self.damping_deadband_rad_s, bool)
                or not isinstance(self.damping_deadband_rad_s, (int, float))
                or not math.isfinite(self.damping_deadband_rad_s)
                or self.damping_deadband_rad_s < 0):
            raise ValueError("damping_deadband_rad_s must be finite and nonnegative")
        self.slew = vector(d["current_slew_a_s"], "current_slew_a_s", positive=True)
        self.damping = vector(d["damping_nm_s_rad"], "damping_nm_s_rad", nonnegative=True)
        self.gravity = vector(d.get("gravity_m_s2", [0, 0, -9.81]), "gravity_m_s2", 3)
        if not 9.7 < np.linalg.norm(self.gravity) < 9.9:
            raise ValueError("gravity_m_s2 must describe the measured base orientation")
        for key, default in [
            ("rate_hz", 100),
            ("gain", 0.1),
            ("ramp_s", 2),
            ("state_timeout_s", 0.05),
            ("temperature_limit_c", 55),
        ]:
            value = d.get(key, default)
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{key} must be finite and positive")
            setattr(self, key, float(value))
        if (
            self.gain > 1
            or not 20 <= self.rate_hz <= 200
            or self.state_timeout_s < 2 / self.rate_hz
        ):
            raise ValueError("Require gain <= 1, rate 20..200 Hz, timeout >= two periods")
        if self.temperature_limit_c > 65:
            raise ValueError("Commissioning temperature limit must not exceed 65 C")
        self.watchdog_ms = d.get("watchdog_ms", 100)
        if (
            type(self.watchdog_ms) is not int
            or not 20 <= self.watchdog_ms <= 200
            or self.watchdog_ms % 20
        ):
            raise ValueError("watchdog_ms must be 20..200 in multiples of 20")
        if self.watchdog_ms <= self.state_timeout_s * 1000:
            raise ValueError("Hardware watchdog must exceed the software state timeout")
        self.urdf = (self.path.parent / d["urdf"]).resolve()
        if not self.urdf.is_file():
            raise ValueError(f"Missing URDF: {self.urdf}")

    def validate_experiment(self):
        """Permit a bounded trial of estimates without declaring them commissioned."""
        if self.data.get("urdf_sha256") != hashlib.sha256(self.urdf.read_bytes()).hexdigest():
            raise ValueError("Experimental URDF checksum differs from the reviewed model")
        directions = self.data.get("direction_verification", {})
        if (
            directions.get("model_signs_verified") is not True
            or directions.get("urdf_sha256") != self.data.get("urdf_sha256")
            or directions.get("model_signs") != self.signs.tolist()
            or self.data.get("alignment_user_accepted") is not True
        ):
            raise ValueError("Experiment requires user-confirmed directions and pose alignment")
        communication = self.data.get("communication_verification", {})
        if (
            self.baudrate < 1000000
            or communication.get("baudrate_verified") != self.baudrate
            or communication.get("target_100hz_verified") is not True
            or self.rate_hz != 100
        ):
            raise ValueError("Experiment requires verified 1 Mbps or faster / 100 Hz communication")
        if not np.isfinite(self.gain) or self.gain < 0:
            raise ValueError("Experiment requires a finite nonnegative gain")
        if np.any(self.limits > 1.0) or np.any(self.slew > 0.05):
            raise ValueError("Experiment requires current <= 1 A, slew <= 0.05 A/s")
        if self.ramp_s < 2 or self.temperature_limit_c > 50:
            raise ValueError("Experiment requires ramp >= 2 s and temperature limit <= 50 C")
        if self.state_timeout_s > 0.05 or self.watchdog_ms > 100:
            raise ValueError("Experiment requires state timeout <= 50 ms and watchdog <= 100 ms")

    def validate_live(self, *, experimental=False):
        if experimental:
            self.validate_experiment()
            return
        required = (
            "geometry_verified",
            "mass_com_verified",
            "encoder_verified",
            "current_calibrated",
            "rubber_bands_removed",
            "thermal_limits_verified",
        )
        checks = self.data.get("commissioning", {})
        missing = [key for key in required if checks.get(key) is not True]
        if missing:
            raise ValueError("Profile is not commissioned: " + ", ".join(missing))
        if "UNVERIFIED_GEOMETRY" in self.urdf.read_text():
            raise ValueError(
                "URDF has UNVERIFIED_GEOMETRY (offline fixture or assembly estimate); "
                "validate the assembled model before commissioning"
            )
        digest = hashlib.sha256(self.urdf.read_bytes()).hexdigest()
        if self.data.get("urdf_sha256") != digest:
            raise ValueError("URDF checksum differs from the commissioned model")
        if self.baudrate < 1000000:
            raise ValueError("Live compensation requires a separately verified baudrate >= 1 Mbps")

    def model_state(self, position, velocity):
        return ((position[:7] - self.zeros) * self.signs, velocity[:7] * self.signs)
