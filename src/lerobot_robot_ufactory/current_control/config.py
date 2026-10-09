"""Per-device current settings and viewer calibration; angles are radians and currents are amperes."""

import math
from copy import deepcopy
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
class CurrentControlConfig:
    """Optional fixed-current and constant-magnitude damping support."""
    enabled: bool = False
    profile_path: str | None = None
    experimental: bool = True
    log_dir: str | None = None
    running_current_slew_a_s: list[float] | None = None

    def __post_init__(self):
        if self.enabled and not self.profile_path:
            raise ValueError("current_control.profile_path is required when enabled")
        if self.running_current_slew_a_s is not None:
            self.running_current_slew_a_s = tuning_slew(self.running_current_slew_a_s)
            if not self.enabled:
                raise ValueError("Running slew overrides require enabled current control")
        if self.log_dir is not None and not self.log_dir.strip():
            raise ValueError("current_control.log_dir must not be empty")

    def load_profile(self):
        profile = DeviceProfile(self.profile_path)
        profile.running_current_slew_a_s = (self.running_current_slew_a_s
                                          if self.running_current_slew_a_s is not None
                                          else profile.default_running_current_slew_a_s)
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


def current_targets(currents, damping, limits):
    """Validate a complete pair before applying either current setting."""
    constant = vector(currents, "constant_current_a")
    passive = vector(damping, "constant_damping_a", nonnegative=True)
    if np.any(np.abs(constant) + passive > limits):
        raise ValueError("每轴恒流绝对值与阻尼之和不能超过该轴电流上限")
    return {"constant_current_a": constant.tolist(),
            "constant_damping_a": passive.tolist()}


class DeviceProfile:
    """A profile binds electrical limits to a USB serial and a viewer model."""

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
        self.limits = vector(d["current_limit_a"], "current_limit_a", positive=True)
        if np.any(self.limits > 1.75):
            raise ValueError("XL330 current_limit_a exceeds the model maximum")
        # Motor-coordinate amperes; every joint has an explicit fixed target.
        currents = vector(d.get("constant_current_a", [0.0] * 7), "constant_current_a")
        if np.any(np.abs(currents) > self.limits):
            raise ValueError("constant_current_a exceeds current_limit_a")
        self.constant_current_a = currents.tolist()
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
        saved_slew = d.get("running_current_slew_a_s")
        # Saved continuous-mode defaults do not affect short commissioning runs.
        self.default_running_current_slew_a_s = None if saved_slew is None else tuning_slew(saved_slew)
        for key, default in [
            ("rate_hz", 100),
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
            not 20 <= self.rate_hz <= 200
            or self.state_timeout_s < 2 / self.rate_hz
        ):
            raise ValueError("Require rate 20..200 Hz, timeout >= two periods")
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

    def arm_only(self):
        """Select the seven arm motors for callers that do not use GELLO ID8."""
        profile = deepcopy(self)
        profile.gripper_id = -1
        profile.all_ids = profile.ids
        profile.model_numbers = profile.model_numbers[:len(profile.ids)]
        profile.data["gripper_id"] = -1
        profile.data["model_numbers"] = list(profile.model_numbers)
        return profile

    def validate_experiment(self):
        """Electrical bounds for fixed-current control, independent of the model."""
        if self.baudrate < 1000000 or self.rate_hz != 100:
            raise ValueError("Current control requires >= 1 Mbps / 100 Hz")
        if np.any(self.limits > 1.0) or np.any(self.slew > 0.05):
            raise ValueError("Current control requires current <= 1 A, startup slew <= 0.05 A/s")
        if self.ramp_s < 2 or self.temperature_limit_c > 50:
            raise ValueError("Current control requires ramp >= 2 s and temperature <= 50 C")
        if self.state_timeout_s > 0.05 or self.watchdog_ms > 100:
            raise ValueError("Current control requires state timeout <= 50 ms and watchdog <= 100 ms")
        constant = vector(self.constant_current_a, "constant_current_a")
        damping = vector(self.constant_damping_a.tolist(), "constant_damping_a", nonnegative=True)
        if np.any(np.abs(constant) + damping > self.limits):
            raise ValueError("Current plus damping exceeds current_limit_a")

    def validate_live(self, *, experimental=False):
        self.validate_experiment()

    def model_state(self, position, velocity):
        return ((position[:7] - self.zeros) * self.signs, velocity[:7] * self.signs)
