"""GELLO J6 (Dynamixel ID6) fixed-current sign calibration diagnostic.

Applies +I then -I mA for at most one second each on ID6 ONLY (default
I = 5 mA, hard cap 30 mA via --current-ma), while continuously sampling the
ID6 Present Position register. Direction per pulse is decided from the net
displacement plus a least-squares trend slope over the whole pulse (robust to
encoder quantization, where most adjacent-sample velocities are zero). The
motor-position direction is mapped through joint_signs[5] so each pulse is
classified as +leader_q6 / -leader_q6 / AMBIGUOUS. This script never connects
to xArm, never starts ArmFeedbackWorker or the bilateral feedback loop, never
writes to Dynamixel IDs 1-5, 7 or 8, and never decides config ``sign[5]``.

Run from the repo root:
    python -m lerobot_robot_ufactory.scripts.uf_test_arm_j6_sign \
        --config-path config/gello/xarm7_gello_test.yaml --current-ma 20
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from lerobot_robot_ufactory.teleoperators.gello_teleop.arm_adapter import (
    ARM_MODELS,
    GelloArmFeedbackAdapter,
)
from lerobot_robot_ufactory.teleoperators.gello_teleop.gello_teleop import GelloTeleop
from lerobot_robot_ufactory.teleoperators.gello_teleop.gello_teleop_config import (
    GelloTeleopConfig,
)
from lerobot_robot_ufactory.utils.arm_feedback import ArmFeedbackConfig

TARGET_DXL_ID = 6
JOINT_INDEX = TARGET_DXL_ID - 1
# XL330-M288-T, 1.0 mA/raw, as recorded in logs/arm_stage_a_*.metadata.json.
EXPECTED_MODEL_NUMBER = 1200
DEFAULT_CURRENT_MA = 5.0
MAX_CURRENT_MA = 30.0
MAX_HOLD_S = 1.0
ENABLED_JOINTS = (False, False, False, False, False, True, False)

# Present Position(132), 4 bytes, multi-turn capable per ROBOTIS XL330 eManual.
ADDR_PRESENT_POSITION = 132
RAD_PER_COUNT = 2.0 * math.pi / 4096.0
# Displacement floor: 15 counts (~23 mrad, ~1.3 deg) sits far above the
# +/-1-2 count read noise, and is cross-checked against measured zero-current
# drift (5x). Slope floor: 5 counts/s. Least-squares slope noise of a
# quantized signal is ~sigma_q * sqrt(12/N) / T; with sigma_q ~= 1 count,
# N ~= 56, T ~= 1 s that is ~0.5 counts/s, so 5 counts/s is ~10 sigma above
# pure quantization noise while a real 1 s pulse moving >= 15 counts must
# average ~15 counts/s, 3x above the floor.
MIN_DELTA_COUNTS = 15
MIN_SLOPE_COUNTS_PER_S = 5.0
NOISE_FACTOR = 5.0
MIN_SAMPLES = 10
# Stop sampling this long before the hold deadline so that one in-flight
# position read (observed ~18 ms at ~56 Hz, including lock contention with
# the background reader) cannot push the zero-current write past hold_s.
READ_ABORT_MARGIN_S = 0.05


def require_j6(dxl_id: int) -> None:
    if dxl_id != TARGET_DXL_ID:
        raise PermissionError(
            f"this diagnostic is hard-restricted to Dynamixel ID{TARGET_DXL_ID}; "
            f"refusing ID{dxl_id}"
        )


def validate_current_ma(current_ma: float) -> float:
    """Reject anything outside (0, MAX_CURRENT_MA] before any torque enable."""
    if isinstance(current_ma, bool) or not math.isfinite(current_ma):
        raise ValueError("J6 current must be a finite number")
    if current_ma <= 0:
        raise ValueError(f"test current must be positive: {current_ma:g} mA")
    if current_ma > MAX_CURRENT_MA:
        raise ValueError(
            f"refusing test current above hard cap {MAX_CURRENT_MA:g} mA: "
            f"{current_ma:g} mA"
        )
    return float(current_ma)


def j6_command_vector(current_ma: float) -> np.ndarray:
    if isinstance(current_ma, bool) or not math.isfinite(current_ma):
        raise ValueError("J6 current must be a finite number")
    if abs(current_ma) > MAX_CURRENT_MA:
        raise ValueError(
            f"refusing |current| above hard cap {MAX_CURRENT_MA:g} mA: "
            f"{current_ma:g} mA"
        )
    vector = np.zeros(7)
    vector[JOINT_INDEX] = current_ma
    return vector


def write_j6(
    adapter: GelloArmFeedbackAdapter, current_ma: float, raw_limit: int, unit_ma: float
) -> np.ndarray:
    applied = adapter.write(j6_command_vector(current_ma))
    raw = int(round(current_ma / unit_ma))
    raw = max(-raw_limit, min(raw_limit, raw))
    if current_ma == 0.0:
        print("J6 command: 0.0 mA")
    else:
        print(f"J6 command: {current_ma:+.1f} mA (raw={raw:+d})")
    return applied


def read_id6_position_raw(driver) -> tuple[int, int]:
    """Single Present Position(132) read of ID6 in the MOTOR coordinate frame."""
    require_j6(TARGET_DXL_ID)
    with driver._lock:
        value, code, error = driver._packetHandler.read4ByteTxRx(
            driver._portHandler, TARGET_DXL_ID, ADDR_PRESENT_POSITION
        )
        driver._check_sdk_result(code, error, "ID6 Present Position(132)")
        stamp_ns = time.monotonic_ns()
    # Same 32-bit two's complement handling as the driver read loops.
    if value >= 2**31:
        value -= 2**32
    return value, stamp_ns


def sample_window(
    driver, duration_s: float, deadline: float | None = None
) -> list[tuple[int, int]]:
    """Sample ID6 raw position as fast as the serial port allows.

    The deadline check happens before every read; at most one in-flight read
    can start before the deadline, which READ_ABORT_MARGIN_S accounts for.
    """
    samples = []
    if deadline is None:
        deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        samples.append(read_id6_position_raw(driver))
    return samples


def unwrap_motor_rad(
    samples: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cumulative multi-turn unwrap of consecutive raw counts -> motor rad."""
    times_ns = np.array([stamp for _, stamp in samples], dtype=np.int64)
    raws = [raw for raw, _ in samples]
    counts = np.empty(len(raws), dtype=np.int64)
    cumulative = 0
    previous = raws[0]
    counts[0] = 0
    for i in range(1, len(raws)):
        delta = raws[i] - previous
        # Wrap-safe delta for the signed 32-bit multi-turn counter.
        if delta > 2**31:
            delta -= 2**32
        elif delta < -(2**31):
            delta += 2**32
        cumulative += delta
        counts[i] = cumulative
        previous = raws[i]
    return (times_ns - times_ns[0]) / 1e9, counts * RAD_PER_COUNT, counts


def trend_slope_rad_s(t: np.ndarray, q: np.ndarray) -> float:
    """Least-squares slope of q(t) ~= a*t + b; robust to quantization stairs."""
    if len(t) < 2 or t[-1] <= t[0]:
        return 0.0
    t_centered = t - np.mean(t)
    denominator = float(np.sum(t_centered**2))
    if denominator <= 0:
        return 0.0
    return float(np.sum(t_centered * (q - np.mean(q))) / denominator)


def analyze_window(samples: list[tuple[int, int]], joint_sign: int) -> dict:
    """Leader-frame delta q, median dq/dt and trend slope for one window."""
    t, q_motor, counts_motor = unwrap_motor_rad(samples)
    q_leader = joint_sign * q_motor
    delta_leader = float(q_leader[-1] - q_leader[0])
    dt = np.diff(t)
    valid = dt > 0
    velocities = np.diff(q_leader)[valid] / dt[valid]
    median_dq_dt = float(np.median(velocities)) if velocities.size else 0.0
    hz = (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 and t[-1] > t[0] else 0.0
    return dict(
        q_start=float(q_leader[0]),
        q_end=float(q_leader[-1]),
        delta_q=delta_leader,
        delta_counts=int(joint_sign * counts_motor[-1]),
        motor_delta_q=float(q_motor[-1] - q_motor[0]),
        median_dq_dt=median_dq_dt,
        trend_slope=trend_slope_rad_s(t, q_leader),
        hz=float(hz),
        samples=len(t),
    )


def current_pulse(
    adapter: GelloArmFeedbackAdapter,
    driver,
    current_ma: float,
    raw_limit: int,
    unit_ma: float,
    hold_s: float,
) -> tuple[list[tuple[int, int]], dict]:
    """Apply current while sampling position.

    Timing semantics:
      t0 = nonzero Goal Current write COMPLETE (current physically commanded)
      t1 = zero-current write BEGIN (sampling stopped before this)
      t2 = zero-current write COMPLETE
    commanded_nonzero_duration_s = t1 - t0, kept <= hold_s by stopping
    sampling READ_ABORT_MARGIN_S early. The write latency t2 - t1 is
    reported separately instead of being hidden inside the hold time.
    """
    write_j6(adapter, current_ma, raw_limit, unit_ma)
    applied_at = time.monotonic()
    deadline = applied_at + max(0.0, hold_s - READ_ABORT_MARGIN_S)
    samples = sample_window(driver, hold_s, deadline=deadline)
    zero_begin = time.monotonic()
    write_j6(adapter, 0.0, raw_limit, unit_ma)
    zero_done = time.monotonic()
    timing = dict(
        commanded_nonzero_duration_s=zero_begin - applied_at,
        zero_write_latency_s=zero_done - zero_begin,
        total_until_zero_write_complete_s=zero_done - applied_at,
    )
    return samples, timing


def classify_pulse(stats: dict, min_delta_rad: float, slope_threshold: float) -> str:
    """'+leader_q6' / '-leader_q6' / 'AMBIGUOUS'; never force a decision."""
    if stats["samples"] < MIN_SAMPLES or stats["hz"] <= 0:
        return "AMBIGUOUS"
    if abs(stats["delta_q"]) < min_delta_rad:
        return "AMBIGUOUS"
    if abs(stats["trend_slope"]) < slope_threshold:
        return "AMBIGUOUS"
    if math.copysign(1.0, stats["trend_slope"]) != math.copysign(
        1.0, stats["delta_q"]
    ):
        return "AMBIGUOUS"
    return "+leader_q6" if stats["delta_q"] > 0 else "-leader_q6"


def final_result(plus_dir: str, minus_dir: str) -> str:
    if (
        plus_dir != "AMBIGUOUS"
        and minus_dir != "AMBIGUOUS"
        and plus_dir != minus_dir
    ):
        return "CONFIRMED"
    return "AMBIGUOUS"


def print_pulse_report(
    label: str,
    stats: dict,
    timing: dict,
    min_delta_rad: float,
    slope_threshold: float,
    direction: str,
):
    print(f"current_ma = {label}")
    print(f"sample_count = {stats['samples']}")
    print(f"q_start = {stats['q_start']:+.6f} rad (leader frame)")
    print(f"q_end = {stats['q_end']:+.6f} rad (leader frame)")
    print(f"delta_q = {stats['delta_q']:+.6f} rad (leader frame)")
    print(f"delta_counts = {stats['delta_counts']:+d} counts (leader frame)")
    print(f"motor_delta_q = {stats['motor_delta_q']:+.6f} rad (motor frame)")
    print(f"median_dq_dt = {stats['median_dq_dt']:+.6f} rad/s (diagnostic only)")
    print(f"trend_slope_rad_s = {stats['trend_slope']:+.6f}")
    print(f"actual_position_sample_hz = {stats['hz']:.1f}")
    print(f"min_delta_threshold = {min_delta_rad:.6f} rad")
    print(f"slope_threshold = {slope_threshold:.6f} rad/s")
    print(f"commanded_nonzero_duration_s = {timing['commanded_nonzero_duration_s']:.3f}")
    print(f"zero_write_latency_s = {timing['zero_write_latency_s']:.3f}")
    print(
        f"total_until_zero_write_complete_s = "
        f"{timing['total_until_zero_write_complete_s']:.3f}"
    )
    print(f"direction = {direction}")


def load_teleop_settings(config_path: Path) -> dict:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    teleop = data.get("teleop") if isinstance(data, dict) else None
    if not isinstance(teleop, dict):
        raise ValueError(f"{config_path} does not contain a teleop configuration")
    allowed = (
        "port",
        "joint_ids",
        "joint_signs",
        "gripper_id",
        "gripper_open_deg",
        "gripper_close_deg",
    )
    settings = {key: teleop[key] for key in allowed if key in teleop}
    if "joint_ids" in settings:
        settings["joint_ids"] = tuple(settings["joint_ids"])
    if "joint_signs" in settings:
        settings["joint_signs"] = tuple(settings["joint_signs"])
    if settings.get("joint_ids") != (1, 2, 3, 4, 5, 6, 7):
        raise ValueError("this diagnostic requires joint_ids [1, 2, 3, 4, 5, 6, 7]")
    signs = settings.get("joint_signs")
    if signs is None or len(signs) != 7 or signs[JOINT_INDEX] not in (-1, 1):
        raise ValueError("teleop.joint_signs must contain seven +/-1 values")
    if settings.get("gripper_id", 8) != 8:
        raise ValueError("this diagnostic requires gripper_id 8 (left untouched)")
    return settings


def make_j6_feedback_config(current_ma: float) -> ArmFeedbackConfig:
    current_ma = validate_current_ma(current_ma)
    config = ArmFeedbackConfig(
        enabled=True,
        observe_only=False,
        source="bias_compensated_joint_effort",
        current_limit_ma=(current_ma,) * 7,
        enabled_joints=ENABLED_JOINTS,
        # These acknowledgements only satisfy ArmFeedbackConfig active-mode
        # validation; this script never runs the feedback processor. The real
        # baseline/sign calibration is exactly what remains to be done.
        baseline_verified=True,
        sign_verified=True,
    )
    if tuple(config.enabled_joints) != ENABLED_JOINTS:
        raise RuntimeError("enabled_joints mismatch; refusing to continue")
    return config


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config-path", type=Path, required=True)
    p.add_argument(
        "--current-ma",
        type=float,
        default=DEFAULT_CURRENT_MA,
        help=f"test current amplitude in mA, in (0, {MAX_CURRENT_MA:g}] "
        f"(default: {DEFAULT_CURRENT_MA:g})",
    )
    p.add_argument(
        "--hold-s",
        type=float,
        default=MAX_HOLD_S,
        help=f"duration per current step, in (0, {MAX_HOLD_S:g}] seconds",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="skip the typed confirmation after the physical ID6 model is displayed",
    )
    return p


def main() -> int:
    args = parser().parse_args()
    if not math.isfinite(args.hold_s) or not 0 < args.hold_s <= MAX_HOLD_S:
        parser().error(f"--hold-s must be finite and in (0, {MAX_HOLD_S:g}] seconds")
    try:
        # Hard rejection of NaN/inf/<=0/>MAX_CURRENT_MA before any hardware
        # access or torque enable.
        current_ma = validate_current_ma(args.current_ma)
    except ValueError as exc:
        parser().error(str(exc))

    settings = load_teleop_settings(args.config_path)
    joint_sign6 = int(settings["joint_signs"][JOINT_INDEX])
    feedback_config = make_j6_feedback_config(current_ma)
    teleop = GelloTeleop(GelloTeleopConfig(id="gello_j6_sign_probe", **settings))
    adapter = None
    connected = False
    enabled = False
    try:
        teleop.connect(calibrate=False)
        connected = True
        driver = teleop.gello_agent._robot._driver
        adapter = GelloArmFeedbackAdapter(driver, feedback_config)
        # Lets driver.close() run adapter cleanup and blocks position writes
        # and global torque enable while the current session is active.
        driver._arm_feedback_adapter = adapter

        infos = adapter.discover()  # read-only: ping + register reads, no writes
        info = infos[JOINT_INDEX]
        require_j6(info["id"])
        unit_ma = float(info["current_unit_ma"])
        raw_limit = int(math.floor(current_ma / unit_ma))
        raw_command = max(
            -raw_limit, min(raw_limit, int(round(current_ma / unit_ma)))
        )
        print("Discovered arm motors (read-only):")
        for item in infos:
            print(
                f"  ID{item['id']}: model_number={item['model']} {item['name']}, "
                f"operating_mode={item['mode']}, "
                f"hardware_current_limit_raw={item['hardware_limit_raw']}, "
                f"current_unit={item['current_unit_ma']:g} mA/raw"
            )
        if info["model"] != EXPECTED_MODEL_NUMBER:
            raise RuntimeError(
                f"ID{TARGET_DXL_ID} model {info['model']} ({info['name']}) is not the "
                f"expected {EXPECTED_MODEL_NUMBER} "
                f"({ARM_MODELS[EXPECTED_MODEL_NUMBER].name}); re-check the mA/raw "
                "conversion before enabling torque"
            )
        if not 0 < raw_limit <= info["hardware_limit_raw"]:
            raise RuntimeError(
                f"ID{TARGET_DXL_ID} software limit raw={raw_limit} is outside "
                f"hardware Current Limit {info['hardware_limit_raw']}; "
                "refusing to enable torque"
            )

        print("=" * 56)
        print(f"TARGET: GELLO ID{TARGET_DXL_ID} ONLY")
        print(f"TEST CURRENT: +/-{current_ma:.1f} mA")
        print(f"HOLD: {args.hold_s:.1f} s per step")
        print(f"CURRENT UNIT: {unit_ma:g} mA/raw (from live discovery)")
        print(f"RAW COMMAND: {raw_command:+d} (limit raw={raw_limit})")
        print(f"HARDWARE CURRENT LIMIT: {info['hardware_limit_raw']} raw")
        print(f"JOINT_SIGNS[5] = {joint_sign6:+d} (from config YAML)")
        print("NO XARM CONNECTION")
        print("=" * 56)

        if not args.yes:
            confirmation = input(
                f"Type 'ID{TARGET_DXL_ID}' to enable Current Control Mode on "
                f"ID{TARGET_DXL_ID} only: "
            ).strip()
            if confirmation != f"ID{TARGET_DXL_ID}":
                print("Cancelled before enabling torque/current output.")
                return 1

        adapter.enable()
        enabled = True

        # Zero-current settle window doubles as the noise/drift reference.
        write_j6(adapter, 0.0, raw_limit, unit_ma)
        settle = analyze_window(sample_window(driver, args.hold_s), joint_sign6)
        min_delta_rad = max(
            MIN_DELTA_COUNTS * RAD_PER_COUNT, NOISE_FACTOR * abs(settle["delta_q"])
        )
        slope_threshold = max(
            MIN_SLOPE_COUNTS_PER_S * RAD_PER_COUNT,
            NOISE_FACTOR * abs(settle["trend_slope"]),
        )
        print(
            f"zero-current settle: drift={settle['delta_q']:+.6f} rad, "
            f"slope={settle['trend_slope']:+.6f} rad/s, hz={settle['hz']:.1f}"
        )
        print(
            f"min_delta_threshold = {min_delta_rad:.6f} rad "
            f"({min_delta_rad / RAD_PER_COUNT:.1f} counts); "
            f"slope_threshold = {slope_threshold:.6f} rad/s "
            f"({slope_threshold / RAD_PER_COUNT:.1f} counts/s)"
        )

        plus_samples, plus_timing = current_pulse(
            adapter, driver, +current_ma, raw_limit, unit_ma, args.hold_s
        )
        plus = analyze_window(plus_samples, joint_sign6)
        plus_dir = classify_pulse(plus, min_delta_rad, slope_threshold)
        print_pulse_report(
            f"+{current_ma:.1f}", plus, plus_timing,
            min_delta_rad, slope_threshold, plus_dir,
        )

        time.sleep(args.hold_s)  # zero-current gap between pulses

        minus_samples, minus_timing = current_pulse(
            adapter, driver, -current_ma, raw_limit, unit_ma, args.hold_s
        )
        minus = analyze_window(minus_samples, joint_sign6)
        minus_dir = classify_pulse(minus, min_delta_rad, slope_threshold)
        print_pulse_report(
            f"-{current_ma:.1f}", minus, minus_timing,
            min_delta_rad, slope_threshold, minus_dir,
        )

        write_j6(adapter, 0.0, raw_limit, unit_ma)

        print("=" * 56)
        result = final_result(plus_dir, minus_dir)
        if result == "CONFIRMED":
            print(f"+{current_ma:.1f} mA -> {plus_dir}")
            print(f"-{current_ma:.1f} mA -> {minus_dir}")
            print(
                f"(motor frame: +current -> "
                f"{'+' if plus['motor_delta_q'] > 0 else '-'}motor_q6; "
                f"joint_signs[5]={joint_sign6:+d} already applied above)"
            )
            print("RESULT: CONFIRMED")
        else:
            print(
                f"RESULT: AMBIGUOUS (plus={plus_dir}, minus={minus_dir}); "
                "do NOT infer a sign from this run"
            )
        print("sign[5] NOT modified by this diagnostic.")
        print("=" * 56)
    except KeyboardInterrupt:
        print("\nInterrupted; applying safety cleanup.", file=sys.stderr)
    except Exception as exc:
        print(f"J6 sign diagnostic failed: {exc}", file=sys.stderr)
        return 2
    finally:
        if adapter is not None and enabled:
            try:
                adapter.write(np.zeros(7))
            except Exception as exc:
                print(f"zero-current cleanup reported: {exc}", file=sys.stderr)
            try:
                # Goal Current=0, Torque Disable, restore prior Operating Mode.
                adapter.disable()
            except Exception as exc:
                print(f"adapter cleanup reported: {exc}", file=sys.stderr)
        if connected:
            try:
                teleop.disconnect()
            except Exception as exc:
                print(f"disconnect reported: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
