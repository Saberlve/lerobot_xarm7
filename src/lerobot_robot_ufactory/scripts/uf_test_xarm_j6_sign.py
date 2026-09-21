"""Read-only xArm J6 q/effort direction calibration logger.

Connects ONLY to xArm and ONLY calls get_joint_states(is_radian=True, num=3)
(a read-only GET_JOINT_POS register read). It never sends motion, torque,
current, mode or enable/disable commands, never connects to GELLO, and never
starts any feedback worker. The operator manually applies a gentle rotational
tendency to xArm J6 in two opposite phases; the script logs q1..q7 and
effort1..7 at ~100 Hz and reports per-phase q6 direction and effort6 response
direction (+/-/AMBIGUOUS) using whole-window robust statistics. It never
decides or writes config ``sign[5]``; the printed evidence is for the human
calibration decision.

Effort units are unspecified by xArm SDK 1.18.4 (same caveat as
utils/arm_feedback.py). Run from the repo root:
    python -m lerobot_robot_ufactory.scripts.uf_test_xarm_j6_sign
"""

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import yaml

TARGET_JOINT = 6
JOINT_INDEX = TARGET_JOINT - 1

# Absolute floors; the effective thresholds are max(floor, 5x baseline std).
# q6 floor 1 mrad (~0.06 deg): far above xArm encoder resolution, only a
# guard for an implausibly perfect baseline. effort floor 0.05 in SDK effort
# units: heuristic floor, since SDK does not specify the unit; the baseline
# std term normally dominates (observed free-space contact swing was ~3.5
# units, baseline noise far below that).
MIN_Q6_DELTA_RAD = 0.001
MIN_EFFORT_DELTA = 0.05
NOISE_FACTOR = 5.0
# J6 must carry at least half of the largest per-joint effort response,
# otherwise the manual excitation mostly coupled into other joints.
COUPLING_RATIO = 0.5
EDGE_FRACTION = 0.2  # median of first/last 20% for robust window delta

PHASES = ("baseline", "phase_a", "rest", "phase_b")


def trend_slope(t: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope of y(t) ~= a*t + b over the whole window."""
    if len(t) < 2 or t[-1] <= t[0]:
        return 0.0
    t_centered = t - np.mean(t)
    denominator = float(np.sum(t_centered**2))
    if denominator <= 0:
        return 0.0
    return float(np.sum(t_centered * (y - np.mean(y))) / denominator)


def edge_delta(y: np.ndarray) -> float:
    """median(last 20%) - median(first 20%); robust to single-sample spikes."""
    k = max(1, int(len(y) * EDGE_FRACTION))
    return float(np.median(y[-k:]) - np.median(y[:k]))


def classify(delta: float, slope: float, delta_threshold: float, slope_threshold: float):
    """Return +1 / -1 / None(AMBIGUOUS); never force a decision."""
    if not all(map(math.isfinite, (delta, slope))):
        return None
    if abs(delta) < delta_threshold:
        return None
    if abs(slope) < slope_threshold:
        return None
    if math.copysign(1.0, slope) != math.copysign(1.0, delta):
        return None
    return 1 if delta > 0 else -1


def direction_label(value, positive: str, negative: str) -> str:
    if value is None:
        return "AMBIGUOUS"
    return positive if value > 0 else negative


def read_once(api) -> tuple[np.ndarray, np.ndarray]:
    """Single read-only sample of (q[7] rad, effort[7] SDK units)."""
    code, states = api.get_joint_states(is_radian=True, num=3)
    if code != 0 or len(states) != 3:
        raise RuntimeError(f"get_joint_states failed: code={code}")
    q = np.asarray(states[0], dtype=float)
    effort = np.asarray(states[2], dtype=float)
    if q.shape != (7,) or effort.shape != (7,):
        raise RuntimeError("get_joint_states returned unexpected shapes")
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(effort)):
        raise RuntimeError("get_joint_states returned non-finite values")
    return q, effort


def sample_window(api, duration_s: float, phase: str, hz: float) -> list:
    """Paced read-only sampling; returns [(timestamp_ns, phase, q, effort)]."""
    samples = []
    period = 1.0 / hz
    start = time.monotonic()
    while time.monotonic() - start < duration_s:
        tick = time.monotonic()
        q, effort = read_once(api)
        samples.append((time.monotonic_ns(), phase, q, effort))
        delay = period - (time.monotonic() - tick)
        if delay > 0:
            time.sleep(delay)
    return samples


def window_times(samples: list) -> np.ndarray:
    t0 = samples[0][0]
    return np.array([(ts - t0) / 1e9 for ts, _, _, _ in samples])


def actual_hz(samples: list) -> float:
    if len(samples) < 2:
        return 0.0
    span = (samples[-1][0] - samples[0][0]) / 1e9
    return (len(samples) - 1) / span if span > 0 else 0.0


def analyze_phase(
    samples: list,
    q6_baseline: float,
    effort_baselines: np.ndarray,
    q6_delta_threshold: float,
    effort_threshold: float,
) -> dict:
    t = window_times(samples)
    duration = t[-1] if len(t) else 0.0
    q = np.array([s[2] for s in samples])
    effort = np.array([s[3] for s in samples])
    q6 = q[:, JOINT_INDEX]
    eff6 = effort[:, JOINT_INDEX] - effort_baselines[JOINT_INDEX]

    q6_delta = edge_delta(q6)
    q6_slope = trend_slope(t, q6)
    q6_dir = classify(
        q6_delta, q6_slope, q6_delta_threshold, q6_delta_threshold / max(duration, 1e-9)
    )
    eff_delta = float(np.median(eff6))
    eff_slope = trend_slope(t, eff6)
    eff_dir = classify(
        eff_delta, eff_slope, effort_threshold, effort_threshold / max(duration, 1e-9)
    )
    responses = np.abs(np.median(effort, axis=0) - effort_baselines)
    dominant = int(np.argmax(responses)) + 1
    j6_excited = bool(
        responses[JOINT_INDEX] >= effort_threshold
        and responses[JOINT_INDEX] >= COUPLING_RATIO * responses.max()
    )
    return dict(
        samples=len(samples),
        hz=actual_hz(samples),
        q6_delta=q6_delta,
        q6_slope=q6_slope,
        q6_dir=q6_dir,
        eff6_median_delta=eff_delta,
        eff6_slope=eff_slope,
        eff_dir=eff_dir,
        eff6_peak_range=float(np.max(eff6) - np.min(eff6)),
        responses=responses,
        dominant_joint=dominant,
        j6_excited=j6_excited,
    )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config-path",
        type=Path,
        default=Path("config/gello/xarm7_gello_test.yaml"),
        help="only robot.robot_ip is read from this YAML",
    )
    p.add_argument("--robot-ip", help="override robot.robot_ip from the YAML")
    p.add_argument("--hz", type=float, default=100.0, help="target rate, [10, 200]")
    p.add_argument("--baseline-s", type=float, default=3.0, help="in [1, 10]")
    p.add_argument("--phase-s", type=float, default=3.0, help="in [1, 10]")
    p.add_argument("--rest-s", type=float, default=2.0, help="in [1, 10]")
    p.add_argument("--log-path", type=Path)
    return p


def load_robot_ip(config_path: Path) -> str:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    robot = data.get("robot") if isinstance(data, dict) else None
    ip = robot.get("robot_ip") if isinstance(robot, dict) else None
    if not isinstance(ip, str) or not ip:
        raise ValueError(f"{config_path} does not contain robot.robot_ip")
    return ip


def main() -> int:
    args = parser().parse_args()
    if not 10.0 <= args.hz <= 200.0:
        parser().error("--hz must be in [10, 200]")
    for name in ("baseline_s", "phase_s", "rest_s"):
        value = getattr(args, name)
        if not math.isfinite(value) or not 1.0 <= value <= 10.0:
            parser().error(f"--{name.replace('_', '-')} must be finite and in [1, 10] s")
    robot_ip = args.robot_ip or load_robot_ip(args.config_path)

    log_path = args.log_path or Path("logs") / f"xarm_j6_sign_{time.time_ns()}.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        raise ValueError(f"refusing to overwrite existing log: {log_path}")

    from xarm.wrapper import XArmAPI

    print("=" * 56)
    print("READ-ONLY XARM J6 CALIBRATION")
    print(f"TARGET: xArm J6 ONLY (index {JOINT_INDEX})")
    print(f"ROBOT IP: {robot_ip}")
    print("NO GELLO CONNECTION / NO MOTION OR TORQUE COMMANDS")
    print(f"CSV: {log_path}")
    print("=" * 56)

    api = None
    log_file = None
    try:
        api = XArmAPI(robot_ip, is_radian=True, enable_report=False, timeout=1.0)
        # Sanity read before any user interaction; read-only.
        read_once(api)

        input(
            "BASELINE: keep the xArm completely untouched, then press Enter "
            f"to record {args.baseline_s:g} s: "
        )
        baseline_samples = sample_window(api, args.baseline_s, "baseline", args.hz)
        q_base = np.array([s[2] for s in baseline_samples])
        e_base = np.array([s[3] for s in baseline_samples])
        q6_baseline = float(np.median(q_base[:, JOINT_INDEX]))
        effort_baselines = np.median(e_base, axis=0)
        q6_std = float(np.std(q_base[:, JOINT_INDEX]))
        eff6_std = float(np.std(e_base[:, JOINT_INDEX]))
        q6_delta_threshold = max(MIN_Q6_DELTA_RAD, NOISE_FACTOR * q6_std)
        effort_threshold = max(MIN_EFFORT_DELTA, NOISE_FACTOR * eff6_std)
        print(
            f"baseline: hz={actual_hz(baseline_samples):.1f}, "
            f"q6_baseline={q6_baseline:+.6f} rad, "
            f"effort6_baseline={effort_baselines[JOINT_INDEX]:+.5f}, "
            f"q6_std={q6_std:.6f}, effort6_std={eff6_std:.5f}"
        )
        print(
            f"thresholds: q6_delta={q6_delta_threshold:.6f} rad, "
            f"effort6_delta={effort_threshold:.5f}"
        )

        input(
            "PHASE A: gently apply a manual rotational tendency to xArm J6 in "
            "ONE direction (do not force, do not move other joints), then "
            f"press Enter to record {args.phase_s:g} s: "
        )
        phase_a = sample_window(api, args.phase_s, "phase_a", args.hz)
        input(
            f"REST: release the robot, then press Enter to record {args.rest_s:g} s: "
        )
        rest = sample_window(api, args.rest_s, "rest", args.hz)
        input(
            "PHASE B: gently apply the OPPOSITE rotational tendency to xArm "
            f"J6, then press Enter to record {args.phase_s:g} s: "
        )
        phase_b = sample_window(api, args.phase_s, "phase_b", args.hz)

        all_samples = baseline_samples + phase_a + rest + phase_b
        fieldnames = (
            ["timestamp_ns", "phase"]
            + [f"q{j}" for j in range(1, 8)]
            + [f"effort{j}" for j in range(1, 8)]
            + ["delta_q6", "delta_effort6"]
        )
        log_file = log_path.open("x", newline="", encoding="utf-8")
        writer = csv.DictWriter(log_file, fieldnames=fieldnames)
        writer.writeheader()
        for ts, phase, q, effort in all_samples:
            row = dict(
                timestamp_ns=ts,
                phase=phase,
                delta_q6=float(q[JOINT_INDEX] - q6_baseline),
                delta_effort6=float(effort[JOINT_INDEX] - effort_baselines[JOINT_INDEX]),
            )
            row.update({f"q{j}": float(q[j - 1]) for j in range(1, 8)})
            row.update({f"effort{j}": float(effort[j - 1]) for j in range(1, 8)})
            writer.writerow(row)
        log_file.flush()
        print(f"CSV written: {log_path} ({len(all_samples)} rows)")

        stats = {}
        for name, window in (("PHASE A", phase_a), ("PHASE B", phase_b)):
            stats[name] = analyze_phase(
                window, q6_baseline, effort_baselines,
                q6_delta_threshold, effort_threshold,
            )
        valid = True
        for name in ("PHASE A", "PHASE B"):
            s = stats[name]
            print(f"{name}")
            print(f"  samples={s['samples']} hz={s['hz']:.1f}")
            print(f"  delta_q6 = {s['q6_delta']:+.6f} rad, "
                  f"q6_trend_slope = {s['q6_slope']:+.6f} rad/s")
            print(f"  q6 direction = "
                  f"{direction_label(s['q6_dir'], '+q6', '-q6')}")
            print(f"  delta_effort6_median = {s['eff6_median_delta']:+.5f}, "
                  f"effort6_trend = {s['eff6_slope']:+.5f}/s, "
                  f"peak_range = {s['eff6_peak_range']:.5f}")
            print(f"  effort response = "
                  f"{direction_label(s['eff_dir'], '+effort6', '-effort6')}")
            responses = " ".join(
                f"J{j}={s['responses'][j - 1]:.4f}" for j in range(1, 8)
            )
            print(f"  |delta_effort| per joint: {responses}")
            print(f"  dominant effort response: J{s['dominant_joint']}")
            if not s["j6_excited"]:
                valid = False
                print(
                    f"  WARNING: J6 NOT SUFFICIENTLY EXCITED "
                    f"(|J6|={s['responses'][JOINT_INDEX]:.4f} < "
                    f"{COUPLING_RATIO:g} x |J{s['dominant_joint']}|="
                    f"{s['responses'].max():.4f} or below threshold)"
                )
        a, b = stats["PHASE A"], stats["PHASE B"]
        if a["q6_dir"] is None or b["q6_dir"] is None or a["q6_dir"] == b["q6_dir"]:
            valid = False
        if a["eff_dir"] is None or b["eff_dir"] is None:
            valid = False
        print("=" * 56)
        if valid:
            print("XARM J6 CALIBRATION DATA VALID")
            print(
                "evidence: phase A q6 "
                f"{direction_label(a['q6_dir'], '+', '-')} / effort6 "
                f"{direction_label(a['eff_dir'], '+', '-')}; phase B q6 "
                f"{direction_label(b['q6_dir'], '+', '-')} / effort6 "
                f"{direction_label(b['eff_dir'], '+', '-')}"
            )
        else:
            print("XARM J6 CALIBRATION AMBIGUOUS")
        print("sign[5] NOT modified by this diagnostic; decide manually.")
        print("=" * 56)
    except KeyboardInterrupt:
        print("\nInterrupted; closing read-only session.", file=sys.stderr)
    except Exception as exc:
        print(f"xArm J6 calibration failed: {exc}", file=sys.stderr)
        return 2
    finally:
        if log_file is not None:
            try:
                log_file.close()
            except Exception as exc:
                print(f"log close reported: {exc}", file=sys.stderr)
        if api is not None:
            try:
                api.disconnect()
            except Exception as exc:
                print(f"disconnect reported: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
