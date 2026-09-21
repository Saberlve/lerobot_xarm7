"""Read-only xArm J2 effort sign calibration logger.

Connects ONLY to xArm and ONLY calls get_joint_states(is_radian=True, num=3)
(a read-only GET_JOINT_POS register read). It never sends motion, torque,
current, mode or enable/disable commands, never connects to GELLO, and never
starts any feedback worker. The report stream (enable_report=True) is kept
on because the controller only refreshes the velocity/effort half of
GET_JOINT_POS while a report client is attached; with report off the effort
values can stay bit-frozen for the whole session (observed on hardware).

Fixed 25 s protocol (timed, no keypresses between windows):
  0-5 s    FREE (baseline median computed automatically from this window)
  5-10 s   operator applies a gentle +q2 external tendency
  10-15 s  FREE
  15-20 s  operator applies a gentle -q2 external tendency
  20-25 s  FREE

Logs timestamp / q2 / effort2 / delta_effort2 (= effort2 - baseline) at
~100 Hz, then reports median/min/max of delta_effort2 for the +q2 and -q2
windows. The baseline is a FIXED-POSE baseline: it is only valid while the
arm stays in the exact pose held during the 0-5 s FREE window; any pose
change (especially J2, which carries gravity load) invalidates it.

Effort units are unspecified by xArm SDK 1.18.4 (same caveat as
utils/arm_feedback.py). This script never decides or writes config
``sign[1]``; the printed evidence is for the human calibration decision.

Run from the repo root:
    python -m lerobot_robot_ufactory.scripts.uf_test_xarm_j2_sign
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import yaml

TARGET_JOINT = 2
JOINT_INDEX = TARGET_JOINT - 1

# 25 s fixed timeline: (t_start_s, t_end_s, phase label, operator instruction).
WINDOWS = (
    (0.0, 5.0, "free_1", "FREE: keep the xArm completely untouched"),
    (5.0, 10.0, "plus_q2", "apply a gentle +q2 external tendency to J2"),
    (10.0, 15.0, "free_2", "FREE: release the robot"),
    (15.0, 20.0, "minus_q2", "apply a gentle -q2 external tendency to J2"),
    (20.0, 25.0, "free_3", "FREE: keep the xArm completely untouched"),
)
TOTAL_S = 25.0
# Announce the next window this many seconds before it starts.
ANNOUNCE_AHEAD_S = 2.0


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
    p.add_argument("--log-path", type=Path)
    return p


def load_robot_ip(config_path: Path) -> str:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    robot = data.get("robot") if isinstance(data, dict) else None
    ip = robot.get("robot_ip") if isinstance(robot, dict) else None
    if not isinstance(ip, str) or not ip:
        raise ValueError(f"{config_path} does not contain robot.robot_ip")
    return ip


def summarize(label: str, deltas: np.ndarray) -> None:
    print(
        f"{label}: samples={deltas.size} "
        f"delta_effort2 median={np.median(deltas):+.5f} "
        f"min={np.min(deltas):+.5f} max={np.max(deltas):+.5f}"
    )


def main() -> int:
    args = parser().parse_args()
    if not 10.0 <= args.hz <= 200.0:
        parser().error("--hz must be in [10, 200]")
    robot_ip = args.robot_ip or load_robot_ip(args.config_path)

    log_path = args.log_path or Path("logs") / f"xarm_j2_sign_{time.time_ns()}.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        raise ValueError(f"refusing to overwrite existing log: {log_path}")

    from xarm.wrapper import XArmAPI

    print("=" * 56)
    print("READ-ONLY XARM J2 EFFORT SIGN CALIBRATION")
    print(f"TARGET: xArm J2 ONLY (index {JOINT_INDEX})")
    print(f"ROBOT IP: {robot_ip}")
    print("NO GELLO CONNECTION / NO MOTION OR TORQUE COMMANDS")
    print(f"CSV: {log_path}")
    print("PROTOCOL (25 s, timed):")
    for t0, t1, _, instruction in WINDOWS:
        print(f"  {t0:4.0f}-{t1:<4.0f}s {instruction}")
    print("NOTE: the baseline is computed from the 0-5 s FREE window and is")
    print("only valid for the pose held during that window. Do not change")
    print("the arm pose before or during the run.")
    print("=" * 56)

    api = None
    log_file = None
    try:
        api = XArmAPI(robot_ip, is_radian=True, enable_report=True, report_type="rich", timeout=1.0)
        # Sanity read before any user interaction; read-only.
        read_once(api)

        input(
            "Hold the arm in the calibration pose, keep it untouched, then "
            "press Enter to start the 25 s experiment: "
        )

        period = 1.0 / args.hz
        start = time.monotonic()
        samples = []  # (t_s, phase, q2, effort2)
        announced_next = set()
        started = set()
        while True:
            now = time.monotonic()
            t = now - start
            if t >= TOTAL_S:
                break
            for t0, t1, phase, instruction in WINDOWS:
                if (
                    phase not in announced_next
                    and t0 - ANNOUNCE_AHEAD_S <= t < t0
                ):
                    print(f"[{t:5.1f}s] next ({t0:.0f}-{t1:.0f}s): {instruction}")
                    announced_next.add(phase)
                if t0 <= t < t1 and phase not in started:
                    print(f"[{t:5.1f}s] START {phase}: {instruction}")
                    started.add(phase)
            phase = next(p for t0, t1, p, _ in WINDOWS if t0 <= t < t1)
            q, effort = read_once(api)
            samples.append((t, phase, float(q[JOINT_INDEX]), float(effort[JOINT_INDEX])))
            delay = period - (time.monotonic() - now)
            if delay > 0:
                time.sleep(delay)
        print(f"[{TOTAL_S:.1f}s] DONE")

        # Fixed-pose baseline: median of effort2 over the first FREE window.
        free1 = np.array([e for t, p, _, e in samples if p == "free_1"])
        if free1.size < 10:
            raise RuntimeError(
                f"too few baseline samples ({free1.size}); cannot calibrate"
            )
        baseline = float(np.median(free1))
        print(
            f"baseline (fixed pose, free_1 window): effort2 median = "
            f"{baseline:+.6f} (n={free1.size})"
        )
        print(
            "WARNING: this baseline is valid ONLY for the current fixed "
            "pose; re-run after any pose change."
        )

        log_file = log_path.open("x", newline="", encoding="utf-8")
        writer = csv.DictWriter(
            log_file,
            fieldnames=["timestamp_s", "phase", "q2", "effort2", "delta_effort2"],
        )
        writer.writeheader()
        for t, phase, q2, effort2 in samples:
            writer.writerow(
                dict(
                    timestamp_s=f"{t:.4f}",
                    phase=phase,
                    q2=f"{q2:+.6f}",
                    effort2=f"{effort2:+.5f}",
                    delta_effort2=f"{effort2 - baseline:+.5f}",
                )
            )
        log_file.flush()
        print(f"CSV written: {log_path} ({len(samples)} rows)")

        print("=" * 56)
        for label, phase in (("+q2 window", "plus_q2"), ("-q2 window", "minus_q2")):
            deltas = np.array([e - baseline for _, p, _, e in samples if p == phase])
            if deltas.size == 0:
                print(f"{label}: NO SAMPLES")
            else:
                summarize(label, deltas)
        print("sign[1] NOT modified by this diagnostic; decide manually.")
        print("=" * 56)
    except KeyboardInterrupt:
        print("\nInterrupted; closing read-only session.", file=sys.stderr)
    except Exception as exc:
        print(f"xArm J2 calibration failed: {exc}", file=sys.stderr)
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
