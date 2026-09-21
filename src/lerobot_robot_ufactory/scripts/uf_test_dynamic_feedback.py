"""Stage C1 dynamic feedback observation; never commands GELLO current.

Records the report-synchronized (q, qd, measured_effort) plus
estimated_external_torque to CSV. C1 math is unchanged:
estimated_external_torque = measured_effort - baseline, and the end-of-run
summary re-verifies that identity on the recorded data.

Protocol:
  1. Static: hold the arm still; per-joint tau_ext std should sit near the
     effort quantization floor.
  2. Motion: move the arm slowly through the workspace; q/effort columns
     capture how far the fixed baseline drifts with pose (the C2 gravity
     model design input).

Run from the repo root (observe-only; no --active flag exists here):
    python -m lerobot_robot_ufactory.scripts.uf_test_dynamic_feedback \
        --config-path config/gello/xarm7_gello_test.yaml \
        --feedback-config config/arm_feedback/dynamic_feedback.yaml \
        --duration 60
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import yaml

from ..utils.arm_feedback import ArmFeedbackConfig


def summarize_dynamic(path):
    with Path(path).open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    if not rows:
        return result
    baseline = np.array([float(rows[0][f"baseline_{j}"]) for j in range(1, 8)])
    ext = np.array(
        [[float(r[f"estimated_external_torque_{j}"]) for j in range(1, 8)] for r in rows]
    )
    effort = np.array([[float(r[f"raw_effort_{j}"]) for j in range(1, 8)] for r in rows])
    identity_error = np.abs(ext - (effort - baseline)).max()
    result["tau_ext_equals_effort_minus_baseline_max_abs_error"] = float(identity_error)
    result["tau_ext_mean"] = [float(x) for x in ext.mean(axis=0)]
    result["tau_ext_std"] = [float(x) for x in ext.std(axis=0)]
    q = np.array([[float(r[f"joint_position_{j}"]) for j in range(1, 8)] for r in rows])
    result["joint_position_range_rad"] = [
        float(q[:, j].max() - q[:, j].min()) for j in range(7)
    ]
    result["faults"] = sorted({r["fault"] for r in rows if r["fault"]})
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config-path", type=Path, required=True, help="robot/teleop YAML")
    p.add_argument("--feedback-config", type=Path, required=True)
    p.add_argument("--duration", type=float, default=60)
    p.add_argument("--log-path", type=Path)
    p.add_argument("--summarize", type=Path, help="analyze existing CSV, no hardware")
    return p


def main():
    args = parser().parse_args()
    if args.summarize:
        print(json.dumps(summarize_dynamic(args.summarize), indent=2))
        return 0
    if not np.isfinite(args.duration) or args.duration <= 0:
        raise ValueError("duration must be finite and positive")
    base = yaml.safe_load(args.config_path.read_text(encoding="utf-8"))
    values = yaml.safe_load(args.feedback_config.read_text(encoding="utf-8"))["arm_feedback"]
    # C1 is observation-only by construction: force the safety flags here so
    # an edited yaml can never turn this script into a current-output tool.
    values.update(enabled=True, observe_only=True, dynamic_mode=True)
    values["log_path"] = str(args.log_path or Path("logs") / f"arm_dynamic_{time.time_ns()}.csv")
    config = ArmFeedbackConfig(**values)  # validation before any hardware access

    from ..teleoperators.gello_teleop.gello_teleop import GelloTeleop
    from ..teleoperators.gello_teleop.gello_teleop_config import GelloTeleopConfig

    source_config = base["teleop"]
    allowed = (
        "port",
        "joint_ids",
        "joint_signs",
        "gripper_id",
        "gripper_open_deg",
        "gripper_close_deg",
    )
    settings = {key: source_config[key] for key in allowed if key in source_config}
    teleop = GelloTeleop(GelloTeleopConfig(**settings, arm_feedback=config))
    worker = None
    try:
        teleop.connect()
        teleop.start_arm_feedback(base["robot"]["robot_ip"])
        worker = teleop._arm_feedback_worker
        print(f"Dynamic feedback OBSERVE ONLY; CSV: {worker.log_path}")
        end = time.monotonic() + args.duration
        while time.monotonic() < end:
            if worker.fault:
                raise RuntimeError(worker.fault)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        teleop.disconnect()
    if worker is not None:
        print(json.dumps(summarize_dynamic(worker.log_path), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
