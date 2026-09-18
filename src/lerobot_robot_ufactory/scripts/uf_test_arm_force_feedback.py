"""Read-only xArm probe + optional GELLO haptics; never commands follower motion.

Use --summarize CSV without hardware. For position teleoperation put the same
arm_feedback mapping under teleop.arm_feedback in your existing config.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import yaml

from ..utils.arm_feedback import ArmFeedbackConfig, ArmFeedbackProcessor, ArmFeedbackSample


def benchmark(iterations=10000):
    """CPU-only processor timing; not a hardware/control-frequency benchmark."""
    config = ArmFeedbackConfig(enabled=True, enabled_joints=(True,) * 7, gain_ma_per_unit=(5,) * 7)
    processor = ArmFeedbackProcessor(config)
    timings = []
    for k in range(iterations + 200):
        sample = ArmFeedbackSample(time.monotonic_ns(), np.ones(7), np.ones(7))
        before = time.monotonic_ns()
        result = processor.process(sample, np.zeros(7), time.monotonic_ns())
        if result.fault:
            raise RuntimeError(result.fault)
        if k >= 200:
            timings.append((time.monotonic_ns() - before) / 1e6)
    return dict(
        kind="CPU ONLY, no hardware",
        iterations=iterations,
        processing_p50_ms=float(np.percentile(timings, 50)),
        processing_p95_ms=float(np.percentile(timings, 95)),
        xarm_sampling="NOT MEASURED",
        dynamixel_read="NOT MEASURED",
        dynamixel_write="NOT MEASURED",
        end_to_end="NOT MEASURED",
    )


def summarize(path):
    with Path(path).open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    result = {}

    def stats(name, values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        result[name] = (
            dict(
                count=len(values),
                mean=float(np.mean(values)),
                p50=float(np.percentile(values, 50)),
                p95=float(np.percentile(values, 95)),
            )
            if len(values)
            else "NOT MEASURED"
        )

    samples, leaders = {}, {}
    for row in rows:
        if row["sample_timestamp_ns"]:
            samples[row["sample_timestamp_ns"]] = row
        if row["leader_timestamp_ns"]:
            leaders[row["leader_timestamp_ns"]] = row
    for name, selected, key in (
        ("xarm_response_period_ms", list(samples.values()), "sample_period_ms"),
        ("processing_latency_ms", rows, "processing_latency_ms"),
        ("dynamixel_write_latency_ms", rows, "write_latency_ms"),
        (
            "sample_to_command_ms",
            [r for r in rows if r["observe_only"] == "False"],
            "sample_to_command_ms",
        ),
        ("loop_dt_ms", rows, "loop_dt_ms"),
        ("leader_read_latency_ms", list(leaders.values()), "leader_read_latency_ms"),
        ("leader_period_ms", list(leaders.values()), "leader_period_ms"),
        ("serial_transaction_ms", rows, "serial_transaction_ms"),
    ):
        stats(name, [float(r[key]) for r in selected if r[key] and float(r[key]) > 0])
    for label, selected, key in (
        ("xarm_response_mean_hz", samples, "sample_timestamp_ns"),
        ("gello_read_mean_hz", leaders, "leader_timestamp_ns"),
    ):
        ts = sorted(int(r[key]) for r in selected.values() if r[key])
        sequence_key = "sample_sequence" if selected is samples else "leader_sequence"
        sequences = [int(r[sequence_key]) for r in selected.values()]
        count = max(sequences) - min(sequences) if sequences else 0
        result[label] = (
            count * 1e9 / (ts[-1] - ts[0]) if len(ts) > 1 and ts[-1] > ts[0] else "NOT MEASURED"
        )
    result["gello_write_mean_hz"] = "NOT MEASURED"
    writes = [int(r["timestamp_ns"]) for r in rows if r["write_latency_ms"]]
    if len(writes) > 1:
        result["gello_write_mean_hz"] = (len(writes) - 1) * 1e9 / (writes[-1] - writes[0])
    if rows:
        for key in ("stale_count", "write_error_count", "loop_overrun_count"):
            result[key] = max(int(r[key]) for r in rows)
        result["faults"] = sorted({r["fault"] for r in rows if r["fault"]})
        result["raw_effort_median_not_an_automatic_baseline"] = [
            float(np.median([float(r[f"raw_effort_{j}"]) for r in samples.values()]))
            for j in range(1, 8)
        ]
    status = Path(path).with_suffix(".status.json")
    if status.exists():
        result["session_status"] = json.loads(status.read_text(encoding="utf-8"))
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config-path", type=Path, help="existing robot/teleop YAML (IP/port)")
    p.add_argument("--feedback-config", type=Path, help="YAML containing arm_feedback")
    p.add_argument("--duration", type=float, default=60)
    p.add_argument("--log-path", type=Path)
    p.add_argument(
        "--active", action="store_true", help="explicitly permit configured motor current"
    )
    p.add_argument(
        "--joints", type=int, nargs="+", default=[7], help="physical IDs, active or hypothetical"
    )
    p.add_argument("--summarize", type=Path, help="analyze existing CSV, no hardware")
    p.add_argument("--benchmark", action="store_true", help="CPU-only processor benchmark")
    return p


def main():
    args = parser().parse_args()
    if args.benchmark:
        print(json.dumps(benchmark(), indent=2))
        return 0
    if args.summarize:
        print(json.dumps(summarize(args.summarize), indent=2))
        return 0
    if not args.config_path or not args.feedback_config:
        raise ValueError("--config-path and --feedback-config required")
    if not np.isfinite(args.duration) or args.duration <= 0:
        raise ValueError("duration must be finite and positive")
    if not args.joints or any(j not in range(1, 8) for j in args.joints):
        raise ValueError("joints must be IDs 1--7")
    base = yaml.safe_load(args.config_path.read_text(encoding="utf-8"))
    values = yaml.safe_load(args.feedback_config.read_text(encoding="utf-8"))["arm_feedback"]
    values.update(
        enabled=True,
        observe_only=not args.active,
        enabled_joints=tuple(j in args.joints for j in range(1, 8)),
    )
    values["log_path"] = str(args.log_path or Path("logs") / f"arm_feedback_{time.time_ns()}.csv")
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
        print(f"Arm feedback {'ACTIVE' if args.active else 'OBSERVE ONLY'}; CSV: {worker.log_path}")
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
        print(json.dumps(summarize(worker.log_path), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
