"""Phase 4 G2-current to GELLO-ID8 conditioning and calibration tool."""

import argparse
import csv
import math
import shutil
import statistics
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from lerobot_robot_ufactory.scripts.uf_read_g2_current import (
    load_probe_settings,
    make_probe_robot,
)
from lerobot_robot_ufactory.scripts.uf_test_gello_gripper_current import (
    load_gello_probe_settings,
)
from lerobot_robot_ufactory.teleoperators.gello_teleop.gello_teleop import (
    GRIPPER_CURRENT_FEEDBACK_KEY,
    GelloTeleop,
)
from lerobot_robot_ufactory.teleoperators.gello_teleop.gello_teleop_config import (
    GelloTeleopConfig,
    GelloFeedbackConfig,
)
from lerobot_robot_ufactory.utils.realtime_teleop import (
    GripperFeedbackDiagnostic,
    GripperFeedbackProcessor,
)


MAX_INITIAL_OUTPUT_LIMIT_MA = 20.0
CALIBRATION_STAGES = (
    "open_idle",
    "closing_free",
    "contact_soft_object",
    "holding_soft_object",
    "reopening",
)
CSV_FIELDS = (
    "elapsed_s",
    "stage",
    "raw_current_ma",
    "bias_corrected_ma",
    "input_clamped_ma",
    "deadzone_output_ma",
    "filtered_ma",
    "target_ma",
    "command_ma",
    "sample_age_s",
    "available",
    "stale",
    "gripper_state",
    "reason",
    "monitor_error",
    "output_active",
)


@dataclass(frozen=True)
class StageStatistics:
    count: int
    mean_ma: float
    median_ma: float
    std_ma: float
    minimum_ma: float
    maximum_ma: float
    peak_abs_ma: float


def calculate_stage_statistics(values: list[float]) -> StageStatistics | None:
    if not values:
        return None
    return StageStatistics(
        count=len(values),
        mean_ma=statistics.fmean(values),
        median_ma=statistics.median(values),
        std_ma=statistics.pstdev(values),
        minimum_ma=min(values),
        maximum_ma=max(values),
        peak_abs_ma=max(abs(value) for value in values),
    )


def suggest_idle_calibration(
    stage_values: dict[str, list[float]],
) -> tuple[float, float] | None:
    """Suggest, but never apply, bias/deadzone from explicitly staged idle data."""
    idle_values = stage_values.get("open_idle", [])
    idle_stats = calculate_stage_statistics(idle_values)
    if idle_stats is None:
        return None
    suggested_bias_ma = idle_stats.median_ma
    residuals = [value - suggested_bias_ma for value in idle_values]
    suggested_deadzone_ma = max(
        max((abs(value) for value in residuals), default=0.0),
        3.0 * statistics.pstdev(residuals),
    )
    return suggested_bias_ma, suggested_deadzone_ma


def make_processor(args) -> GripperFeedbackProcessor:
    return GripperFeedbackProcessor(
        bias_ma=args.bias_ma,
        deadzone_ma=args.deadzone_ma,
        input_limit_ma=args.input_limit_ma,
        ema_beta=args.ema_beta,
        gain=args.gain,
        output_sign=args.output_sign,
        output_limit_ma=args.output_limit_ma,
        slew_rate_ma_s=args.slew_rate_ma_s,
        timeout_s=args.timeout_s,
    )


def make_output_teleop(args) -> GelloTeleop:
    settings = load_gello_probe_settings(args.config_path)
    return GelloTeleop(
        GelloTeleopConfig(
            id="gello_gripper_feedback_probe",
            port=settings.port,
            joint_ids=settings.joint_ids,
            joint_signs=settings.joint_signs,
            gripper_id=settings.gripper_id,
            gripper_open_deg=settings.gripper_open_deg,
            gripper_close_deg=settings.gripper_close_deg,
            gripper_current_control_enabled=True,
            gripper_current_limit_ma=args.phase2_current_limit_ma,
            feedback=GelloFeedbackConfig(
                enabled=True, bias_ma=args.bias_ma, deadzone_ma=args.deadzone_ma,
                input_limit_ma=args.input_limit_ma, ema_beta=args.ema_beta,
                gain=args.gain, output_sign=args.output_sign,
                output_limit_ma=args.output_limit_ma,
                slew_rate_ma_s=args.slew_rate_ma_s, timeout_s=args.timeout_s,
            ),
        )
    )


def _format_value(value: float | None, precision: int = 2) -> str:
    return "N/A" if value is None else f"{value:+.{precision}f}"


def render_multisignal_chart(
    series: dict[str, list[float | None]],
    *,
    width: int,
    height: int = 11,
    fixed_peak_ma: float | None = None,
) -> str:
    if width < 10 or height < 5:
        raise ValueError("chart dimensions are too small")
    visible = {name: values[-width:] for name, values in series.items()}
    numeric = [
        abs(value)
        for values in visible.values()
        for value in values
        if value is not None and math.isfinite(value)
    ]
    peak_ma = fixed_peak_ma or max(numeric, default=1.0)
    peak_ma = max(float(peak_ma), 1.0)
    grid = [[" " for _ in range(width)] for _ in range(height)]
    zero_row = round((height - 1) / 2)
    grid[zero_row] = ["-" for _ in range(width)]

    for symbol, values in visible.items():
        x_offset = width - len(values)
        for index, value in enumerate(values):
            if value is None or not math.isfinite(value):
                continue
            clipped = max(-peak_ma, min(peak_ma, value))
            row = round((peak_ma - clipped) / (2.0 * peak_ma) * (height - 1))
            column = x_offset + index
            existing = grid[row][column]
            grid[row][column] = symbol if existing in (" ", "-") else "#"

    lines = []
    for row, cells in enumerate(grid):
        level_ma = peak_ma - row * (2.0 * peak_ma / (height - 1))
        lines.append(f"{level_ma:8.1f} |" + "".join(cells))
    lines.append("         +" + "-" * width)
    return "\n".join(lines)


def format_dashboard(
    diagnostic: GripperFeedbackDiagnostic,
    histories: dict[str, deque],
    *,
    args,
    stage: str,
    output_active: bool,
) -> str:
    terminal_width = shutil.get_terminal_size((120, 30)).columns
    chart_width = max(10, min(args.chart_width, terminal_width - 12))
    input_chart = render_multisignal_chart(
        {
            "R": list(histories["raw"]),
            "B": list(histories["bias"]),
            "F": list(histories["filtered"]),
        },
        width=chart_width,
        fixed_peak_ma=args.input_limit_ma,
    )
    output_chart = render_multisignal_chart(
        {
            "T": list(histories["target"]),
            "C": list(histories["command"]),
        },
        width=chart_width,
        fixed_peak_ma=args.output_limit_ma,
    )
    age_ms = (
        "N/A"
        if diagnostic.sample_age_s is None
        else f"{diagnostic.sample_age_s * 1000.0:.1f}"
    )
    return "\n".join(
        (
            "Phase 4 xArm G2 -> GELLO ID8 feedback diagnostics",
            f"stage={stage} output={'ON' if output_active else 'OFF'} "
            f"active={diagnostic.feedback_active} reason={diagnostic.reason}",
            f"available={diagnostic.sample_available} stale={diagnostic.sample_stale} "
            f"state={diagnostic.gripper_state} age_ms={age_ms}",
            f"raw={_format_value(diagnostic.raw_current_ma)} mA  "
            f"bias_corrected={_format_value(diagnostic.bias_corrected_ma)} mA  "
            f"filtered={_format_value(diagnostic.filtered_ma)} mA",
            f"target={diagnostic.target_ma:+.3f} mA  "
            f"command={diagnostic.command_ma:+.3f} mA",
            f"bias={args.bias_ma:+g} deadzone={args.deadzone_ma:g} "
            f"gain={args.gain:g} sign={args.output_sign:+d} "
            f"output_limit={args.output_limit_ma:g} mA "
            f"slew={args.slew_rate_ma_s:g} mA/s beta={args.ema_beta:g}",
            "Input curves: R=raw, B=bias-corrected, F=filtered (#=overlap)",
            input_chart,
            "Output curves: T=target, C=command (#=overlap)",
            output_chart,
            "Ctrl+C stops and applies zero-current cleanup.",
        )
    )


def write_csv_row(
    writer: csv.DictWriter | None,
    *,
    elapsed_s: float,
    stage: str,
    diagnostic: GripperFeedbackDiagnostic,
    monitor_error: str | None,
    output_active: bool,
) -> None:
    if writer is None:
        return
    writer.writerow(
        {
            "elapsed_s": f"{elapsed_s:.6f}",
            "stage": stage,
            "raw_current_ma": diagnostic.raw_current_ma,
            "bias_corrected_ma": diagnostic.bias_corrected_ma,
            "input_clamped_ma": diagnostic.input_clamped_ma,
            "deadzone_output_ma": diagnostic.deadzone_output_ma,
            "filtered_ma": diagnostic.filtered_ma,
            "target_ma": diagnostic.target_ma,
            "command_ma": diagnostic.command_ma,
            "sample_age_s": diagnostic.sample_age_s,
            "available": diagnostic.sample_available,
            "stale": diagnostic.sample_stale,
            "gripper_state": diagnostic.gripper_state,
            "reason": diagnostic.reason,
            "monitor_error": monitor_error,
            "output_active": output_active,
        }
    )


def print_stage_report(stage_values: dict[str, list[float]]) -> None:
    print("\nStage raw-current statistics (mA):")
    print(
        "stage                     n       mean     median        std        min"
        "        max   peak_abs"
    )
    for stage in CALIBRATION_STAGES:
        stats = calculate_stage_statistics(stage_values.get(stage, []))
        if stats is None:
            print(f"{stage:24s}  no valid samples")
            continue
        print(
            f"{stage:24s} {stats.count:5d} "
            f"{stats.mean_ma:10.2f} {stats.median_ma:10.2f} "
            f"{stats.std_ma:10.2f} {stats.minimum_ma:10.2f} "
            f"{stats.maximum_ma:10.2f} {stats.peak_abs_ma:10.2f}"
        )
    suggestion = suggest_idle_calibration(stage_values)
    if suggestion is None:
        print("No open_idle samples; bias/deadzone suggestions are unavailable.")
        return
    bias_ma, deadzone_ma = suggestion
    print(
        "Suggested starting values from open_idle only (not applied): "
        f"bias_ma={bias_ma:.2f}, deadzone_ma={deadzone_ma:.2f}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize and calibrate the Phase 4 G2-current feedback chain."
    )
    parser.add_argument("--config-path", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--observe-only",
        action="store_true",
        help="calculate and display all signals without connecting GELLO (default)",
    )
    mode.add_argument(
        "--enable-output",
        action="store_true",
        help="enable ID8 only after a second typed confirmation",
    )
    parser.add_argument("--bias-ma", type=float, required=True)
    parser.add_argument("--deadzone-ma", type=float, required=True)
    parser.add_argument("--input-limit-ma", type=float, required=True)
    parser.add_argument("--ema-beta", type=float, required=True)
    parser.add_argument("--gain", type=float, required=True)
    parser.add_argument("--output-sign", type=int, choices=(-1, 1), required=True)
    parser.add_argument("--output-limit-ma", type=float, required=True)
    parser.add_argument("--slew-rate-ma-s", type=float, required=True)
    parser.add_argument("--timeout-s", type=float, required=True)
    parser.add_argument(
        "--phase2-current-limit-ma",
        type=float,
        help="required only with --enable-output; must be >= output limit and <=20 mA",
    )
    parser.add_argument("--duration-s", type=float, default=0.0)
    parser.add_argument("--display-hz", type=float, default=20.0)
    parser.add_argument("--chart-width", type=int, default=72)
    parser.add_argument("--plain", action="store_true")
    parser.add_argument("--csv", type=Path, help="optional CSV output path")
    parser.add_argument(
        "--staged-calibration",
        action="store_true",
        help="interactively capture the five predefined stages; observe-only only",
    )
    parser.add_argument("--stage-duration-s", type=float, default=5.0)
    return parser


def validate_args(parser: argparse.ArgumentParser, args) -> None:
    try:
        make_processor(args)
    except ValueError as exc:
        parser.error(str(exc))
    if not math.isfinite(args.duration_s) or args.duration_s < 0:
        parser.error("--duration-s must be finite and non-negative")
    if not math.isfinite(args.display_hz) or not 1 <= args.display_hz <= 100:
        parser.error("--display-hz must be finite and in [1, 100]")
    if args.chart_width < 10:
        parser.error("--chart-width must be at least 10")
    if not math.isfinite(args.stage_duration_s) or args.stage_duration_s <= 0:
        parser.error("--stage-duration-s must be finite and positive")
    if args.staged_calibration and args.enable_output:
        parser.error("--staged-calibration is restricted to observe-only mode")
    if args.enable_output:
        if not 0 < args.duration_s <= 60:
            parser.error("--enable-output requires --duration-s in (0, 60]")
        if args.phase2_current_limit_ma is None:
            parser.error("--phase2-current-limit-ma is required with --enable-output")
        if (
            not math.isfinite(args.phase2_current_limit_ma)
            or not 0 < args.phase2_current_limit_ma <= MAX_INITIAL_OUTPUT_LIMIT_MA
        ):
            parser.error("--phase2-current-limit-ma must be finite and in (0, 20]")
        if args.output_limit_ma > args.phase2_current_limit_ma:
            parser.error("--output-limit-ma cannot exceed --phase2-current-limit-ma")
        if args.output_limit_ma > MAX_INITIAL_OUTPUT_LIMIT_MA:
            parser.error("initial closed-loop output is restricted to <=20 mA")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    processor = make_processor(args)
    robot = make_probe_robot(load_probe_settings(args.config_path))
    teleop = None
    teleop_connected = False
    output_active = False
    csv_stream = None
    csv_writer = None
    histories = {
        name: deque(maxlen=args.chart_width)
        for name in ("raw", "bias", "filtered", "target", "command")
    }
    stage_values = {stage: [] for stage in CALIBRATION_STAGES}
    stage_report_printed = False
    started_s = time.monotonic()

    def capture(stage: str, duration_s: float) -> None:
        nonlocal output_active
        stage_started_s = time.monotonic()
        next_tick_s = stage_started_s
        while duration_s == 0 or time.monotonic() - stage_started_s < duration_s:
            sample = robot.get_gripper_current_sample()
            diagnostic = processor.process(sample, now_monotonic_s=time.monotonic())
            if teleop is not None and output_active:
                teleop.send_feedback(
                    {GRIPPER_CURRENT_FEEDBACK_KEY: diagnostic.command_ma}
                )
                worker_active, worker_error = teleop.get_feedback_output_status()
                if worker_error is not None:
                    raise RuntimeError(f"GELLO feedback worker stopped: {worker_error}")
                if not worker_active:
                    raise RuntimeError("GELLO feedback worker stopped unexpectedly")

            histories["raw"].append(diagnostic.raw_current_ma)
            histories["bias"].append(diagnostic.bias_corrected_ma)
            histories["filtered"].append(diagnostic.filtered_ma)
            histories["target"].append(diagnostic.target_ma)
            histories["command"].append(diagnostic.command_ma)
            if stage in stage_values and diagnostic.feedback_active:
                stage_values[stage].append(diagnostic.raw_current_ma)
            write_csv_row(
                csv_writer,
                elapsed_s=time.monotonic() - started_s,
                stage=stage,
                diagnostic=diagnostic,
                monitor_error=sample.error,
                output_active=output_active,
            )

            if args.plain or not sys.stdout.isatty():
                print(
                    f"stage={stage} raw_ma={diagnostic.raw_current_ma} "
                    f"bias_ma={diagnostic.bias_corrected_ma} "
                    f"filtered_ma={diagnostic.filtered_ma} "
                    f"target_ma={diagnostic.target_ma:+.3f} "
                    f"command_ma={diagnostic.command_ma:+.3f} "
                    f"available={diagnostic.sample_available} "
                    f"stale={diagnostic.sample_stale} state={diagnostic.gripper_state} "
                    f"age_s={diagnostic.sample_age_s} reason={diagnostic.reason} "
                    f"output={'ON' if output_active else 'OFF'}"
                )
            else:
                dashboard = format_dashboard(
                    diagnostic,
                    histories,
                    args=args,
                    stage=stage,
                    output_active=output_active,
                )
                sys.stdout.write("\x1b[2J\x1b[H" + dashboard + "\n")
                sys.stdout.flush()

            next_tick_s += 1.0 / args.display_hz
            time.sleep(max(0.0, next_tick_s - time.monotonic()))

    try:
        if args.csv is not None:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            csv_stream = args.csv.open("w", newline="", encoding="utf-8")
            csv_writer = csv.DictWriter(csv_stream, fieldnames=CSV_FIELDS)
            csv_writer.writeheader()

        robot.connect()
        if args.enable_output:
            teleop = make_output_teleop(args)
            teleop.connect(calibrate=False)
            teleop_connected = True
            info = teleop.probe_gripper_dynamixel()
            print(
                f"Physical target: ID{info.dxl_id} {info.model_name}, "
                f"model={info.model_number}, hardware_limit={info.current_limit_ma:g} mA"
            )
            print(
                f"Phase2 limit={args.phase2_current_limit_ma:g} mA; "
                f"Phase4 output limit={args.output_limit_ma:g} mA; "
                f"duration={args.duration_s:g} s. Initial Goal Current is 0 mA."
            )
            phrase = input("Type 'ENABLE ID8 PHASE4' to energize ID8: ").strip()
            if phrase != "ENABLE ID8 PHASE4":
                print("Cancelled before Current Control Mode was enabled.")
                return 1
            processor.reset("output_start")
            teleop.start_feedback()
            output_active = True
        else:
            print("OBSERVE ONLY: GELLO is not connected; Goal Current cannot be written.")

        if args.staged_calibration:
            for stage in CALIBRATION_STAGES:
                input(
                    f"\nPrepare '{stage}', ensure the scene is safe, then press Enter "
                    f"to record {args.stage_duration_s:g} s..."
                )
                capture(stage, args.stage_duration_s)
            print_stage_report(stage_values)
            stage_report_printed = True
        else:
            capture("continuous", args.duration_s)
    except KeyboardInterrupt:
        print("\nStopped by user; applying zero-current cleanup.")
    except Exception as exc:
        processor.fail_safe_zero("tool_exception")
        print(f"Phase 4 diagnostic tool failed: {exc}", file=sys.stderr)
        return 2
    finally:
        processor.fail_safe_zero("tool_stopped")
        if (
            args.staged_calibration
            and not stage_report_printed
            and any(stage_values.values())
        ):
            print_stage_report(stage_values)
        try:
            if teleop is not None and teleop_connected:
                try:
                    teleop.send_feedback({GRIPPER_CURRENT_FEEDBACK_KEY: 0.0})
                except Exception:
                    pass
                try:
                    teleop.stop_feedback()
                except Exception as exc:
                    print(f"ID8 feedback cleanup reported: {exc}", file=sys.stderr)
                teleop.disconnect()
        finally:
            robot.disconnect()
            if csv_stream is not None:
                csv_stream.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
