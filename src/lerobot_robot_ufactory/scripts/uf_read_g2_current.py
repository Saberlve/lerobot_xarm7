import argparse
import logging
import math
import shutil
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TextIO

import yaml

from lerobot_robot_ufactory.robots.uf_robot.uf_robot import G2CurrentSample, UFRobot
from lerobot_robot_ufactory.robots.uf_robot.uf_robot_config import UFRobotConfig


G2_STATE_LABELS = {
    0: "moving / no object",
    1: "object detected while opening",
    2: "object detected while closing",
    3: "target reached / no object",
}


@dataclass(frozen=True)
class ProbeRobotSettings:
    robot_ip: str
    robot_dof: int
    monitor_frequency_hz: int
    stale_timeout_s: float


@dataclass(frozen=True)
class MonitorStats:
    display_cycles: int
    fresh_cycles: int
    unique_samples: int
    minimum_ma: int | None
    maximum_ma: int | None


def load_probe_settings(config_path: Path) -> ProbeRobotSettings:
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    robot_config = config.get("robot") if isinstance(config, dict) else None
    if not isinstance(robot_config, dict):
        raise ValueError(f"{config_path} does not contain a robot configuration")
    robot_ip = robot_config.get("robot_ip")
    robot_dof = robot_config.get("robot_dof")
    gripper_type = robot_config.get("gripper_type")
    if not isinstance(robot_ip, str) or not robot_ip:
        raise ValueError(f"{config_path} does not define robot.robot_ip")
    if robot_dof not in (5, 6, 7):
        raise ValueError(f"{config_path} has invalid robot.robot_dof: {robot_dof}")
    if gripper_type != 2:
        raise ValueError(f"{config_path} must configure robot.gripper_type: 2 for Gripper G2")

    monitor_frequency_hz = robot_config.get("gripper_current_monitor_frequency_hz", 250)
    stale_timeout_s = robot_config.get("gripper_current_stale_timeout_s", 0.25)
    if (
        not isinstance(monitor_frequency_hz, int)
        or isinstance(monitor_frequency_hz, bool)
        or monitor_frequency_hz <= 0
    ):
        raise ValueError("gripper_current_monitor_frequency_hz must be a positive integer")
    if not isinstance(stale_timeout_s, (int, float)) or not math.isfinite(stale_timeout_s):
        raise ValueError("gripper_current_stale_timeout_s must be finite and positive")
    if stale_timeout_s <= 0:
        raise ValueError("gripper_current_stale_timeout_s must be finite and positive")
    return ProbeRobotSettings(
        robot_ip=robot_ip,
        robot_dof=int(robot_dof),
        monitor_frequency_hz=monitor_frequency_hz,
        stale_timeout_s=float(stale_timeout_s),
    )


def make_probe_robot(settings: ProbeRobotSettings) -> UFRobot:
    """Build a no-camera manual-mode robot; no arm or gripper position is commanded."""
    return UFRobot(
        UFRobotConfig(
            id="g2_current_probe",
            robot_ip=settings.robot_ip,
            robot_dof=settings.robot_dof,
            control_space="joint",
            joint_command_mode=6,
            gripper_type=2,
            manual_mode=True,
            cameras={},
            gripper_error_log_path=None,
            gripper_current_monitor=True,
            gripper_current_monitor_frequency_hz=settings.monitor_frequency_hz,
            gripper_current_stale_timeout_s=settings.stale_timeout_s,
        )
    )


def render_ascii_chart(
    samples_ma: list[int | None],
    *,
    width: int = 72,
    height: int = 13,
    minimum_scale_ma: int = 100,
) -> str:
    """Render a signed-current chart with a symmetric zero-centered scale."""
    if width < 10:
        raise ValueError("chart width must be at least 10")
    if height < 5:
        raise ValueError("chart height must be at least 5")
    if minimum_scale_ma <= 0:
        raise ValueError("minimum_scale_ma must be positive")

    visible = list(samples_ma[-width:])
    numeric = [value for value in visible if value is not None]
    if not numeric:
        return "Waiting for a fresh G2 current sample..."

    peak_ma = max(minimum_scale_ma, max(abs(value) for value in numeric))
    grid = [[" " for _ in range(width)] for _ in range(height)]
    zero_row = round((height - 1) / 2)
    grid[zero_row] = ["-" for _ in range(width)]
    x_offset = width - len(visible)
    for index, value in enumerate(visible):
        if value is None:
            continue
        clipped = max(-peak_ma, min(peak_ma, value))
        row = round((peak_ma - clipped) / (2 * peak_ma) * (height - 1))
        grid[row][x_offset + index] = "*"

    lines = []
    for row, cells in enumerate(grid):
        level_ma = peak_ma - row * (2 * peak_ma / (height - 1))
        label = f"{level_ma:7.0f} |"
        lines.append(label + "".join(cells))
    lines.append("        +" + "-" * width)
    return "\n".join(lines)


def format_dashboard(
    sample: G2CurrentSample,
    history_ma: list[int | None],
    *,
    unique_sample_rate_hz: int,
    chart_width: int,
) -> str:
    if sample.available:
        status = "FRESH"
    elif sample.stale:
        status = "STALE"
    else:
        status = "UNAVAILABLE"
    current_text = "N/A" if sample.current_ma is None else f"{sample.current_ma:+d} mA"
    age_text = "N/A" if sample.age_s is None else f"{sample.age_s * 1000:.1f} ms"
    state_text = G2_STATE_LABELS.get(sample.gripper_state, "unknown")
    header = [
        "xArm Gripper G2 actual Q-axis current (TCP 30000)",
        f"status={status}  current={current_text}  age={age_text}",
        f"state={sample.gripper_state} ({state_text})",
        f"fresh snapshots observed={unique_sample_rate_hz}/s  reason={sample.reason}",
    ]
    if sample.error:
        header.append(f"error={sample.error}")
    header.append(render_ascii_chart(history_ma, width=chart_width))
    header.append(
        "Ctrl+C to stop. This viewer sends no arm, gripper, or GELLO position/current goal."
    )
    return "\n".join(header)


def _format_plain_sample(sample: G2CurrentSample) -> str:
    timestamp = "N/A" if sample.sample_monotonic_s is None else f"{sample.sample_monotonic_s:.6f}"
    age_ms = "N/A" if sample.age_s is None else f"{sample.age_s * 1000:.1f}"
    return (
        f"sample_s={timestamp} available={sample.available} stale={sample.stale} "
        f"current_ma={sample.current_ma} state={sample.gripper_state} age_ms={age_ms} "
        f"reason={sample.reason} error={sample.error}"
    )


def monitor_g2_current(
    robot: UFRobot,
    *,
    duration_s: float = 0.0,
    display_hz: float = 20.0,
    chart_width: int = 72,
    dashboard: bool = True,
    stream: TextIO = sys.stdout,
    clock: Callable[[], float] = time.perf_counter,
    sleeper: Callable[[float], None] = time.sleep,
) -> MonitorStats:
    if not math.isfinite(duration_s) or duration_s < 0:
        raise ValueError("duration_s must be finite and non-negative")
    if not math.isfinite(display_hz) or display_hz <= 0:
        raise ValueError("display_hz must be finite and positive")
    if chart_width < 10:
        raise ValueError("chart_width must be at least 10")

    history_ma: deque[int | None] = deque(maxlen=chart_width)
    recent_sample_times: deque[float] = deque()
    current_values = []
    display_cycles = 0
    fresh_cycles = 0
    unique_samples = 0
    last_sample_timestamp = None
    started_s = clock()
    next_display_s = started_s

    while duration_s == 0 or clock() - started_s < duration_s:
        sample = robot.get_gripper_current_sample()
        now_s = clock()
        display_cycles += 1
        if sample.available:
            fresh_cycles += 1
            current_values.append(sample.current_ma)
            history_ma.append(sample.current_ma)
        else:
            history_ma.append(None)

        if (
            sample.sample_monotonic_s is not None
            and sample.sample_monotonic_s != last_sample_timestamp
        ):
            last_sample_timestamp = sample.sample_monotonic_s
            unique_samples += 1
            recent_sample_times.append(now_s)
        while recent_sample_times and recent_sample_times[0] < now_s - 1.0:
            recent_sample_times.popleft()

        if dashboard:
            terminal_width = shutil.get_terminal_size((100, 24)).columns
            effective_width = max(10, min(chart_width, terminal_width - 10))
            output = format_dashboard(
                sample,
                list(history_ma),
                unique_sample_rate_hz=len(recent_sample_times),
                chart_width=effective_width,
            )
            stream.write("\x1b[2J\x1b[H" + output + "\n")
        else:
            stream.write(_format_plain_sample(sample) + "\n")
        stream.flush()

        next_display_s += 1.0 / display_hz
        sleeper(max(0.0, next_display_s - clock()))

    numeric_values = [value for value in current_values if value is not None]
    return MonitorStats(
        display_cycles=display_cycles,
        fresh_cycles=fresh_cycles,
        unique_samples=unique_samples,
        minimum_ma=min(numeric_values) if numeric_values else None,
        maximum_ma=max(numeric_values) if numeric_values else None,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize cached xArm Gripper G2 actual current without commanding motion."
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        required=True,
        help="YAML configuration containing the xArm IP, DOF, and gripper_type=2",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=0.0,
        help="monitor duration; 0 runs until Ctrl+C (default: 0)",
    )
    parser.add_argument("--display-hz", type=float, default=20.0)
    parser.add_argument("--chart-width", type=int, default=72)
    parser.add_argument(
        "--plain",
        action="store_true",
        help="print one sample per line instead of clearing an interactive chart",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip the interactive hardware-state confirmation",
    )
    args = parser.parse_args()

    settings = load_probe_settings(args.config_path)
    if not args.yes:
        print(f"Target: xArm{settings.robot_dof} at {settings.robot_ip}, Gripper G2")
        print("The viewer enables G2 reporting and puts the arm in manual/teach mode.")
        print("It sends no arm motion, gripper position, GELLO current, or GELLO position goal.")
        if input("Type 'yes' to connect: ").strip().lower() != "yes":
            print("Cancelled.")
            return 1

    logging.basicConfig(level=logging.WARNING)
    robot = make_probe_robot(settings)
    stats = None
    try:
        robot.connect()
        stats = monitor_g2_current(
            robot,
            duration_s=args.duration_s,
            display_hz=args.display_hz,
            chart_width=args.chart_width,
            dashboard=not args.plain and sys.stdout.isatty(),
        )
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        robot.disconnect()

    if stats is not None:
        print(
            "Summary: "
            f"fresh={stats.fresh_cycles}/{stats.display_cycles}, "
            f"unique_samples={stats.unique_samples}, "
            f"range_ma=({stats.minimum_ma}, {stats.maximum_ma})"
        )
        if stats.fresh_cycles == 0:
            print("No fresh G2 current sample was observed.", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
