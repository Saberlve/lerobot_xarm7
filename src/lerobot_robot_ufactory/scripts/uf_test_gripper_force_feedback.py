"""Phase 3 G2-current to GELLO-ID8 integration probe.

The default mode is observe-only. No GELLO current mode or output is enabled
unless --enable-output is supplied and the operator types the safety phrase.
"""

import argparse
import math
import sys
import time
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
)
from lerobot_robot_ufactory.utils.realtime_teleop import (
    map_gripper_current_feedback,
)


MAX_INITIAL_OUTPUT_LIMIT_MA = 20.0


def make_output_teleop(
    config_path: Path, *, gain: float, output_sign: int, limit_ma: float
) -> GelloTeleop:
    settings = load_gello_probe_settings(config_path)
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
            gripper_current_limit_ma=limit_ma,
            gripper_force_feedback_enabled=True,
            gripper_feedback_gain=gain,
            gripper_feedback_output_sign=output_sign,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Observe or briefly test the Phase 3 G2-current to GELLO-ID8 mapping."
    )
    parser.add_argument("--config-path", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--observe-only",
        action="store_true",
        help="print mapped current without connecting to or writing GELLO (default)",
    )
    mode.add_argument(
        "--enable-output",
        action="store_true",
        help="after typed confirmation, send the mapped current to GELLO ID8",
    )
    parser.add_argument("--gain", type=float, required=True)
    parser.add_argument("--output-sign", type=int, choices=(-1, 1), required=True)
    parser.add_argument("--output-limit-ma", type=float, required=True)
    parser.add_argument(
        "--duration-s",
        type=float,
        default=0.0,
        help="0 runs observe-only until Ctrl+C; output mode requires a value in (0, 5]",
    )
    parser.add_argument("--display-hz", type=float, default=20.0)
    args = parser.parse_args()

    if not math.isfinite(args.gain) or args.gain < 0.0:
        parser.error("--gain must be finite and non-negative")
    if (
        not math.isfinite(args.output_limit_ma)
        or not 0.0 < args.output_limit_ma <= MAX_INITIAL_OUTPUT_LIMIT_MA
    ):
        parser.error(
            f"--output-limit-ma must be finite and in (0, {MAX_INITIAL_OUTPUT_LIMIT_MA:g}] "
            "for this initial integration probe"
        )
    if not math.isfinite(args.display_hz) or not 1.0 <= args.display_hz <= 100.0:
        parser.error("--display-hz must be finite and in [1, 100] Hz")
    if not math.isfinite(args.duration_s) or args.duration_s < 0.0:
        parser.error("--duration-s must be finite and non-negative")
    if args.enable_output and not 0.0 < args.duration_s <= 5.0:
        parser.error("--enable-output requires --duration-s in (0, 5]")

    robot = make_probe_robot(load_probe_settings(args.config_path))
    teleop = None
    teleop_connected = False
    output_active = False
    try:
        robot.connect()
        if args.enable_output:
            teleop = make_output_teleop(
                args.config_path,
                gain=args.gain,
                output_sign=args.output_sign,
                limit_ma=args.output_limit_ma,
            )
            teleop.connect(calibrate=False)
            teleop_connected = True
            info = teleop.probe_gripper_dynamixel()
            print(
                f"Physical target: ID{info.dxl_id} {info.model_name} "
                f"(model {info.model_number}), hardware limit={info.current_limit_ma:g} mA"
            )
            print(
                f"Mapping: xArm current * {args.gain:g} * {args.output_sign:+d}, "
                f"clamped to +/-{args.output_limit_ma:g} mA for {args.duration_s:g} s."
            )
            phrase = input("Type 'ENABLE ID8 FEEDBACK' to energize ID8: ").strip()
            if phrase != "ENABLE ID8 FEEDBACK":
                print("Cancelled before Current Control Mode was enabled.")
                return 1
            teleop.start_feedback()
            output_active = True
        else:
            print("OBSERVE ONLY: GELLO is not connected and no current output will be written.")

        started_s = time.monotonic()
        period_s = 1.0 / args.display_hz
        while args.duration_s == 0.0 or time.monotonic() - started_s < args.duration_s:
            sample = robot.get_gripper_current_sample()
            mapped_ma = map_gripper_current_feedback(
                sample,
                gain=args.gain,
                output_sign=args.output_sign,
                output_limit_ma=args.output_limit_ma,
            )
            if teleop is not None and output_active:
                teleop.send_feedback({GRIPPER_CURRENT_FEEDBACK_KEY: mapped_ma})
                worker_active, worker_error = teleop.get_feedback_output_status()
                if worker_error is not None:
                    raise RuntimeError(f"GELLO feedback worker stopped: {worker_error}")
                if not worker_active:
                    raise RuntimeError("GELLO feedback worker stopped unexpectedly")
            age_ms = "N/A" if sample.age_s is None else f"{sample.age_s * 1000:.1f}"
            current_ma = "N/A" if sample.current_ma is None else f"{sample.current_ma:+d}"
            print(
                f"xarm_current_ma={current_ma} age_ms={age_ms} "
                f"available={sample.available} stale={sample.stale} "
                f"reason={sample.reason} error={sample.error} "
                f"mapped_feedback_ma={mapped_ma:+.3f} "
                f"output={'ON' if output_active else 'OFF'}"
            )
            time.sleep(period_s)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as exc:
        print(f"Phase 3 integration probe failed: {exc}", file=sys.stderr)
        return 2
    finally:
        try:
            if teleop is not None and teleop_connected:
                try:
                    teleop.stop_feedback()
                except Exception as exc:
                    print(f"ID8 feedback cleanup reported: {exc}", file=sys.stderr)
                teleop.disconnect()
        finally:
            robot.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
