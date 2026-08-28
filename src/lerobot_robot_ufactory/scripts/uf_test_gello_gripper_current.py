import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from lerobot_robot_ufactory.teleoperators.gello_teleop.gello_teleop import GelloTeleop
from lerobot_robot_ufactory.teleoperators.gello_teleop.gello_teleop_config import (
    GelloTeleopConfig,
)


@dataclass(frozen=True)
class GelloProbeSettings:
    port: str
    joint_ids: tuple[int, ...]
    joint_signs: tuple[int, ...]
    gripper_id: int
    gripper_open_deg: float
    gripper_close_deg: float


def load_gello_probe_settings(config_path: Path) -> GelloProbeSettings:
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    teleop = config.get("teleop") if isinstance(config, dict) else None
    if not isinstance(teleop, dict):
        raise ValueError(f"{config_path} does not contain a teleop configuration")

    port = teleop.get("port")
    joint_ids = tuple(teleop.get("joint_ids", ()))
    joint_signs = tuple(teleop.get("joint_signs", ()))
    gripper_id = teleop.get("gripper_id")
    gripper_open_deg = teleop.get("gripper_open_deg")
    gripper_close_deg = teleop.get("gripper_close_deg")
    if not isinstance(port, str) or not port:
        raise ValueError("teleop.port must be configured")
    if joint_ids != (1, 2, 3, 4, 5, 6, 7):
        raise ValueError("this Phase 2 probe requires joint_ids [1, 2, 3, 4, 5, 6, 7]")
    if len(joint_signs) != len(joint_ids) or any(sign not in (-1, 1) for sign in joint_signs):
        raise ValueError("teleop.joint_signs must contain one +/-1 value per arm joint")
    if gripper_id != 8:
        raise ValueError("this Phase 2 probe is restricted to gripper_id 8")
    if not isinstance(gripper_open_deg, (int, float)) or not math.isfinite(gripper_open_deg):
        raise ValueError("teleop.gripper_open_deg must be finite")
    if not isinstance(gripper_close_deg, (int, float)) or not math.isfinite(gripper_close_deg):
        raise ValueError("teleop.gripper_close_deg must be finite")
    if gripper_open_deg == gripper_close_deg:
        raise ValueError("gripper open and close positions must differ")
    return GelloProbeSettings(
        port=port,
        joint_ids=joint_ids,
        joint_signs=joint_signs,
        gripper_id=gripper_id,
        gripper_open_deg=float(gripper_open_deg),
        gripper_close_deg=float(gripper_close_deg),
    )


def make_gello_probe(
    settings: GelloProbeSettings, *, current_limit_ma: float | None
) -> GelloTeleop:
    current_enabled = current_limit_ma is not None
    return GelloTeleop(
        GelloTeleopConfig(
            id="gello_id8_current_probe",
            port=settings.port,
            joint_ids=settings.joint_ids,
            joint_signs=settings.joint_signs,
            gripper_id=settings.gripper_id,
            gripper_open_deg=settings.gripper_open_deg,
            gripper_close_deg=settings.gripper_close_deg,
            gripper_current_control_enabled=current_enabled,
            gripper_current_limit_ma=current_limit_ma,
        )
    )


def read_gripper_position(teleop: GelloTeleop) -> float:
    action = teleop.gello_agent.act(
        {"joint_state": np.zeros(teleop.dof + 1, dtype=float)}
    )
    gripper_pos = float(action[teleop.dof])
    if not math.isfinite(gripper_pos):
        raise RuntimeError(f"gripper.pos is not finite: {gripper_pos}")
    return gripper_pos


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Probe GELLO ID8 and optionally apply a short manual current while "
            "printing gripper.pos. This command never connects to xArm."
        )
    )
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="only identify ID8 and read gripper.pos; do not enter Current Control Mode",
    )
    parser.add_argument(
        "--limit-ma",
        type=float,
        help="required software current limit for mode/output testing; maximum 100 mA",
    )
    parser.add_argument(
        "--current-ma",
        type=float,
        default=0.0,
        help="short signed manual test current (default: 0 mA)",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=0.5,
        help="test output duration, in (0, 2] seconds (default: 0.5)",
    )
    parser.add_argument("--sample-hz", type=float, default=20.0)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip the typed confirmation after the physical ID8 model is displayed",
    )
    args = parser.parse_args()

    if args.probe_only:
        if args.limit_ma is not None or args.current_ma != 0.0:
            parser.error("--probe-only cannot be combined with --limit-ma or nonzero --current-ma")
    else:
        if args.limit_ma is None:
            parser.error("--limit-ma is required unless --probe-only is used")
        if not math.isfinite(args.limit_ma) or not 0 < args.limit_ma <= 100.0:
            parser.error("--limit-ma must be finite and in (0, 100] mA")
        if not math.isfinite(args.current_ma) or abs(args.current_ma) > args.limit_ma:
            parser.error("absolute --current-ma must be finite and no greater than --limit-ma")
        if not math.isfinite(args.duration_s) or not 0 < args.duration_s <= 2.0:
            parser.error("--duration-s must be finite and in (0, 2] seconds")
    if not math.isfinite(args.sample_hz) or not 1.0 <= args.sample_hz <= 100.0:
        parser.error("--sample-hz must be finite and in [1, 100] Hz")

    settings = load_gello_probe_settings(args.config_path)
    teleop = make_gello_probe(
        settings,
        current_limit_ma=None if args.probe_only else float(args.limit_ma),
    )
    connected = False
    enabled = False
    samples = []
    try:
        teleop.connect(calibrate=False)
        connected = True
        info = teleop.probe_gripper_dynamixel()
        print(
            f"ID{info.dxl_id}: {info.model_name}, model_number={info.model_number}, "
            f"operating_mode={info.operating_mode}, "
            f"hardware_current_limit={info.current_limit_ma:g} mA, "
            f"current_unit={info.current_unit_ma:g} mA/raw"
        )
        initial_pos = read_gripper_position(teleop)
        print(f"gripper.pos before current mode: {initial_pos:.6f}")
        if args.probe_only:
            return 0

        if not args.yes:
            print(
                f"Requested ID8 test: limit={args.limit_ma:g} mA, "
                f"current={args.current_ma:+g} mA, duration={args.duration_s:g} s."
            )
            confirmation = input("Type 'ID8' to enable Current Control Mode: ").strip()
            if confirmation != "ID8":
                print("Cancelled before enabling torque/current output.")
                return 1

        teleop.enable_gripper_current_mode()
        enabled = True
        applied_ma = teleop.write_gripper_current_ma(float(args.current_ma))
        print(f"Applied current (after conversion/clamp): {applied_ma:+g} mA")
        started = time.monotonic()
        period_s = 1.0 / args.sample_hz
        while time.monotonic() - started < args.duration_s:
            gripper_pos = read_gripper_position(teleop)
            samples.append(gripper_pos)
            print(f"elapsed_s={time.monotonic() - started:.3f} gripper.pos={gripper_pos:.6f}")
            time.sleep(period_s)
        teleop.zero_gripper_current()
        print("Goal Current returned to 0 mA.")
    except KeyboardInterrupt:
        print("\nInterrupted; applying safety cleanup.", file=sys.stderr)
    except Exception as exc:
        print(f"Phase 2 GELLO ID8 test failed: {exc}", file=sys.stderr)
        return 2
    finally:
        if connected:
            if enabled:
                try:
                    teleop.zero_gripper_current()
                except Exception as exc:
                    print(f"Zero-current cleanup reported: {exc}", file=sys.stderr)
                try:
                    teleop.disable_gripper_current_mode()
                except Exception as exc:
                    print(f"Current-mode cleanup reported: {exc}", file=sys.stderr)
            teleop.disconnect()

    if samples:
        print(
            f"gripper.pos remained readable for {len(samples)} samples; "
            f"range=[{min(samples):.6f}, {max(samples):.6f}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
