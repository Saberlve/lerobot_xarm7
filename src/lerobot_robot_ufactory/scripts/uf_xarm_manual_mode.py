#!/usr/bin/env python3

import argparse
import sys
from dataclasses import dataclass

import lerobot_robot_ufactory  # noqa: F401  # register UFACTORY plugins
from xarm.wrapper import XArmAPI
from lerobot_robot_ufactory.configs import parser as config_parser


@dataclass
class ManualModeConfig:
    robot_ip: str = "192.168.1.127"
    manual_mode: bool = True
    teach_sensitivity: int | None = 3
    return_to_initial: bool = True
    reset_speed: float = 30.0


@config_parser.wrap()
def get_cfg(cfg: ManualModeConfig) -> ManualModeConfig:
    return cfg


def run(cfg: ManualModeConfig):
    if cfg.teach_sensitivity is not None and not 1 <= cfg.teach_sensitivity <= 5:
        raise ValueError("teach_sensitivity must be between 1 and 5")
    if cfg.reset_speed <= 0:
        raise ValueError("reset_speed must be greater than 0")

    arm = XArmAPI(cfg.robot_ip)
    initial_point = None
    try:
        if not arm.connected:
            raise ConnectionError(f"Failed to connect to xArm at {cfg.robot_ip}")

        arm.motion_enable(enable=True)
        arm.clean_error()
        arm.set_mode(0)
        arm.set_state(0)

        code, initial_point = arm.get_initial_point()
        if code != 0:
            raise RuntimeError(f"get_initial_point failed, code={code}")
        if not initial_point or len(initial_point) < arm.axis:
            raise RuntimeError(
                f"Invalid initial point returned by xArm: {initial_point}"
            )
        initial_point = list(initial_point[:arm.axis])
        print(f"Initial point loaded from xArm Studio: {initial_point}")

        if cfg.manual_mode:
            if cfg.teach_sensitivity is not None:
                code = arm.set_teach_sensitivity(cfg.teach_sensitivity)
                if code != 0:
                    raise RuntimeError(f"set_teach_sensitivity failed, code={code}")

            code = arm.set_mode(2)
            if code != 0:
                raise RuntimeError(f"set_mode(2) failed, code={code}")
            code = arm.set_state(0)
            if code != 0:
                raise RuntimeError(f"set_state(0) failed, code={code}")

            print("Joint teaching mode enabled. Drag the arm manually.")
            input("Press Enter to disable mode 2... ")
        else:
            code = arm.set_mode(0)
            if code != 0:
                raise RuntimeError(f"set_mode(0) failed, code={code}")
            code = arm.set_state(0)
            if code != 0:
                raise RuntimeError(f"set_state(0) failed, code={code}")
            print("Joint teaching mode disabled.")
    finally:
        if arm.connected:
            try:
                arm.set_mode(0)
                arm.set_state(0)
                if cfg.return_to_initial and initial_point is not None:
                    _move_to_initial(arm, initial_point, cfg.reset_speed)
            finally:
                arm.disconnect()


def _move_to_initial(arm: XArmAPI, initial_point: list[float], speed: float):
    code = arm.set_servo_angle(
        angle=initial_point,
        speed=speed,
        is_radian=False,
        wait=True,
    )
    if code != 0:
        raise RuntimeError(f"Failed to move to xArm initial point, code={code}")


def main():
    cli_parser = argparse.ArgumentParser(description="Control xArm joint teaching mode from YAML")
    _, unknown = cli_parser.parse_known_args()
    sys.argv = [sys.argv[0]] + unknown
    cfg = get_cfg()
    run(cfg)


if __name__ == "__main__":
    main()
