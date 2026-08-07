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


@config_parser.wrap()
def get_cfg(cfg: ManualModeConfig) -> ManualModeConfig:
    return cfg


def run(cfg: ManualModeConfig):
    if cfg.teach_sensitivity is not None and not 1 <= cfg.teach_sensitivity <= 5:
        raise ValueError("teach_sensitivity must be between 1 and 5")

    arm = XArmAPI(cfg.robot_ip)
    try:
        if not arm.connected:
            raise ConnectionError(f"Failed to connect to xArm at {cfg.robot_ip}")

        arm.motion_enable(enable=True)
        arm.clean_error()
        arm.set_mode(0)
        arm.set_state(0)

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
            # Always leave the robot in normal position-control mode.
            arm.set_mode(0)
            arm.set_state(0)
            arm.disconnect()


def main():
    cli_parser = argparse.ArgumentParser(description="Control xArm joint teaching mode from YAML")
    _, unknown = cli_parser.parse_known_args()
    sys.argv = [sys.argv[0]] + unknown
    cfg = get_cfg()
    run(cfg)


if __name__ == "__main__":
    main()
