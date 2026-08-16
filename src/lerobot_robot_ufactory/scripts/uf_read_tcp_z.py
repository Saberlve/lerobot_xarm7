import argparse
import math
from pathlib import Path
from typing import Callable

import numpy as np
import yaml
from xarm.wrapper import XArmAPI


def _load_robot_config(config_path: Path) -> tuple[str, int]:
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    robot_config = config.get("robot") if isinstance(config, dict) else None
    if not isinstance(robot_config, dict):
        raise ValueError(f"{config_path} does not contain a robot configuration")

    robot_ip = robot_config.get("robot_ip")
    robot_dof = robot_config.get("robot_dof")
    if not isinstance(robot_ip, str) or not robot_ip:
        raise ValueError(f"{config_path} does not define robot.robot_ip")
    if robot_dof not in (5, 6, 7):
        raise ValueError(f"{config_path} has invalid robot.robot_dof: {robot_dof}")
    return robot_ip, int(robot_dof)


def read_tcp_z(
    config_path: Path,
    margin_mm: float = 5.0,
    arm_factory: Callable[[str], object] = XArmAPI,
) -> tuple[float, float]:
    """Read the current TCP z using the same FK API as the runtime guard."""
    if not math.isfinite(margin_mm) or margin_mm < 0:
        raise ValueError("margin-mm must be a finite, non-negative number")

    robot_ip, robot_dof = _load_robot_config(config_path)
    arm = arm_factory(robot_ip)
    try:
        if not arm.connected:
            raise ConnectionError(f"Unable to connect to xArm at {robot_ip}")
        if arm.axis != robot_dof:
            raise RuntimeError(
                f"Connected xArm has {arm.axis} axes, but config specifies {robot_dof}"
            )

        code, states = arm.get_joint_states(is_radian=True, num=1)
        if code != 0 or not states or len(states[0]) < robot_dof:
            raise RuntimeError(f"get_joint_states failed, code={code}")
        joints = np.asarray(states[0][:robot_dof], dtype=np.float64)
        if not np.all(np.isfinite(joints)):
            raise RuntimeError("get_joint_states returned NaN/Inf")

        code, pose = arm.get_forward_kinematics(
            joints.tolist(), input_is_radian=True, return_is_radian=True
        )
        pose = np.asarray(pose, dtype=np.float64)
        if code != 0 or pose.shape[0] < 3 or not np.all(np.isfinite(pose)):
            raise RuntimeError(f"get_forward_kinematics failed, code={code}")

        tcp_z_mm = float(pose[2])
        return tcp_z_mm, tcp_z_mm + margin_mm
    finally:
        arm.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read the current xArm TCP z without moving the robot."
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        required=True,
        help="GELLO YAML configuration containing robot.robot_ip and robot.robot_dof",
    )
    parser.add_argument(
        "--margin-mm",
        type=float,
        default=5.0,
        help="safety margin added to the measured z (default: 5 mm)",
    )
    args = parser.parse_args()

    tcp_z_mm, recommended_mm = read_tcp_z(args.config_path, args.margin_mm)
    print(f"Current TCP z: {tcp_z_mm:.3f} mm")
    print(f"Safety margin: {args.margin_mm:.3f} mm")
    print(f"Recommended YAML value: min_tcp_z_mm: {recommended_mm:.3f}")


if __name__ == "__main__":
    main()
