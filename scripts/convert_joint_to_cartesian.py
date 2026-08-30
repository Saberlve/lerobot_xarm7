"""Convert a joint-space LeRobot dataset (xArm7 GELLO recordings) to Cartesian TCP poses.

action / observation.state change from
    [J1..J7 (rad), gripper.pos]
to
    [pose.x/y/z (mm), pose.rx/ry/rz (axis-angle, rad), gripper.pos]

matching the robot's cartesian feature convention in
``robots/uf_robot/uf_robot.py`` (CARTESIAN_OBS_KEYS / CARTESIAN_ACTION_KEYS,
compatible with ``set_position_aa(is_radian=True)``).

The FK model uses the controller-calibrated joint origins plus the
controller's own FK to identify the TCP endpoint (same procedure as
``UFRobot._initialize_local_kinematics``), so converted poses match what
``control_space: "cartesian"`` recording would have produced.

Usage:
    python scripts/convert_joint_to_cartesian.py \
        --input datasets/xarm7_gello_pick_ball \
        --output datasets/xarm7_gello_pick_ball_cartesian \
        --robot-ip 192.168.1.245
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LOCAL_KINEMATICS_PATH = (
    Path(__file__).resolve().parent.parent
    / "src/lerobot_robot_ufactory/robots/uf_robot/local_kinematics.py"
)

CARTESIAN_NAMES = [
    "pose.x", "pose.y", "pose.z", "pose.rx", "pose.ry", "pose.rz", "gripper.pos",
]
FK_MAX_ERROR_MM = 2.0  # same limit as local_kinematics_max_error_mm default


def _load_local_kinematics_module():
    """Import local_kinematics.py directly, avoiding the heavy package __init__."""
    spec = importlib.util.spec_from_file_location("local_kinematics", LOCAL_KINEMATICS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_LOCAL_KINEMATICS_MODULE = None


def _local_kinematics():
    """Return a cached local_kinematics module instance."""
    global _LOCAL_KINEMATICS_MODULE
    if _LOCAL_KINEMATICS_MODULE is None:
        _LOCAL_KINEMATICS_MODULE = _load_local_kinematics_module()
    return _LOCAL_KINEMATICS_MODULE


def _rotation_to_axis_angle(rotation: np.ndarray) -> np.ndarray:
    """Rotation matrix -> axis-angle vector (rad), angle in [0, pi]."""
    return _local_kinematics().rotation_to_axis_angle(rotation)


def _to_axis_angle_continuous(rotation: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
    """Pick the axis-angle representative closest to the previous frame.

    R(axis, angle) == R(-axis, 2*pi - angle); choosing per frame between the
    two avoids pi-flip discontinuities (e.g. TCP pointing straight down).
    """
    return _local_kinematics().axis_angle_continuous(rotation, previous)


def _joints_to_tcp_pose(kinematics, joints: np.ndarray) -> np.ndarray:
    transform = kinematics.forward_matrix(joints)
    return transform[:3, 3], transform[:3, :3]


def build_kinematics(robot_ip: str):
    """Connect to the controller and build the calibrated FK model."""
    from xarm.wrapper import XArmAPI

    lk = _load_local_kinematics_module()
    joint_origins = lk.read_xarm7_kinematics(robot_ip)

    arm = XArmAPI(robot_ip)
    try:
        if not arm.connected:
            raise RuntimeError(f"Unable to connect to xArm controller at {robot_ip}")
        code, states = arm.get_joint_states(is_radian=True, num=1)
        if code != 0 or not states or len(states[0]) < 7:
            raise RuntimeError(f"get_joint_states failed, code={code}")
        current = np.asarray(states[0][:7], dtype=np.float64)

        world_offset = np.asarray(arm.world_offset, dtype=np.float64)
        if world_offset.shape != (6,) or not np.all(np.isfinite(world_offset)):
            raise RuntimeError(f"Controller returned an invalid world_offset: {world_offset}")
        if not arm.default_is_radian:
            world_offset[3:6] = np.radians(world_offset[3:6])

        code, pose = arm.get_forward_kinematics(
            current.tolist(), input_is_radian=True, return_is_radian=True
        )
        if code != 0 or pose is None or len(pose) < 6:
            raise RuntimeError(f"Controller FK failed, code={code}")
        controller_transform = lk.xarm_rpy_transform(pose[:6])

        chain = lk.XArm7Kinematics(joint_origins, world_offset=world_offset)
        endpoint = np.linalg.inv(chain.forward_matrix(current)) @ controller_transform
        kinematics = lk.XArm7Kinematics(
            joint_origins, world_offset=world_offset, end_transform=endpoint
        )
        calibration = {
            "robot_ip": robot_ip,
            "joint_origins": joint_origins.tolist(),
            "world_offset": world_offset.tolist(),
            "end_transform": endpoint.tolist(),
        }
        return kinematics, calibration, arm
    except Exception:
        arm.disconnect()
        raise


def validate_against_controller(kinematics, arm, joints_samples: np.ndarray) -> float:
    """Compare local FK against the controller FK on recorded joint samples."""
    lk = _load_local_kinematics_module()
    max_error_mm = 0.0
    for row in joints_samples:
        joints = np.asarray(row[:7], dtype=np.float64)  # drop the gripper column
        code, pose = arm.get_forward_kinematics(
            joints.tolist(), input_is_radian=True, return_is_radian=True
        )
        if code != 0 or pose is None or len(pose) < 3:
            raise RuntimeError(f"Controller FK failed during validation, code={code}")
        local_position = kinematics.tcp_position(joints)
        error_mm = float(np.linalg.norm(local_position - np.asarray(pose[:3], dtype=np.float64)))
        max_error_mm = max(max_error_mm, error_mm)
    if max_error_mm > FK_MAX_ERROR_MM:
        raise RuntimeError(
            f"Local FK differs from controller FK by {max_error_mm:.3f} mm "
            f"(limit {FK_MAX_ERROR_MM:.3f} mm); refusing to convert."
        )
    return max_error_mm


def convert_column(kinematics, values: np.ndarray) -> np.ndarray:
    """(N, 8) joint+gripper rows -> (N, 7) cartesian+gripper rows."""
    out = np.empty((values.shape[0], 7), dtype=np.float64)
    previous_aa = None
    for i, row in enumerate(values):
        position, rotation = _joints_to_tcp_pose(kinematics, row[:7])
        aa = _to_axis_angle_continuous(rotation, previous_aa)
        previous_aa = aa
        out[i, :3] = position  # mm (origins are scaled by 1000)
        out[i, 3:6] = aa
        out[i, 6] = row[7]  # gripper.pos passthrough
    return out


def _feature_stats(values: np.ndarray) -> dict:
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [int(values.shape[0])],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="joint-space dataset root")
    parser.add_argument("--output", required=True, type=Path, help="cartesian dataset root")
    parser.add_argument("--robot-ip", default="192.168.1.245")
    parser.add_argument(
        "--calibration-cache",
        type=Path,
        default=None,
        help="optional JSON cache for the FK calibration (read if exists, written otherwise)",
    )
    parser.add_argument("--validation-samples", type=int, default=50)
    args = parser.parse_args()

    src = args.input.resolve()
    dst = args.output.resolve()
    if not src.is_dir():
        raise SystemExit(f"input dataset not found: {src}")
    if dst.exists():
        raise SystemExit(f"output already exists, refusing to overwrite: {dst}")

    kinematics = None
    arm = None
    if args.calibration_cache and args.calibration_cache.is_file():
        lk = _load_local_kinematics_module()
        calib = json.loads(args.calibration_cache.read_text())
        kinematics = lk.XArm7Kinematics(
            calib["joint_origins"],
            world_offset=calib["world_offset"],
            end_transform=np.asarray(calib["end_transform"], dtype=np.float64),
        )
        print(f"Loaded FK calibration from {args.calibration_cache}")
    else:
        kinematics, calibration, arm = build_kinematics(args.robot_ip)
        print(f"Built FK calibration from controller at {args.robot_ip}")
        if args.calibration_cache:
            args.calibration_cache.parent.mkdir(parents=True, exist_ok=True)
            args.calibration_cache.write_text(json.dumps(calibration, indent=2))
            print(f"Cached FK calibration to {args.calibration_cache}")

    data_files = sorted(src.glob("data/chunk-*/*.parquet"))
    if not data_files:
        raise SystemExit(f"no data parquets found under {src}/data")
    frames = pd.concat([pd.read_parquet(f) for f in data_files], ignore_index=True)
    state_joints = np.asarray(frames["observation.state"].tolist(), dtype=np.float64)
    action_joints = np.asarray(frames["action"].tolist(), dtype=np.float64)
    print(f"Loaded {len(frames)} frames from {len(data_files)} parquet file(s)")

    if arm is not None:
        rng = np.random.default_rng(0)
        sample_idx = rng.choice(
            len(frames), size=min(args.validation_samples, len(frames)), replace=False
        )
        max_err = validate_against_controller(kinematics, arm, state_joints[sample_idx])
        print(f"Controller FK validation on {len(sample_idx)} frames: max error {max_err:.3f} mm")
        arm.disconnect()

    state_cart = convert_column(kinematics, state_joints)
    action_cart = convert_column(kinematics, action_joints)

    # Copy everything, then patch data + meta.
    shutil.copytree(src, dst)
    frames["observation.state"] = list(state_cart.astype(np.float32))
    frames["action"] = list(action_cart.astype(np.float32))
    frames = frames.sort_values("index", ignore_index=True)
    out_data = dst / "data/chunk-000/file-000.parquet"
    if len(data_files) > 1:
        for f in dst.glob("data/chunk-*/*.parquet"):
            f.unlink()
    out_data.parent.mkdir(parents=True, exist_ok=True)
    frames.to_parquet(out_data, index=False)

    info_path = dst / "meta/info.json"
    info = json.loads(info_path.read_text())
    for key in ("action", "observation.state"):
        info["features"][key]["names"] = CARTESIAN_NAMES
        info["features"][key]["shape"] = [len(CARTESIAN_NAMES)]
    info["data_path"] = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    info_path.write_text(json.dumps(info, indent=4))

    stats_path = dst / "meta/stats.json"
    stats = json.loads(stats_path.read_text())
    stats["action"] = _feature_stats(action_cart.astype(np.float64))
    stats["observation.state"] = _feature_stats(state_cart.astype(np.float64))
    stats_path.write_text(json.dumps(stats, indent=4))

    print(f"Wrote cartesian dataset to {dst}")
    print(f"state range x/y/z mm: "
          f"[{state_cart[:, 0].min():.1f}, {state_cart[:, 0].max():.1f}] / "
          f"[{state_cart[:, 1].min():.1f}, {state_cart[:, 1].max():.1f}] / "
          f"[{state_cart[:, 2].min():.1f}, {state_cart[:, 2].max():.1f}]")


if __name__ == "__main__":
    sys.exit(main())
