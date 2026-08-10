#!/usr/bin/env python3
"""Calibrate and reset a GELLO leader to the xArm7 SDK initial pose."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml
from dynamixel_sdk import COMM_SUCCESS, GroupSyncWrite, PacketHandler, PortHandler


DEFAULT_PORT = (
    "/dev/serial/by-id/"
    "usb-FTDI_USB__-__Serial_Converter_FTB9HYVD-if00-port0"
)
DEFAULT_CALIBRATION = Path("config/gello/xarm7_gello_reset_calibration.yaml")
ROBOT_INITIAL_JOINTS_DEG = [0.0, -30.0, 0.0, 0.0, 0.0, 30.0, 0.0]
JOINT_IDS = list(range(1, 8))
GRIPPER_ID = 8
BAUDRATE = 57600

ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_HARDWARE_ERROR = 70
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_CURRENT = 126
ADDR_PRESENT_POSITION = 132
POSITION_CONTROL_MODE = 3


class GelloBus:
    def __init__(self, port: str, ids: Sequence[int]) -> None:
        self.ids = list(ids)
        self.port = PortHandler(port)
        self.packet = PacketHandler(2.0)
        self.writer = GroupSyncWrite(
            self.port, self.packet, ADDR_GOAL_POSITION, 4
        )

        if not self.port.openPort():
            raise RuntimeError(f"Failed to open GELLO port: {port}")
        if not self.port.setBaudRate(BAUDRATE):
            self.port.closePort()
            raise RuntimeError(f"Failed to set GELLO baud rate to {BAUDRATE}")

    def close(self) -> None:
        self.writer.clearParam()
        self.port.closePort()

    def _check(self, operation: str, dxl_id: int, comm: int, error: int) -> None:
        if comm != COMM_SUCCESS:
            detail = self.packet.getTxRxResult(comm)
            raise RuntimeError(
                f"{operation} failed for Dynamixel {dxl_id}: {detail} ({comm})"
            )
        if error != 0:
            detail = self.packet.getRxPacketError(error)
            raise RuntimeError(
                f"{operation} failed for Dynamixel {dxl_id}: {detail} ({error})"
            )

    def read_u8(self, dxl_id: int, address: int) -> int:
        value, comm, error = self.packet.read1ByteTxRx(
            self.port, dxl_id, address
        )
        self._check("read", dxl_id, comm, error)
        return value

    def read_i16(self, dxl_id: int, address: int) -> int:
        value, comm, error = self.packet.read2ByteTxRx(
            self.port, dxl_id, address
        )
        self._check("read", dxl_id, comm, error)
        return value - 0x10000 if value >= 0x8000 else value

    def read_i32(self, dxl_id: int, address: int) -> int:
        value, comm, error = self.packet.read4ByteTxRx(
            self.port, dxl_id, address
        )
        self._check("read", dxl_id, comm, error)
        return value - 0x100000000 if value >= 0x80000000 else value

    def positions(self) -> np.ndarray:
        return np.asarray(
            [self.read_i32(dxl_id, ADDR_PRESENT_POSITION) for dxl_id in self.ids],
            dtype=float,
        )

    def set_torque(self, enabled: bool) -> None:
        value = 1 if enabled else 0
        for dxl_id in self.ids:
            comm, error = self.packet.write1ByteTxRx(
                self.port, dxl_id, ADDR_TORQUE_ENABLE, value
            )
            self._check("set torque", dxl_id, comm, error)

    def verify_position_mode(self) -> None:
        for dxl_id in self.ids:
            mode = self.read_u8(dxl_id, ADDR_OPERATING_MODE)
            if mode != POSITION_CONTROL_MODE:
                raise RuntimeError(
                    f"Dynamixel {dxl_id} is in mode {mode}, expected position mode 3"
                )

    def write_positions(self, raw_positions: Sequence[int]) -> None:
        self.writer.clearParam()
        try:
            for dxl_id, raw_position in zip(self.ids, raw_positions, strict=True):
                encoded = int(raw_position) & 0xFFFFFFFF
                data = list(encoded.to_bytes(4, byteorder="little", signed=False))
                if not self.writer.addParam(dxl_id, data):
                    raise RuntimeError(
                        f"Failed to add goal position for Dynamixel {dxl_id}"
                    )
            comm = self.writer.txPacket()
            if comm != COMM_SUCCESS:
                detail = self.packet.getTxRxResult(comm)
                raise RuntimeError(
                    f"SyncWrite failed: {detail} ({comm})"
                )
        finally:
            self.writer.clearParam()


def counts_to_degrees(values: Sequence[float]) -> np.ndarray:
    return np.asarray(values, dtype=float) / 4096.0 * 360.0


def save_calibration(path: Path, ids: Sequence[int], targets: Sequence[int]) -> None:
    data = {
        "schema_version": 1,
        "description": "GELLO pose matching the xArm7 SDK initial point",
        "robot_initial_joints_deg": ROBOT_INITIAL_JOINTS_DEG,
        "dynamixel_ids": list(ids),
        "target_raw_counts": [int(value) for value in targets],
        "target_encoder_deg": [
            round(float(value), 6) for value in counts_to_degrees(targets)
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def load_calibration(path: Path, expected_ids: Sequence[int]) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(
            f"Calibration file not found: {path}. Run this script with --calibrate first."
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    ids = data.get("dynamixel_ids")
    targets = data.get("target_raw_counts")
    if ids != list(expected_ids):
        raise ValueError(f"Calibration IDs {ids} do not match expected IDs {expected_ids}")
    if not isinstance(targets, list) or len(targets) != len(expected_ids):
        raise ValueError("Calibration target_raw_counts has an invalid length")
    return np.asarray(targets, dtype=float)


def calibrate(args: argparse.Namespace) -> None:
    ids = JOINT_IDS + ([GRIPPER_ID] if not args.no_gripper else [])
    bus = GelloBus(args.port, ids)
    try:
        bus.set_torque(False)
        print("GELLO torque is disabled.")
        if not args.yes:
            print(
                "Manually place GELLO in the physical pose matching xArm7 "
                f"{ROBOT_INITIAL_JOINTS_DEG} degrees."
            )
            print("Keep the gripper open, then press Enter to capture this pose.")
            input()
        targets = np.rint(bus.positions()).astype(int)
        save_calibration(args.calibration, ids, targets)
        print(f"Saved calibration: {args.calibration}")
        print("Target encoder degrees:", counts_to_degrees(targets).round(2).tolist())
    finally:
        try:
            bus.set_torque(False)
        finally:
            bus.close()


def reset(args: argparse.Namespace) -> None:
    ids = JOINT_IDS + ([GRIPPER_ID] if not args.no_gripper else [])
    targets = load_calibration(args.calibration, ids)
    bus = GelloBus(args.port, ids)
    torque_enabled = False
    completed = False
    try:
        bus.set_torque(False)
        bus.verify_position_mode()
        current = bus.positions()
        move_deg = counts_to_degrees(targets - current)
        max_move_deg = float(np.max(np.abs(move_deg)))
        print("Current encoder degrees:", counts_to_degrees(current).round(2).tolist())
        print("Target encoder degrees: ", counts_to_degrees(targets).round(2).tolist())
        print("Required move degrees:  ", move_deg.round(2).tolist())
        print(f"Maximum move: {max_move_deg:.2f} degrees")

        if max_move_deg > args.max_move_deg:
            raise RuntimeError(
                f"Refusing reset: {max_move_deg:.2f} degree move exceeds "
                f"--max-move-deg={args.max_move_deg:.2f}"
            )
        if not args.yes:
            answer = input("Type 'yes' to enable GELLO torque and reset: ").strip()
            if answer.lower() != "yes":
                print("Reset cancelled.")
                return

        bus.write_positions(np.rint(current).astype(int))
        for remaining in (3, 2, 1):
            print(f"Enabling torque in {remaining}...")
            time.sleep(1.0)
        bus.set_torque(True)
        torque_enabled = True

        duration = max(0.5, max_move_deg / args.speed_deg_s)
        steps = max(1, int(duration * args.control_hz))
        started = time.monotonic()
        for step in range(1, steps + 1):
            u = step / steps
            smooth = u * u * (3.0 - 2.0 * u)
            command = np.rint(current + (targets - current) * smooth).astype(int)
            bus.write_positions(command)

            if step % max(1, int(args.control_hz / 5)) == 0:
                currents = [
                    bus.read_i16(dxl_id, ADDR_PRESENT_CURRENT) for dxl_id in ids
                ]
                errors = [
                    bus.read_u8(dxl_id, ADDR_HARDWARE_ERROR) for dxl_id in ids
                ]
                if any(errors):
                    raise RuntimeError(f"Dynamixel hardware errors: {errors}")
                if max(abs(value) for value in currents) > args.max_current_raw:
                    raise RuntimeError(
                        f"Current safety threshold exceeded: {currents}"
                    )

            delay = started + step / args.control_hz - time.monotonic()
            if delay > 0:
                time.sleep(delay)

        deadline = time.monotonic() + args.settle_timeout_s
        final = bus.positions()
        while time.monotonic() < deadline:
            error_deg = counts_to_degrees(final - targets)
            if float(np.max(np.abs(error_deg))) <= args.tolerance_deg:
                break
            bus.write_positions(np.rint(targets).astype(int))
            time.sleep(1.0 / args.control_hz)
            final = bus.positions()
        else:
            error_deg = counts_to_degrees(final - targets)
            raise RuntimeError(
                "GELLO did not reach the calibrated pose; final errors: "
                f"{error_deg.round(2).tolist()} degrees"
            )

        print("Final encoder degrees:", counts_to_degrees(final).round(2).tolist())
        print("GELLO reset completed.")
        completed = True
    finally:
        if torque_enabled and (not completed or args.release_after_reset):
            try:
                bus.set_torque(False)
                print("GELLO torque disabled.")
            except Exception as exc:
                print(f"WARNING: failed to disable GELLO torque: {exc}")
        bus.close()

    if completed and not args.release_after_reset:
        print("GELLO torque remains enabled and is holding the calibrated pose.")


def release(args: argparse.Namespace) -> None:
    ids = JOINT_IDS + ([GRIPPER_ID] if not args.no_gripper else [])
    bus = GelloBus(args.port, ids)
    try:
        bus.set_torque(False)
        print("GELLO torque disabled.")
    finally:
        bus.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate or reset GELLO to the physical pose matching the xArm7 "
            "SDK initial joints [0, -30, 0, 0, 0, 30, 0] degrees."
        )
    )
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--calibrate", action="store_true")
    mode.add_argument(
        "--release",
        action="store_true",
        help="Disable GELLO torque without moving it",
    )
    parser.add_argument("--no-gripper", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Skip interactive confirmation")
    parser.add_argument("--speed-deg-s", type=float, default=20.0)
    parser.add_argument("--control-hz", type=float, default=50.0)
    parser.add_argument("--max-move-deg", type=float, default=90.0)
    parser.add_argument("--max-current-raw", type=int, default=300)
    parser.add_argument("--tolerance-deg", type=float, default=3.0)
    parser.add_argument("--settle-timeout-s", type=float, default=5.0)
    parser.add_argument(
        "--release-after-reset",
        action="store_true",
        help="Disable torque after a successful reset instead of holding the pose",
    )
    args = parser.parse_args()
    if args.speed_deg_s <= 0 or args.control_hz <= 0:
        parser.error("--speed-deg-s and --control-hz must be positive")
    if args.max_move_deg <= 0 or args.tolerance_deg <= 0:
        parser.error("--max-move-deg and --tolerance-deg must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.calibrate:
        calibrate(args)
    elif args.release:
        release(args)
    else:
        reset(args)


if __name__ == "__main__":
    main()
