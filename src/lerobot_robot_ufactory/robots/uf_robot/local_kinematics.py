"""Small, allocation-light local kinematics helpers for xArm7 safety checks."""

from __future__ import annotations

import math
import socket
import struct

import numpy as np


_KINEMATICS_REQUEST = bytes((0x00, 0x01, 0x00, 0x02, 0x00, 0x01, 0x08))
_KINEMATICS_RESPONSE_SIZE = 179


def read_xarm7_kinematics(robot_ip: str, timeout_s: float = 2.0) -> np.ndarray:
    """Read the controller-calibrated joint origins used by UFACTORY's ROS model.

    Returns seven rows of ``x, y, z, roll, pitch, yaw``. Translation is in
    metres and rotation is in radians.
    """
    with socket.create_connection((robot_ip, 502), timeout=timeout_s) as sock:
        sock.settimeout(timeout_s)
        sock.sendall(_KINEMATICS_REQUEST)
        response = bytearray()
        while len(response) < _KINEMATICS_RESPONSE_SIZE:
            chunk = sock.recv(_KINEMATICS_RESPONSE_SIZE - len(response))
            if not chunk:
                break
            response.extend(chunk)

    if len(response) != _KINEMATICS_RESPONSE_SIZE or response[8] == 0:
        raise RuntimeError(
            f"Unable to read xArm kinematics calibration: bytes={len(response)}, "
            f"valid={bool(response[8]) if len(response) > 8 else False}"
        )
    if response[9] != 7:
        raise RuntimeError(f"Expected xArm7 kinematics, controller reported {response[9]} axes")
    origins = np.asarray(struct.unpack("<42f", response[11:179]), dtype=np.float64).reshape(7, 6)
    if not np.all(np.isfinite(origins)):
        raise RuntimeError("Controller returned non-finite kinematics calibration")
    return origins


def _rpy_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _pose_transform(pose: np.ndarray, translation_scale: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _rpy_rotation(*pose[3:6])
    transform[:3, 3] = pose[:3] * translation_scale
    return transform


def xarm_rpy_transform(pose: list[float] | np.ndarray) -> np.ndarray:
    """Convert xArm FK ``x, y, z, roll, pitch, yaw`` output to a matrix."""
    value = np.asarray(pose, dtype=np.float64)
    if value.shape != (6,) or not np.all(np.isfinite(value)):
        raise ValueError("RPY pose must contain six finite values")
    return _pose_transform(value, 1.0)


def rotation_to_axis_angle(rotation: np.ndarray) -> np.ndarray:
    """Rotation matrix -> axis-angle vector (rad), angle in [0, pi]."""
    cos_angle = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.acos(cos_angle)
    if angle < 1e-9:
        return np.zeros(3)
    if math.pi - angle < 1e-4:
        # Near pi: R + I = 2 * axis * axis^T; recover axis from the diagonal
        # and resolve signs with the off-diagonal terms.
        axis = np.sqrt(np.clip((np.diag(rotation) + 1.0) / 2.0, 0.0, None))
        axis[0] = math.copysign(axis[0], rotation[0, 1] * axis[1] if abs(axis[1]) > 1e-9 else rotation[0, 1])
        axis[1] = math.copysign(axis[1], rotation[0, 1])
        axis[2] = math.copysign(axis[2], rotation[0, 2])
        norm = np.linalg.norm(axis)
        if norm < 1e-9:
            return np.zeros(3)
        return axis / norm * angle
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    ) / (2.0 * math.sin(angle))
    return axis * angle


def axis_angle_continuous(rotation: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
    """Pick the axis-angle representative closest to the previous frame.

    R(axis, angle) == R(-axis, 2*pi - angle); choosing per frame between the
    two avoids pi-flip discontinuities (e.g. TCP pointing straight down).
    """
    aa = rotation_to_axis_angle(rotation)
    if previous is None:
        return aa
    angle = np.linalg.norm(aa)
    if angle < 1e-9:
        return previous.copy()
    alt = -aa / angle * (2.0 * math.pi - angle)
    return alt if np.linalg.norm(alt - previous) < np.linalg.norm(aa - previous) else aa


def rotation_to_6d(rotation: np.ndarray) -> np.ndarray:
    """Rotation matrix -> continuous 6D representation ``[r11, r21, r31, r12, r22, r32]``.

    The 6D representation is the first two columns of the rotation matrix,
    from Zhou et al., "On the Continuity of Rotation Representations in
    Neural Networks" (CVPR 2019, arXiv:1812.07035). Unlike Euler angles,
    quaternions, or axis-angle it has no discontinuities, which makes it
    suitable as a learning target.
    """
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    return np.concatenate([rotation[:, 0], rotation[:, 1]])


def rot6d_to_rotation(rotation_6d: np.ndarray) -> np.ndarray:
    """6D representation -> rotation matrix via Gram-Schmidt orthonormalization.

    Inverse of ``rotation_to_6d``, following the reference implementation of
    Zhou et al.: b1 = N(a1), b2 = N(a2 - (b1 . a2) b1), b3 = b1 x b2.
    """
    value = np.asarray(rotation_6d, dtype=np.float64)
    if value.shape != (6,) or not np.all(np.isfinite(value)):
        raise ValueError("rotation_6d must be a finite six-element vector")
    a1, a2 = value[:3], value[3:6]
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


class XArm7Kinematics:
    """Forward kinematics and TCP-height Jacobian for one calibrated xArm7."""

    def __init__(
        self,
        joint_origins: np.ndarray,
        tcp_offset: list[float] | np.ndarray | None = None,
        world_offset: list[float] | np.ndarray | None = None,
        end_transform: np.ndarray | None = None,
    ) -> None:
        origins = np.asarray(joint_origins, dtype=np.float64)
        if origins.shape != (7, 6) or not np.all(np.isfinite(origins)):
            raise ValueError("joint_origins must be a finite 7x6 array")
        tcp = np.zeros(6) if tcp_offset is None else np.asarray(tcp_offset, dtype=np.float64)
        world = np.zeros(6) if world_offset is None else np.asarray(world_offset, dtype=np.float64)
        if tcp.shape != (6,) or world.shape != (6,) or not np.all(np.isfinite([*tcp, *world])):
            raise ValueError("TCP and world offsets must be finite six-element poses")

        self._origins = tuple(_pose_transform(row, 1000.0) for row in origins)
        if end_transform is None:
            self._tcp = _pose_transform(tcp, 1.0)
        else:
            endpoint = np.asarray(end_transform, dtype=np.float64)
            if endpoint.shape != (4, 4) or not np.all(np.isfinite(endpoint)):
                raise ValueError("end_transform must be a finite 4x4 matrix")
            self._tcp = endpoint.copy()
        self._world = _pose_transform(world, 1.0)

    def forward_matrix(self, joints: list[float] | np.ndarray) -> np.ndarray:
        q = np.asarray(joints, dtype=np.float64)
        if q.shape != (7,) or not np.all(np.isfinite(q)):
            raise ValueError("joints must be a finite seven-element vector")
        transform = self._world.copy()
        for origin, angle in zip(self._origins, q, strict=True):
            transform = transform @ origin
            c, s = np.cos(angle), np.sin(angle)
            rotation_z = np.asarray(
                [[c, -s, 0.0, 0.0], [s, c, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            transform = transform @ rotation_z
        return transform @ self._tcp

    def tcp_position(self, joints: list[float] | np.ndarray) -> np.ndarray:
        return self.forward_matrix(joints)[:3, 3]

    def tcp_z_and_jacobian(self, joints: list[float] | np.ndarray) -> tuple[float, np.ndarray]:
        q = np.asarray(joints, dtype=np.float64)
        if q.shape != (7,) or not np.all(np.isfinite(q)):
            raise ValueError("joints must be a finite seven-element vector")

        transform = self._world.copy()
        axes = np.empty((7, 3), dtype=np.float64)
        points = np.empty((7, 3), dtype=np.float64)
        for index, (origin, angle) in enumerate(zip(self._origins, q, strict=True)):
            transform = transform @ origin
            points[index] = transform[:3, 3]
            axes[index] = transform[:3, 2]
            c, s = np.cos(angle), np.sin(angle)
            rotation_z = np.asarray(
                [[c, -s, 0.0, 0.0], [s, c, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            transform = transform @ rotation_z
        tcp_position = (transform @ self._tcp)[:3, 3]
        jacobian_z = np.cross(axes, tcp_position - points)[:, 2]
        return float(tcp_position[2]), jacobian_z
