#!/usr/bin/env python

import time
import math
import logging
import struct
import numpy as np
from datetime import datetime
from enum import IntEnum
from dataclasses import dataclass
from pathlib import Path
from threading import Thread, Event, Lock
from lerobot.robots import Robot
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot_robot_ufactory.devices.pika import PikaDevice
from .uf_robot_config import UFRobotConfig
from xarm.wrapper import XArmAPI
from xarm.core.utils import convert

from .local_kinematics import (
    XArm7Kinematics,
    read_xarm7_kinematics,
    xarm_rpy_transform,
)


logger = logging.getLogger(__name__)

## Configurations:
INIT_SYNC_JOINT_VELOCITY_RAD = 0.2
ROBOT_RESET_SPEED_DEG = 20
TCP_Z_CLAMP_TOLERANCE_MM = 1e-3
TCP_Z_LOG_INTERVAL_S = 1.0
TCP_Z_MAX_IK_JOINT_STEP_RAD = math.radians(10.0)
LOCAL_GUARD_MAX_ITERATIONS = 8
LOCAL_GUARD_JACOBIAN_DAMPING = 1e-6
XARM7_JOINT_LOWER_RAD = np.asarray(
    [-2 * math.pi, -2.059, -2 * math.pi, -0.19198, -2 * math.pi, -1.69297, -2 * math.pi]
)
XARM7_JOINT_UPPER_RAD = np.asarray(
    [2 * math.pi, 2.0944, 2 * math.pi, 3.927, 2 * math.pi, math.pi, 2 * math.pi]
)

CARTESIAN_OBS_KEYS = [
    "pose.x", "pose.y", "pose.z", "pose.rx", "pose.ry", "pose.rz",
    # un-comment if you need more features below:
    # "velo.x", "velo.y", "velo.z", "velo.rx", "velo.ry", "velo.rz",
]

CARTESIAN_ACTION_KEYS = [
    "pose.x", "pose.y", "pose.z", "pose.rx", "pose.ry", "pose.rz",
]

class GripperType(IntEnum):
    NoGripper = 0
    xArmGripper = 1
    xArmGripperG2 = 2
    BioGripperG2 = 3
    PikaGripper = 10
    RobotiqGripper = 11


@dataclass
class GripperParam:
    name: str
    open_pos: int
    close_pos: int
    speed: int = 0
    force: int = 0
    gripper_norm: float = 0

    def get_grippos(self, gripper_norm):
        pos = self.open_pos + gripper_norm * (self.close_pos - self.open_pos)
        min_pos, max_pos = min(self.open_pos, self.close_pos), max(self.open_pos, self.close_pos)
        return int(min(max(min_pos, pos), max_pos))

    def get_gripper_norm(self, grippos):
        if grippos is None:
            return self.gripper_norm
        self.gripper_norm = (self.open_pos - grippos) / (self.open_pos - self.close_pos)
        return self.gripper_norm


class UFRobot(Robot, Thread):

    config_class = UFRobotConfig
    name = "UFACTORY Robot"

    def __init__(self, config: UFRobotConfig, prefix=''):
        super().__init__(config)
        Thread.__init__(self)
        self.prefix = '' if not prefix else f"{prefix}."
        self.config = config
        self._dof = config.robot_dof 
        if self._dof == None or (not self._dof in (5,6,7)):
            raise ValueError(f"Please specify the correct DOF uf_robot!, got {self._dof}")
        # LeRobot uses this value in dataset metadata and policy processors.
        # The vendor name is not specific enough to distinguish xArm variants.
        self.robot_type = f"xarm{self._dof}"
        
        self._control_space = self.config.control_space

        self.real_arm = None
        self._initial_point = None
        cameras_args = self.config.cameras_args or {}
        self.camera_width = cameras_args.get('w', 0)
        self.camera_height = cameras_args.get('h', 0)
        self.cameras = make_cameras_from_configs(config.cameras)

        self._is_connected = False
        self._is_calibrated =True

        self.enable_logs = bool(getattr(config, "enable_logs", False))
        self.logs = {}

        self._cmd_cnt = 0
        self._last_gripper_command = None
        self._last_gripper_command_attempt_s = float("-inf")
        self._last_logged_controller_error = 0

        self._max_joint_velocity = math.radians(self.config.max_joint_velocity)
        self._max_linear_velocity = self.config.max_linear_velocity

        self._min_tcp_z_mm = self.config.min_tcp_z_mm
        self._tcp_z_guard_activation_margin_mm = self.config.tcp_z_guard_activation_margin_mm
        self._tcp_z_guard_backend = self.config.tcp_z_guard_backend
        self._tcp_z_soft_floor_mm = (
            None
            if self._min_tcp_z_mm is None
            else self._min_tcp_z_mm + self.config.tcp_z_soft_margin_mm
        )
        self._local_kinematics = None
        self._local_joint_origins = None
        self._local_model_fault_count = 0
        self._last_safe_joint_target = None
        self._last_guard_path = "not_run"
        self._tcp_z_is_clamped = False
        self._tcp_z_last_log_time = 0.0
        self._tcp_z_last_error_log_time = 0.0

        self.report_stop_event = Event()
        self._rt_report_normal = False
        self._update_lock = Lock()
        self._last_rt_report_monotonic_s = None
        self._last_realtime_sync_timing = {}
        self._realtime_camera_frame_index = {key: 0 for key in self.cameras}
        # Cartesian observations and the joint-mode TCP z guard use the
        # asynchronous RT report.
        self._use_rt_report = (
            self._control_space == "cartesian" or self._min_tcp_z_mm is not None
        )
        self._cart_obs_has_vel = any('velo.' in key for key in CARTESIAN_OBS_KEYS)
        self._jnt_obs_has_vel = self.config.observe_joint_vel

        self._gripper_type = self.config.gripper_type
        if self._gripper_type == GripperType.xArmGripper:
            gripper_speed = 5000 if self.config.gripper_speed < 0 else min(max(50, self.config.gripper_speed), 5000)
            gripper_force = 50 if self.config.gripper_force < 0 else self.config.gripper_force # # not support
            self._gripper_param = GripperParam('xArmGripper', open_pos=800, close_pos=0, speed=gripper_speed, force=gripper_force)
        elif self._gripper_type == GripperType.xArmGripperG2:
            speed = 225 if self.config.gripper_speed < 0 else min(max(15, self.config.gripper_speed), 225)
            gripper_speed = int(((speed * 60) / 9.88235 + 140) / 0.4)
            # Keep the SDK-facing speed in mm/s. GripperParam.speed retains
            # the original low-level register conversion used by this repo.
            self._gripper_g2_speed = speed
            gripper_force = 50 if self.config.gripper_force < 0 else min(max(1, self.config.gripper_force), 100)
            self._gripper_param = GripperParam('xArmGripperG2', open_pos=84, close_pos=0, speed=gripper_speed, force=gripper_force)
        elif self._gripper_type == GripperType.BioGripperG2:
            gripper_speed = 2000 if self.config.gripper_speed < 0 else min(max(500, self.config.gripper_speed), 4500)
            gripper_force = 100 if self.config.gripper_force < 0 else min(max(1, self.config.gripper_force), 100)
            self._gripper_param = GripperParam('BioGripperG2', open_pos=150, close_pos=71, speed=gripper_speed, force=gripper_force)
        elif self._gripper_type == GripperType.PikaGripper:
            self.pika_device = PikaDevice(2, pika_gripper_port=self.config.gripper_port)
            self.pika_gripper = self.pika_device.pika_gripper
            logger = logging.getLogger('pika.gripper')
            logger.setLevel(logging.WARNING)
            gripper_speed = 0 if self.config.gripper_speed < 0 else self.config.gripper_speed # not support
            gripper_force = 0 if self.config.gripper_force < 0 else self.config.gripper_force # not support
            self._gripper_param = GripperParam('PikaGripper', open_pos=100, close_pos=0, speed=gripper_speed, force=gripper_force)
        elif self._gripper_type == GripperType.RobotiqGripper:
            gripper_speed = 255 if self.config.gripper_speed < 0 else min(max(1, self.config.gripper_speed), 255)
            gripper_force = 255 if self.config.gripper_force < 0 else min(max(1, self.config.gripper_force), 255)
            self._gripper_param = GripperParam('RobotiqGripper', open_pos=0, close_pos=0xFF, speed=gripper_speed, force=gripper_force)
        else: # no gripper or not support
            self._gripper_type = 0
            self._gripper_param = GripperParam('NoGripper', open_pos=0, close_pos=0, speed=0, force=0)

    @property
    def _robot_state_features(self)-> dict:
        if self._control_space == "joint":
            state_features = {f"{self.prefix}J{motor}.pos": float for motor in range(1, self._dof+1)}
            if self._jnt_obs_has_vel:
                state_features.update({f"{self.prefix}J{motor}.vel": float for motor in range(1, self._dof+1)})
            if self._gripper_type > GripperType.NoGripper:
                state_features.update({f"{self.prefix}gripper.pos": float})
        elif self._control_space == "cartesian":
            state_features = {f"{self.prefix}{key}": float for key in CARTESIAN_OBS_KEYS}
            if self._gripper_type > GripperType.NoGripper:
                state_features.update({f"{self.prefix}gripper.pos": float})
        else:
            raise ValueError(f"Please check the given control space of uf_robot! got {self._control_space}")
        return state_features

    @property
    # CHECK!! channel first or last?
    def _cam_features(self) -> dict:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            camera_width = self.camera_width if self.camera_width != 0 else cam.width
            camera_height = self.camera_height if self.camera_height != 0 else cam.height
            cam_ft[f"{self.prefix}{cam_key}"] = (camera_height, camera_width, 3)
            # cam_ft[f"{self.prefix}{cam_key}"] = (cam.height, cam.width, 3)
        return cam_ft

    @property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._robot_state_features, **self._cam_features}

    @property
    def action_features(self)-> dict:
        if self._control_space == "joint":
            action_ft = {f"{self.prefix}J{motor}.pos": float for motor in range(1, self._dof+1)}
        elif self._control_space == "cartesian":
            action_ft = {f"{self.prefix}{key}": float for key in CARTESIAN_ACTION_KEYS}
        else:
            raise ValueError(f"Please check the given control space of uf_robot! got {self._control_space}")
        # Consider adding velocity configuration ??
        if self._gripper_type > GripperType.NoGripper:
            action_ft.update({f"{self.prefix}gripper.pos": float})
        return action_ft

    def connect(self, calibrate: bool = True) -> None:
        if self._tcp_z_guard_backend == "local_projection":
            self._local_joint_origins = read_xarm7_kinematics(self.config.robot_ip)
        self.real_arm = XArmAPI(self.config.robot_ip)
        time.sleep(0.2)
        self._is_connected = self.real_arm.connected
        if not self._is_connected:
            print(f"UF Robot connection Failed, please check the hardware availability at ip: {self.config.robot_ip}")
            raise ConnectionError()

        if not self._dof == self.real_arm.axis:
            print(f"[ERROR: ] Real Robot DOF({self.real_arm.axis}) does not match configuration ({self._dof})!")
            self._is_connected = False
            raise ConnectionError()

        code, initial_point = self.real_arm.get_initial_point()
        if code != 0:
            raise RuntimeError(f"get_initial_point failed, code={code}")
        if initial_point is None or len(initial_point) < self._dof:
            raise RuntimeError(f"Invalid initial point returned by xArm: {initial_point}")
        self._initial_point = list(initial_point[:self._dof])

        if self._tcp_z_guard_backend == "local_projection":
            self._initialize_local_kinematics()

        for cam in self.cameras.values():
            cam.connect()
            self._is_connected = self._is_connected and cam.is_connected

        if not self._is_connected:
            print("Could not connect to the cameras, check that all cameras are plugged-in.")
            raise ConnectionError()

        # if self._gripper_type == GripperType.PikaGripper:
        #     if not self.pika_gripper.connect():
        #         print('Could not connect to pika gripper.')
        #         raise ConnectionError()

        if self.config.manual_mode:
            self.configure()
        else:
            self.reset_to_initial()
        # reset_to_initial clears any pre-existing controller error before
        # configuring APIs that may otherwise be rejected with code 1.
        if (
            self._tcp_z_guard_backend == "local_projection"
            and self.config.controller_safety_boundary
        ):
            self._configure_controller_safety_boundary()
        if self._min_tcp_z_mm is not None and self.config.manual_mode:
            self._initialize_tcp_z_guard()
        if calibrate:  
            self.calibrate()

        self.real_arm.set_linear_spd_limit_factor(2.0)

        self._is_connected = True

    def reset_to_initial(self) -> None:
        if not self._is_connected or self.real_arm is None:
            raise ConnectionError("UF Robot is not connected")
        if self._initial_point is None:
            raise RuntimeError("xArm initial point has not been loaded")

        # The controller requires motion to be enabled again after an
        # emergency stop has been released, before any reset motion command.
        code = self.real_arm.motion_enable(enable=True)
        self._check_motion_code("motion_enable", code)
        code = self.real_arm.clean_error()
        self._check_motion_code("clean_error", code)
        code = self.real_arm.set_mode(0)
        self._check_motion_code("set_mode(0)", code)
        code = self.real_arm.set_state(0)
        self._check_motion_code("set_state(0)", code)
        code = self.real_arm.set_servo_angle(
            angle=self._initial_point,
            speed=ROBOT_RESET_SPEED_DEG,
            is_radian=False,
            wait=True,
        )
        if code != 0:
            raise RuntimeError(f"Failed to move to xArm initial point, code={code}")

        self.configure()
        if self._min_tcp_z_mm is not None:
            self._initialize_tcp_z_guard()

    def _initialize_tcp_z_guard(self) -> None:
        """Initialize guard state from the robot's current physical target."""
        if self._control_space != "joint" or self._min_tcp_z_mm is None or self.real_arm is None:
            return

        code, states = self.real_arm.get_joint_states(is_radian=True, num=1)
        if code != 0 or not states or len(states[0]) < self._dof:
            raise RuntimeError(f"Unable to initialize TCP z guard from joint state, code={code}")
        target = np.asarray(states[0][:self._dof], dtype=np.float64)
        if not np.all(np.isfinite(target)):
            raise RuntimeError("Unable to initialize TCP z guard from non-finite joint state")
        self._last_safe_joint_target = target
        if self._tcp_z_guard_backend == "local_projection":
            if self._local_kinematics is None:
                raise RuntimeError("Local xArm7 kinematics has not been initialized")
            current_z = float(self._local_kinematics.tcp_position(target)[2])
            if current_z < self._min_tcp_z_mm:
                raise RuntimeError(
                    f"Current TCP z {current_z:.2f} mm is below the hard floor "
                    f"{self._min_tcp_z_mm:.2f} mm"
                )
        self._tcp_z_is_clamped = False
        self._tcp_z_last_log_time = 0.0
        self._tcp_z_last_error_log_time = 0.0

    def _controller_pose_offset(self, name: str) -> np.ndarray:
        pose = np.asarray(getattr(self.real_arm, name), dtype=np.float64)
        if pose.shape != (6,) or not np.all(np.isfinite(pose)):
            raise RuntimeError(f"Controller returned an invalid {name}: {pose}")
        if not self.real_arm.default_is_radian:
            pose[3:6] = np.radians(pose[3:6])
        return pose

    def _initialize_local_kinematics(self) -> None:
        if self._local_joint_origins is None:
            raise RuntimeError("xArm7 calibration parameters were not loaded")
        code, states = self.real_arm.get_joint_states(is_radian=True, num=1)
        if code != 0 or not states or len(states[0]) < 7:
            raise RuntimeError(f"Unable to validate local kinematics from joint state, code={code}")
        current = np.asarray(states[0][:7], dtype=np.float64)
        world_offset = self._controller_pose_offset("world_offset")
        chain_kinematics = XArm7Kinematics(
            self._local_joint_origins,
            world_offset=world_offset,
        )
        code, current_pose = self.real_arm.get_forward_kinematics(
            current.tolist(), input_is_radian=True, return_is_radian=True
        )
        if code != 0 or current_pose is None or len(current_pose) < 6:
            raise RuntimeError(f"Controller FK failed while identifying its TCP endpoint, code={code}")
        controller_transform = xarm_rpy_transform(current_pose[:6])
        endpoint_transform = (
            np.linalg.inv(chain_kinematics.forward_matrix(current)) @ controller_transform
        )
        self._local_kinematics = XArm7Kinematics(
            self._local_joint_origins,
            world_offset=world_offset,
            end_transform=endpoint_transform,
        )

        current_z = float(self._local_kinematics.tcp_position(current)[2])
        if current_z < self._min_tcp_z_mm:
            raise RuntimeError(
                f"Current TCP z {current_z:.2f} mm is below the configured hard floor "
                f"{self._min_tcp_z_mm:.2f} mm"
            )
        validation_points = [current.copy(), current.copy()]
        validation_points[0][1] = np.clip(
            validation_points[0][1] + 0.05,
            XARM7_JOINT_LOWER_RAD[1],
            XARM7_JOINT_UPPER_RAD[1],
        )
        validation_points[1][3] = np.clip(
            validation_points[1][3] + 0.05,
            XARM7_JOINT_LOWER_RAD[3],
            XARM7_JOINT_UPPER_RAD[3],
        )
        for joints in validation_points:
            code, pose = self.real_arm.get_forward_kinematics(
                joints.tolist(), input_is_radian=True, return_is_radian=True
            )
            if code != 0 or pose is None or len(pose) < 3:
                raise RuntimeError(f"Controller FK failed during local model validation, code={code}")
            local_position = self._local_kinematics.tcp_position(joints)
            error_mm = float(np.linalg.norm(local_position - np.asarray(pose[:3], dtype=np.float64)))
            if not math.isfinite(error_mm) or error_mm > self.config.local_kinematics_max_error_mm:
                raise RuntimeError(
                    f"Local kinematics differs from controller FK by {error_mm:.3f} mm "
                    f"(limit {self.config.local_kinematics_max_error_mm:.3f} mm)"
                )

    @staticmethod
    def _motion_code(code):
        return code[0] if isinstance(code, (tuple, list)) else code

    def _configure_controller_safety_boundary(self) -> None:
        floor = int(math.ceil(self._min_tcp_z_mm))
        boundary = [9999, -9999, 9999, -9999, 9999, floor]
        code = self._motion_code(self.real_arm.set_reduced_tcp_boundary(boundary))
        self._check_motion_code("set_reduced_tcp_boundary", code)
        code = self._motion_code(self.real_arm.set_fence_mode(True))
        self._check_motion_code("set_fence_mode(True)", code)

        code, states = self.real_arm.get_reduced_states(is_radian=True)
        self._check_motion_code("get_reduced_states", code)
        if len(states) < 2 or list(map(int, states[1][:6])) != boundary:
            raise RuntimeError(f"Controller safety boundary readback mismatch: {states}")
        if len(states) >= 6 and not bool(states[5]):
            raise RuntimeError("Controller fence mode did not remain enabled")

    def _log_tcp_z_clamp(self, clamped: bool, requested_z: float | None = None) -> None:
        """Log clamp state changes while avoiding per-cycle console spam."""
        now = time.monotonic()
        if clamped:
            should_log = not self._tcp_z_is_clamped or now - self._tcp_z_last_log_time >= TCP_Z_LOG_INTERVAL_S
            if should_log:
                logger.warning(
                    "TCP z safety clamp active: requested %.2f mm, limiting to %.2f mm",
                    requested_z if requested_z is not None else float("nan"),
                    self._min_tcp_z_mm,
                )
                self._tcp_z_last_log_time = now
        elif self._tcp_z_is_clamped:
            logger.info("TCP z safety clamp released")
        self._tcp_z_is_clamped = clamped

    def _log_tcp_z_guard_error(self, message: str) -> None:
        now = time.monotonic()
        if now - self._tcp_z_last_error_log_time >= TCP_Z_LOG_INTERVAL_S:
            logger.error("TCP z safety guard rejected target: %s", message)
            self._tcp_z_last_error_log_time = now

    def _guard_joint_target(self, command: list[float]) -> np.ndarray | None:
        if getattr(self, "_tcp_z_guard_backend", "controller_rpc") == "local_projection":
            return self._guard_joint_target_local(command)
        return self._guard_joint_target_controller_rpc(command)

    def _guard_joint_target_local(self, command: list[float]) -> np.ndarray | None:
        """Project a joint update onto the local TCP-height constraint."""
        desired = np.asarray(command, dtype=np.float64)
        fallback = self._last_safe_joint_target
        try:
            if self._local_kinematics is None:
                raise RuntimeError("local kinematics is unavailable")
            if not self._local_model_matches_rt():
                self._last_guard_path = "model_fault"
                return None if fallback is None else np.asarray(fallback, dtype=np.float64).copy()
            if desired.shape != (7,) or not np.all(np.isfinite(desired)):
                raise ValueError("joint target has invalid shape or contains NaN/Inf")
            if fallback is None:
                raise RuntimeError("last safe joint target is unavailable")

            previous = np.asarray(fallback, dtype=np.float64)
            if previous.shape != (7,) or not np.all(np.isfinite(previous)):
                raise RuntimeError("last safe joint target is invalid")
            delta = (desired - previous + math.pi) % (2 * math.pi) - math.pi
            delta = np.clip(delta, -TCP_Z_MAX_IK_JOINT_STEP_RAD, TCP_Z_MAX_IK_JOINT_STEP_RAD)
            candidate = np.clip(previous + delta, XARM7_JOINT_LOWER_RAD, XARM7_JOINT_UPPER_RAD)
            soft_floor = float(self._tcp_z_soft_floor_mm)

            requested_z = float(self._local_kinematics.tcp_position(candidate)[2])
            if requested_z >= soft_floor:
                self._last_guard_path = "local_safe"
                self._last_safe_joint_target = candidate.copy()
                self._log_tcp_z_clamp(False)
                return candidate

            projected = candidate
            for _ in range(LOCAL_GUARD_MAX_ITERATIONS):
                z_value, jacobian_z = self._local_kinematics.tcp_z_and_jacobian(projected)
                error = soft_floor - z_value
                if error <= TCP_Z_CLAMP_TOLERANCE_MM:
                    break
                norm_sq = float(jacobian_z @ jacobian_z)
                if not math.isfinite(norm_sq) or norm_sq < 1e-10:
                    raise RuntimeError("TCP-height Jacobian is singular")
                projected = projected + jacobian_z * error / (
                    norm_sq + LOCAL_GUARD_JACOBIAN_DAMPING
                )
                projected = np.clip(projected, XARM7_JOINT_LOWER_RAD, XARM7_JOINT_UPPER_RAD)

            projected_delta = (projected - previous + math.pi) % (2 * math.pi) - math.pi
            max_delta = float(np.max(np.abs(projected_delta)))
            if max_delta > TCP_Z_MAX_IK_JOINT_STEP_RAD:
                projected = previous + projected_delta * (TCP_Z_MAX_IK_JOINT_STEP_RAD / max_delta)

            projected_z = float(self._local_kinematics.tcp_position(projected)[2])
            if projected_z < soft_floor - TCP_Z_CLAMP_TOLERANCE_MM:
                # The linearized correction can overshoot near high curvature.
                # Backtrack toward the known-safe previous command without an RPC.
                accepted = None
                for alpha in np.linspace(0.875, 0.0, 8):
                    trial = previous + alpha * (projected - previous)
                    if float(self._local_kinematics.tcp_position(trial)[2]) >= soft_floor:
                        accepted = trial
                        break
                if accepted is None:
                    self._last_guard_path = "local_hold"
                    self._log_tcp_z_clamp(True, requested_z)
                    return previous.copy()
                projected = accepted

            self._last_guard_path = "local_projected"
            self._last_safe_joint_target = projected.copy()
            self._log_tcp_z_clamp(True, requested_z)
            return projected
        except Exception as exc:
            self._last_guard_path = "local_hold"
            self._log_tcp_z_guard_error(str(exc))
            return None if fallback is None else np.asarray(fallback, dtype=np.float64).copy()

    def _local_model_matches_rt(self) -> bool:
        """Compare co-timed RT joint/TCP feedback without making an SDK request."""
        if not getattr(self, "_rt_report_normal", False):
            return True
        with self._update_lock:
            joints = np.asarray(self.rt_actual_joint_pos, dtype=np.float64).copy()
            reported = np.asarray(self.rt_actual_tcp_pose[:3], dtype=np.float64).copy()
        try:
            local = self._local_kinematics.tcp_position(joints)
            error_mm = float(np.linalg.norm(local - reported))
            limit = float(self.config.local_kinematics_max_error_mm)
            if getattr(self, "enable_logs", True):
                self.logs["local_kinematics_error_mm"] = error_mm
            if math.isfinite(error_mm) and error_mm <= limit:
                self._local_model_fault_count = 0
                return True
        except Exception as exc:
            self._log_tcp_z_guard_error(f"RT model validation failed: {exc}")
        self._local_model_fault_count += 1
        if self._local_model_fault_count >= 3:
            self._log_tcp_z_guard_error(
                f"local/RT TCP mismatch persisted for {self._local_model_fault_count} frames"
            )
            return False
        return True

    def _guard_joint_target_controller_rpc(self, command: list[float]) -> np.ndarray | None:
        """Return a safe joint target, or None when motion must be skipped."""
        desired = np.asarray(command, dtype=np.float64)
        if self._min_tcp_z_mm is None:
            self._last_guard_path = "disabled"
            return desired

        fallback = self._last_safe_joint_target
        try:
            if desired.shape != (self._dof,) or not np.all(np.isfinite(desired)):
                raise ValueError("joint target has invalid shape or contains NaN/Inf")

            # Far above the floor, the current TCP height is available from
            # the RT report. Bypass the synchronous controller FK call so the
            # normal GELLO path keeps a stable command cadence. A large
            # activation margin absorbs ordinary per-cycle motion changes.
            if self._rt_actual_tcp_is_far_above_floor():
                self._last_guard_path = "rt_fast_path"
                self._last_safe_joint_target = desired.copy()
                self._log_tcp_z_clamp(False)
                return desired

            code, pose = self.real_arm.get_forward_kinematics(
                desired.tolist(), input_is_radian=True, return_is_radian=True
            )
            pose = np.asarray(pose, dtype=np.float64)
            if code != 0 or pose.shape[0] < 6 or not np.all(np.isfinite(pose)):
                raise RuntimeError(f"forward kinematics failed, code={code}")

            requested_z = float(pose[2])
            if requested_z >= self._min_tcp_z_mm:
                self._last_guard_path = "fk_safe"
                self._validate_guard_joint_target(desired, fallback, "GELLO target")
                self._last_safe_joint_target = desired.copy()
                self._log_tcp_z_clamp(False)
                return desired

            clamped_pose = pose[:6].copy()
            clamped_pose[2] = self._min_tcp_z_mm
            ik_reference = fallback if fallback is not None else desired
            code, inverse = self.real_arm.get_inverse_kinematics(
                clamped_pose.tolist(),
                input_is_radian=True,
                return_is_radian=True,
                limited=True,
                ref_angles=np.asarray(ik_reference, dtype=np.float64).tolist(),
            )
            inverse = np.asarray(inverse, dtype=np.float64)
            if code != 0 or inverse.shape[0] < self._dof or not np.all(np.isfinite(inverse)):
                raise RuntimeError(f"inverse kinematics failed, code={code}")

            safe_target = inverse[:self._dof].copy()
            self._last_guard_path = "fk_ik_clamp"
            self._validate_guard_joint_target(safe_target, fallback, "clamped IK target")
            code, verified_pose = self.real_arm.get_forward_kinematics(
                safe_target.tolist(), input_is_radian=True, return_is_radian=True
            )
            verified_pose = np.asarray(verified_pose, dtype=np.float64)
            if (
                code != 0
                or verified_pose.shape[0] < 3
                or not np.all(np.isfinite(verified_pose))
                or verified_pose[2] < self._min_tcp_z_mm - TCP_Z_CLAMP_TOLERANCE_MM
            ):
                raise RuntimeError(f"inverse-kinematics result is below the TCP z floor, code={code}")

            self._last_safe_joint_target = safe_target
            self._log_tcp_z_clamp(True, requested_z)
            return safe_target
        except Exception as exc:
            self._last_guard_path = "fallback"
            self._log_tcp_z_guard_error(str(exc))
            if fallback is None:
                return None
            return np.asarray(fallback, dtype=np.float64).copy()

    def _validate_guard_joint_target(
        self,
        target: np.ndarray,
        previous_safe_target: np.ndarray | None,
        label: str,
    ) -> None:
        code, is_limited = self.real_arm.is_joint_limit(target.tolist(), is_radian=True)
        if code != 0 or is_limited is not False:
            raise RuntimeError(
                f"{label} violates a joint limit, code={code}, limited={is_limited}, "
                f"target={target.tolist()}"
            )

        if previous_safe_target is None:
            return
        previous = np.asarray(previous_safe_target, dtype=np.float64)
        if previous.shape != target.shape or not np.all(np.isfinite(previous)):
            raise RuntimeError("previous safe joint target is invalid")
        delta = (target - previous + math.pi) % (2 * math.pi) - math.pi
        max_delta = float(np.max(np.abs(delta)))
        if max_delta > TCP_Z_MAX_IK_JOINT_STEP_RAD:
            raise RuntimeError(
                f"{label} jumps {math.degrees(max_delta):.1f} deg from the previous safe target"
            )

    def _rt_actual_tcp_is_far_above_floor(self) -> bool:
        if self._min_tcp_z_mm is None or not getattr(self, "_rt_report_normal", False):
            return False
        update_lock = getattr(self, "_update_lock", None)
        if update_lock is None:
            return False
        with update_lock:
            pose = getattr(self, "rt_actual_tcp_pose", None)
            if pose is None or len(pose) < 3:
                return False
            actual_z = float(pose[2])
        activation_margin = getattr(self, "_tcp_z_guard_activation_margin_mm", 100.0)
        return (
            math.isfinite(actual_z)
            and actual_z > self._min_tcp_z_mm + activation_margin
        )

    def configure(self) -> None:
        self.real_arm.motion_enable()
        self.real_arm.clean_error()
        self.real_arm.set_mode(0)  # set to idle mode
        self.real_arm.set_state(0)  # set to start state
        time.sleep(0.5)

        _, err_warn = self.real_arm.get_err_warn_code()
        if err_warn[0] != 0:
            raise RuntimeError(f"Failed to set correct state to UF robot! Controller Error code: {err_warn[0]} !")

        if self._gripper_type > GripperType.NoGripper:
            self._configure_gripper(move_to_open=not self.config.manual_mode)

        if self.config.manual_mode:
            if self.config.teach_sensitivity is not None:
                code = self.real_arm.set_teach_sensitivity(self.config.teach_sensitivity)
                if code != 0:
                    raise RuntimeError(f"set_teach_sensitivity failed, code={code}")

            code = self.real_arm.set_mode(2)
            if code != 0:
                raise RuntimeError(f"set_mode(2) failed, code={code}")
            code = self.real_arm.set_state(0)
            if code != 0:
                raise RuntimeError(f"set_state(0) failed, code={code}")

            _, err_warn = self.real_arm.get_err_warn_code()
            if err_warn[0] != 0:
                raise RuntimeError(
                    f"Failed to set manual mode for UF robot! Controller Error code: {err_warn[0]} !"
                )
            return

        if self._control_space == "joint":
            code = self.real_arm.set_mode(self.config.joint_command_mode)
            if code != 0:
                raise RuntimeError(f"set_mode({self.config.joint_command_mode}) failed, code={code}")
        elif self._control_space == "cartesian":
            self.real_arm.set_mode(7)
        else:
            raise ValueError(f"Please check the given control space of uf_robot! got {self._control_space}")

        self.real_arm.set_state(0)

        _, err_warn = self.real_arm.get_err_warn_code()
        if err_warn[0] != 0:
            raise RuntimeError(f"Failed to set correct state to UF robot! Controller Error code: {err_warn[0]} !")

        if self._use_rt_report and not self._rt_report_normal:
            self.start()
        time.sleep(0.2)

    def _configure_gripper(self, move_to_open: bool) -> None:
        """Initialize the configured gripper without moving it in manual mode."""
        self.real_arm._arm._baud_checkset = True
        try:
            if self._gripper_type == GripperType.xArmGripper:
                self._check_gripper_code("set_gripper_enable", self.real_arm.set_gripper_enable(True))
                self._check_gripper_code("set_gripper_mode", self.real_arm.set_gripper_mode(0))
                self._check_gripper_code("set_gripper_speed", self.real_arm.set_gripper_speed(self._gripper_param.speed))
                if move_to_open:
                    self._check_gripper_code(
                        "set_gripper_position",
                        self.real_arm.set_gripper_position(
                            self._gripper_param.open_pos,
                            wait=True,
                        ),
                    )
            elif self._gripper_type == GripperType.xArmGripperG2:
                self._check_gripper_code("set_gripper_enable", self.real_arm.set_gripper_enable(True))
                self._check_gripper_code("set_gripper_mode", self.real_arm.set_gripper_mode(0))
                if move_to_open:
                    self._check_gripper_code(
                        "set_gripper_g2_position",
                        self.real_arm.set_gripper_g2_position(
                            self._gripper_param.open_pos,
                            speed=self._gripper_g2_speed,
                            force=self._gripper_param.force,
                            wait=True,
                            check_baud=False,
                        ),
                    )
            elif self._gripper_type == GripperType.BioGripperG2:
                _, mode = self.real_arm.get_bio_gripper_control_mode()
                if mode != 1:
                    self.real_arm.set_bio_gripper_control_mode(1)
                self.real_arm.set_bio_gripper_enable(True)
                if move_to_open:
                    self.real_arm.open_bio_gripper()
            elif self._gripper_type == GripperType.PikaGripper:
                self.pika_gripper.enable()
                if move_to_open:
                    time.sleep(0.5)
                    self.pika_gripper.set_gripper_distance(self._gripper_param.open_pos)
            elif self._gripper_type == GripperType.RobotiqGripper:
                self.real_arm.robotiq_reset()
                self.real_arm.robotiq_set_activate(wait=True)
                if move_to_open:
                    self.real_arm.robotiq_set_position(self._gripper_param.open_pos, wait=True)
        finally:
            self.real_arm._arm._baud_checkset = False

        _, err_warn = self.real_arm.get_err_warn_code()
        if err_warn[0] != 0:
            raise RuntimeError(f"Failed to set correct state to Gripper! Controller Error code: {err_warn[0]} !")

        if move_to_open:
            self._gripper_param.grippos = self._gripper_param.open_pos
            self._gripper_param.gripper_norm = 0.0
            self._last_gripper_command = 0.0

    def calibrate(self) -> None:
        self._is_calibrated = True
        pass # CHECK! currently No-op

    def get_observation(self) -> dict[str, np.ndarray]:
        obs_dict = {}
        self._log_controller_error_if_changed("get_observation")
        logs_enabled = bool(getattr(self, "enable_logs", True))

        # Read robot state
        before_read_t = time.perf_counter() if logs_enabled else None
        if self._control_space == "joint":
            code, states = self.real_arm.get_joint_states(is_radian=True, num=3)
            pos_list = states[0].copy()
            obs_dict = {f"{self.prefix}J{k+1}.pos": pos_list[k] for k in range(self._dof)}
            if self._jnt_obs_has_vel:
                vel_list = states[1].copy()
                obs_dict.update({f"{self.prefix}J{k+1}.vel": vel_list[k] for k in range(self._dof)})
        elif self._control_space == "cartesian":
            if not self._rt_report_normal:
                raise ConnectionError("RT Report for target robot NOT READY! ")

            with self._update_lock:
                pos_list = self.rt_actual_tcp_pose.copy()
                vel_list = self.rt_actual_tcp_speed.copy()
                # pos_cmd_list = self.rt_cmd_tcp_pose.copy()
                # vel_cmd_list = self.rt_cmd_tcp_vel.copy()
                # jpos_fbk_list = self.rt_actual_joint_pos.copy()
                # jvel_fbk_list = self.rt_actual_joint_speed.copy()

            obs_dict = {f"{self.prefix}pose.x": pos_list[0], f"{self.prefix}pose.y": pos_list[1], f"{self.prefix}pose.z": pos_list[2], f"{self.prefix}pose.rx": pos_list[3], f"{self.prefix}pose.ry": pos_list[4], f"{self.prefix}pose.rz": pos_list[5]}
            if self._cart_obs_has_vel:
                obs_dict.update({f"{self.prefix}velo.x": vel_list[0], f"{self.prefix}velo.y": vel_list[1], f"{self.prefix}velo.z": vel_list[2], f"{self.prefix}velo.rx": vel_list[3], f"{self.prefix}velo.ry": vel_list[4], f"{self.prefix}velo.rz": vel_list[5]})
        else:
            ValueError(f"Please check the given control space of uf_robot! got {self._control_space}")
        
        if self._gripper_type > GripperType.NoGripper:
            if self._gripper_type == GripperType.xArmGripper:
                code, grippos = self.real_arm.get_gripper_position()
                if code != 0 or grippos is None:
                    self._log_gripper_error("get_gripper_position", code, f"position={grippos}")
                grippos_norm = self._gripper_param.get_gripper_norm(grippos)
            elif self._gripper_type == GripperType.xArmGripperG2:
                code, grippos = self.real_arm.get_gripper_g2_position()
                if code != 0 or not isinstance(grippos, (int, float, np.number)):
                    self._log_gripper_error("get_gripper_g2_position", code, f"position={grippos}")
                    grippos = None
                grippos_norm = self._gripper_param.get_gripper_norm(grippos)
            elif self._gripper_type == GripperType.BioGripperG2:
                code, grippos = self.real_arm.get_bio_gripper_g2_position()
                grippos_norm = self._gripper_param.get_gripper_norm(grippos)
            elif self._gripper_type == GripperType.PikaGripper:
                grippos = self.pika_gripper.get_gripper_distance()
                grippos_norm = self._gripper_param.get_gripper_norm(grippos)
            elif self._gripper_type == GripperType.RobotiqGripper:
                self.real_arm.robotiq_get_status(number_of_registers=3)
                grippos = self.real_arm.robotiq_status['gPO']  # 0..255
                grippos_norm = self._gripper_param.get_gripper_norm(grippos) # 0=open, 1=closed
            if logs_enabled:
                self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t
            obs_dict[f"{self.prefix}gripper.pos"] = grippos_norm

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            before_camread_t = time.perf_counter() if logs_enabled else None
            frame = cam.async_read()
            shape = frame.shape
            if (self.camera_height > 0 and self.camera_height != shape[0]) or (self.camera_width > 0 and self.camera_width != shape[1]):
                camera_width = self.camera_width if self.camera_width != 0 else shape[1]
                camera_height = self.camera_height if self.camera_height != 0 else shape[0]
                import cv2
                frame = cv2.resize(frame, (camera_height, camera_width), interpolation=cv2.INTER_AREA)
            obs_dict[f"{self.prefix}{cam_key}"] = frame
            if logs_enabled:
                self.logs[f"async_read_camera_{cam_key}_dt_s"] = (
                    time.perf_counter() - before_camread_t
                )

        return obs_dict

    def get_realtime_observation(self) -> dict[str, np.ndarray]:
        """Build a recording observation without controller command-channel reads.

        Joint feedback comes from the asynchronous RT report, while gripper
        feedback uses the latest commanded/cached value. Camera reads remain
        outside the realtime joint control thread.
        """
        if self._control_space != "joint":
            return self.get_observation()
        if not self._rt_report_normal:
            raise ConnectionError("RT Report for target robot NOT READY!")
        logs_enabled = bool(getattr(self, "enable_logs", True))
        with self._update_lock:
            positions = self.rt_actual_joint_pos.copy()
            velocities = self.rt_actual_joint_speed.copy()
            # Timestamp the state snapshot before the slower camera reads.
            self._last_realtime_observation_monotonic_s = time.perf_counter()
            state_rt_receive_s = self._last_rt_report_monotonic_s
            camera_timing = {}
        obs_dict = {
            f"{self.prefix}J{index + 1}.pos": positions[index]
            for index in range(self._dof)
        }
        if self._jnt_obs_has_vel:
            obs_dict.update(
                {
                    f"{self.prefix}J{index + 1}.vel": velocities[index]
                    for index in range(self._dof)
                }
            )
        if self._gripper_type > GripperType.NoGripper:
            gripper = self._last_gripper_command
            if gripper is None:
                gripper = self._gripper_param.gripper_norm
            obs_dict[f"{self.prefix}gripper.pos"] = float(gripper)

        for camera_key, camera in self.cameras.items():
            before_camera_t = time.perf_counter()
            frame = camera.async_read()
            after_camera_t = time.perf_counter()
            shape = frame.shape
            if (
                self.camera_height > 0
                and self.camera_height != shape[0]
                or self.camera_width > 0
                and self.camera_width != shape[1]
            ):
                import cv2

                width = self.camera_width if self.camera_width != 0 else shape[1]
                height = self.camera_height if self.camera_height != 0 else shape[0]
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            obs_dict[f"{self.prefix}{camera_key}"] = frame
            camera_timing[camera_key] = {
                "frame_index": self._realtime_camera_frame_index[camera_key],
                "read_start_s": before_camera_t,
                "read_end_s": after_camera_t,
            }
            self._realtime_camera_frame_index[camera_key] += 1
            if logs_enabled:
                self.logs[f"async_read_camera_{camera_key}_dt_s"] = (
                    after_camera_t - before_camera_t
                )
                if not hasattr(self, "_last_realtime_camera_timings"):
                    self._last_realtime_camera_timings = {}
                self._last_realtime_camera_timings[camera_key] = (
                    before_camera_t,
                    after_camera_t,
                )
        self._last_realtime_observation_end_monotonic_s = time.perf_counter()
        self._last_realtime_sync_timing = {
            "state_sample_s": self._last_realtime_observation_monotonic_s,
            "state_rt_receive_s": state_rt_receive_s,
            "camera": camera_timing,
        }
        return obs_dict

    def _send_gripper_action(self, gripper_norm: float) -> None:
        gripper_norm = min(max(float(gripper_norm), 0.0), 1.0)
        logs_enabled = bool(getattr(self, "enable_logs", True))
        if (
            self._last_gripper_command is not None
            and abs(gripper_norm - self._last_gripper_command)
            < self.config.gripper_command_threshold
        ):
            return

        # The gripper goes through the controller RS485 bridge. Driving that
        # bridge at the 60 Hz joint-command rate can trigger controller error 19.
        # Intermediate targets are coalesced and failed attempts are also
        # rate-limited so an error cannot cause a retry storm.
        now = time.perf_counter()
        if now - self._last_gripper_command_attempt_s < self.config.gripper_command_interval_s:
            return
        self._last_gripper_command_attempt_s = now
        command_start_s = now if logs_enabled else None

        if self._gripper_type == GripperType.xArmGripper:
            grippos = self._gripper_param.get_grippos(gripper_norm)
            # Use the SDK's dedicated gripper command instead of injecting a
            # generic RS485 packet through set_rs485_data. During continuous
            # Online joint motion may not complete the default wait_motion check, so
            # explicitly bypass it while retaining a non-blocking write.
            result = self.real_arm.set_gripper_position(
                grippos,
                wait=False,
                wait_motion=False,
                check_baud=False,
                check_err=False,
            )
        elif self._gripper_type == GripperType.xArmGripperG2:
            grippos = self._gripper_param.get_grippos(gripper_norm)
            result = self.real_arm.set_gripper_g2_position(
                grippos,
                speed=self._gripper_g2_speed,
                force=self._gripper_param.force,
                wait=False,
                wait_motion=False,
                check_baud=False,
                check_err=False,
            )
        elif self._gripper_type == GripperType.BioGripperG2:
            grippos = self._gripper_param.get_grippos(gripper_norm)
            grippos = int(grippos * 3.7342 - 265.13)
            modbus_datas = [0x08, 0x10, 0x0C, 0x00, 0x00, 0x05, 0x0A, 0x00, 0x01]
            modbus_datas.extend(list(struct.pack('>h', self._gripper_param.speed)))
            modbus_datas.extend(list(struct.pack('>h', self._gripper_param.force)))
            modbus_datas.extend(list(struct.pack('>i', grippos)))
            result = self.real_arm.getset_tgpio_modbus_data(modbus_datas)
        elif self._gripper_type == GripperType.PikaGripper:
            grippos = self._gripper_param.get_grippos(gripper_norm)
            self.pika_gripper.set_gripper_distance(grippos)
            result = 0
        elif self._gripper_type == GripperType.RobotiqGripper:
            grippos = self._gripper_param.get_grippos(gripper_norm)
            modbus_datas = [0x09, 0x10, 0x03, 0xE8, 0x00, 0x03, 0x06, 0x09, 0x00, 0x00, grippos, self._gripper_param.speed, self._gripper_param.force]
            result = self.real_arm.getset_tgpio_modbus_data(modbus_datas)

        code = result[0] if isinstance(result, (tuple, list)) else result
        command_dt_ms = (
            (time.perf_counter() - command_start_s) * 1000
            if logs_enabled
            else None
        )
        if code not in (None, 0):
            detail = f"target={gripper_norm:.6f}, position={grippos}"
            if command_dt_ms is not None:
                detail += f", dt_ms={command_dt_ms:.3f}"
            self._log_gripper_error(
                "send_gripper_action",
                code,
                detail,
            )
            return
        if logs_enabled:
            self._log_gripper_command(gripper_norm, grippos, command_dt_ms)
        self._last_gripper_command = gripper_norm

    def _log_gripper_command(self, target: float, position: int, dt_ms: float) -> None:
        log_path = self.config.gripper_error_log_path
        if not log_path:
            return
        try:
            path = Path(log_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
            with path.open("a", encoding="utf-8") as stream:
                stream.write(
                    f"{timestamp} gripper command: target={target:.6f}, "
                    f"position={position}, dt_ms={dt_ms:.3f}, code=0\n"
                )
        except OSError:
            logging.exception("Failed to write gripper command log to %s", log_path)

    def _log_gripper_error(self, operation: str, code, detail: str = "") -> None:
        controller_error = getattr(self.real_arm, "error_code", None)
        message = (
            f"gripper communication error: operation={operation}, code={code}, "
            f"controller_error={controller_error}, {detail}"
        ).rstrip(", ")
        logging.error(message)

        log_path = self.config.gripper_error_log_path
        if not log_path:
            return
        try:
            path = Path(log_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
            with path.open("a", encoding="utf-8") as stream:
                stream.write(f"{timestamp} {message}\n")
        except OSError:
            logging.exception("Failed to write gripper error log to %s", log_path)

    def _check_gripper_code(self, operation: str, code) -> None:
        if code in (None, 0):
            return
        self._log_gripper_error(operation, code)
        raise RuntimeError(f"{operation} failed, code={code}, {self._motion_status()}")

    def _log_controller_error_if_changed(self, operation: str) -> None:
        controller_error = getattr(self.real_arm, "error_code", 0)
        if controller_error and controller_error != self._last_logged_controller_error:
            self._log_gripper_error(operation, "controller", "controller error became active")
        self._last_logged_controller_error = controller_error

    def _motion_status(self) -> str:
        """Return controller state details for a failed motion command."""
        arm = self.real_arm
        mode = getattr(arm, "mode", "unknown")
        state = getattr(arm, "state", "unknown")
        error_code = getattr(arm, "error_code", "unknown")
        warn_code = getattr(arm, "warn_code", "unknown")
        return f"mode={mode}, state={state}, error_code={error_code}, warn_code={warn_code}"

    def _check_motion_code(self, command: str, code: int | None) -> None:
        """Fail loudly when the SDK explicitly rejects a motion command."""
        if code is not None and code != 0:
            raise RuntimeError(f"{command} failed, code={code}, {self._motion_status()}")

    def send_action(self, action: dict) -> np.ndarray:
        if not self._is_connected:
            raise ConnectionError()
        self._log_controller_error_if_changed("send_action")
        if self.config.manual_mode:
            gripper_key = f"{self.prefix}gripper.pos"
            if (
                self._gripper_type > GripperType.NoGripper
                and gripper_key in action
                and self.real_arm.error_code == 0
                and not self.config.no_action
            ):
                self._send_gripper_action(action[gripper_key])
            return action
        if self.real_arm.error_code != 0:
            return action
        if self.config.no_action:
            return action

        logs_enabled = bool(getattr(self, "enable_logs", True))
        before_write_t = time.perf_counter() if logs_enabled else None
        safe_action = dict(action)
        if self._control_space == "joint":
            # first sync with gello or other control device SLOWLY!
            jnt_spd = INIT_SYNC_JOINT_VELOCITY_RAD if self._cmd_cnt < 20 else self._max_joint_velocity
            wait_ = True if self._cmd_cnt == 0 else False

            cmd_list = [0]*(self._dof)
            for i in range(self._dof):
                cmd_list[i] = action[f"{self.prefix}J{i+1}.pos"]
            guard_start_t = time.perf_counter() if logs_enabled else None
            safe_cmd = self._guard_joint_target(cmd_list)
            if logs_enabled:
                self.logs["safety_guard_dt_s"] = time.perf_counter() - guard_start_t
                self.logs["safety_guard_path"] = self._last_guard_path
            if safe_cmd is None:
                # Do not send an unverified arm target. Gripper handling below
                # remains independent and can continue safely.
                safe_cmd = None
            else:
                for i in range(self._dof):
                    safe_action[f"{self.prefix}J{i+1}.pos"] = float(safe_cmd[i])

            if safe_cmd is not None:
                # All joint-space control uses xArm mode 6 online trajectory
                # planning. Targets are absolute and sent without waiting.
                if wait_ == False and self.real_arm.mode != 6:
                    code = self.real_arm.set_mode(6)
                    self._check_motion_code("set_mode(6)", code)
                    code = self.real_arm.set_state(0)
                    self._check_motion_code("set_state(0)", code)
                    time.sleep(0.1)
                elif wait_ and self.real_arm.mode != 0:
                    code = self.real_arm.set_mode(0)
                    self._check_motion_code("set_mode(0)", code)
                    code = self.real_arm.set_state(0)
                    self._check_motion_code("set_state(0)", code)
                    time.sleep(0.1)

                code = self.real_arm.set_servo_angle(
                    angle=safe_cmd[:self._dof].tolist(),
                    speed=jnt_spd,
                    is_radian=True,
                    wait=wait_,
                )
                self._check_motion_code("set_servo_angle", code)
        elif self._control_space == "cartesian": # unit: mm?
            lin_spd = self._max_linear_velocity

            if not self._rt_report_normal:
                raise ConnectionError("RT Report for target robot NOT READY! ")
            cmd_list = [action[f"{self.prefix}pose.x"], action[f"{self.prefix}pose.y"], action[f"{self.prefix}pose.z"], action[f"{self.prefix}pose.rx"], action[f"{self.prefix}pose.ry"], action[f"{self.prefix}pose.rz"]]
            self.real_arm.set_position_aa(axis_angle_pose=cmd_list, speed=lin_spd, is_radian=True, wait=False)
            # self.real_arm.set_position(*cmd_list, radius=0, speed=lin_spd, is_radian=True, wait=False)

        if self._cmd_cnt < 99999:
            self._cmd_cnt += 1 # CHECK!! possibility of overflow?
        if self._gripper_type > GripperType.NoGripper:
            self._send_gripper_action(safe_action[f"{self.prefix}gripper.pos"])

        if logs_enabled:
            self.logs["write_pos_dt_s"] = time.perf_counter() - before_write_t
        return safe_action

    def print_logs(self) -> None:
        pass

    def disconnect(self) -> None:
        if not self._is_connected and self.real_arm is None:
            return
        if self._use_rt_report:
            self.report_stop_event.set()
            if self.is_alive():
                self.join()
        if self.real_arm is not None and getattr(self.real_arm, "connected", False):
            self.real_arm.set_state(4) # stop
            self.real_arm.set_mode(0)
            self.real_arm.disconnect()
        # CHECK!! how about gripper? 

        for cam in self.cameras.values():
            if getattr(cam, "is_connected", False):
                cam.disconnect()

        self._is_connected = False

    @property
    def is_calibrated(self) -> bool:
        """Whether the robot is currently calibrated or not. Should be always `True` if not applicable"""
        return self._is_calibrated

    @property
    def is_connected(self) -> bool:
        """Whether the robot is currently calibrated or not. Should be always `True` if not applicable"""
        return self._is_connected

    def run(self):
        import socket
        
        robot_port = 30000 # DO NOT CHANGE
        # create socket connection
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setblocking(True)
        sock.settimeout(1)
        sock.connect((self.config.robot_ip, robot_port))

        buffer = sock.recv(4)
        print(buffer)
        while len(buffer) < 4:
            buffer += sock.recv(4 - len(buffer))
        size = convert.bytes_to_u32(buffer[:4])
        print(f"UFACTORY Robot ({self.config.robot_ip}) RT Report Thread starts!! =======")
        while not self.report_stop_event.is_set():
            buffer += sock.recv(size - len(buffer))
            if len(buffer) < size:
                continue
            data = buffer[:size]
            buffer = buffer[size:]
            with self._update_lock:
                self.rt_actual_joint_pos = convert.bytes_to_fp32s(data[116:144], 7)
                self.rt_actual_joint_speed = convert.bytes_to_fp32s(data[144:172], 7)
                self.rt_cmd_tcp_pose = convert.bytes_to_fp32s(data[424:448], 6)
                self.rt_cmd_tcp_vel = convert.bytes_to_fp32s(data[448:472], 6)
                self.rt_actual_tcp_pose = convert.bytes_to_fp32s(data[472:496], 6)
                self.rt_actual_tcp_speed = convert.bytes_to_fp32s(data[496:520], 6)
                # This is the host arrival/decode time, deliberately kept in
                # the same monotonic clock domain as the recorder.
                self._last_rt_report_monotonic_s = time.perf_counter()
            self._rt_report_normal = True

        self._rt_report_normal = False
        print(f"UFACTORY Robot ({self.config.robot_ip}) RT Report Thread Exit!! =======")
