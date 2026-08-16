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


logger = logging.getLogger(__name__)

## Configurations:
INIT_SYNC_JOINT_VELOCITY_RAD = 0.2
ROBOT_RESET_SPEED_DEG = 60
TCP_Z_CLAMP_TOLERANCE_MM = 1e-3
TCP_Z_LOG_INTERVAL_S = 1.0
TCP_Z_MAX_IK_JOINT_STEP_RAD = math.radians(10.0)

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
        
        self._control_space = self.config.control_space

        self.real_arm = None
        self._initial_point = None
        cameras_args = self.config.cameras_args or {}
        self.camera_width = cameras_args.get('w', 0)
        self.camera_height = cameras_args.get('h', 0)
        self.cameras = make_cameras_from_configs(config.cameras)

        self._is_connected = False
        self._is_calibrated =True

        self.logs = {}

        self._cmd_cnt = 0
        self._last_gripper_command = None
        self._last_logged_controller_error = 0

        self._max_joint_velocity = math.radians(self.config.max_joint_velocity)
        self._max_linear_velocity = self.config.max_linear_velocity

        self._min_tcp_z_mm = self.config.min_tcp_z_mm
        self._tcp_z_guard_activation_margin_mm = self.config.tcp_z_guard_activation_margin_mm
        self._last_safe_joint_target = None
        self._last_safe_cartesian_target = None
        self._tcp_z_is_clamped = False
        self._tcp_z_last_log_time = 0.0
        self._tcp_z_last_error_log_time = 0.0

        self.report_stop_event = Event()
        self._rt_report_normal = False
        self._update_lock = Lock()
        # The TCP z guard uses the asynchronous RT report to avoid a blocking
        # FK request on every GELLO servo cycle.
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
        if self._min_tcp_z_mm is None or self.real_arm is None:
            return

        if self._control_space == "joint":
            code, states = self.real_arm.get_joint_states(is_radian=True, num=1)
            if code != 0 or not states or len(states[0]) < self._dof:
                raise RuntimeError(f"Unable to initialize TCP z guard from joint state, code={code}")
            target = np.asarray(states[0][:self._dof], dtype=np.float64)
            if not np.all(np.isfinite(target)):
                raise RuntimeError("Unable to initialize TCP z guard from non-finite joint state")
            self._last_safe_joint_target = target
        else:
            code, pose = self.real_arm.get_position_aa(is_radian=True)
            if code != 0 or len(pose) < 6:
                raise RuntimeError(f"Unable to initialize TCP z guard from TCP pose, code={code}")
            target = np.asarray(pose[:6], dtype=np.float64)
            if not np.all(np.isfinite(target)):
                raise RuntimeError("Unable to initialize TCP z guard from non-finite TCP pose")
            self._last_safe_cartesian_target = target

        self._tcp_z_is_clamped = False
        self._tcp_z_last_log_time = 0.0
        self._tcp_z_last_error_log_time = 0.0

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
        """Return a safe joint target, or None when motion must be skipped."""
        desired = np.asarray(command, dtype=np.float64)
        if self._min_tcp_z_mm is None:
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

    def _guard_cartesian_target(self, command: list[float]) -> np.ndarray | None:
        """Clamp a Cartesian target without changing its other five components."""
        target = np.asarray(command, dtype=np.float64)
        if self._min_tcp_z_mm is None:
            return target

        fallback = self._last_safe_cartesian_target
        try:
            if target.shape != (6,) or not np.all(np.isfinite(target)):
                raise ValueError("Cartesian target has invalid shape or contains NaN/Inf")
            requested_z = float(target[2])
            if requested_z < self._min_tcp_z_mm:
                target[2] = self._min_tcp_z_mm
                self._log_tcp_z_clamp(True, requested_z)
            else:
                self._log_tcp_z_clamp(False)
            self._last_safe_cartesian_target = target.copy()
            return target
        except Exception as exc:
            self._log_tcp_z_guard_error(str(exc))
            if fallback is None:
                return None
            return np.asarray(fallback, dtype=np.float64).copy()

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
                        self.real_arm.set_gripper_position(self._gripper_param.open_pos),
                    )
            elif self._gripper_type == GripperType.xArmGripperG2:
                self.real_arm.set_gripper_enable(True)
                self.real_arm.set_gripper_mode(0)
                if move_to_open:
                    self.real_arm.set_gripper_g2_position(self._gripper_param.open_pos)
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

    def calibrate(self) -> None:
        self._is_calibrated = True
        pass # CHECK! currently No-op

    def get_observation(self) -> dict[str, np.ndarray]:
        obs_dict = {}
        self._log_controller_error_if_changed("get_observation")

        # Read Stretch state
        before_read_t = time.perf_counter()
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
            self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t
            obs_dict[f"{self.prefix}gripper.pos"] = grippos_norm

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            before_camread_t = time.perf_counter()
            frame = cam.async_read()
            shape = frame.shape
            if (self.camera_height > 0 and self.camera_height != shape[0]) or (self.camera_width > 0 and self.camera_width != shape[1]):
                camera_width = self.camera_width if self.camera_width != 0 else shape[1]
                camera_height = self.camera_height if self.camera_height != 0 else shape[0]
                import cv2
                frame = cv2.resize(frame, (camera_height, camera_width), interpolation=cv2.INTER_AREA)
            obs_dict[f"{self.prefix}{cam_key}"] = frame
            self.logs[f"async_read_camera_{cam_key}_dt_s"] = time.perf_counter() - before_camread_t

        return obs_dict

    def _send_gripper_action(self, gripper_norm: float) -> None:
        gripper_norm = min(max(float(gripper_norm), 0.0), 1.0)
        if (
            self._last_gripper_command is not None
            and abs(gripper_norm - self._last_gripper_command)
            < self.config.gripper_command_threshold
        ):
            return

        if self._gripper_type == GripperType.xArmGripper:
            grippos = self._gripper_param.get_grippos(gripper_norm)
            modbus_datas = [0x08, 0x10, 0x07, 0x00, 0x00, 0x02, 0x04]
            modbus_datas.extend(list(struct.pack('>i', grippos)))
            result = self.real_arm.getset_tgpio_modbus_data(modbus_datas)
        elif self._gripper_type == GripperType.xArmGripperG2:
            grippos = self._gripper_param.get_grippos(gripper_norm)
            grippos = int((math.degrees(math.asin((grippos - 16) / 110)) + 8.33) * 18.28)
            modbus_datas = [0x08, 0x10, 0x0C, 0x00, 0x00, 0x05, 0x0A, 0x00, 0x01]
            modbus_datas.extend(list(struct.pack('>h', self._gripper_param.speed)))
            modbus_datas.extend(list(struct.pack('>h', self._gripper_param.force)))
            modbus_datas.extend(list(struct.pack('>i', grippos)))
            result = self.real_arm.getset_tgpio_modbus_data(modbus_datas)
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
        if code not in (None, 0):
            self._log_gripper_error("send_gripper_action", code, f"target={gripper_norm:.6f}")
            return
        self._last_gripper_command = gripper_norm

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

        before_write_t = time.perf_counter()
        safe_action = dict(action)
        if self._control_space == "joint":
            # first sync with gello or other control device SLOWLY!
            jnt_spd = INIT_SYNC_JOINT_VELOCITY_RAD if self._cmd_cnt < 20 else self._max_joint_velocity
            wait_ = True if self._cmd_cnt == 0 else False

            cmd_list = [0]*(self._dof)
            for i in range(self._dof):
                cmd_list[i] = action[f"{self.prefix}J{i+1}.pos"]
            safe_cmd = self._guard_joint_target(cmd_list)
            if safe_cmd is None:
                # Do not send an unverified arm target. Gripper handling below
                # remains independent and can continue safely.
                safe_cmd = None
            else:
                for i in range(self._dof):
                    safe_action[f"{self.prefix}J{i+1}.pos"] = float(safe_cmd[i])

            if safe_cmd is not None and self.config.joint_command_mode == 1:
                # set_servo_angle_j is an absolute target command. It is the
                # SDK's high-frequency interface and executes only the latest
                # target, so it must be used with servo motion mode (1).
                if self.real_arm.mode != 1:
                    code = self.real_arm.set_mode(1)
                    self._check_motion_code("set_mode(1)", code)
                    code = self.real_arm.set_state(0)
                    self._check_motion_code("set_state(0)", code)
                    time.sleep(0.1)
                code = self.real_arm.set_servo_angle_j(
                    safe_cmd[:self._dof].tolist(), speed=jnt_spd, is_radian=True
                )
                self._check_motion_code("set_servo_angle_j", code)
            elif safe_cmd is not None:
                # The legacy mode-6 path uses the absolute move_joint API.
                # The first blocking command must be sent in position mode.
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
            safe_cmd = self._guard_cartesian_target(cmd_list)
            if safe_cmd is not None:
                safe_cmd = safe_cmd.tolist()
                for i, key in enumerate(("x", "y", "z", "rx", "ry", "rz")):
                    safe_action[f"{self.prefix}pose.{key}"] = float(safe_cmd[i])
                self.real_arm.set_position_aa(axis_angle_pose=safe_cmd, speed=lin_spd, is_radian=True, wait=False)
            # self.real_arm.set_position(*cmd_list, radius=0, speed=lin_spd, is_radian=True, wait=False)

        if self._cmd_cnt < 99999:
            self._cmd_cnt += 1 # CHECK!! possibility of overflow?
        if self._gripper_type > GripperType.NoGripper:
            self._send_gripper_action(safe_action[f"{self.prefix}gripper.pos"])

        self.logs["write_pos_dt_s"] = time.perf_counter() - before_write_t
        return safe_action

    def print_logs(self) -> None:
        pass

    def disconnect(self) -> None:
        self.real_arm.set_state(4) # stop
        self.real_arm.set_mode(0)
        if self._use_rt_report:
            self.report_stop_event.set()
            self.join()
        self.real_arm.disconnect()
        # CHECK!! how about gripper? 

        for cam in self.cameras.values():
            cam.disconnect()

        self._is_connected = False

    def is_calibrated(self) -> bool:
        """Whether the robot is currently calibrated or not. Should be always `True` if not applicable"""
        return self._is_calibrated

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
            self._rt_report_normal = True

        self._rt_report_normal = False
        print(f"UFACTORY Robot ({self.config.robot_ip}) RT Report Thread Exit!! =======")
