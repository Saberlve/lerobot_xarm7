#!/usr/bin/env python
import logging
import math
import threading
import time
import numpy as np
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from ..base_teleop import UFBaseTeleop
from .gello_adapter import GripperDynamixelInfo, PatchedDynamixelRobotConfig
from .gello_teleop_config import GelloTeleopConfig


logger = logging.getLogger(__name__)
GRIPPER_CURRENT_FEEDBACK_KEY = "gripper.current_ma"
FEEDBACK_COMMAND_WATCHDOG_S = 0.1

class GelloTeleop(UFBaseTeleop):
    """
    GELLO for xArm tele-op, ref: https://wuphilipp.github.io/gello_site/
    """

    config_class = GelloTeleopConfig
    name = "Gello Teleop For xArm"

    def __init__(self, config: GelloTeleopConfig):
        super().__init__(config)
        self.config = config
        self._is_connected = False
        self._teleop_enabled = False
        self._needs_alignment = True
        self._is_calibrated = True # CHECK!!
        self._keyboard_gripper_state = {"close": False, "open": False}
        self._keyboard_gripper_target = None
        self._keyboard_gripper_speed = 1.0
        self._keyboard_gripper_stroke_mm = None
        self._keyboard_gripper_last_update = None
        self._keyboard_gripper_lock = threading.Lock()
        self._keyboard_press_time = {"close": None, "open": None}
        self._keyboard_step_pending = {"close": False, "open": False}
        # The realtime controller only publishes the latest current target to
        # this in-memory slot. Dynamixel I/O stays on the worker thread.
        self._feedback_lock = threading.Lock()
        self._feedback_event = threading.Event()
        self._feedback_stop = threading.Event()
        self._feedback_thread = None
        self._feedback_pending_ma = 0.0
        self._feedback_output_active = False
        self._feedback_output_error = None

        joint_offsets = [0.0] * len(self.config.joint_ids)
        self._align_gripper_to_current = self.config.gripper_open_deg is None
        if self.config.gripper_id >= 0:
            if self.config.gripper_open_deg is not None:
                gripper_open_deg = self.config.gripper_open_deg
                gripper_close_deg = self.config.gripper_close_deg
            else:
                # Only the range matters. It is shifted to the current GELLO
                # gripper position whenever teleoperation is enabled.
                gripper_open_deg = 0.0
                gripper_close_deg = -42.0
            gripper_config = [
                self.config.gripper_id,
                gripper_open_deg,
                gripper_close_deg,
            ]
        else:
            gripper_config = None

        param_dict = {
                "joint_ids": self.config.joint_ids,
                "joint_signs": self.config.joint_signs,
                "joint_offsets": joint_offsets,
                "gripper_config": gripper_config
        }
        self._dynamixel_robo_config = PatchedDynamixelRobotConfig(**param_dict)
        print(self._dynamixel_robo_config)
        self.dof = len(self.config.joint_ids)

    @property
    def action_features(self) -> dict:
        # Add one more dof for gripper
        # act_ft = {
        #     "joint_position": {
        #     "dtype": "float",
        #     "shape": (self.dof+1,)
        #     }
        # }
        act_ft = { f"J{i+1}.pos": float for i in range(self.dof) } | {"gripper.pos": float}
        return act_ft

    @property
    def feedback_features(self) -> dict:
        # fbk_ft = {
        #     "joint_position": {
        #     "dtype": "float",
        #     "shape": (self.dof+1,)
        #     }
        # }
        fbk_ft = { f"J{i+1}.pos": float for i in range(self.dof) } | {"gripper.pos": float}
        return fbk_ft

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self._is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        from gello.agents.gello_agent import GelloAgent

        try:
            self.gello_agent = GelloAgent(port=self.config.port, dynamixel_config=self._dynamixel_robo_config)
            self.gello_agent._robot.set_torque_mode(False)
            if not self._is_calibrated and calibrate:
                logger.info(
                    "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
                )
                self.calibrate()

            self.configure()
            self._is_connected = True
            super().connect(calibrate)
        except BaseException:
            self._is_connected = False
            self._close_gello_driver()
            raise
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        # TODO: Go to sync position slowly? Can not 
        pass

    def probe_gripper_dynamixel(self) -> GripperDynamixelInfo:
        if not self._is_connected:
            raise DeviceNotConnectedError("Gello teleop is not connected")
        return self.gello_agent._robot.probe_gripper_dynamixel()

    def enable_gripper_current_mode(self) -> GripperDynamixelInfo:
        """Enable manually commanded Current Control Mode on GELLO ID8 only."""
        if not self._is_connected:
            raise DeviceNotConnectedError("Gello teleop is not connected")
        if not self.config.gripper_current_control_enabled:
            raise RuntimeError("gripper current control is disabled by configuration")
        current_limit_ma = self.config.gripper_current_limit_ma
        if current_limit_ma is None:
            raise RuntimeError("gripper_current_limit_ma is not configured")
        return self.gello_agent._robot.enable_gripper_current_mode(current_limit_ma)

    def write_gripper_current_ma(self, current_ma: float) -> float:
        """Write a signed, safety-clamped manual current command to ID8."""
        if not self._is_connected:
            raise DeviceNotConnectedError("Gello teleop is not connected")
        if not self.config.gripper_current_control_enabled:
            raise RuntimeError("gripper current control is disabled by configuration")
        return self.gello_agent._robot.write_gripper_current_ma(current_ma)

    def zero_gripper_current(self) -> None:
        if not self._is_connected:
            raise DeviceNotConnectedError("Gello teleop is not connected")
        self.gello_agent._robot.zero_gripper_current()

    def disable_gripper_current_mode(self) -> None:
        if not self._is_connected:
            raise DeviceNotConnectedError("Gello teleop is not connected")
        self._stop_feedback_worker()
        self.gello_agent._robot.disable_gripper_current_mode()

    def start_feedback(self) -> GripperDynamixelInfo | None:
        """Enable ID8 current mode and start the non-blocking output worker."""
        if not self.config.gripper_force_feedback_enabled:
            return None
        if not self._is_connected:
            raise DeviceNotConnectedError("Gello teleop is not connected")
        with self._feedback_lock:
            if self._feedback_output_active:
                return None

        info = self.enable_gripper_current_mode()
        worker = threading.Thread(
            target=self._feedback_output_loop,
            name="gello-id8-current-feedback",
            daemon=True,
        )
        try:
            with self._feedback_lock:
                self._feedback_pending_ma = 0.0
                self._feedback_output_error = None
                self._feedback_stop.clear()
                self._feedback_event.clear()
                self._feedback_thread = worker
                self._feedback_output_active = True
            worker.start()
        except BaseException:
            with self._feedback_lock:
                self._feedback_output_active = False
                self._feedback_thread = None
            try:
                self.gello_agent._robot.zero_gripper_current()
            finally:
                self.gello_agent._robot.disable_gripper_current_mode()
            raise
        return info

    def _feedback_output_loop(self) -> None:
        last_written_ma = 0.0
        while True:
            self._feedback_event.wait(timeout=FEEDBACK_COMMAND_WATCHDOG_S)
            with self._feedback_lock:
                command_ready = self._feedback_event.is_set()
                self._feedback_event.clear()
                if self._feedback_stop.is_set() or not self._feedback_output_active:
                    return
                current_ma = self._feedback_pending_ma if command_ready else 0.0
            if not command_ready and last_written_ma == 0.0:
                continue
            try:
                last_written_ma = self.gello_agent._robot.write_gripper_current_ma(
                    current_ma
                )
            except Exception as exc:
                with self._feedback_lock:
                    self._feedback_output_active = False
                    self._feedback_output_error = exc
                logger.exception(
                    "GELLO ID8 feedback write failed; zeroing and disabling current mode"
                )
                try:
                    self.gello_agent._robot.zero_gripper_current()
                except Exception:
                    logger.exception("Failed to write zero after GELLO feedback failure")
                try:
                    self.gello_agent._robot.disable_gripper_current_mode()
                except Exception:
                    logger.exception("Failed to disable ID8 after GELLO feedback failure")
                return

    def _stop_feedback_worker(self) -> None:
        with self._feedback_lock:
            self._feedback_output_active = False
            self._feedback_pending_ma = 0.0
            self._feedback_stop.set()
            self._feedback_event.set()
            worker = self._feedback_thread
            self._feedback_thread = None
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=2.0)
            if worker.is_alive():
                logger.error("GELLO ID8 feedback worker did not stop within 2 seconds")

    def stop_feedback(self) -> None:
        """Stop accepting targets, then zero and disable ID8 current mode."""
        self._stop_feedback_worker()
        if not hasattr(self, "gello_agent"):
            return
        failures = []
        try:
            self.gello_agent._robot.zero_gripper_current()
        except Exception as exc:
            failures.append(f"zero current failed: {exc}")
        try:
            self.gello_agent._robot.disable_gripper_current_mode()
        except Exception as exc:
            failures.append(f"disable current mode failed: {exc}")
        if failures:
            raise RuntimeError("; ".join(failures))

    def get_feedback_output_status(self) -> tuple[bool, str | None]:
        """Return the worker state without performing Dynamixel I/O."""
        with self._feedback_lock:
            error = self._feedback_output_error
            return self._feedback_output_active, None if error is None else str(error)

    def _safely_disable_gripper_current_mode(self, context: str) -> None:
        if not hasattr(self, "gello_agent"):
            return
        gello_robot = self.gello_agent._robot
        if not hasattr(gello_robot, "disable_gripper_current_mode"):
            return
        try:
            self.stop_feedback()
        except Exception:
            # The driver has already attempted Goal Current=0 and torque disable.
            logger.exception(
                "Failed to cleanly disable GELLO ID8 current mode during %s", context
            )
            try:
                gello_robot.disable_gripper_current_mode()
            except Exception:
                logger.exception(
                    "Fallback GELLO ID8 disable also failed during %s", context
                )

    def reset_to_robot_observation(self, obs):
        """Map the current passive GELLO pose to the robot's current pose."""
        if not self._is_connected:
            raise DeviceNotConnectedError("Gello teleop is not connected")

        self._teleop_enabled = False
        self._safely_disable_gripper_current_mode("reset")
        gello_robot = self.gello_agent._robot
        driver = gello_robot._driver
        gello_robot.set_torque_mode(False)
        current_raw = np.asarray(driver.get_joints(), dtype=float)
        signs = np.asarray(gello_robot._joint_signs, dtype=float)

        robot_joints = np.asarray(
            [obs[f"J{i + 1}.pos"] for i in range(self.dof)], dtype=float
        )
        gello_robot._joint_offsets[: self.dof] = (
            current_raw[: self.dof] - robot_joints * signs[: self.dof]
        )

        if (
            self._align_gripper_to_current
            and gello_robot.gripper_open_close is not None
            and len(current_raw) > self.dof
        ):
            gripper_pos = float(obs.get("gripper.pos", 0.0))
            gripper_open, gripper_close = gello_robot.gripper_open_close
            gripper_pos = min(max(gripper_pos, 0.0), 1.0)
            gripper_span = gripper_close - gripper_open
            gripper_open = current_raw[self.dof] - gripper_pos * gripper_span
            gello_robot.gripper_open_close = (
                gripper_open,
                gripper_open + gripper_span,
            )

        gello_robot._last_pos = None
        if self.config.gripper_control_mode == "keyboard":
            with self._keyboard_gripper_lock:
                self._keyboard_gripper_target = float(obs.get("gripper.pos", 0.0))
                self._keyboard_gripper_last_update = time.monotonic()
                self._keyboard_press_time = {"close": None, "open": None}
                self._keyboard_step_pending = {"close": False, "open": False}
        self._needs_alignment = False
        logger.info("Current GELLO pose aligned to current robot observation")

    def set_teleop_enabled(self, enabled: bool, obs=None):
        if enabled and not self._is_connected:
            raise DeviceNotConnectedError("Gello teleop is not connected")
        if enabled and self._needs_alignment:
            if obs is None:
                raise ValueError("Robot observation is required to enable GELLO teleoperation")
            self.reset_to_robot_observation(obs)
        if not enabled and self._is_connected and hasattr(self, "gello_agent"):
            self._safely_disable_gripper_current_mode("pause")
            self.gello_agent._robot.set_torque_mode(False)
            self._needs_alignment = True
        self._teleop_enabled = enabled
        logger.info("Gello teleoperation %s", "enabled" if enabled else "disabled")

    def set_gripper_keyboard_state(self, *, close: bool, open: bool) -> None:
        if self.config.gripper_control_mode != "keyboard":
            return
        now = time.monotonic()
        with self._keyboard_gripper_lock:
            for name, pressed in (("close", close), ("open", open)):
                pressed = bool(pressed)
                was_pressed = self._keyboard_gripper_state[name]
                if pressed and not was_pressed:
                    # Press edge: queue one fixed step and start the hold timer.
                    self._keyboard_press_time[name] = now
                    self._keyboard_step_pending[name] = True
                elif not pressed and was_pressed:
                    # Release edge: stop the hold timer, but keep any queued
                    # step so a tap shorter than one control cycle still counts.
                    self._keyboard_press_time[name] = None
                self._keyboard_gripper_state[name] = pressed

    def set_gripper_motion_parameters(self, speed_mm_s: float, stroke_mm: float) -> None:
        if stroke_mm <= 0:
            raise ValueError("gripper stroke must be positive")
        self._keyboard_gripper_speed = max(float(speed_mm_s), 0.0) / float(stroke_mm)
        self._keyboard_gripper_stroke_mm = float(stroke_mm)

    def _keyboard_gripper_action(self, fallback: float) -> float:
        now = time.monotonic()
        with self._keyboard_gripper_lock:
            if self._keyboard_gripper_target is None:
                self._keyboard_gripper_target = min(max(float(fallback), 0.0), 1.0)
            last = self._keyboard_gripper_last_update
            self._keyboard_gripper_last_update = now
            # Apply queued tap steps even if the key was already released.
            stroke = self._keyboard_gripper_stroke_mm
            for name, direction in (("close", 1.0), ("open", -1.0)):
                if self._keyboard_step_pending[name]:
                    self._keyboard_step_pending[name] = False
                    if stroke:
                        step = self.config.gripper_keyboard_step_mm / stroke
                        self._keyboard_gripper_target = min(
                            max(self._keyboard_gripper_target + direction * step, 0.0),
                            1.0,
                        )
            if last is not None:
                close = self._keyboard_gripper_state["close"]
                open_ = self._keyboard_gripper_state["open"]
                if close != open_:
                    name = "close" if close else "open"
                    direction = 1.0 if close else -1.0
                    press_time = self._keyboard_press_time[name]
                    if (
                        press_time is not None
                        and now - press_time >= self.config.gripper_keyboard_hold_delay_s
                    ):
                        # Held past the delay: continuous motion at gripper speed.
                        self._keyboard_gripper_target = min(
                            max(self._keyboard_gripper_target + direction * self._keyboard_gripper_speed * (now - last), 0.0),
                            1.0,
                        )
            return self._keyboard_gripper_target

    def get_action(self) -> dict[str, np.ndarray]:
        if not self._teleop_enabled:
            raise RuntimeError("Gello teleop is disabled")
        fake_obs = dict({"joint_state": np.array([0.0]*(self.dof+1))}) # for agent.act() argument, actually no use
        action_array = self.gello_agent.act(fake_obs) # current gello joint pos as np.ndarray

        action = {}
        for i in range(self.dof):
            action.update({f"J{i+1}.pos": action_array[i]})
        gripper_pos = action_array[self.dof]
        if self.config.gripper_control_mode == "keyboard":
            gripper_pos = self._keyboard_gripper_action(gripper_pos)
        action.update({"gripper.pos": gripper_pos})
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        """Publish the current frame's ID8 target without performing serial I/O."""
        if not self.config.gripper_force_feedback_enabled:
            return
        current_ma = feedback.get(GRIPPER_CURRENT_FEEDBACK_KEY, 0.0)
        if isinstance(current_ma, bool):
            current_ma = 0.0
        try:
            current_ma = float(current_ma)
        except (TypeError, ValueError):
            current_ma = 0.0
        if not math.isfinite(current_ma):
            current_ma = 0.0

        limit_ma = self.config.gripper_current_limit_ma
        if limit_ma is None:
            current_ma = 0.0
        else:
            current_ma = max(-limit_ma, min(limit_ma, current_ma))
        with self._feedback_lock:
            if not self._feedback_output_active:
                return
            self._feedback_pending_ma = current_ma
            self._feedback_event.set()

    def _close_gello_driver(self) -> None:
        if not hasattr(self, "gello_agent"):
            return
        gello_robot = self.gello_agent._robot
        try:
            self._safely_disable_gripper_current_mode("disconnect")
            gello_robot.set_torque_mode(False)
        finally:
            gello_robot._driver.close()

    def disconnect(self) -> None:
        try:
            self._close_gello_driver()
        finally:
            self._is_connected = False
            self._teleop_enabled = False
            self._needs_alignment = True
        logger.info(f"{self} disconnected.")
