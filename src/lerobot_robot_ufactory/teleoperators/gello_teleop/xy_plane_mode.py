"""Opt-in base-frame XY translation, evaluated only on the control owner."""

import logging
import threading
import time

import numpy as np

from ...robots.uf_robot.local_kinematics import xarm_rpy_transform

logger = logging.getLogger(__name__)


class XYPlaneMode:
    def __init__(self):
        self.lock = threading.Lock()
        self.pressed = False
        self.reset()

    def reset(self):
        with self.lock:
            self.pending = False
            self.active = False
            self.offset = np.zeros(7)
            self.realign = None
            self.anchor = None
            self.source_xy = None
            self.warning = None
            self._controller_fk_validation = False

    def key(self, pressed, enabled):
        with self.lock:
            if pressed and not self.pressed and enabled:
                self.pending = not self.pending
            self.pressed = pressed

    def prepare(self, robot, observation):
        with self.lock:
            if not self.pending:
                return
            if getattr(robot, '_control_space', None) != 'joint':
                raise ValueError('XY-plane mode requires joint-space robot control')
            self.kinematics = getattr(robot, '_local_kinematics', None)
            if self.kinematics is None:
                raise ValueError('XY-plane mode requires local kinematics (record_space: both or tcp)')
            self.arm = robot.real_arm
            latest = getattr(robot, 'latest_state_before', None)
            if callable(latest):
                joints = np.asarray(latest(time.perf_counter()).joint_positions, dtype=float)
            else:
                prefix = getattr(robot, 'prefix', '')
                joints = np.asarray([observation[f'{prefix}J{i}.pos'] for i in range(1, 8)])
            if joints.shape != (7,) or not np.isfinite(joints).all():
                raise ValueError('XY-plane mode requires seven finite robot joint positions')
            if self.active:
                self.realign = joints.copy()
                self.active = False
            else:
                code, pose = self.arm.get_forward_kinematics(
                    joints.tolist(), input_is_radian=True, return_is_radian=True,
                )
                if code != 0:
                    raise RuntimeError(f'XY-plane mode FK failed: {code}')
                self.anchor = xarm_rpy_transform(pose)
                self.pose = np.asarray(pose, dtype=float)
                self.last_joints = joints.copy()
                self.source_xy = None
                self.command_xy = self.pose[:2].copy()
                self.active = True
            self.warning = None
            self._controller_fk_validation = False
            self.pending = False
            logger.info('GELLO mode: %s', 'base XY translation only' if self.active else 'all joints')

    def apply(self, action):
        with self.lock:
            result = np.asarray(action, dtype=float).copy()
            if self.realign is not None:
                self.offset = self.realign - result[:7]
                self.realign = None
            mapped = result[:7] + self.offset
            if not self.active:
                result[:7] = mapped
                return result
            xy = self.kinematics.forward_matrix(mapped)[:2, 3].copy()
            if self.source_xy is None:
                self.source_xy = xy
                result[:7] = self.last_joints
                return result
            # Consume input even when blocked: rejected displacement must not
            # accumulate and prevent immediate motion back into reachable space.
            displacement = xy - self.source_xy
            self.source_xy = xy
            distance = float(np.linalg.norm(displacement))
            # Bound the total planar step, preserving the requested direction.
            step = displacement * min(1.0, 2.0 / distance) if distance > 0.0 else displacement
            reason = None
            if distance > 1e-9:
                for _ in range(4):
                    target = self.pose.copy()
                    target[:2] = self.command_xy + step
                    code, solution = self.arm.get_inverse_kinematics(
                        target.tolist(), input_is_radian=True, return_is_radian=True,
                        limited=False, ref_angles=self.last_joints.tolist(),
                    )
                    if code != 0:
                        reason = f'逆解失败（code={code}）'
                    else:
                        solution = np.asarray(solution, dtype=float)
                        if solution.shape != (7,) or not np.isfinite(solution).all():
                            reason = '逆解返回无效关节角'
                            break
                        if np.max(np.abs(solution - self.last_joints)) > 0.2:
                            reason = '逆解关节跳变超过 0.2 rad'
                        else:
                            reason = self._validate_solution(solution, target)
                            if reason is not None:
                                break
                            self.last_joints = solution.copy()
                            self.command_xy = target[:2].copy()
                            reason = None
                            break
                    # Retry a smaller Cartesian step, never interpolate joints
                    # because that could violate the locked Z and orientation.
                    step *= 0.5
                if reason is None and distance > 2.0:
                    reason = '输入位移过大，已限制为每周期 2 mm；超出部分不累计'
                self._set_warning(reason)
            result[:7] = self.last_joints
            return result

    @staticmethod
    def _pose_errors(actual, desired):
        position_mm = float(np.linalg.norm(actual[:3, 3] - desired[:3, 3]))
        rotation_error = float(np.linalg.norm(actual[:3, :3] - desired[:3, :3]))
        return position_mm, rotation_error

    @staticmethod
    def _error_detail(errors):
        position_mm, rotation_error = errors
        angle_deg = np.degrees(2 * np.arcsin(np.clip(rotation_error / (2 * np.sqrt(2)), 0, 1)))
        return f'位置误差 {position_mm:.3f} mm，姿态误差 {angle_deg:.4f}°'

    def _validate_solution(self, solution, target):
        desired = self.anchor.copy()
        desired[:2, 3] = target[:2]
        local_errors = None
        if not self._controller_fk_validation:
            local_errors = self._pose_errors(self.kinematics.forward_matrix(solution), desired)
            if local_errors[0] <= 0.1 and local_errors[1] <= 1e-3:
                return None
        # The local safety model allows up to 2 mm disagreement with controller
        # FK. It cannot independently reject a controller IK at 0.1 mm precision.
        # Verify with the same controller model that produced the IK instead;
        # keep the original pose tolerances and joint-continuity guard intact.
        code, pose = self.arm.get_forward_kinematics(
            solution.tolist(), input_is_radian=True, return_is_radian=True,
        )
        if code != 0:
            return f'控制器正解复核失败（code={code}）'
        try:
            actual = xarm_rpy_transform(pose)
        except (ValueError, TypeError):
            return '控制器正解复核返回无效位姿'
        errors = self._pose_errors(actual, desired)
        if errors[0] > 0.1 or errors[1] > 1e-3:
            detail = self._error_detail(errors)
            if local_errors is not None:
                detail += '；本地模型：' + self._error_detail(local_errors)
            return f'控制器逆解位姿校验不通过（{detail}）'
        if not self._controller_fk_validation:
            logger.info(
                'XY 平移模式：本地模型与控制器不一致（%s），控制器正解复核通过（%s）；'
                '本次模式后续使用控制器正解复核',
                self._error_detail(local_errors), self._error_detail(errors),
            )
            self._controller_fk_validation = True
        return None

    def _set_warning(self, reason):
        if reason != self.warning:
            if reason:
                logger.warning('XY 平移模式：%s', reason)
            elif self.warning:
                logger.info('XY 平移模式：已恢复正常跟随')
        self.warning = reason
