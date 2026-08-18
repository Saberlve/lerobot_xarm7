import math
from dataclasses import dataclass, field
from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig

@RobotConfig.register_subclass("uf::robot")
@dataclass
class UFRobotConfig(RobotConfig):
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {}
    )
    cameras_args: dict = None
    robot_ip: str = "192.168.1.127"
    robot_dof: int | None = None  # Set it correctly if controlling in joint space!
    control_space: str = "joint"
    gripper_type: int = 1       # 1: xArm Gripper, 2: xArm Gripper G2, 10: Pika Gripper, 11: Robotiq 2F-85
    gripper_port: str = None    # only used by pika gripper (gripper_type=10)
    gripper_speed: int = -1     # auto
    gripper_force: int = -1     # auto
    gripper_command_threshold: float = 0.01  # normalized change required before sending a new command
    gripper_command_interval_s: float = 0.1  # minimum interval between tool RS485 goals
    gripper_error_log_path: str | None = "logs/xarm_gripper_errors.log"
    enable_logs: bool = False  # optional per-cycle timing and diagnostic logs
    observe_joint_vel: bool = False # only effective in joint control mode
    manual_mode: bool = False  # xArm joint teaching mode; records state and optional gripper actions
    manual_gripper_speed: float = 0.5  # normalized gripper position per second in manual mode
    teach_sensitivity: int | None = None  # xArm teaching sensitivity, valid range: 1-5
    joint_command_mode: int = 6  # 1: servo-angle-j, 6: online trajectory planning
    # start_joints and start_tcp_pose are intentionally disabled.
    # Reset uses the xArm SDK initial_point instead of configuration poses.
    max_joint_velocity: int = 90   # °/s, only effective in joint control mode
    max_linear_velocity: int = 200 # mm/s, only effective in cartesian control mode
    no_action: bool = False # only for debug
    # Optional TCP height floor in the xArm base coordinate system (mm).
    # The value should include any desired safety margin above the table.
    min_tcp_z_mm: float | None = None
    # Skip synchronous FK while the actual TCP is this far above the floor.
    # The RT report keeps this fast path asynchronous and avoids jitter during
    # normal teleoperation; FK/IK remains active near the configured floor.
    tcp_z_guard_activation_margin_mm: float = 100.0
    # ``local_projection`` performs all per-cycle FK/Jacobian work on the CPU.
    # ``controller_rpc`` retains the legacy controller FK/IK implementation.
    tcp_z_guard_backend: str = "controller_rpc"
    tcp_z_soft_margin_mm: float = 5.0
    local_kinematics_max_error_mm: float = 2.0
    controller_safety_boundary: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.id = 'uf_robot' if self.id is None else self.id
        if self.manual_mode:
            if self.control_space != "joint":
                raise ValueError("manual_mode requires control_space='joint'")
            if self.teach_sensitivity is not None and not 1 <= self.teach_sensitivity <= 5:
                raise ValueError("teach_sensitivity must be between 1 and 5")
        if self.manual_gripper_speed < 0:
            raise ValueError("manual_gripper_speed must be non-negative")
        if not 0 <= self.gripper_command_threshold <= 1:
            raise ValueError("gripper_command_threshold must be between 0 and 1")
        if not math.isfinite(self.gripper_command_interval_s) or self.gripper_command_interval_s < 0:
            raise ValueError("gripper_command_interval_s must be finite and non-negative")
        if self.control_space == "joint" and self.joint_command_mode not in (1, 6):
            raise ValueError("joint_command_mode must be 1 or 6 for joint control")
        if self.min_tcp_z_mm is not None and not math.isfinite(self.min_tcp_z_mm):
            raise ValueError("min_tcp_z_mm must be finite when provided")
        if (
            not math.isfinite(self.tcp_z_guard_activation_margin_mm)
            or self.tcp_z_guard_activation_margin_mm < 0
        ):
            raise ValueError("tcp_z_guard_activation_margin_mm must be finite and non-negative")
        if self.tcp_z_guard_backend not in ("controller_rpc", "local_projection"):
            raise ValueError("tcp_z_guard_backend must be 'controller_rpc' or 'local_projection'")
        if self.tcp_z_guard_backend == "local_projection":
            if self.control_space != "joint" or self.robot_dof != 7:
                raise ValueError("local_projection requires joint control on an xArm7")
            if self.min_tcp_z_mm is None:
                raise ValueError("local_projection requires min_tcp_z_mm")
        if not math.isfinite(self.tcp_z_soft_margin_mm) or self.tcp_z_soft_margin_mm < 0:
            raise ValueError("tcp_z_soft_margin_mm must be finite and non-negative")
        if (
            not math.isfinite(self.local_kinematics_max_error_mm)
            or self.local_kinematics_max_error_mm <= 0
        ):
            raise ValueError("local_kinematics_max_error_mm must be finite and positive")
