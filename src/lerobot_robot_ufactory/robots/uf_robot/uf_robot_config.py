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
    # Representation of the saved dataset, only effective with control_space="joint".
    # "joint" records J1..Jn (rad); "tcp" records the FK-converted TCP pose
    # (mm; continuous 6D rotation, Zhou et al. CVPR 2019); "both" records
    # joints and TCP pose side by side. Control stays in joint space.
    record_space: str = "joint"
    gripper_type: int = 1       # 1: xArm Gripper, 2: xArm Gripper G2, 10: Pika Gripper, 11: Robotiq 2F-85
    gripper_port: str = None    # only used by pika gripper (gripper_type=10)
    gripper_speed: int = -1     # auto
    gripper_force: int = -1     # auto
    gripper_command_threshold: float = 0.01  # normalized change required before sending a new command
    gripper_command_interval_s: float = 0.1  # minimum interval between tool RS485 goals
    gripper_error_log_path: str | None = "logs/xarm_gripper_errors.log"
    # Opt-in asynchronous Gripper G2 actual-current reporting via TCP 30000.
    # This remains outside the LeRobot observation/dataset schema.
    gripper_current_monitor: bool = False
    gripper_current_monitor_frequency_hz: int = 250
    gripper_current_stale_timeout_s: float = 0.25
    enable_logs: bool = False  # optional per-cycle timing and diagnostic logs
    # Software-bounded pairing for robot state, every RGB camera and Photon.
    # These are host monotonic-clock bounds, not a common hardware trigger.
    sync_max_skew_ms: float = 70.0
    sync_pair_max_skew_ms: float = 90.0
    sync_wait_ms: float = 70.0
    sync_history_size: int = 30
    observe_joint_vel: bool = False # only effective in joint control mode
    manual_mode: bool = False  # xArm joint teaching mode; records state and optional gripper actions
    manual_gripper_speed: float = 0.5  # normalized gripper position per second in manual mode
    teach_sensitivity: int | None = None  # xArm teaching sensitivity, valid range: 1-5
    joint_command_mode: int = 6  # xArm online joint trajectory planning
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
        for name in (
            "sync_max_skew_ms",
            "sync_pair_max_skew_ms",
            "sync_wait_ms",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not isinstance(self.sync_history_size, int)
            or isinstance(self.sync_history_size, bool)
            or self.sync_history_size <= 0
        ):
            raise ValueError("sync_history_size must be a positive integer")
        if not 0 <= self.gripper_command_threshold <= 1:
            raise ValueError("gripper_command_threshold must be between 0 and 1")
        if not math.isfinite(self.gripper_command_interval_s) or self.gripper_command_interval_s < 0:
            raise ValueError("gripper_command_interval_s must be finite and non-negative")
        if self.gripper_type == 2:
            if self.gripper_speed != -1 and not 15 <= self.gripper_speed <= 225:
                raise ValueError("xArm Gripper G2 gripper_speed must be -1 or between 15 and 225 mm/s")
            if self.gripper_force != -1 and not 1 <= self.gripper_force <= 100:
                raise ValueError("xArm Gripper G2 gripper_force must be -1 or between 1 and 100")
        if self.gripper_current_monitor and self.gripper_type != 2:
            raise ValueError("gripper_current_monitor requires gripper_type=2 (xArm Gripper G2)")
        if (
            not isinstance(self.gripper_current_monitor_frequency_hz, int)
            or isinstance(self.gripper_current_monitor_frequency_hz, bool)
            or self.gripper_current_monitor_frequency_hz <= 0
        ):
            raise ValueError("gripper_current_monitor_frequency_hz must be a positive integer")
        if (
            not math.isfinite(self.gripper_current_stale_timeout_s)
            or self.gripper_current_stale_timeout_s <= 0
        ):
            raise ValueError("gripper_current_stale_timeout_s must be finite and positive")
        if self.control_space == "joint" and self.joint_command_mode != 6:
            raise ValueError("joint_command_mode must be 6 for joint control")
        if self.record_space not in ("joint", "tcp", "both"):
            raise ValueError("record_space must be 'joint', 'tcp' or 'both'")
        if self.record_space in ("tcp", "both"):
            if self.control_space != "joint" or self.robot_dof != 7:
                raise ValueError(f"record_space='{self.record_space}' requires joint control on an xArm7")
            if self.manual_mode:
                raise ValueError(f"record_space='{self.record_space}' is not supported in manual_mode")
        if self.record_space == "tcp" and self.observe_joint_vel:
            raise ValueError("record_space='tcp' does not support observe_joint_vel")
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
