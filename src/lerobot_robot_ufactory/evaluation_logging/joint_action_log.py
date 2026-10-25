"""Joint evaluation CSV telemetry, including action chunk transitions."""
import csv
from pathlib import Path
import time


def _rt_joint_state(robot) -> list[float] | None:
    """Latest joint positions from the 250 Hz RT report cache (non-blocking).

    This is the same feedback source GELLO recordings use for the state, so
    logging it at 30 Hz does not perturb the control loop. Returns None while
    the report thread has not delivered the first packet.
    """
    if not getattr(robot, "_rt_report_normal", False):
        return None
    with robot._update_lock:
        return list(robot.rt_actual_joint_pos)


class JointActionLog:
    def __init__(self, directory):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = Path(directory)
        log_path.mkdir(parents=True, exist_ok=True)
        self.step_file = (log_path / f"starvla_eval_{stamp}_steps.csv").open("w", newline="", buffering=1)
        self.seam_file = (log_path / f"starvla_eval_{stamp}_seams.csv").open("w", newline="", buffering=1)
        self.step_writer = csv.writer(self.step_file)
        self.seam_writer = csv.writer(self.seam_file)
        self.step_writer.writerow(
            ["t_s", "episode", "chunk", "step_in_chunk"]
            + [f"s{i}" for i in range(1, 8)]
            + ["sg"]
            + [f"a{i}" for i in range(1, 8)]
            + ["ag"]
        )
        self.seam_writer.writerow(
            ["t_s", "episode", "chunk", "inference_ms"]
            + [f"d{i}" for i in range(1, 8)]
            + ["dg"]
            + [f"last{i}" for i in range(1, 8)]
            + ["last_g"]
            + [f"new{i}" for i in range(1, 8)]
            + ["new_g"]
        )
        self.started = time.perf_counter()
        print(f"Diagnostic logs: {self.step_file.name}, {self.seam_file.name}")

    def record_seam(self, *, episode, chunk, inference_ms, previous, action):
        if previous is None:
            return
        self.seam_writer.writerow(
            [f"{time.perf_counter() - self.started:.6f}", episode, chunk, f"{inference_ms:.1f}"]
            + [f"{v:.6f}" for v in (action - previous)]
            + [f"{v:.6f}" for v in previous]
            + [f"{v:.6f}" for v in action]
        )

    def record_step(self, *, robot, episode, chunk, step, gripper, action):
        # Read only the existing RT cache; no blocking controller requests.
        joints = _rt_joint_state(robot) or [float("nan")] * 7
        self.step_writer.writerow(
            [f"{time.perf_counter() - self.started:.6f}", episode, chunk, step]
            + [f"{v:.6f}" for v in joints]
            + ["" if gripper is None else f"{gripper:.4f}"]
            + [f"{v:.6f}" for v in action]
        )

    def close(self):
        try:
            self.step_file.close()
        finally:
            self.seam_file.close()
