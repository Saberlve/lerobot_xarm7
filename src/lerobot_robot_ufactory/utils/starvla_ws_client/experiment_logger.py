"""Low-overhead, hardware-agnostic logging for real-robot policy comparisons."""

from __future__ import annotations

import csv
import json
import statistics
import subprocess
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _json_safe(value.value)
    return value


def _git_commit() -> str | None:
    """Return the client checkout commit without requiring a particular cwd."""
    try:
        result = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


class ExperimentLogger:
    """Write one self-contained experiment directory and its final summary."""

    ROBOT_STATE_FIELDS = (
        ["timestamp", "control_step", "episode", "chunk", "selected_action_index"]
        + [f"commanded_j{i}" for i in range(1, 8)]
        + ["commanded_gripper"]
        + [f"actual_j{i}" for i in range(1, 8)]
        + [f"actual_j{i}_velocity" for i in range(1, 8)]
        + [
            "actual_tcp_x",
            "actual_tcp_y",
            "actual_tcp_z",
            "actual_tcp_rx",
            "actual_tcp_ry",
            "actual_tcp_rz",
        ]
    )

    def __init__(
        self,
        root_dir: str,
        experiment_name: str,
        *,
        config: dict[str, Any],
        control_frequency: float,
        inference_delay: int,
    ) -> None:
        if not root_dir:
            raise ValueError(
                "experiment_log_dir must be non-empty when experiment logging is enabled"
            )
        if not experiment_name or Path(experiment_name).name != experiment_name:
            raise ValueError("experiment_name must be a single non-empty directory name")

        root = Path(root_dir).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        self.run_dir = root / experiment_name
        self.run_dir.mkdir(exist_ok=False)

        self._start_perf = time.perf_counter()
        self._epoch_offset = time.time() - self._start_perf
        self._control_frequency = float(control_frequency)
        self._inference_delay = int(inference_delay)
        self._latencies_ms: list[float] = []
        self._boundary_jumps: list[float] = []
        self._request_count = 0
        self._deadline_miss_count = 0
        self._timeout_count = 0
        self._chunk_switch_count = 0
        self._last_target_hold_count = 0
        self._total_control_steps = 0
        self._closed = False

        config_payload = dict(config)
        config_payload.setdefault("timestamp", self._timestamp(self._start_perf))
        client_git_commit = _git_commit()
        config_payload.setdefault(
            "git_commit", config_payload.get("server_git_commit") or client_git_commit
        )
        config_payload.setdefault("client_git_commit", client_git_commit)
        (self.run_dir / "config.json").write_text(
            json.dumps(_json_safe(config_payload), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        self._events_file = (self.run_dir / "rtc_events.jsonl").open(
            "w", encoding="utf-8", buffering=1
        )
        self._state_file = (self.run_dir / "robot_state.csv").open(
            "w", encoding="utf-8", newline="", buffering=1
        )
        self._state_writer = csv.DictWriter(self._state_file, fieldnames=self.ROBOT_STATE_FIELDS)
        self._state_writer.writeheader()

    def _timestamp(self, perf_time: float | None = None) -> str:
        if perf_time is None:
            perf_time = time.perf_counter()
        epoch = self._epoch_offset + float(perf_time)
        return datetime.fromtimestamp(epoch).astimezone().isoformat(timespec="milliseconds")

    def record_inference_event(
        self,
        *,
        control_step: int,
        request_id: int | str | None,
        request_send_perf: float | None,
        response_receive_perf: float | None,
        request_anchor_step: int,
        response_arrival_step: int | None,
        actual_elapsed_steps: int | None,
        chunk_switch_step: int | None,
        selected_action_index: int | None,
        timeout: bool,
        last_target_hold: bool,
        boundary_action_jump_l2: float | None,
        request_skipped: bool = False,
        stale_response: bool = False,
        used_prefix: bool | None = None,
        event_type: str = "inference",
        error: str | None = None,
    ) -> None:
        latency_ms = None
        if request_send_perf is not None and response_receive_perf is not None:
            latency_ms = max(0.0, (response_receive_perf - request_send_perf) * 1000.0)
        deadline_budget_ms = (
            1000.0 * self._inference_delay / self._control_frequency
            if self._control_frequency > 0 and self._inference_delay > 0
            else None
        )
        deadline_miss = bool(
            not request_skipped
            and latency_ms is not None
            and deadline_budget_ms is not None
            and latency_ms > deadline_budget_ms
        )

        event = {
            "event_type": event_type,
            "timestamp": self._timestamp(response_receive_perf or request_send_perf),
            "control_step": int(control_step),
            "request_id": request_id,
            "request_send_time": self._timestamp(request_send_perf)
            if request_send_perf is not None
            else None,
            "response_receive_time": (
                self._timestamp(response_receive_perf)
                if response_receive_perf is not None
                else None
            ),
            "request_latency_ms": latency_ms,
            "request_rtt_ms": latency_ms,
            "request_anchor_step": int(request_anchor_step),
            "response_arrival_step": (
                int(response_arrival_step) if response_arrival_step is not None else None
            ),
            "actual_elapsed_steps": (
                int(actual_elapsed_steps) if actual_elapsed_steps is not None else None
            ),
            "chunk_switch_step": int(chunk_switch_step) if chunk_switch_step is not None else None,
            "selected_action_index": (
                int(selected_action_index) if selected_action_index is not None else None
            ),
            "deadline_miss": deadline_miss,
            "timeout": bool(timeout),
            "last_target_hold": bool(last_target_hold),
            "boundary_action_jump_l2": (
                float(boundary_action_jump_l2) if boundary_action_jump_l2 is not None else None
            ),
            "request_skipped": bool(request_skipped),
            "stale_response": bool(stale_response),
            "used_prefix": used_prefix,
            "error": error,
        }
        self._events_file.write(json.dumps(event, ensure_ascii=False) + "\n")

        if not request_skipped:
            self._request_count += 1
            if latency_ms is not None:
                self._latencies_ms.append(latency_ms)
            self._deadline_miss_count += int(deadline_miss)
            self._timeout_count += int(timeout)
            self._last_target_hold_count += int(last_target_hold)
        if chunk_switch_step is not None:
            self._chunk_switch_count += 1
        if boundary_action_jump_l2 is not None:
            self._boundary_jumps.append(float(boundary_action_jump_l2))

    def record_robot_state(
        self,
        *,
        control_step: int,
        episode: int,
        chunk: int,
        selected_action_index: int,
        commanded_action: np.ndarray,
        actual_joint_position: list[float] | None,
        actual_joint_velocity: list[float] | None,
        actual_tcp_pose: list[float] | None,
    ) -> None:
        action = np.asarray(commanded_action, dtype=np.float64).reshape(-1)
        positions = list(actual_joint_position or [])
        velocities = list(actual_joint_velocity or [])
        tcp_pose = list(actual_tcp_pose or [])
        row: dict[str, Any] = {
            "timestamp": self._timestamp(),
            "control_step": int(control_step),
            "episode": int(episode),
            "chunk": int(chunk),
            "selected_action_index": int(selected_action_index),
        }
        for index in range(7):
            row[f"commanded_j{index + 1}"] = float(action[index]) if index < len(action) else ""
            row[f"actual_j{index + 1}"] = float(positions[index]) if index < len(positions) else ""
            row[f"actual_j{index + 1}_velocity"] = (
                float(velocities[index]) if index < len(velocities) else ""
            )
        row["commanded_gripper"] = float(action[7]) if len(action) > 7 else ""
        for index, name in enumerate(("x", "y", "z", "rx", "ry", "rz")):
            row[f"actual_tcp_{name}"] = float(tcp_pose[index]) if index < len(tcp_pose) else ""
        self._state_writer.writerow(row)
        self._total_control_steps += 1

    @staticmethod
    def _percentile(values: list[float], quantile: float) -> float | None:
        if not values:
            return None
        return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))

    def close(self, *, status: str = "completed", error: str | None = None) -> None:
        if self._closed:
            return
        duration = time.perf_counter() - self._start_perf
        summary = {
            "status": status,
            "error": error,
            "total_duration_s": duration,
            "total_control_steps": self._total_control_steps,
            "inference_request_count": self._request_count,
            "latency_mean_ms": (
                float(statistics.fmean(self._latencies_ms)) if self._latencies_ms else None
            ),
            "latency_p50_ms": self._percentile(self._latencies_ms, 50),
            "latency_p95_ms": self._percentile(self._latencies_ms, 95),
            "deadline_miss_count": self._deadline_miss_count,
            "deadline_miss_rate": (
                self._deadline_miss_count / self._request_count if self._request_count else 0.0
            ),
            "timeout_count": self._timeout_count,
            "chunk_switch_count": self._chunk_switch_count,
            "mean_boundary_action_jump_l2": (
                float(statistics.fmean(self._boundary_jumps)) if self._boundary_jumps else None
            ),
            "max_boundary_action_jump_l2": max(self._boundary_jumps, default=None),
            "last_target_hold_count": self._last_target_hold_count,
        }
        (self.run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        self._events_file.close()
        self._state_file.close()
        self._closed = True
