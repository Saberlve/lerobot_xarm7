"""External-torque estimators shared by runtime, training and offline tests."""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .arm_feedback import vector7

logger = logging.getLogger(__name__)


class HistoryBuffer:
    """Official NEXT rolling input: ``[q, qdot, q_cmd - q]``."""

    def __init__(self, history: int):
        self.history = int(history)
        if self.history < 1:
            raise ValueError("history must be >= 1")
        self.rows = deque(maxlen=self.history)

    @property
    def ready(self) -> bool:
        return len(self.rows) == self.history

    def reset(self) -> None:
        self.rows.clear()

    def append(self, joint_pos, joint_vel, joint_command) -> None:
        q = vector7(joint_pos, "joint_pos").astype(np.float32)
        qdot = vector7(joint_vel, "joint_vel").astype(np.float32)
        command = vector7(joint_command, "joint_command").astype(np.float32)
        self.rows.append(np.concatenate((q, qdot, command - q)))

    def array(self) -> np.ndarray:
        if not self.ready:
            raise ValueError(f"HistoryBuffer needs {self.history} rows, has {len(self.rows)}")
        return np.stack(self.rows).astype(np.float32)


@dataclass(frozen=True)
class ExternalTorqueEstimate:
    external_torque: np.ndarray = field(default_factory=lambda: np.zeros(7))
    predicted_free_torque: np.ndarray = field(default_factory=lambda: np.zeros(7))
    ready: bool = True
    model_valid: bool = True
    estimator_mode: str = "baseline"
    inference_latency_ms: float = 0.0
    status: str = "ok"
    baseline_external_torque: np.ndarray | None = None


class ArmExternalTorqueEstimator:
    """Common estimator interface; outputs retain measured-torque units."""

    def reset(self) -> None:
        raise NotImplementedError

    def update(
        self,
        joint_pos,
        joint_vel,
        joint_command,
        measured_torque,
        timestamp: int,
    ) -> ExternalTorqueEstimate:
        raise NotImplementedError

    def get_diagnostics(self) -> dict:
        raise NotImplementedError


class BaselineExternalTorqueEstimator(ArmExternalTorqueEstimator):
    def __init__(self, baseline):
        self.baseline = vector7(baseline, "baseline")
        self.last = ExternalTorqueEstimate()

    def reset(self) -> None:
        self.last = ExternalTorqueEstimate()

    def update(self, joint_pos, joint_vel, joint_command, measured_torque, timestamp):
        vector7(joint_pos, "joint_pos")
        vector7(joint_vel, "joint_vel")
        vector7(joint_command, "joint_command")
        measured = vector7(measured_torque, "measured_torque")
        external = measured - self.baseline
        self.last = ExternalTorqueEstimate(
            external_torque=external,
            predicted_free_torque=self.baseline.copy(),
            estimator_mode="baseline",
            baseline_external_torque=external.copy(),
        )
        return self.last

    def get_diagnostics(self) -> dict:
        return _diagnostic_dict(self.last)


class DisabledExternalTorqueEstimator(ArmExternalTorqueEstimator):
    def __init__(self, reason: str):
        self.reason = reason
        self.last = self._estimate()

    def _estimate(self):
        return ExternalTorqueEstimate(
            ready=False,
            model_valid=False,
            estimator_mode="disabled",
            status=self.reason,
        )

    def reset(self) -> None:
        self.last = self._estimate()

    def update(self, joint_pos, joint_vel, joint_command, measured_torque, timestamp):
        self.last = self._estimate()
        return self.last

    def get_diagnostics(self) -> dict:
        return _diagnostic_dict(self.last)


class NextExternalTorqueEstimator(ArmExternalTorqueEstimator):
    """Checkpoint-compatible, ROS-free NEXT inference backend."""

    def __init__(
        self,
        checkpoint: str | Path,
        normalization: str | Path | None = None,
        config: str | Path | None = None,
        device: str = "cpu",
        inference_timeout_ms: float = 50.0,
        shadow_baseline=None,
    ):
        self.inference_timeout_ms = float(inference_timeout_ms)
        self.shadow = (
            None
            if shadow_baseline is None
            else BaselineExternalTorqueEstimator(shadow_baseline)
        )
        self._load(checkpoint, normalization, config, device)
        self.buffer = HistoryBuffer(self.history)
        self.last_timestamp = None
        self.last = ExternalTorqueEstimate(
            ready=False,
            estimator_mode="next",
            status="history_not_ready",
        )

    def _load(self, checkpoint, normalization, config, device):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("NEXT requires PyTorch; install the 'next' extra") from exc
        from .next_models import build_model

        checkpoint_path = Path(checkpoint).expanduser()
        if checkpoint_path.is_dir():
            run_dir = checkpoint_path
            checkpoint_path = run_dir / "model.pt"
        else:
            run_dir = checkpoint_path.parent
        normalization_path = (
            Path(normalization).expanduser()
            if normalization
            else run_dir / "normalization.npz"
        )
        config_path = Path(config).expanduser() if config else run_dir / "config.yaml"
        for path in (checkpoint_path, normalization_path, config_path):
            if not path.is_file():
                raise FileNotFoundError(f"NEXT artifact not found: {path}")

        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("NEXT device is cuda but CUDA is unavailable")
        checkpoint_data = torch.load(checkpoint_path, map_location=requested, weights_only=False)
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        joint_order = checkpoint_data.get("joint_order", cfg.get("joint_order"))
        if joint_order is not None and tuple(joint_order) != tuple(
            f"J{joint}" for joint in range(1, 8)
        ):
            raise ValueError(
                f"NEXT checkpoint joint_order must be J1--J7, got {joint_order}"
            )
        model_cfg = checkpoint_data.get("model", cfg.get("model", {}))
        self.history = int(checkpoint_data["history"])
        input_size = int(checkpoint_data["input_size"])
        output_size = int(checkpoint_data["output_size"])
        if input_size != 21 or output_size != 7:
            raise ValueError(
                f"xArm7 NEXT checkpoint must have input_size=21/output_size=7, got "
                f"{input_size}/{output_size}"
            )
        self.model = build_model(model_cfg, input_size, output_size, self.history).to(requested)
        self.model.load_state_dict(checkpoint_data["model_state_dict"])
        self.model.eval()
        with np.load(normalization_path) as archive:
            self.normalization = {key: archive[key].astype(np.float32) for key in archive.files}
        _validate_normalization(self.normalization)
        self.device = requested
        self.torch = torch
        self.config = cfg

    def reset(self) -> None:
        self.buffer.reset()
        self.last_timestamp = None
        if self.shadow is not None:
            self.shadow.reset()
        self.last = ExternalTorqueEstimate(
            ready=False,
            estimator_mode="next",
            status="history_not_ready",
        )

    def update(self, joint_pos, joint_vel, joint_command, measured_torque, timestamp):
        measured = vector7(measured_torque, "measured_torque")
        if self.last_timestamp is not None:
            if timestamp < self.last_timestamp:
                raise ValueError("NEXT samples must be timestamp ordered")
            if timestamp == self.last_timestamp:
                return self.last
        self.last_timestamp = timestamp
        self.buffer.append(joint_pos, joint_vel, joint_command)
        baseline_external = None
        if self.shadow is not None:
            baseline_external = self.shadow.update(
                joint_pos, joint_vel, joint_command, measured, timestamp
            ).external_torque
        if not self.buffer.ready:
            self.last = ExternalTorqueEstimate(
                ready=False,
                estimator_mode="next",
                status="history_not_ready",
                baseline_external_torque=baseline_external,
            )
            return self.last

        start = time.perf_counter_ns()
        norm = self.normalization
        inputs = (self.buffer.array() - norm["x_mean"]) / norm["x_std"]
        tensor = self.torch.from_numpy(inputs).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            normalized_output = self.model(tensor).detach().cpu().numpy()[0]
        predicted = normalized_output * norm["y_std"] + norm["y_mean"]
        predicted = vector7(predicted, "predicted_free_torque")
        latency_ms = (time.perf_counter_ns() - start) / 1e6
        if latency_ms > self.inference_timeout_ms:
            raise TimeoutError(
                f"NEXT inference timeout: {latency_ms:.2f} ms > {self.inference_timeout_ms:g} ms"
            )
        external = measured - predicted
        self.last = ExternalTorqueEstimate(
            external_torque=external,
            predicted_free_torque=predicted,
            estimator_mode="next",
            inference_latency_ms=latency_ms,
            baseline_external_torque=baseline_external,
        )
        return self.last

    def get_diagnostics(self) -> dict:
        return _diagnostic_dict(self.last)


class FallbackExternalTorqueEstimator(ArmExternalTorqueEstimator):
    """Use baseline after a NEXT load/inference failure, never silently."""

    def __init__(self, primary, baseline, fallback: str, load_error: Exception | None = None):
        self.primary = primary
        self.baseline = BaselineExternalTorqueEstimator(baseline)
        self.fallback = fallback
        self.load_error = (
            None if load_error is None else f"{type(load_error).__name__}: {load_error}"
        )
        self.last = ExternalTorqueEstimate(ready=False, status="not_started")

    @property
    def permanently_disabled(self) -> bool:
        return self.fallback == "disable" and self.primary is None

    def reset(self) -> None:
        if self.primary is not None:
            self.primary.reset()
        self.baseline.reset()

    def update(self, joint_pos, joint_vel, joint_command, measured_torque, timestamp):
        error = self.load_error
        if self.primary is not None:
            try:
                self.last = self.primary.update(
                    joint_pos, joint_vel, joint_command, measured_torque, timestamp
                )
                return self.last
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                logger.exception(
                    "NEXT inference failed; applying configured fallback=%s", self.fallback
                )
                self.primary.reset()
                # Latch the failure for the session. Re-enabling a model after
                # an inference exception could create a discontinuous current
                # command; recovery requires an explicit worker restart.
                self.primary = None
                self.load_error = error

        if self.fallback == "disable":
            self.last = ExternalTorqueEstimate(
                ready=False,
                model_valid=False,
                estimator_mode="next_disabled",
                status=error or "NEXT unavailable",
            )
            return self.last
        baseline = self.baseline.update(
            joint_pos, joint_vel, joint_command, measured_torque, timestamp
        )
        self.last = ExternalTorqueEstimate(
            external_torque=baseline.external_torque,
            predicted_free_torque=baseline.predicted_free_torque,
            ready=True,
            model_valid=False,
            estimator_mode="baseline_fallback",
            status=error or "NEXT unavailable",
            baseline_external_torque=baseline.external_torque.copy(),
        )
        return self.last

    def get_diagnostics(self) -> dict:
        return _diagnostic_dict(self.last)


def make_external_torque_estimator(config) -> ArmExternalTorqueEstimator:
    """Construct the configured backend with explicit load-failure policy."""
    if config.estimator.mode == "baseline":
        return BaselineExternalTorqueEstimator(config.baseline)
    try:
        primary = NextExternalTorqueEstimator(
            checkpoint=config.next.checkpoint,
            normalization=config.next.normalization or None,
            config=config.next.config or None,
            device=config.next.device,
            inference_timeout_ms=config.next.inference_timeout_ms,
            shadow_baseline=config.baseline if config.estimator.shadow_baseline else None,
        )
        return FallbackExternalTorqueEstimator(
            primary, config.baseline, config.next.fallback
        )
    except Exception as exc:
        logger.error("NEXT model loading failed: %s", exc)
        return FallbackExternalTorqueEstimator(
            None, config.baseline, config.next.fallback, load_error=exc
        )


def _validate_normalization(norm) -> None:
    expected = {"x_mean": (21,), "x_std": (21,), "y_mean": (7,), "y_std": (7,)}
    missing = sorted(set(expected) - set(norm))
    if missing:
        raise ValueError(f"NEXT normalization missing arrays: {', '.join(missing)}")
    for name, shape in expected.items():
        value = np.asarray(norm[name])
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise ValueError(f"NEXT normalization {name} must be finite shape={shape}")
    if np.any(np.asarray(norm["x_std"]) <= 0) or np.any(np.asarray(norm["y_std"]) <= 0):
        raise ValueError("NEXT normalization standard deviations must be positive")


def _diagnostic_dict(estimate: ExternalTorqueEstimate) -> dict:
    return {
        "history_ready": estimate.ready,
        "model_valid": estimate.model_valid,
        "estimator_mode": estimate.estimator_mode,
        "inference_latency_ms": estimate.inference_latency_ms,
        "status": estimate.status,
    }
