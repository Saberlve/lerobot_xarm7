import csv
import threading

import numpy as np
import pytest
import torch
import yaml

from lerobot_robot_ufactory.teleoperators.gello_teleop.gello_teleop import GelloTeleop
from lerobot_robot_ufactory.utils.arm_external_torque import (
    BaselineExternalTorqueEstimator,
    FallbackExternalTorqueEstimator,
    HistoryBuffer,
    NextExternalTorqueEstimator,
    make_external_torque_estimator,
)
from lerobot_robot_ufactory.utils.arm_feedback import (
    ArmContactConfig,
    ArmFeedbackConfig,
    ArmFeedbackProcessor,
    ArmFeedbackSample,
    ContactGate,
)
from lerobot_robot_ufactory.utils.next_models import build_model
from lerobot_robot_ufactory.utils.next_training import (
    JOINT_ORDER,
    TrainOptions,
    load_training_csv,
    make_windows,
    train_next,
)


def make_run(tmp_path, *, history=2, predicted=(0.5,) * 7):
    run = tmp_path / "run"
    run.mkdir()
    model_cfg = {
        "type": "mlp",
        "hidden_size": 4,
        "num_layers": 1,
        "dropout": 0.0,
    }
    model = build_model(model_cfg, 21, 7, history)
    for parameter in model.parameters():
        parameter.data.zero_()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": 21,
            "output_size": 7,
            "history": history,
            "model": model_cfg,
            "joint_order": list(JOINT_ORDER),
        },
        run / "model.pt",
    )
    np.savez(
        run / "normalization.npz",
        x_mean=np.zeros(21, dtype=np.float32),
        x_std=np.ones(21, dtype=np.float32),
        y_mean=np.asarray(predicted, dtype=np.float32),
        y_std=np.ones(7, dtype=np.float32),
    )
    (run / "config.yaml").write_text(
        yaml.safe_dump({"model": model_cfg, "joint_order": list(JOINT_ORDER)}),
        encoding="utf-8",
    )
    return run


def test_history_buffer_matches_official_feature_order():
    buffer = HistoryBuffer(2)
    q = np.arange(7, dtype=float)
    buffer.append(q, q + 10, q + 20)
    assert not buffer.ready
    buffer.append(q + 1, q + 11, q + 31)
    assert buffer.ready
    np.testing.assert_allclose(buffer.array()[0], np.r_[q, q + 10, np.full(7, 20)])
    np.testing.assert_allclose(buffer.array()[1], np.r_[q + 1, q + 11, np.full(7, 30)])
    buffer.reset()
    assert not buffer.ready
    with pytest.raises(ValueError, match="needs 2 rows"):
        buffer.array()


def test_next_loading_warmup_output_and_joint_order(tmp_path):
    run = make_run(tmp_path)
    estimator = NextExternalTorqueEstimator(run)
    q = np.arange(7, dtype=float)
    first = estimator.update(q, q * 0, q + 0.2, np.ones(7), 1)
    assert not first.ready and first.status == "history_not_ready"
    duplicate = estimator.update(q, q * 0, q + 0.2, np.ones(7), 1)
    assert not duplicate.ready  # a 100 Hz worker must not duplicate a 20 Hz report sample
    second = estimator.update(q, q * 0, q + 0.2, np.ones(7), 2)
    assert second.ready and second.model_valid
    assert second.external_torque.shape == (7,)
    np.testing.assert_allclose(second.predicted_free_torque, 0.5)
    np.testing.assert_allclose(second.external_torque, 0.5)


def test_nonfinite_input_rejected_and_load_failure_fallback(tmp_path):
    baseline = np.arange(7, dtype=float)
    config = ArmFeedbackConfig(
        estimator={"mode": "next"},
        next={
            "enabled": True,
            "checkpoint": str(tmp_path / "missing.pt"),
            "fallback": "baseline",
        },
        baseline=tuple(baseline),
    )
    estimator = make_external_torque_estimator(config)
    output = estimator.update(np.zeros(7), np.zeros(7), np.zeros(7), baseline + 2, 1)
    assert output.ready and not output.model_valid
    assert output.estimator_mode == "baseline_fallback"
    np.testing.assert_allclose(output.external_torque, 2)
    with pytest.raises(ValueError, match="finite"):
        BaselineExternalTorqueEstimator(np.zeros(7)).update(
            np.r_[np.nan, np.zeros(6)], np.zeros(7), np.zeros(7), np.zeros(7), 1
        )


def test_runtime_exception_obeys_disable_fallback():
    class Broken:
        def reset(self):
            pass

        def update(self, *args):
            raise RuntimeError("injected inference failure")

    estimator = FallbackExternalTorqueEstimator(Broken(), np.zeros(7), "disable")
    output = estimator.update(*([np.zeros(7)] * 4), 1)
    assert not output.ready and not output.model_valid
    assert output.estimator_mode == "next_disabled"
    assert "injected inference failure" in output.status
    assert estimator.permanently_disabled


def test_contact_hysteresis_debounce_and_ramp():
    gate = ContactGate(
        ArmContactConfig(
            enabled=True,
            threshold_nm=(2,) * 7,
            release_threshold_nm=(1,) * 7,
            debounce_ms=10,
            ramp_up_ms=20,
            ramp_down_ms=20,
        )
    )
    state, amount = gate.update(np.full(7, 3.0), 1_000_000_000)
    assert not state.any() and not amount.any()
    state, amount = gate.update(np.full(7, 3.0), 1_010_000_000)
    assert state.all() and np.allclose(amount, 0.5)
    state, amount = gate.update(np.full(7, 1.5), 1_020_000_000)
    assert state.all() and np.allclose(amount, 1.0)
    gate.update(np.zeros(7), 1_030_000_000)
    state, amount = gate.update(np.zeros(7), 1_040_000_000)
    assert not state.any() and np.allclose(amount, 0.5)


def test_abnormal_torque_spike_fails_safe_zero():
    config = ArmFeedbackConfig(
        spike_limit=(2,) * 7,
        gain_ma_per_unit=(10,) * 7,
        enabled_joints=(True,) * 7,
    )
    processor = ArmFeedbackProcessor(config)
    sample = ArmFeedbackSample(1, np.full(7, 3.0), np.full(7, 3.0))
    result = processor.process(sample, np.zeros(7), 1)
    assert result.fault == "torque_spike"
    assert not result.command_current_ma.any()


def test_gello_command_snapshot_uses_sent_joint_order():
    teleop = GelloTeleop.__new__(GelloTeleop)
    teleop._arm_command_lock = threading.Lock()
    teleop._arm_command_snapshot = (0, np.zeros(7), "command_unavailable")
    action = {f"J{joint}.pos": float(joint) for joint in range(1, 8)}
    teleop.update_arm_feedback_command(action, timestamp_ns=123)
    stamp, command, error = teleop.arm_feedback_command_snapshot()
    assert stamp == 123 and error is None
    np.testing.assert_allclose(command, np.arange(1, 8))
    command[:] = 0
    assert teleop.arm_feedback_command_snapshot()[1][0] == 1


def write_training_csv(path, rows=80):
    fields = ["timestamp_ns", "command_valid", "fault"]
    for prefix in ("q", "qdot", "qcmd", "tau_measured"):
        fields.extend(f"{prefix}_{joint}" for joint in range(1, 8))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in range(rows):
            row = {"timestamp_ns": index + 1, "command_valid": True, "fault": ""}
            for joint in range(1, 8):
                q = 0.01 * index + joint
                row[f"q_{joint}"] = q
                row[f"qdot_{joint}"] = 0.01
                row[f"qcmd_{joint}"] = q + 0.1
                row[f"tau_measured_{joint}"] = 0.2 * q
            writer.writerow(row)


def test_csv_windows_and_training_artifacts(tmp_path):
    path = tmp_path / "free.csv"
    write_training_csv(path)
    segments = load_training_csv(path)
    x, y = make_windows(segments, history=3)
    assert x.shape == (78, 3, 21) and y.shape == (78, 7)
    np.testing.assert_allclose(x[0, 0, 14:], 0.1, atol=1e-6)

    output = tmp_path / "trained"
    metrics = train_next(
        [path],
        output,
        TrainOptions(history=3, model_type="mlp", hidden_size=8, epochs=1, batch_size=16),
    )
    assert metrics["best_epoch"] == 1
    for name in ("model.pt", "config.yaml", "normalization.npz", "metrics.json"):
        assert (output / name).is_file()
