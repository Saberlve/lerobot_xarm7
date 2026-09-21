"""ROS-free xArm7 NEXT dataset, training, and evaluation helpers."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .next_models import build_model

JOINT_ORDER = tuple(f"J{joint}" for joint in range(1, 8))


def _columns(prefix: str, fieldnames) -> list[str]:
    alternatives = (
        [f"{prefix}_{joint}" for joint in range(1, 8)],
        [f"{prefix}{joint}" for joint in range(1, 8)],
    )
    for names in alternatives:
        if all(name in fieldnames for name in names):
            return names
    raise ValueError(f"training CSV is missing {prefix} joint columns")


def load_training_csv(path: str | Path) -> list[dict[str, np.ndarray]]:
    """Load contiguous valid segments from an arm-feedback diagnostic CSV."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"empty training CSV: {path}")
        fields = reader.fieldnames
        q_names = (
            _columns("q", fields)
            if "q_1" in fields or "q1" in fields
            else _columns("joint_position", fields)
        )
        qdot_names = (
            _columns("qdot", fields)
            if "qdot_1" in fields or "qdot1" in fields
            else _columns("joint_velocity", fields)
        )
        qcmd_names = _columns("qcmd", fields)
        torque_names = (
            _columns("tau_measured", fields)
            if "tau_measured_1" in fields
            else _columns("raw_effort", fields)
        )
        rows = list(reader)

    segments = []
    current = {name: [] for name in ("q", "qdot", "qcmd", "torque", "timestamp_ns")}

    def finish():
        nonlocal current
        if current["q"]:
            segments.append(
                {
                    name: np.asarray(
                        values,
                        dtype=np.int64 if name == "timestamp_ns" else np.float32,
                    )
                    for name, values in current.items()
                }
            )
        current = {name: [] for name in current}

    previous_timestamp = None
    for row in rows:
        valid = str(row.get("command_valid", "true")).lower() in ("1", "true")
        valid = valid and not row.get("fault", "")
        try:
            timestamp = int(row.get("sample_timestamp_ns") or row.get("timestamp_ns"))
            q = [float(row[name]) for name in q_names]
            qdot = [float(row[name]) for name in qdot_names]
            qcmd = [float(row[name]) for name in qcmd_names]
            torque = [float(row[name]) for name in torque_names]
            finite = np.all(np.isfinite(q + qdot + qcmd + torque))
        except (TypeError, ValueError):
            finite = False
        if (
            not valid
            or not finite
            or (previous_timestamp is not None and timestamp <= previous_timestamp)
        ):
            finish()
            previous_timestamp = None
            continue
        current["q"].append(q)
        current["qdot"].append(qdot)
        current["qcmd"].append(qcmd)
        current["torque"].append(torque)
        current["timestamp_ns"].append(timestamp)
        previous_timestamp = timestamp
    finish()
    if not segments:
        raise ValueError(f"no valid contact-free rows in {path}")
    return segments


def make_windows(segments, history: int):
    """Create official NEXT windows and final-timestep torque targets."""
    history = int(history)
    if history < 1:
        raise ValueError("history must be >= 1")
    windows, targets = [], []
    for segment in segments:
        q = np.asarray(segment["q"], dtype=np.float32)
        qdot = np.asarray(segment["qdot"], dtype=np.float32)
        qcmd = np.asarray(segment["qcmd"], dtype=np.float32)
        torque = np.asarray(segment["torque"], dtype=np.float32)
        if not (q.shape[1:] == qdot.shape[1:] == qcmd.shape[1:] == torque.shape[1:] == (7,)):
            raise ValueError("NEXT data must use ordered J1--J7 vectors")
        features = np.concatenate((q, qdot, qcmd - q), axis=1)
        for start in range(0, len(features) - history + 1):
            windows.append(features[start : start + history])
            targets.append(torque[start + history - 1])
    if not windows:
        raise ValueError("no NEXT windows; collect more rows or reduce --history")
    return np.asarray(windows, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def fit_normalization(x, y):
    """Official train-only per-feature/per-joint standardization."""
    return {
        "x_mean": x.mean(axis=(0, 1)).astype(np.float32),
        "x_std": (x.std(axis=(0, 1)) + 1e-6).astype(np.float32),
        "y_mean": y.mean(axis=0).astype(np.float32),
        "y_std": (y.std(axis=0) + 1e-6).astype(np.float32),
    }


@dataclass
class TrainOptions:
    history: int = 50
    model_type: str = "lstm"
    hidden_size: int = 128
    num_layers: int = 2
    head_hidden: int = 256
    head_layers: int = 2
    dropout: float = 0.1
    epochs: int = 20
    batch_size: int = 2048
    learning_rate: float = 1e-3
    val_fraction: float = 0.1
    seed: int = 0
    device: str = "cpu"
    plot: bool = False


def train_next(csv_paths, output_dir, options: TrainOptions):
    """Train NEXT, save the best validation checkpoint, and return metrics."""
    import torch
    import torch.nn.functional as functional
    from torch.utils.data import DataLoader, TensorDataset

    if not 0 < options.val_fraction < 1:
        raise ValueError("val_fraction must be in (0, 1)")
    if options.epochs < 1 or options.batch_size < 1 or options.learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")

    random.seed(options.seed)
    np.random.seed(options.seed)
    torch.manual_seed(options.seed)
    torch.cuda.manual_seed_all(options.seed)
    segments = []
    for path in csv_paths:
        segments.extend(load_training_csv(path))
    x, y = make_windows(segments, options.history)
    validation_count = max(1, int(len(x) * options.val_fraction))
    validation_start = len(x) - validation_count
    train_end = max(0, validation_start - options.history)
    if train_end < 1:
        raise ValueError("dataset is too small for a leakage-buffered validation split")
    train_x, train_y = x[:train_end], y[:train_end]
    validation_x, validation_y = x[validation_start:], y[validation_start:]
    norm = fit_normalization(train_x, train_y)
    def normalize_x(value):
        return (value - norm["x_mean"]) / norm["x_std"]

    def normalize_y(value):
        return (value - norm["y_mean"]) / norm["y_std"]

    requested = torch.device(options.device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("training device is cuda but CUDA is unavailable")
    model_cfg = {
        "type": options.model_type,
        "state_mode": "stateless",
        "hidden_size": options.hidden_size,
        "num_layers": options.num_layers,
        "head_hidden": options.head_hidden,
        "head_layers": options.head_layers,
        "bidirectional": False,
        "dropout": options.dropout,
    }
    model = build_model(model_cfg, 21, 7, options.history).to(requested)
    optimizer = torch.optim.Adam(model.parameters(), lr=options.learning_rate)
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(normalize_x(train_x)),
            torch.from_numpy(normalize_y(train_y)),
        ),
        batch_size=options.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(options.seed),
    )
    validation_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(normalize_x(validation_x)),
            torch.from_numpy(normalize_y(validation_y)),
        ),
        batch_size=options.batch_size,
    )
    metrics = {
        "joint_order": list(JOINT_ORDER),
        "train_loss": [],
        "validation_loss": [],
        "validation_rmse_per_joint": [],
        "best_epoch": None,
        "best_validation_loss": None,
    }
    best_state = None
    best_loss = float("inf")
    for epoch in range(1, options.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(requested), batch_y.to(requested)
            prediction = model(batch_x)
            loss = functional.mse_loss(prediction, batch_y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch_x)
            count += len(batch_x)
        train_loss = total / count
        validation_loss, rmse = _evaluate_model(
            model, validation_loader, requested, norm["y_mean"], norm["y_std"]
        )
        metrics["train_loss"].append(train_loss)
        metrics["validation_loss"].append(validation_loss)
        metrics["validation_rmse_per_joint"].append(rmse.tolist())
        print(
            f"epoch {epoch:03d} train={train_loss:.6f} "
            f"validation={validation_loss:.6f} rmse={np.round(rmse, 4).tolist()}"
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            metrics["best_epoch"] = epoch
            metrics["best_validation_loss"] = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": best_state,
        "input_size": 21,
        "output_size": 7,
        "history": options.history,
        "model": model_cfg,
        "joint_order": list(JOINT_ORDER),
    }
    torch.save(checkpoint, output / "model.pt")
    np.savez(output / "normalization.npz", **norm)
    config = {
        "data": {"paths": [str(Path(path)) for path in csv_paths], "history": options.history},
        "model": model_cfg,
        "train": {key: value for key, value in vars(options).items() if key != "history"},
        "joint_order": list(JOINT_ORDER),
    }
    (output / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if options.plot:
        _save_training_plot(metrics, output / "loss.png")
    return metrics


def _evaluate_model(model, loader, device, y_mean, y_std):
    import torch
    import torch.nn.functional as functional

    model.eval()
    total, count, errors = 0.0, 0, []
    with torch.inference_mode():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            prediction = model(batch_x)
            total += functional.mse_loss(prediction, batch_y).item() * len(batch_x)
            count += len(batch_x)
            errors.append((prediction.cpu().numpy() - batch_y.cpu().numpy()) * y_std)
    error = np.concatenate(errors)
    return total / count, np.sqrt(np.mean(error**2, axis=0))


def _save_training_plot(metrics, path):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("--plot requires matplotlib") from exc
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(metrics["train_loss"], label="train")
    axis.plot(metrics["validation_loss"], label="validation")
    axis.set(xlabel="epoch", ylabel="normalized MSE", title="xArm7 NEXT training")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
