"""Offline evaluation of an xArm7 NEXT run against recorded CSV data."""

import argparse
import json
from pathlib import Path

import numpy as np

from ..utils.arm_external_torque import NextExternalTorqueEstimator
from ..utils.next_training import load_training_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    estimator = NextExternalTorqueEstimator(
        args.run_dir, device=args.device, inference_timeout_ms=10_000
    )
    predictions, measured, residuals = [], [], []
    for path in args.data:
        for segment in load_training_csv(path):
            estimator.reset()
            for q, qdot, qcmd, torque, timestamp in zip(
                segment["q"],
                segment["qdot"],
                segment["qcmd"],
                segment["torque"],
                segment["timestamp_ns"],
                strict=True,
            ):
                result = estimator.update(q, qdot, qcmd, torque, int(timestamp))
                if result.ready:
                    predictions.append(result.predicted_free_torque)
                    measured.append(torque)
                    residuals.append(result.external_torque)
    if not predictions:
        raise ValueError("no predictions; dataset is shorter than the checkpoint history")
    predictions = np.asarray(predictions)
    measured = np.asarray(measured)
    residuals = np.asarray(residuals)
    metrics = {
        "samples": len(predictions),
        "per_joint_rmse": np.sqrt(np.mean((predictions - measured) ** 2, axis=0)).tolist(),
        "residual_mean": residuals.mean(axis=0).tolist(),
        "residual_std": residuals.std(axis=0).tolist(),
    }
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
