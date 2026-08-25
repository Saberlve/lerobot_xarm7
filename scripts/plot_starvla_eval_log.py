"""Plot diagnostic logs written by uf_starvla_eval.

Usage:
    python scripts/plot_starvla_eval_log.py logs/starvla_eval_20260824_120000_steps.csv [max_t_s]

Reads the matching *_seams.csv next to the steps file automatically.
Outputs three PNGs next to the input:
    *_joints.png   per-joint RT state vs sent action (seams marked)
    *_velocity.png per-joint state velocity (zero plateaus = inference pauses)
    *_seams.png    per-seam jump: new_chunk[0] - last_executed_action
"""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

JOINT_LABELS = [f"J{i}" for i in range(1, 8)]
RAD2DEG = 180.0 / np.pi


def _load(path: Path) -> dict[str, np.ndarray] | None:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    cols = {}
    for key in rows[0]:
        cols[key] = np.array(
            [float(r[key]) if r[key] not in ("", None) else np.nan for r in rows]
        )
    return cols


def _mark_seams(axes, seam_t):
    for ax in axes:
        for t in seam_t:
            ax.axvline(t, color="r", ls="--", lw=0.8, alpha=0.35)


def main() -> None:
    steps_path = Path(sys.argv[1])
    max_t_s = float(sys.argv[2]) if len(sys.argv) > 2 else None
    seams_path = steps_path.with_name(steps_path.name.replace("_steps.csv", "_seams.csv"))
    steps = _load(steps_path)
    seams = _load(seams_path) if seams_path.exists() else None
    if steps is None:
        raise SystemExit(f"no rows in {steps_path}")
    if max_t_s is not None:
        step_mask = steps["t_s"] <= max_t_s
        steps = {key: values[step_mask] for key, values in steps.items()}
        if seams is not None:
            seam_mask = seams["t_s"] <= max_t_s
            seams = {key: values[seam_mask] for key, values in seams.items()}
        if not len(steps["t_s"]):
            raise SystemExit(f"no rows at or before t_s={max_t_s:g} in {steps_path}")
    seam_t = seams["t_s"] if seams else []

    t = steps["t_s"]

    # --- Figure 1: state vs action, per joint + gripper ---------------------
    fig, axes = plt.subplots(8, 1, figsize=(14, 16), sharex=True)
    for i in range(7):
        ax = axes[i]
        ax.plot(t, steps[f"s{i + 1}"] * RAD2DEG, lw=0.8, label="state (RT)")
        ax.plot(t, steps[f"a{i + 1}"] * RAD2DEG, lw=0.8, label="action sent")
        ax.set_ylabel(f"{JOINT_LABELS[i]} (deg)")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right")
    axes[7].plot(t, steps["sg"], lw=0.8, label="gripper state (cmd)")
    axes[7].plot(t, steps["ag"], lw=0.8, label="gripper action")
    axes[7].set_ylabel("gripper")
    axes[7].set_xlabel("t (s)")
    axes[7].grid(alpha=0.3)
    axes[7].legend(loc="upper right")
    _mark_seams(axes, seam_t)
    fig.suptitle(f"State vs sent action @30 Hz (red dashed = chunk seam)\n{steps_path.name}")
    fig.tight_layout()
    out1 = steps_path.with_name(steps_path.stem + "_joints.png")
    fig.savefig(out1, dpi=600)

    # --- Figure 2: state velocity (plateaus = inference pauses) -------------
    fig, axes = plt.subplots(7, 1, figsize=(14, 12), sharex=True)
    dt = np.diff(t)
    for i in range(7):
        vel = np.diff(steps[f"s{i + 1}"]) / np.maximum(dt, 1e-6) * RAD2DEG
        axes[i].plot(t[1:], vel, lw=0.8)
        axes[i].set_ylabel(f"{JOINT_LABELS[i]} (deg/s)")
        axes[i].grid(alpha=0.3)
    axes[-1].set_xlabel("t (s)")
    _mark_seams(axes, seam_t)
    fig.suptitle("Joint velocity from RT state (zero plateaus = arm paused during inference)")
    fig.tight_layout()
    out2 = steps_path.with_name(steps_path.stem + "_velocity.png")
    fig.savefig(out2, dpi=600)

    # --- Figure 3: seam jumps -------------------------------------------------
    if seams:
        d = np.stack([seams[f"d{i}"] for i in range(1, 8)], axis=1) * RAD2DEG
        chunks = seams["chunk"].astype(int)
        episodes = seams["episode"].astype(int)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        im = ax1.imshow(d, aspect="auto", cmap="RdBu_r",
                        vmin=-np.nanmax(np.abs(d)), vmax=np.nanmax(np.abs(d)))
        ax1.set_xticks(range(7), JOINT_LABELS)
        ax1.set_yticks(range(len(chunks)), [f"ep{e} ck{c}" for e, c in zip(episodes, chunks)])
        ax1.set_title("seam jump per joint: new_chunk[0] - last_executed (deg)")
        fig.colorbar(im, ax=ax1, label="deg")
        ax2.bar(range(len(chunks)), seams["dg"])
        ax2.set_xticks(range(len(chunks)), [f"ep{e} ck{c}" for e, c in zip(episodes, chunks)],
                       rotation=90, fontsize=7)
        ax2.set_title("seam jump: gripper")
        ax2.grid(alpha=0.3)
        ax3.bar(range(len(chunks)), seams["inference_ms"])
        ax3.set_xticks(range(len(chunks)), [f"ep{e} ck{c}" for e, c in zip(episodes, chunks)],
                       rotation=90, fontsize=7)
        ax3.set_title("blocking inference latency (ms)")
        ax3.grid(alpha=0.3)
        fig.tight_layout()
        out3 = steps_path.with_name(steps_path.stem + "_seams.png")
        fig.savefig(out3, dpi=600)
    else:
        out3 = None

    print(f"saved: {out1}")
    print(f"saved: {out2}")
    if out3:
        print(f"saved: {out3}")


if __name__ == "__main__":
    main()
