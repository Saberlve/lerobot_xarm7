#!/usr/bin/env python3
"""
Compare ACT model inference at two checkpoints against ground truth.

Usage:
    cd /home/lizhuoyuan/project/lerobot_xarm7
    python scripts/compare_act.py --num-samples 8 --episode 0
"""

import argparse
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec

# ── lerobot imports ──────────────────────────────────────────────
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.converters import (
    batch_to_transition,
    transition_to_batch,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset

JOINT_NAMES = ["J1", "J2", "J3", "J4", "J5", "J6", "J7", "Gripper"]


def parse_args():
    p = argparse.ArgumentParser(description="Compare ACT 40K vs 60K inference")
    p.add_argument("--dataset-root", default="./datasets/xarm7-pick-bottle",
                   help="Path to dataset root")
    p.add_argument("--checkpoint-40k",
                   default="./outputs/train/2026-08-08/09-57-03_act/checkpoints/040000/pretrained_model",
                   help="Path to 40K checkpoint pretrained_model dir")
    p.add_argument("--checkpoint-60k",
                   default="./outputs/train/2026-08-08/09-57-03_act/checkpoints/060000/pretrained_model",
                   help="Path to 60K checkpoint pretrained_model dir")
    p.add_argument("--episodes", type=str, default="0",
                   help="Comma-separated episode indices, e.g. '0,10,20,30,40,50'")
    p.add_argument("--num-samples", type=int, default=8, help="Number of frames to sample per episode")
    p.add_argument("--output-dir", default="./outputs/compare_act",
                   help="Output directory for results")
    return p.parse_args()


def load_policy_and_processors(checkpoint_path: str):
    """Load ACT policy, preprocessor, and postprocessor from a checkpoint."""
    print(f"  Loading policy from {checkpoint_path} ...")
    policy = ACTPolicy.from_pretrained(checkpoint_path)
    policy.to("cpu")
    policy.reset()

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=checkpoint_path,
        config_filename="policy_preprocessor.json",
        overrides={"device_processor": {"device": "cpu"}},
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=checkpoint_path,
        config_filename="policy_postprocessor.json",
        overrides={"device_processor": {"device": "cpu"}},
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return policy, preprocessor, postprocessor


def get_episode_frame_range(dataset: LeRobotDataset, episode_idx: int):
    """Get (from_idx, to_idx) for a specific episode.

    dataset.meta.episodes is a HuggingFace Dataset with columns:
    episode_index, dataset_from_index, dataset_to_index, length, ...
    """
    eps = dataset.meta.episodes
    for i in range(len(eps)):
        if int(eps[i]["episode_index"]) == episode_idx:
            from_idx = int(eps[i]["dataset_from_index"])
            to_idx = int(eps[i]["dataset_to_index"])
            return from_idx, to_idx
    raise ValueError(f"Episode {episode_idx} not found in dataset")


def run_inference(policy, preprocessor, postprocessor, obs_dict: dict) -> np.ndarray:
    """Run inference on a single observation dict, return predicted action as (8,) numpy."""
    batch = preprocessor(obs_dict)
    with torch.inference_mode():
        action_chunk = policy.predict_action_chunk(batch)  # (1, chunk_size, 8)
    action = action_chunk[:, 0, :]  # (1, 8)
    action = postprocessor(action)
    return action.cpu().numpy().squeeze(0)  # (8,)


def run_one_episode(args, episode_idx, dataset, policy_40k, preproc_40k, postproc_40k,
                     policy_60k, preproc_60k, postproc_60k):
    """Run inference on one episode. Returns (results, images, mae_40k, mae_60k, l2_40k, l2_60k)."""
    from_idx, to_idx = get_episode_frame_range(dataset, episode_idx)
    total_frames = to_idx - from_idx
    print(f"  Episode {episode_idx}: frames [{from_idx}, {to_idx}), total={total_frames}")

    sample_indices = np.linspace(from_idx, to_idx - 1, args.num_samples, dtype=int)
    print(f"  Sampling {args.num_samples} frames at indices: {list(sample_indices)}")

    results = []
    images = []

    for i, global_idx in enumerate(sample_indices):
        print(f"    Frame {i+1}/{args.num_samples} (global idx={global_idx}) ...")
        frame = dataset[global_idx]

        gt_action = frame["action"].numpy().squeeze()

        img = frame["observation.images.camera"].numpy()
        img = np.transpose(img, (1, 2, 0))
        img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
        images.append(img_uint8)

        obs_dict = {
            "observation.state": frame["observation.state"],
            "observation.images.camera": frame["observation.images.camera"],
        }

        policy_40k.reset()
        policy_60k.reset()

        pred_40k = run_inference(policy_40k, preproc_40k, postproc_40k, obs_dict)
        pred_60k = run_inference(policy_60k, preproc_60k, postproc_60k, obs_dict)

        results.append({
            "frame": i,
            "global_idx": int(global_idx),
            "gt": gt_action,
            "pred_40k": pred_40k,
            "pred_60k": pred_60k,
        })

    # Compute metrics
    n_joints = 8
    gt_all = np.stack([r["gt"] for r in results])
    pred_40k_all = np.stack([r["pred_40k"] for r in results])
    pred_60k_all = np.stack([r["pred_60k"] for r in results])

    err_40k = np.abs(pred_40k_all - gt_all)
    err_60k = np.abs(pred_60k_all - gt_all)
    mae_40k = err_40k.mean(axis=0)
    mae_60k = err_60k.mean(axis=0)
    l2_40k = np.sqrt(((pred_40k_all - gt_all) ** 2).sum(axis=1))
    l2_60k = np.sqrt(((pred_60k_all - gt_all) ** 2).sum(axis=1))

    return results, images, mae_40k, mae_60k, l2_40k.mean(), l2_60k.mean()


def save_episode_plots(args, episode_idx, results, images, mae_40k, mae_60k,
                        gt_all, pred_40k_all, pred_60k_all, out_dir):
    """Generate per-episode plots and CSV."""
    n_joints = 8

    # CSV
    csv_path = os.path.join(out_dir, "comparison.csv")
    err_40k = np.abs(pred_40k_all - gt_all)
    err_60k = np.abs(pred_60k_all - gt_all)
    l2_40k = np.sqrt(((pred_40k_all - gt_all) ** 2).sum(axis=1))
    l2_60k = np.sqrt(((pred_60k_all - gt_all) ** 2).sum(axis=1))

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["frame", "global_idx"]
        for jn in JOINT_NAMES:
            header += [f"GT_{jn}", f"40K_{jn}", f"60K_{jn}", f"err40K_{jn}", f"err60K_{jn}"]
        header += ["L2_40K", "L2_60K"]
        writer.writerow(header)
        for i, r in enumerate(results):
            row = [r["frame"], r["global_idx"]]
            for j in range(n_joints):
                row += [f"{r['gt'][j]:.6f}", f"{r['pred_40k'][j]:.6f}",
                        f"{r['pred_60k'][j]:.6f}",
                        f"{err_40k[i][j]:.6f}", f"{err_60k[i][j]:.6f}"]
            row += [f"{l2_40k[i]:.6f}", f"{l2_60k[i]:.6f}"]
            writer.writerow(row)

    # 图1: Trajectory curves
    fig1, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()
    colors = {"GT": "black", "40K": "#2196F3", "60K": "#FF9800"}
    x = np.arange(args.num_samples)
    for j in range(n_joints):
        ax = axes[j]
        ax.plot(x, gt_all[:, j], "o-", color=colors["GT"], label="GT", linewidth=2, markersize=5)
        ax.plot(x, pred_40k_all[:, j], "s--", color=colors["40K"], label="40K", linewidth=1.5, markersize=5)
        ax.plot(x, pred_60k_all[:, j], "d-.", color=colors["60K"], label="60K", linewidth=1.5, markersize=5)
        ax.set_title(JOINT_NAMES[j], fontsize=12, fontweight="bold")
        ax.set_xlabel("Frame index")
        ax.set_ylabel("Joint value (rad)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig1.suptitle(f"ACT Inference — Episode {episode_idx} ({args.num_samples} frames)",
                  fontsize=14, fontweight="bold")
    fig1.tight_layout()
    fig1.savefig(os.path.join(out_dir, "trajectory_curves.png"), dpi=150)
    plt.close(fig1)

    # 图2: MAE bar chart
    fig2, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(n_joints)
    width = 0.35
    bars1 = ax.bar(x_pos - width/2, mae_40k, width, label="40K", color="#2196F3", edgecolor="white")
    bars2 = ax.bar(x_pos + width/2, mae_60k, width, label="60K", color="#FF9800", edgecolor="white")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(JOINT_NAMES)
    ax.set_ylabel("MAE (rad)")
    ax.set_title(f"Per-Joint MAE: 40K vs 60K — Episode {episode_idx}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.001, f"{h:.4f}",
                ha="center", va="bottom", fontsize=7)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.001, f"{h:.4f}",
                ha="center", va="bottom", fontsize=7)
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "error_comparison.png"), dpi=150)
    plt.close(fig2)

    # 图3: Image collage with action tables
    n_cols = min(4, args.num_samples)
    n_rows = (args.num_samples + n_cols - 1) // n_cols
    fig3 = plt.figure(figsize=(4 * n_cols, 4.5 * n_rows))
    gs = GridSpec(n_rows * 2, n_cols, figure=fig3, height_ratios=[3, 1] * n_rows)

    for i in range(args.num_samples):
        row = (i // n_cols) * 2
        col = i % n_cols

        ax_img = fig3.add_subplot(gs[row, col])
        ax_img.imshow(images[i])
        ax_img.set_title(f"Frame {i}", fontsize=10)
        ax_img.axis("off")

        ax_tbl = fig3.add_subplot(gs[row + 1, col])
        ax_tbl.axis("off")

        table_data = [["Joint", "GT", "40K", "60K"]]
        for j in range(n_joints):
            gt_val = results[i]["gt"][j]
            p40_val = results[i]["pred_40k"][j]
            p60_val = results[i]["pred_60k"][j]

            def fmt(v, err, thresh=0.05):
                s = f"{v:.3f}"
                return f"!{s}" if err > thresh else s

            table_data.append([
                JOINT_NAMES[j],
                fmt(gt_val, 0),
                fmt(p40_val, abs(p40_val - gt_val)),
                fmt(p60_val, abs(p60_val - gt_val)),
            ])

        tbl = ax_tbl.table(cellText=table_data, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7)
        tbl.scale(1.0, 1.1)

        for j in range(n_joints):
            err40 = abs(results[i]["pred_40k"][j] - results[i]["gt"][j])
            err60 = abs(results[i]["pred_60k"][j] - results[i]["gt"][j])
            if err40 > 0.05:
                tbl[(j + 1, 2)].set_facecolor("#FFCDD2")
            if err60 > 0.05:
                tbl[(j + 1, 3)].set_facecolor("#FFCDD2")

    fig3.suptitle(f"Frame-by-Frame Comparison — Episode {episode_idx}",
                  fontsize=14, fontweight="bold")
    fig3.tight_layout()
    fig3.savefig(os.path.join(out_dir, "frame_comparison.png"), dpi=150)
    plt.close(fig3)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Parse episodes
    episode_list = [int(x.strip()) for x in args.episodes.split(",")]
    print(f"Episodes to evaluate: {episode_list}")

    # ── 1. Load dataset (full, no episode filter) ────────────────
    print("\n" + "=" * 60)
    print("Loading dataset (full) ...")
    dataset = LeRobotDataset(
        repo_id="xarm7-pick-bottle",
        root=args.dataset_root,
    )
    print(f"  Total episodes: {len(dataset.meta.episodes)}")

    # ── 2. Load models (once) ────────────────────────────────────
    print("\nLoading models (shared across episodes) ...")
    print("  Loading 40K model ...")
    policy_40k, preproc_40k, postproc_40k = load_policy_and_processors(args.checkpoint_40k)
    print("  Loading 60K model ...")
    policy_60k, preproc_60k, postproc_60k = load_policy_and_processors(args.checkpoint_60k)

    # ── 3. Evaluate each episode ─────────────────────────────────
    n_joints = 8
    all_mae_40k = []
    all_mae_60k = []
    all_l2_40k = []
    all_l2_60k = []

    for ep in episode_list:
        print(f"\n{'─' * 50}")
        print(f"Evaluating Episode {ep}")
        print(f"{'─' * 50}")

        ep_out_dir = os.path.join(args.output_dir, f"ep{ep}")
        os.makedirs(ep_out_dir, exist_ok=True)

        results, images, mae_40k, mae_60k, l2_40k, l2_60k = run_one_episode(
            args, ep, dataset, policy_40k, preproc_40k, postproc_40k,
            policy_60k, preproc_60k, postproc_60k)

        all_mae_40k.append(mae_40k)
        all_mae_60k.append(mae_60k)
        all_l2_40k.append(l2_40k)
        all_l2_60k.append(l2_60k)

        # Print per-episode summary
        print(f"\n  Episode {ep} Summary:")
        print(f"  {'Joint':>10} | {'40K MAE':>10} | {'60K MAE':>10} | {'Δ':>10}")
        print(f"  {'─' * 48}")
        for j in range(n_joints):
            diff = mae_40k[j] - mae_60k[j]
            sign = "▼" if diff > 0 else "▲"
            print(f"  {JOINT_NAMES[j]:>10} | {mae_40k[j]:10.4f} | {mae_60k[j]:10.4f} | {sign}{abs(diff):9.4f}")
        print(f"  {'L2 mean':>10} | {l2_40k:10.4f} | {l2_60k:10.4f} |")

        # Extract arrays for plotting
        gt_all = np.stack([r["gt"] for r in results])
        pred_40k_all = np.stack([r["pred_40k"] for r in results])
        pred_60k_all = np.stack([r["pred_60k"] for r in results])

        save_episode_plots(args, ep, results, images, mae_40k, mae_60k,
                          gt_all, pred_40k_all, pred_60k_all, ep_out_dir)
        print(f"  Plots saved to {ep_out_dir}")

    # ── 4. Aggregate results across episodes ─────────────────────
    mean_mae_40k = np.stack(all_mae_40k).mean(axis=0)
    mean_mae_60k = np.stack(all_mae_60k).mean(axis=0)
    mean_l2_40k = np.mean(all_l2_40k)
    mean_l2_60k = np.mean(all_l2_60k)

    print(f"\n{'=' * 60}")
    print(f"=== AGGREGATE RESULTS ({len(episode_list)} episodes) ===")
    print(f"{'Joint':>10} | {'40K MAE':>10} | {'60K MAE':>10} | {'Δ':>10}")
    print("-" * 48)
    for j in range(n_joints):
        diff = mean_mae_40k[j] - mean_mae_60k[j]
        sign = "▼" if diff > 0 else "▲"
        pct = (diff / mean_mae_40k[j] * 100) if mean_mae_40k[j] > 0 else 0
        print(f"{JOINT_NAMES[j]:>10} | {mean_mae_40k[j]:10.4f} | {mean_mae_60k[j]:10.4f} | {sign}{abs(diff):9.4f} ({pct:+.0f}%)")
    print(f"{'L2 mean':>10} | {mean_l2_40k:10.4f} | {mean_l2_60k:10.4f} |")
    l2_diff_pct = (mean_l2_40k - mean_l2_60k) / mean_l2_40k * 100
    print(f"\n  Overall L2 improvement: {l2_diff_pct:.1f}%")

    # Save aggregate CSV
    agg_csv = os.path.join(args.output_dir, "aggregate_summary.csv")
    with open(agg_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Joint", "40K_MAE", "60K_MAE", "Diff", "Change%"])
        for j in range(n_joints):
            diff = mean_mae_40k[j] - mean_mae_60k[j]
            pct = (diff / mean_mae_40k[j] * 100) if mean_mae_40k[j] > 0 else 0
            writer.writerow([JOINT_NAMES[j], f"{mean_mae_40k[j]:.6f}", f"{mean_mae_60k[j]:.6f}",
                           f"{diff:.6f}", f"{pct:.1f}%"])
        writer.writerow(["L2_mean", f"{mean_l2_40k:.6f}", f"{mean_l2_60k:.6f}", "", f"{l2_diff_pct:.1f}%"])
    print(f"\nAggregate CSV saved to {agg_csv}")

    # Save per-episode summary CSV
    eps_csv = os.path.join(args.output_dir, "per_episode_summary.csv")
    with open(eps_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "L2_40K", "L2_60K"])
        for i, ep in enumerate(episode_list):
            writer.writerow([ep, f"{all_l2_40k[i]:.6f}", f"{all_l2_60k[i]:.6f}"])
    print(f"Per-episode summary saved to {eps_csv}")

    # Aggregate bar chart
    fig_agg, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(n_joints + 1)
    labels = JOINT_NAMES + ["L2"]
    vals_40k = list(mean_mae_40k) + [mean_l2_40k]
    vals_60k = list(mean_mae_60k) + [mean_l2_60k]
    width = 0.35
    bars1 = ax.bar(x_pos - width/2, vals_40k, width, label="40K", color="#2196F3", edgecolor="white")
    bars2 = ax.bar(x_pos + width/2, vals_60k, width, label="60K", color="#FF9800", edgecolor="white")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("MAE / L2 (rad)")
    ax.set_title(f"Aggregate Error: 40K vs 60K (avg over {len(episode_list)} episodes)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.0005, f"{h:.4f}",
                ha="center", va="bottom", fontsize=7)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.0005, f"{h:.4f}",
                ha="center", va="bottom", fontsize=7)
    fig_agg.tight_layout()
    fig_agg.savefig(os.path.join(args.output_dir, "aggregate_error.png"), dpi=150)
    plt.close(fig_agg)
    print("Aggregate bar chart saved: aggregate_error.png")

    print(f"\n{'=' * 60}")
    print(f"All outputs saved to {args.output_dir}/")
    print("Done!")
    print("Done!")


if __name__ == "__main__":
    main()
