"""Compare full-load A/B recording timing sidecars, without opening hardware."""
import argparse
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]

def stats(values):
    if not values:
        return None
    values = np.asarray(values, dtype=float)
    return {"count": len(values), "mean": float(np.mean(values)), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99)),
            "max": float(np.max(values)), "std": float(np.std(values))}

def measure(episodes, fps=30, warmup_s=2):
    groups = {k: [] for k in ("read_begin_to_send_ms", "read_end_to_send_ms", "serial_read_ms",
                              "cache_fetch_ms", "send_call_ms", "send_interval_ms")}
    total = used = missing = repeated = comparable = long_intervals = warmup = 0
    for rows in episodes:
        if not rows:
            continue
        start = rows[0]["action_send_start_ns"]
        previous = None
        for row in rows:
            total += 1
            if row["action_send_start_ns"] - start < warmup_s * 1e9:
                warmup += 1
                continue
            fields = ("sample_start_ns", "sample_end_ns", "sample_sequence")
            if any(row.get(k) is None for k in fields):
                missing += 1
                previous = None
                continue
            a,b,s = row["sample_start_ns"],row["sample_end_ns"],row["action_send_start_ns"]
            if not a <= b <= row["gello_read_end_ns"] <= s <= row["action_send_end_ns"]:
                raise ValueError("Sample/action timestamp ordering is invalid")
            used += 1
            groups["read_begin_to_send_ms"].append((s-a)/1e6)
            groups["read_end_to_send_ms"].append((s-b)/1e6)
            groups["serial_read_ms"].append((b-a)/1e6)
            groups["cache_fetch_ms"].append((row["gello_read_end_ns"]-row["gello_read_start_ns"])/1e6)
            groups["send_call_ms"].append((row["action_send_end_ns"]-s)/1e6)
            if previous is not None:
                interval=(s-previous["action_send_start_ns"])/1e6
                groups["send_interval_ms"].append(interval)
                comparable += 1
                repeated += row["sample_sequence"] == previous["sample_sequence"]
                long_intervals += interval > 1.5 * 1000 / fps
            previous = row
    if not used or missing:
        raise ValueError(f"Missing valid original-sample timing: used={used}, missing={missing}; use newly recorded test data")
    return {"total_actions": total, "warmup_actions_excluded": warmup, "used_actions": used,
            "repeated_sample_pct": 100*repeated/comparable if comparable else None,
            "send_gap_over_1_5_period_count": long_intervals,
            "send_gap_over_1_5_period_pct": 100*long_intervals/comparable if comparable else None,
            "metrics": {k: stats(v) for k,v in groups.items()}}

def latest(mode):
    paths = sorted((ROOT / "datasets").glob(f"gravity_latency_chips_{mode}_*"))
    paths = [p for p in paths if list((p/"timestamps").glob("episode_*_actions.parquet"))]
    if not paths:
        raise ValueError(f"No saved {mode} test episode found")
    return paths[-1]

def load(path):
    files = sorted((path/"timestamps").glob("episode_*_actions.parquet"))
    if not files:
        raise ValueError(f"No saved action timing files in {path}")
    manifest = ROOT / "logs" / "gravity_latency_configs" / f"{path.name}.yaml"
    config = yaml.safe_load(manifest.read_text())
    report = measure([pq.read_table(f).to_pylist() for f in files],config["teleop"]["realtime_control_fps"])
    report.update(dataset=str(path), episode_count=len(files), sensor_summaries=[
        json.loads(f.read_text()) for f in sorted((path/"timestamps").glob("episode_*_summary.json"))])
    return config, report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--off",type=Path)
    parser.add_argument("--on",type=Path)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    off_cfg,off=load(args.off or latest("off"))
    on_cfg,on=load(args.on or latest("on"))
    for key in ("robot", "web_preview", "synchronize", "offline_mesh3dflow"):
        if off_cfg.get(key)!=on_cfg.get(key):
            raise ValueError(f"A/B settings differ: {key}")
    for key in set(off_cfg["teleop"]) | set(on_cfg["teleop"]):
        if key!="gravity_compensation" and off_cfg["teleop"].get(key)!=on_cfg["teleop"].get(key):
            raise ValueError(f"A/B teleop settings differ: {key}")
    if off_cfg["dataset"]["fps"]!=on_cfg["dataset"]["fps"]:
        raise ValueError("A/B dataset rates differ")
    report={"timing_basis":"Host perf_counter_ns; serial read begin/end approximate sensor acquisition. Two seconds excluded per episode. Not physical xArm response latency.",
            "off":off,"on":on,"on_minus_off_ms":{}}
    for metric in off["metrics"]:
        if off["metrics"][metric] and on["metrics"][metric]:
            report["on_minus_off_ms"][metric]={k:on["metrics"][metric][k]-off["metrics"][metric][k] for k in ("median","p95","p99","max")}
    rendered=json.dumps(report,indent=2,allow_nan=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(rendered+"\n")

if __name__ == "__main__":
    main()
