"""Launch the full-load chips A/B recording with a fresh dataset destination."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("off", "on"))
    parser.add_argument("--dry-run", action="store_true", help="Validate and show the configuration without hardware")
    args = parser.parse_args()
    template = ROOT / f"config/gello/xarm7_gello_record_chips_latency_{args.mode}.yaml"
    config = yaml.safe_load(template.read_text())
    run = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    name = f"gravity_latency_chips_{args.mode}_{run}"
    config["dataset"]["root"] = str(ROOT / "datasets" / name)
    config["dataset"]["repo_id"] = f"ufactory/{name}"
    config["resume"] = False
    if args.mode == "on":
        config["teleop"]["gravity_compensation"]["log_dir"] = str(ROOT / "logs" / name)
    print(json.dumps({"mode": args.mode, "control_hz": config["teleop"]["realtime_control_fps"],
                      "dataset_fps": config["dataset"]["fps"],
                      "cameras": {k: {"type": v["type"], "fps": v["fps"]} for k,v in config["robot"]["cameras"].items()},
                      "dataset_root": config["dataset"]["root"]}, indent=2), flush=True)
    if args.dry_run:
        return
    manifest_dir = ROOT / "logs" / "gravity_latency_configs"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    effective = manifest_dir / f"{name}.yaml"
    with effective.open("x") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
    print(f"Saved effective configuration: {effective}", flush=True)
    result = subprocess.run([sys.executable, "-m", "lerobot_robot_ufactory.scripts.uf_lerobot_record",
                             "--config_path", str(effective)], cwd=ROOT)
    raise SystemExit(result.returncode)

if __name__ == "__main__":
    main()
