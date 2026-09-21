"""Run existing safe GELLO/xArm teleoperation while recording NEXT inputs.

The arm-feedback worker writes synchronized q, qdot, safety-checked q_cmd and
measured effort to its configured CSV. Collect only contact-free motion.
"""

import argparse
import sys

from lerobot.utils.import_utils import register_third_party_plugins

from .uf_robot_teleop import get_cfg, teleop_loop


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--output")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    register_third_party_plugins()
    cfg = get_cfg()
    arm = cfg.teleop.arm_feedback
    arm.enabled = True
    arm.observe_only = True
    arm.dynamic_mode = True
    # Training targets do not require a trained model. Avoid a circular
    # dependency on the checkpoint being produced by this dataset.
    arm.estimator.mode = "baseline"
    arm.estimator.shadow_baseline = False
    arm.next.enabled = False
    if args.output:
        arm.log_path = args.output
    print("NEXT DATA COLLECTION: observe-only; keep the xArm completely contact-free.")
    print(f"Joint order: J1--J7; output CSV: {arm.log_path}")
    teleop_loop(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
