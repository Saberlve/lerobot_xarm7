"""FTP1 xArm7 EEF evaluation; no hardware motion unless --execute is supplied."""
import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
import numpy as np
import yaml
from uf_n0_eval import N0EvalConfig, POSE_KEYS, run as run_eef
from starvla_client.ftp1_policy_client import FTP1PolicyClient


@dataclass
class FTP1EvalConfig(N0EvalConfig):
    steps_per_inference: int = 32

    def __post_init__(self):
        super().__post_init__()
        if self.temporal_ensemble:
            raise ValueError('FTP1 runner currently uses blocking chunks; disable temporal_ensemble')


def build_observation(robot, obs, cfg):
    recorded = robot.convert_observation_for_recording(obs)
    prefix = robot.prefix
    raw = np.asarray([recorded[prefix+k] for k in (*POSE_KEYS,'gripper.pos')], dtype=np.float32)
    return dict(eef_record=raw, lang=cfg.single_task,
        image={name:np.asarray(obs[prefix+key]).copy() for name,key in
               zip(('camera_main_rgb_0','right_wrist_camera_rgb_0'),cfg.rgb_keys)},
        tactile=[np.asarray(obs[prefix+key]).copy() for key in cfg.tactile_keys])


def make_policy(cfg):
    return FTP1PolicyClient(host=cfg.server_host,port=cfg.server_port,unnorm_key=cfg.unnorm_key)


def validate_policy(policy):
    if policy.native_dim != 120 or policy.action_dim != 32 or policy.chunk_size < 1:
        raise ValueError('FTP1 xArm7 EEF action recovery is required')


def run(cfg, **kwargs):
    return run_eef(cfg, policy_factory=kwargs.pop('policy_factory',make_policy),
                   observation_builder=build_observation, policy_validator=validate_policy, **kwargs)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config_path',default=str(Path(__file__).resolve().parents[1]/'config/eval/xarm7_ftp1_eval.yaml'))
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--execute',action='store_true')
    mode.add_argument('--dry-run',action='store_true')
    parser.add_argument('--check-config',action='store_true')
    args=parser.parse_args()
    cfg=FTP1EvalConfig(**yaml.safe_load(Path(args.config_path).read_text()))
    if args.execute or args.dry_run: cfg.dry_run=not args.execute
    logging.basicConfig(level=logging.INFO)
    if args.check_config:
        print('FTP1 local config valid; server mapping is checked before connecting the robot')
        return
    try: run(cfg)
    except KeyboardInterrupt: logging.info('Stopped')

if __name__=='__main__': main()
