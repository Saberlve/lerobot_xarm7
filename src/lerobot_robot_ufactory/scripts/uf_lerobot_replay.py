"""Replay absolute joint states from a LeRobot dataset on an xArm robot."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pyarrow.parquet as parquet

import lerobot_robot_ufactory  # noqa: F401
from lerobot.utils.robot_utils import precise_sleep
from lerobot_robot_ufactory.robots.uf_robot.uf_robot import UFRobot
from lerobot_robot_ufactory.robots.uf_robot.uf_robot_config import UFRobotConfig


JOINT_STATE_NAMES = tuple(f"J{index}.pos" for index in range(1, 8))
STATE_NAMES = JOINT_STATE_NAMES + ("gripper.pos",)
STATE_FEATURE = "observation.state"
DEFAULT_DATASET_ROOT = Path("datasets/xarm7_manual_replay")
DEFAULT_ROBOT_IP = "192.168.1.245"


@dataclass(frozen=True)
class ReplayEpisode:
    """Validated, ordered absolute states for one dataset episode."""

    fps: float
    episode_index: int
    frame_indices: tuple[int, ...]
    states: tuple[tuple[float, ...], ...]


def _load_info(dataset_root: Path) -> dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise ValueError(f"LeRobot metadata file does not exist: {info_path}")

    try:
        with info_path.open("r", encoding="utf-8") as info_file:
            info = json.load(info_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read LeRobot metadata: {info_path}") from exc

    if not isinstance(info, dict):
        raise ValueError(f"LeRobot metadata must contain a JSON object: {info_path}")
    return info


def _validate_state_feature(info: dict[str, Any]) -> float:
    try:
        fps = float(info["fps"])
        feature = info["features"][STATE_FEATURE]
        names = tuple(feature["names"])
        shape = tuple(feature["shape"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Dataset metadata must define {STATE_FEATURE!r} and a positive FPS"
        ) from exc

    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Dataset FPS must be a positive finite number, got {fps!r}")
    if names != STATE_NAMES or shape != (len(STATE_NAMES),):
        raise ValueError(
            f"{STATE_FEATURE!r} must contain absolute fields {list(STATE_NAMES)!r}; "
            f"got names={list(names)!r}, shape={list(shape)!r}"
        )
    return fps


def _read_episode_rows(dataset_root: Path, episode_index: int) -> list[dict[str, Any]]:
    data_files = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
    if not data_files:
        raise ValueError(f"No LeRobot data files found below {dataset_root / 'data'}")

    rows: list[dict[str, Any]] = []
    columns = [STATE_FEATURE, "episode_index", "frame_index"]
    for data_file in data_files:
        try:
            table = parquet.read_table(data_file, columns=columns)
        except Exception as exc:
            raise ValueError(f"Could not read LeRobot data file: {data_file}") from exc
        rows.extend(
            row for row in table.to_pylist() if int(row["episode_index"]) == episode_index
        )

    if not rows:
        raise ValueError(f"Episode {episode_index} does not exist in {dataset_root}")
    rows.sort(key=lambda row: int(row["frame_index"]))
    return rows


def load_replay_episode(dataset_root: Path, episode_index: int = 0) -> ReplayEpisode:
    """Load and validate one episode as absolute robot target states."""

    dataset_root = Path(dataset_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {dataset_root}")
    if episode_index < 0:
        raise ValueError(f"Episode index must be non-negative, got {episode_index}")

    info = _load_info(dataset_root)
    fps = _validate_state_feature(info)
    rows = _read_episode_rows(dataset_root, episode_index)

    frame_indices = tuple(int(row["frame_index"]) for row in rows)
    expected_indices = tuple(range(len(rows)))
    if frame_indices != expected_indices:
        raise ValueError(
            f"Episode {episode_index} frame_index must be contiguous from 0; "
            f"got first={frame_indices[0]}, last={frame_indices[-1]}"
        )

    states: list[tuple[float, ...]] = []
    for row_index, row in enumerate(rows):
        raw_state = row[STATE_FEATURE]
        if not isinstance(raw_state, (list, tuple)) or len(raw_state) != len(STATE_NAMES):
            raise ValueError(
                f"Episode {episode_index}, frame {row_index} must contain "
                f"{len(STATE_NAMES)} state values"
            )
        state = tuple(float(value) for value in raw_state)
        if not all(math.isfinite(value) for value in state):
            raise ValueError(
                f"Episode {episode_index}, frame {row_index} contains a non-finite value"
            )
        states.append(state)

    return ReplayEpisode(
        fps=fps,
        episode_index=episode_index,
        frame_indices=frame_indices,
        states=tuple(states),
    )


def state_to_robot_action(state: Sequence[float]) -> dict[str, float]:
    """Map one absolute dataset state to the xArm absolute command format."""

    if len(state) != len(STATE_NAMES):
        raise ValueError(f"Expected {len(STATE_NAMES)} state values, got {len(state)}")
    values = tuple(float(value) for value in state)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Robot state contains a non-finite value")
    # These values are absolute positions. Do not subtract the previous state.
    return dict(zip(STATE_NAMES, values, strict=True))


def replay_episode(robot: Any, episode: ReplayEpisode, show_progress: bool = True) -> None:
    """Send every absolute state once at the dataset FPS."""

    period_s = 1.0 / episode.fps
    next_deadline = time.perf_counter()
    total_frames = len(episode.states)

    for frame_number, state in enumerate(episode.states, start=1):
        robot.send_action(state_to_robot_action(state))
        if show_progress:
            print(f"\rReplaying frame {frame_number}/{total_frames}", end="", flush=True)

        next_deadline += period_s
        precise_sleep(max(next_deadline - time.perf_counter(), 0.0))

    if show_progress:
        print()


def _build_robot(robot_ip: str) -> UFRobot:
    config = UFRobotConfig(
        id="xarm7_replay_robot",
        robot_ip=robot_ip,
        robot_dof=7,
        control_space="joint",
        joint_command_mode=6,
        gripper_type=1,
        manual_mode=False,
        cameras={},
    )
    return UFRobot(config)


def _confirm_start(episode: ReplayEpisode, robot_ip: str, skip_confirmation: bool) -> bool:
    first_state = state_to_robot_action(episode.states[0])
    last_state = state_to_robot_action(episode.states[-1])
    duration_s = (len(episode.states) - 1) / episode.fps
    print(f"Robot: xArm7 at {robot_ip}")
    print(
        f"Episode {episode.episode_index}: {len(episode.states)} frames, "
        f"{episode.fps:g} FPS, about {duration_s:.2f} seconds"
    )
    print(f"First absolute state: {first_state}")
    print(f"Last absolute state:  {last_state}")
    print("The robot will first move to its xArm SDK initial point.")
    print("After replay it will stop at the last state and disconnect.")

    if skip_confirmation:
        return True
    if not sys.stdin.isatty():
        raise RuntimeError("Interactive confirmation is required; use --yes to continue")
    return input("Type 'yes' to connect and start replay: ").strip().lower() == "yes"


def _disconnect_quietly(robot: UFRobot) -> None:
    if getattr(robot, "real_arm", None) is None:
        return
    try:
        robot.disconnect()
    except Exception as exc:
        print(f"Warning: failed to disconnect robot cleanly: {exc}", file=sys.stderr)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay absolute observation.state joint positions on an xArm7."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"LeRobot dataset root (default: {DEFAULT_DATASET_ROOT})",
    )
    parser.add_argument(
        "--robot-ip",
        default=DEFAULT_ROBOT_IP,
        help=f"xArm controller IP (default: {DEFAULT_ROBOT_IP})",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="Episode to replay (default: 0)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation before connecting to the robot",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        episode = load_replay_episode(args.dataset_root, args.episode_index)
        if not _confirm_start(episode, args.robot_ip, args.yes):
            print("Replay cancelled before connecting to the robot.")
            return 0

        robot = _build_robot(args.robot_ip)
        try:
            robot.connect()
            print("Robot connected and moved to the SDK initial point.")
            replay_episode(robot, episode)
            print("Replay complete. The robot will remain at the last state.")
        finally:
            _disconnect_quietly(robot)
    except KeyboardInterrupt:
        print("\nReplay interrupted by user.", file=sys.stderr)
        return 130
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Replay failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
