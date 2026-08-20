import json

import pyarrow as pa
import pyarrow.parquet as parquet
import pytest

from lerobot_robot_ufactory.scripts import uf_lerobot_replay as replay_module
from lerobot_robot_ufactory.scripts.uf_lerobot_replay import (
    JOINT_STATE_NAMES,
    ReplayEpisode,
    STATE_NAMES,
    load_replay_episode,
    replay_episode,
    state_to_robot_action,
)


def _write_dataset(tmp_path, states, episode_indices=None, frame_indices=None, names=None):
    dataset_root = tmp_path / "dataset"
    (dataset_root / "data" / "chunk-000").mkdir(parents=True)
    (dataset_root / "meta").mkdir()

    episode_indices = episode_indices or [0] * len(states)
    frame_indices = frame_indices or list(range(len(states)))
    names = names or list(STATE_NAMES)
    info = {
        "fps": 30,
        "features": {
            "observation.state": {
                "dtype": "float32",
                "names": names,
                "shape": [len(names)],
            }
        },
    }
    (dataset_root / "meta" / "info.json").write_text(
        json.dumps(info), encoding="utf-8"
    )
    table = pa.table(
        {
            "observation.state": states,
            "episode_index": episode_indices,
            "frame_index": frame_indices,
        }
    )
    parquet.write_table(table, dataset_root / "data" / "chunk-000" / "file-000.parquet")
    return dataset_root


def test_load_replay_episode_sorts_and_maps_absolute_states(tmp_path):
    state_a = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, 1.00375]
    state_b = [0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8, 0.4]
    dataset_root = _write_dataset(
        tmp_path,
        states=[state_b, state_a, [9.0] * 8],
        episode_indices=[0, 0, 1],
        frame_indices=[1, 0, 0],
    )

    episode = load_replay_episode(dataset_root, episode_index=0)

    assert episode.fps == 30
    assert episode.frame_indices == (0, 1)
    assert episode.states == (tuple(state_a), tuple(state_b))
    assert state_to_robot_action(state_a) == dict(zip(STATE_NAMES, state_a, strict=True))


def test_replay_sends_absolute_values_without_delta_accumulation(monkeypatch):
    class FakeRobot:
        def __init__(self):
            self.actions = []

        def send_action(self, action):
            self.actions.append(action.copy())

    monkeypatch.setattr(replay_module, "precise_sleep", lambda _: None)
    state_a = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    state_b = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 0.9)
    episode = ReplayEpisode(
        fps=30,
        episode_index=0,
        frame_indices=(0, 1),
        states=(state_a, state_b),
    )
    robot = FakeRobot()

    replay_episode(robot, episode, show_progress=False)

    assert robot.actions == [
        dict(zip(STATE_NAMES, state_a, strict=True)),
        dict(zip(STATE_NAMES, state_b, strict=True)),
    ]
    assert robot.actions[1]["J1.pos"] == state_b[0]


def test_ufactory_robot_sends_absolute_radian_joint_targets(monkeypatch):
    from lerobot_robot_ufactory.robots.uf_robot import uf_robot as uf_robot_module
    from lerobot_robot_ufactory.robots.uf_robot.uf_robot_config import UFRobotConfig

    class FakeArm:
        error_code = 0
        mode = 6

        def __init__(self):
            self.calls = []

        def set_mode(self, mode):
            self.calls.append(("set_mode", mode))
            self.mode = mode
            return 0

        def set_state(self, state):
            self.calls.append(("set_state", state))
            return 0

        def set_servo_angle(self, **kwargs):
            self.calls.append(("set_servo_angle", kwargs))
            return 0

    monkeypatch.setattr(uf_robot_module.time, "sleep", lambda _: None)
    robot = uf_robot_module.UFRobot(
        UFRobotConfig(
            id="replay-test",
            robot_dof=7,
            control_space="joint",
            gripper_type=0,
            cameras={},
        )
    )
    arm = FakeArm()
    robot.real_arm = arm
    robot._is_connected = True
    target = state_to_robot_action((0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, 0.8))

    robot.send_action(target)

    servo_calls = [kwargs for name, kwargs in arm.calls if name == "set_servo_angle"]
    assert servo_calls == [
        {
            "angle": [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7],
            "speed": 0.2,
            "is_radian": True,
            "wait": True,
        }
    ]


def test_reset_to_initial_enables_robot_before_motion(monkeypatch):
    from lerobot_robot_ufactory.robots.uf_robot import uf_robot as uf_robot_module
    from lerobot_robot_ufactory.robots.uf_robot.uf_robot_config import UFRobotConfig

    class FakeArm:
        mode = 0
        state = 0
        error_code = 0
        warn_code = 0

        def __init__(self):
            self.calls = []

        def motion_enable(self, enable=True):
            self.calls.append(("motion_enable", enable))
            return 0

        def clean_error(self):
            self.calls.append(("clean_error",))
            return 0

        def set_mode(self, mode):
            self.calls.append(("set_mode", mode))
            self.mode = mode
            return 0

        def set_state(self, state):
            self.calls.append(("set_state", state))
            self.state = state
            return 0

        def set_servo_angle(self, **kwargs):
            self.calls.append(("set_servo_angle", kwargs))
            return 0

    monkeypatch.setattr(uf_robot_module.time, "sleep", lambda _: None)
    robot = uf_robot_module.UFRobot(
        UFRobotConfig(
            id="replay-reset-test",
            robot_dof=7,
            control_space="joint",
            gripper_type=0,
            cameras={},
        )
    )
    arm = FakeArm()
    robot.real_arm = arm
    robot._is_connected = True
    robot._initial_point = [0.0] * 7
    robot.configure = lambda: None

    robot.reset_to_initial()

    assert [name for name, *_ in arm.calls] == [
        "motion_enable",
        "clean_error",
        "set_mode",
        "set_state",
        "set_servo_angle",
    ]
    assert arm.calls[0] == ("motion_enable", True)


def test_ufactory_robot_replay_path_sends_absolute_mode6_targets():
    from lerobot_robot_ufactory.robots.uf_robot import uf_robot as uf_robot_module
    from lerobot_robot_ufactory.robots.uf_robot.uf_robot_config import UFRobotConfig

    class FakeArm:
        error_code = 0
        mode = 6
        state = 0
        warn_code = 0

        def __init__(self):
            self.calls = []

        def set_mode(self, mode):
            self.calls.append(("set_mode", mode))
            self.mode = mode
            return 0

        def set_state(self, state):
            self.calls.append(("set_state", state))
            return 0

        def set_servo_angle(self, **kwargs):
            self.calls.append(("set_servo_angle", kwargs))
            return 0

    robot = uf_robot_module.UFRobot(
        UFRobotConfig(
            id="replay-mode6-test",
            robot_dof=7,
            control_space="joint",
            joint_command_mode=6,
            gripper_type=0,
            cameras={},
        )
    )
    arm = FakeArm()
    robot.real_arm = arm
    robot._is_connected = True

    target = state_to_robot_action((0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, 0.8))
    robot.send_action(target)

    assert arm.calls == [
        ("set_mode", 0),
        ("set_state", 0),
        (
            "set_servo_angle",
            {
                "angle": [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7],
                "speed": 0.2,
                "is_radian": True,
                "wait": True,
            },
        )
    ]


def test_load_replay_episode_rejects_wrong_state_schema(tmp_path):
    states = [[0.0] * 8]
    dataset_root = _write_dataset(tmp_path, states, names=list(JOINT_STATE_NAMES) + ["wrong"])

    with pytest.raises(ValueError, match="absolute fields"):
        load_replay_episode(dataset_root)


def test_load_replay_episode_rejects_non_contiguous_frames(tmp_path):
    dataset_root = _write_dataset(tmp_path, [[0.0] * 8, [1.0] * 8], frame_indices=[0, 2])

    with pytest.raises(ValueError, match="contiguous"):
        load_replay_episode(dataset_root)


def test_state_to_robot_action_rejects_non_finite_values():
    state = [0.0] * 7 + [float("nan")]

    with pytest.raises(ValueError, match="non-finite"):
        state_to_robot_action(state)
