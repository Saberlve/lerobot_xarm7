import sys
import argparse
import atexit
import csv
import logging
import math
import time
from pathlib import Path
from dataclasses import asdict, dataclass
from datetime import datetime
from pprint import pformat
import lerobot_robot_ufactory # patch
from lerobot.processor import (
    make_default_processors,
)
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    make_robot_from_config,
)
from lerobot.teleoperators import (  # noqa: F401
    TeleoperatorConfig,
    make_teleoperator_from_config,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import (
    init_logging,
)
from lerobot_robot_ufactory.configs import parser
from lerobot_robot_ufactory.utils.utils import is_headless, init_keyboard_listener
from lerobot_robot_ufactory.teleoperators.base_teleop import UFBaseTeleop
from lerobot_robot_ufactory.utils.realtime_teleop import RealtimeTeleopController


@dataclass
class TeleopConfig:
    robot: RobotConfig
    teleop: TeleoperatorConfig
    fps: int = 30
    guard_latency_experiment: bool = False
    experiment_duration_s: float = 60.0
    timing_log_dir: str = "logs"

    def __post_init__(self):
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if not math.isfinite(self.experiment_duration_s) or self.experiment_duration_s <= 0:
            raise ValueError("experiment_duration_s must be finite and positive")
        if hasattr(self.robot, 'robots'):
            for _, robot in self.robot.robots.items():
                robot.cameras = {}
        else:
            self.robot.cameras = {}


@dataclass
class GuardLatencyTiming:
    iteration: int
    elapsed_s: float
    period_ms: float | None
    gello_read_ms: float
    safety_guard_ms: float
    guard_path: str
    servo_j_ms: float
    send_action_ms: float
    work_ms: float
    cycle_ms: float


def _percentile(values: list[float], percentile: float) -> float:
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile / 100
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _write_guard_latency_timings(
    samples: list[GuardLatencyTiming], log_dir: str, fps: int
) -> Path:
    output_dir = Path(log_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"gello_guard_latency_{timestamp}.csv"
    fieldnames = list(GuardLatencyTiming.__dataclass_fields__)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            writer.writerow(asdict(sample))

    target_ms = 1000 / fps
    period_values = [sample.period_ms for sample in samples if sample.period_ms is not None]
    overruns = sum(sample.work_ms > target_ms for sample in samples)
    logging.info("Guard latency timing written to %s", output_path)
    for name, values in (
        ("loop period", period_values),
        ("GELLO read", [sample.gello_read_ms for sample in samples]),
        ("safety guard", [sample.safety_guard_ms for sample in samples]),
        ("joint command", [sample.servo_j_ms for sample in samples]),
        ("send_action", [sample.send_action_ms for sample in samples]),
        ("loop work", [sample.work_ms for sample in samples]),
    ):
        logging.info(
            "%s: p50=%.3f ms, p95=%.3f ms, p99=%.3f ms, max=%.3f ms",
            name,
            _percentile(values, 50),
            _percentile(values, 95),
            _percentile(values, 99),
            max(values, default=float("nan")),
        )
    path_counts = {}
    for sample in samples:
        path_counts[sample.guard_path] = path_counts.get(sample.guard_path, 0) + 1
    logging.info("guard paths: %s", path_counts)
    for path in sorted(path_counts):
        path_samples = [sample for sample in samples if sample.guard_path == path]
        guard_values = [sample.safety_guard_ms for sample in path_samples]
        send_values = [sample.send_action_ms for sample in path_samples]
        path_overruns = sum(sample.work_ms > target_ms for sample in path_samples)
        logging.info(
            "guard path %s: n=%d, guard p50/p95/p99=%.3f/%.3f/%.3f ms, "
            "send p50/p95/p99=%.3f/%.3f/%.3f ms, overruns=%d",
            path,
            len(path_samples),
            _percentile(guard_values, 50),
            _percentile(guard_values, 95),
            _percentile(guard_values, 99),
            _percentile(send_values, 50),
            _percentile(send_values, 95),
            _percentile(send_values, 99),
            path_overruns,
        )
    logging.info(
        "deadline overruns (> %.3f ms work): %d/%d (%.2f%%)",
        target_ms,
        overruns,
        len(samples),
        100 * overruns / len(samples) if samples else 0,
    )
    return output_path


def _validate_guard_latency_config(cfg: TeleopConfig) -> None:
    raise ValueError(
        "guard_latency_experiment is retired: joint-space control uses xArm mode 6"
    )


def teleop_loop(cfg: TeleopConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    if cfg.guard_latency_experiment:
        _validate_guard_latency_config(cfg)
        if hasattr(cfg.robot, "enable_logs") and not cfg.robot.enable_logs:
            raise ValueError(
                "Guard latency experiment requires robot.enable_logs=true"
            )
        logging.warning(
            "Guard latency experiment is retired for mode-6 joint control "
            "(requested duration %.1f seconds)",
            cfg.experiment_duration_s,
        )

    teleop = make_teleoperator_from_config(cfg.teleop)
    if hasattr(cfg.robot, "teleop"):
        cfg.robot.teleop = teleop
    robot = make_robot_from_config(cfg.robot)

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    robot_connected = False
    teleop_connected = False
    listener = None
    cleanup_done = False

    def cleanup_connections():
        nonlocal cleanup_done
        if cleanup_done:
            return
        cleanup_done = True
        if teleop_connected:
            try:
                teleop.disconnect()
            except Exception:
                logging.exception("Failed to disconnect teleoperator cleanly")
        if robot_connected:
            try:
                robot.disconnect()
            except Exception:
                logging.exception("Failed to disconnect robot cleanly")
        if listener is not None:
            try:
                listener.stop()
            except Exception:
                logging.exception("Failed to stop keyboard listener cleanly")

    atexit.register(cleanup_connections)
    robot.connect()
    robot_connected = True
    teleop.connect()
    teleop_connected = True
    if getattr(teleop.config, "gripper_control_mode", "gello") == "keyboard":
        speed, stroke = robot.get_gripper_motion_parameters()
        teleop.set_gripper_motion_parameters(speed, stroke)

    sleep_time_s = 1 / cfg.fps

    is_evt = not is_headless()
    is_uf_teleop = isinstance(teleop, UFBaseTeleop)

    def reset_uf_control():
        if is_uf_teleop:
            # Stop teleop output before handing control to the xArm reset motion.
            teleop.set_teleop_enabled(False)
        reset = getattr(robot, "reset_to_initial", None)
        if reset is None:
            reset = robot.configure
        reset()
        if is_uf_teleop:
            obs = robot.get_observation()
            teleop.set_teleop_enabled(True, obs)

    is_reset = is_uf_teleop
    is_paused = True
    events = {"exit": False}
    key_dict = {}

    if is_evt:
        from pynput import keyboard

        key_dict = {
            keyboard.Key.esc: 0,    # exit
            keyboard.Key.left: 0,   # reset and pause
            keyboard.Key.space: 0,  # start/pause
            keyboard.Key.enter: 0,  # help
        }
        gripper_keys = {"close": False, "open": False}

        def on_press(key):
            char = getattr(key, "char", None)
            if char in ("c", "C"):
                gripper_keys["close"] = True
            elif char in ("o", "O"):
                gripper_keys["open"] = True
            teleop.set_gripper_keyboard_state(**gripper_keys)
            if key_dict.get(key, 1) == 0:
                try:
                    if key == keyboard.Key.esc:
                        events["exit"] = True
                        print("\nEscape key pressed. Stopping ...")
                except Exception as e:
                    print(f"Error handling key press: {e}")
            if key in key_dict:
                key_dict[key] = True

        def on_release(key):
            char = getattr(key, "char", None)
            if char in ("c", "C"):
                gripper_keys["close"] = False
            elif char in ("o", "O"):
                gripper_keys["open"] = False
            teleop.set_gripper_keyboard_state(**gripper_keys)
            try:
                if key == keyboard.Key.enter:
                    if is_paused:
                        if is_reset:
                            print('⌨   [ESC] Exit  [Space] Reset / Start  [←] Reset')
                        else:
                            print('⌨   [ESC] Exit  [Space] Start  [←] Reset')
                    else:
                        print('⌨   [ESC] Exit  [Space] Pause  [←] Pause / Reset')
            except Exception as e:
                print(f"Error handling key release: {e}")
            if key in key_dict:
                key_dict[key] = False

        listener, events = init_keyboard_listener(events=events, on_press=on_press, on_release=on_release)
        print("\n********** Teleop Control Loop Start **********")
        if is_uf_teleop:
            controls = '[ESC] Exit  [Space] Reset / Start  [←] Reset'
            if getattr(teleop.config, "gripper_control_mode", "gello") == "keyboard":
                controls += '  [C] Close  [O] Open'
            print(f'⌨   {controls}')
        else:
            print('⌨   [ESC] Exit  [Space] Start  [←] Reset')
    else:
        input('⌨   Press Enter to start teleop >>> ')
        if is_uf_teleop:
            reset_uf_control()
        is_paused = False
        is_reset = False
        print("\n********** Teleop Control Loop Start **********")

    key_space_pressed = False
    key_left_pressed = False
    latency_samples: list[GuardLatencyTiming] = []
    experiment_start_t = None
    previous_command_t = None
    realtime_controller = None
    realtime_control_fps = int(teleop.config.realtime_control_fps)

    def start_realtime_controller():
        nonlocal realtime_controller
        if (
            cfg.guard_latency_experiment
            or not is_uf_teleop
            or getattr(robot, "_control_space", None) != "joint"
        ):
            return
        obs = robot.get_realtime_observation()
        realtime_controller = RealtimeTeleopController(
            robot,
            teleop,
            teleop_action_processor,
            robot_action_processor,
            realtime_control_fps,
            obs,
        )
        realtime_controller.start()

    def stop_realtime_controller():
        nonlocal realtime_controller
        if realtime_controller is not None:
            realtime_controller.stop()
            realtime_controller = None

    if not is_evt and not is_paused:
        start_realtime_controller()

    while not events["exit"]:
        start_loop_t = time.perf_counter()

        if is_evt:
            if key_dict[keyboard.Key.left] and not key_left_pressed:
                key_left_pressed = True
                is_reset = True
                if not is_paused:
                    is_paused = True
                    stop_realtime_controller()
                    if is_uf_teleop:
                        teleop.set_teleop_enabled(False)
                print('⌨   [ESC] Exit  [Space] Reset / Start  [←] Reset')
            elif not key_dict[keyboard.Key.left] and key_left_pressed:
                key_left_pressed = False

            if key_dict[keyboard.Key.space] and not key_space_pressed:
                key_space_pressed = True
                is_paused = not is_paused
                if is_paused:
                    stop_realtime_controller()
                    if is_uf_teleop:
                        teleop.set_teleop_enabled(False)
                    # print('========== Teleop is paused ==========')
                    print('⌨   [ESC] Exit  [Space] Start  [←] Reset')
                else:
                    if is_reset:
                        reset_uf_control()
                        is_reset = False
                    # print('========== Teleop is start ==========')
                    elif is_uf_teleop:
                        obs = robot.get_observation()
                        teleop.set_teleop_enabled(True, obs)
                    start_realtime_controller()
                    print('⌨   [ESC] Exit  [Space] Pause  [←] Reset')
                continue
            elif not key_dict[keyboard.Key.space] and key_space_pressed:
                key_space_pressed = False

            if is_reset or is_paused:
                continue

        if cfg.guard_latency_experiment:
            if experiment_start_t is None:
                experiment_start_t = start_loop_t
            period_ms = None
            if previous_command_t is not None:
                period_ms = (start_loop_t - previous_command_t) * 1e3
            previous_command_t = start_loop_t

            read_start_t = time.perf_counter()
            act = teleop.get_action()
            read_end_t = time.perf_counter()
            robot.send_action(act)
            send_end_t = time.perf_counter()
            robot_logs = getattr(robot, "logs", {})
            work_s = send_end_t - start_loop_t
            precise_sleep(max(sleep_time_s - work_s, 0.0))
            cycle_end_t = time.perf_counter()
            latency_samples.append(
                GuardLatencyTiming(
                    iteration=len(latency_samples),
                    elapsed_s=start_loop_t - experiment_start_t,
                    period_ms=period_ms,
                    gello_read_ms=(read_end_t - read_start_t) * 1e3,
                    safety_guard_ms=float(robot_logs.get("safety_guard_dt_s", float("nan"))) * 1e3,
                    guard_path=str(robot_logs.get("safety_guard_path", "unknown")),
                    servo_j_ms=float(robot_logs.get("servo_j_dt_s", float("nan"))) * 1e3,
                    send_action_ms=(send_end_t - read_end_t) * 1e3,
                    work_ms=work_s * 1e3,
                    cycle_ms=(cycle_end_t - start_loop_t) * 1e3,
                )
            )
            if cycle_end_t - experiment_start_t >= cfg.experiment_duration_s:
                events["exit"] = True
        else:
            if realtime_controller is not None:
                realtime_controller.heartbeat()
                realtime_controller.raise_if_failed()
                precise_sleep(sleep_time_s)
            else:
                # Generic non-UFACTORY teleoperators retain the standard loop.
                obs = robot.get_observation()
                act = teleop.get_action()
                act_processed_teleop = teleop_action_processor((act, obs))
                robot_action_to_send = robot_action_processor((act_processed_teleop, obs))
                robot.send_action(robot_action_to_send)
                dt_s = time.perf_counter() - start_loop_t
                precise_sleep(max(sleep_time_s - dt_s, 0.0))
    
    print("\n********** Teleop Control Loop Exit **********")
    stop_realtime_controller()
    if latency_samples:
        output_path = _write_guard_latency_timings(
            latency_samples, cfg.timing_log_dir, realtime_control_fps
        )
        print(f"Guard latency timing log: {output_path}")
    cleanup_connections()
    atexit.unregister(cleanup_connections)

@parser.wrap()
def get_cfg(cfg: TeleopConfig) -> TeleopConfig:
    return cfg

def main():
    parser = argparse.ArgumentParser(description='configuration args')
    args, unknown = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + unknown
    register_third_party_plugins()
    cfg = get_cfg()
    teleop_loop(cfg)


if __name__ == "__main__":
    main()
