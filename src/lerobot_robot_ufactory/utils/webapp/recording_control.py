"""Browser control mailbox, independent of GUI/keyboard libraries and robot I/O."""
from __future__ import annotations

from collections import deque
import queue
import threading
import time


class RecordingControl:
    def __init__(self, session_id, connection=None, send=None, clock=time.monotonic):
        self.session_id = session_id
        self.connection = connection
        self.send = send or (connection.send if connection is not None else lambda value: None)
        self.clock = clock
        self.lock = threading.RLock()
        self.send_lock = threading.RLock()
        self.events = {"exit_early": False, "rerecord_episode": False,
                       "stop_recording": False, "pause_recording": False}
        self.keys = {"close": False, "open": False}
        self.state = {"session_id": session_id, "phase": "initializing", "version": 0,
                      "episode": 0, "saved": 0, "frames": 0, "elapsed": 0.0,
                      "joint_mode": "all", "gripper_mode": "keyboard",
                      "j7_enabled": False, "error": None, "has_unsaved": False,
                      "stage": "Connecting devices"}
        self.actions = queue.Queue(maxsize=32)
        self._seen = deque(maxlen=1024)
        self._last_heartbeat = clock()
        self._last_progress = 0.0
        self._closed = threading.Event()
        self.teleop = None
        self.realtime_controller = None
        self.connected = True
        self.preview = None
        self._reader = None
        self._sender = None
        self._outgoing = queue.Queue(maxsize=64)
        if connection is not None:
            self._sender = threading.Thread(target=self._send_messages, daemon=True)
            self._sender.start()
            self._reader = threading.Thread(target=self._receive, daemon=True)
            self._reader.start()

    def emit(self, value):
        if self.connection is not None:
            with self.lock:
                try:
                    self._outgoing.put_nowait(value)
                except queue.Full:
                    try:
                        self._outgoing.get_nowait()
                    except queue.Empty:
                        pass
                    self._outgoing.put_nowait(value)
            return
        self._send_one(value)

    def _send_one(self, value):
        with self.send_lock:
            try:
                self.send(value)
            except (OSError, EOFError, BrokenPipeError):
                self.disconnect()

    def _send_messages(self):
        while not self._closed.is_set() or not self._outgoing.empty():
            try:
                value = self._outgoing.get(timeout=0.1)
            except queue.Empty:
                continue
            self._send_one(value)

    def transition(self, phase, **values):
        with self.lock:
            self.state.update(values)
            self.state["phase"] = phase
            self.state["version"] += 1
            snapshot = dict(self.state)
        self.emit({"type": "state", "state": snapshot})

    def progress(self, **values):
        with self.lock:
            self.state.update(values)
            if self.teleop is not None:
                mode = "j7" if getattr(self.teleop, "_joint7_only_active", False) else "all"
                if mode != self.state["joint_mode"]:
                    self.state["joint_mode"] = mode
                    self.state["version"] += 1
            now = self.clock()
            if now - self._last_progress < 0.2:
                return
            self._last_progress = now
            snapshot = dict(self.state)
        self.emit({"type": "state", "state": snapshot})

    def bind(self, teleop):
        self.teleop = teleop
        cfg = getattr(teleop, "config", None)
        self.state.update(gripper_mode=getattr(cfg, "gripper_control_mode", "keyboard"),
                          j7_enabled=getattr(cfg, "joint7_only_mode_enabled", False))

    def release_keys(self):
        with self.lock:
            self.keys.update(close=False, open=False)
            if self.teleop is not None:
                setter = getattr(self.teleop, "set_gripper_keyboard_state", None)
                if setter:
                    setter(close=False, open=False)
                setter = getattr(self.teleop, "set_joint7_mode_key", None)
                if setter:
                    setter(False)

    def disconnect(self):
        with self.lock:
            self.connected = False
            self.release_keys()
            if self.state["phase"] in ("recording", "resetting"):
                self.events["pause_recording"] = True
                self.events["exit_early"] = True
                if self.realtime_controller is not None:
                    self.realtime_controller.request_pause()

    def heartbeat(self):
        with self.lock:
            self._last_heartbeat = self.clock()
            self.connected = True

    def watchdog(self):
        if self.clock() - self._last_heartbeat > 2.0:
            self.disconnect()

    def command(self, message):
        action = message.get("action")
        if action == "heartbeat":
            self.heartbeat()
            return
        if action == "disconnect":
            self.disconnect()
            return
        if action == "preview":
            if self.preview is not None:
                self.preview.set_subscriptions(message.get("cameras", []))
            return
        request_id = message.get("request_id")
        error = None
        with self.lock:
            try:
                if message.get("session_id") != self.session_id:
                    raise ValueError("Session expired")
                self.heartbeat()
                if not isinstance(request_id, str) or not request_id or request_id in self._seen:
                    raise ValueError("Duplicate or missing request identifier")
                self._seen.append(request_id)
                release = action == "release" or (action == "gripper" and not message.get("pressed"))
                if not release and message.get("version") != self.state["version"]:
                    raise ValueError("State changed; retry using the current state")
                phase = self.state["phase"]
                if action == "release":
                    self.release_keys()
                elif action == "gripper":
                    if message.get("key") not in self.keys or type(message.get("pressed")) is not bool:
                        raise ValueError("Invalid gripper key event")
                    if message["pressed"] and (phase != "recording" or self.state["gripper_mode"] != "keyboard"):
                        raise ValueError("Web gripper control is unavailable")
                    self.keys[message["key"]] = message["pressed"]
                    if self.teleop is not None:
                        self.teleop.set_gripper_keyboard_state(**self.keys)
                elif action == "joint_mode":
                    if phase != "recording" or not self.state["j7_enabled"]:
                        raise ValueError("J7 mode can only be switched during supported recording")
                    self.teleop.set_joint7_mode_key(True)
                    self.teleop.set_joint7_mode_key(False)
                elif action == "start":
                    if phase != "ready":
                        raise ValueError("Start requires ready state")
                    self.transition("resetting", stage="Preparing episode / aligning GELLO")
                    self.actions.put_nowait("start")
                elif action in ("save", "discard"):
                    if phase not in ("recording", "paused"):
                        raise ValueError("No current episode to save/discard")
                    self.release_keys()
                    if phase == "recording":
                        if action == "discard":
                            self.events["rerecord_episode"] = True
                        self.events["exit_early"] = True
                        self.transition("saving" if action == "save" else "resetting",
                                        stage="Finishing capture")
                    else:
                        self.events["pause_recording"] = False
                        self.transition("saving" if action == "save" else "resetting")
                        self.actions.put_nowait(action)
                elif action == "exit":
                    if phase in ("saving", "resetting", "stopping", "finished", "error"):
                        raise ValueError("Wait for the current operation to finish")
                    self.release_keys()
                    self.events["stop_recording"] = True
                    self.events["exit_early"] = True
                    self.transition("stopping", stage="Releasing devices")
                    self.actions.put_nowait("exit")
                else:
                    raise ValueError("Unknown action")
            except (ValueError, queue.Full) as exc:
                error = str(exc)
        self.emit({"type": "ack", "request_id": request_id, "ok": error is None, "error": error})

    def wait_action(self, idle=None):
        while not self.events["stop_recording"]:
            self.watchdog()
            try:
                return self.actions.get(timeout=0.1)
            except queue.Empty:
                if idle:
                    idle()
        return "exit"

    def _receive(self):
        while not self._closed.is_set():
            try:
                if self.connection.poll(0.1):
                    self.command(self.connection.recv())
                self.watchdog()
            except (OSError, EOFError):
                self.disconnect()
                self.events["stop_recording"] = True
                self.events["exit_early"] = True
                return

    def close(self):
        self._closed.set()
        self.release_keys()
        if self._reader:
            self._reader.join(timeout=1)
        if self._sender:
            self._sender.join(timeout=1)


def controlled_recording(
    cfg, robot, teleop, dataset, control, preview,
    teleop_action_processor, robot_action_processor, robot_observation_processor,
    runtime_dir, tactile_cameras,
):
    """Reuse capture and tactile transactions; only the interaction loop differs."""
    from pathlib import Path
    from lerobot_robot_ufactory.scripts import uf_lerobot_record as recording

    control.bind(teleop)
    control.preview = preview
    preview_at = 0.0
    reset_after_discard = False
    synchronization = None

    def idle_preview():
        nonlocal preview_at
        if preview is not None and time.monotonic() >= preview_at:
            preview_at = time.monotonic() + 0.2
            preview.publish(robot_observation_processor(robot.get_observation()))

    def save_current(sync):
        index = recording._current_episode_index(dataset)
        control.transition("saving", stage="Validating images", has_unsaved=True)
        recording.validate_episode_images(dataset, recording._get_episode_buffer(dataset))
        if cfg.offline_mesh3dflow and tactile_cameras:
            from lerobot_robot_ufactory.tactile.deferred import compute_episode_mesh
            control.progress(stage="Computing Mesh3DFlow")
            compute_episode_mesh(dataset, tactile_cameras, runtime_dir, index)
        transactional = sync is not None and sync.tactile_recorder is not None
        if sync is not None and transactional:
            sync.write(Path(dataset.root), index, defer_commit=True)
        control.progress(stage="Saving dataset / encoding video")
        dataset.save_episode()
        if sync is not None:
            if transactional:
                sync.commit()
            else:
                sync.write(Path(dataset.root), index)
        print(f"[Finish] Save episode {index}", flush=True)

    with recording.VideoEncodingManager(dataset), recording._RecordingCleanup(
        robot, teleop, None, None, preview
    ), recording._EpisodeSynchronizationOwner() as owner:
        teleop.set_teleop_enabled(False)
        if dataset.num_episodes >= cfg.dataset.num_episodes:
            return dataset
        control.transition("ready", saved=dataset.num_episodes, episode=dataset.num_episodes,
                           stage="Check previews, then press Space / Start")
        try:
            while not control.events["stop_recording"] and dataset.num_episodes < cfg.dataset.num_episodes:
                action = control.wait_action(idle_preview)
                if action != "start":
                    break
                if not control.connected or control.clock() - control._last_heartbeat > 2.0:
                    control.transition("ready", stage="Reconnect before starting an episode")
                    continue
                control.events.update(exit_early=False, rerecord_episode=False, pause_recording=False)
                control.release_keys()
                try:
                    recording._prepare_recording_episode(
                        robot, teleop, True, False, reset_robot=not reset_after_discard,
                        cancel_check=lambda: control.events["pause_recording"],
                    )
                except InterruptedError:
                    teleop.set_teleop_enabled(False)
                    control.transition("ready", stage="Preparation paused; reconnect to start")
                    continue
                reset_after_discard = False
                if control.events["pause_recording"]:
                    teleop.set_teleop_enabled(False)
                    robot.pause_motion()
                    control.transition("ready", stage="Disconnected during preparation; reconnect to start")
                    continue
                control.transition("recording", frames=0, elapsed=0.0, joint_mode="all",
                                   has_unsaved=False, stage="Recording")
                try:
                    synchronization = recording.record_loop(
                        robot=robot, events=control.events, fps=cfg.dataset.fps,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        teleop=teleop, dataset=dataset,
                        control_time_s=cfg.dataset.episode_time_s,
                        single_task=cfg.dataset.single_task,
                        manual_gripper_keys=control.keys, web_preview=preview,
                        synchronize=cfg.synchronize, synchronization_owner=owner,
                        recording_control=control,
                    )
                except TimeoutError as exc:
                    print(f"Capture timeout; discarding episode: {exc}", flush=True)
                    control.events["rerecord_episode"] = True
                owner.track(synchronization)
                teleop.set_teleop_enabled(False)
                control.release_keys()
                control.progress(joint_mode="all")
                if control.events["stop_recording"]:
                    owner.discard(synchronization)
                    synchronization = None
                    recording._discard_current_episode(dataset)
                    break
                decision = "discard" if control.events["rerecord_episode"] else "save"
                if control.events["pause_recording"]:
                    control.transition("paused", stage="Disconnected: save or discard this episode",
                                       has_unsaved=recording._episode_buffer_size(
                                           recording._get_episode_buffer(dataset)) > 0,
                                       joint_mode="all")
                    decision = control.wait_action(idle_preview)
                if decision == "exit":
                    owner.discard(synchronization)
                    synchronization = None
                    recording._discard_current_episode(dataset)
                    break
                if decision == "discard":
                    owner.discard(synchronization)
                    synchronization = None
                    recording._discard_current_episode(dataset)
                    control.transition("resetting", stage="Opening gripper, then resetting arm")
                    try:
                        recording._reset_recording_robot(robot, open_gripper_first=True,
                            cancel_check=lambda: control.events["pause_recording"])
                        reset_after_discard = True
                    except InterruptedError:
                        reset_after_discard = False
                else:
                    if recording._episode_buffer_size(recording._get_episode_buffer(dataset)) > 0:
                        save_current(synchronization)
                        owner.release(synchronization)
                        synchronization = None
                    else:
                        owner.discard(synchronization)
                        synchronization = None
                        recording._discard_current_episode(dataset)
                control.events.update(exit_early=False, rerecord_episode=False, pause_recording=False)
                if dataset.num_episodes >= cfg.dataset.num_episodes:
                    control.progress(saved=dataset.num_episodes, has_unsaved=False)
                    break
                control.transition("ready", saved=dataset.num_episodes, episode=dataset.num_episodes,
                                   has_unsaved=False, frames=0, elapsed=0.0, joint_mode="all",
                                   stage="Ready for next episode")
        finally:
            control.release_keys()
    return dataset
