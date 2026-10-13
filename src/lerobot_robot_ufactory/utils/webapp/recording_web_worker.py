"""Isolated hardware worker (spawned only after an explicit web launch)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import struct
import sys
import time
import traceback

from lerobot_robot_ufactory.utils.webapp.recording_control import RecordingControl


def worker_main(project, folder, session_id, connection, images, options):
    os.chdir(project)
    folder = Path(folder)
    log = (folder / "console.log").open("a", buffering=1, encoding="utf-8")
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    sys.stdout = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
    sys.stderr = os.fdopen(os.dup(2), "w", buffering=1, encoding="utf-8")
    control = RecordingControl(session_id, connection)
    preview = None
    try:
        from lerobot_robot_ufactory.utils.webapp.recording_web_config import (
            validate_text,
        )
        raw, cfg = validate_text((folder / "effective.yaml").read_text())
        if options["dataset_mode"] == "postprocess":
            from lerobot_robot_ufactory.scripts.uf_lerobot_record import record

            control.state["operation"] = "postprocess"
            control.transition("postprocessing", stage="Postprocessing saved episodes")
            cfg.resume = True
            cfg.display_data = False
            cfg.play_sounds = False
            control.prepare_dataset = lambda: prepare_web_postprocess(project, cfg, raw, options)
            dataset = record(cfg, postprocess_only=True, recording_control=control)
            control.transition("finished", saved=dataset.num_episodes, has_unsaved=False,
                               stage="Postprocessing complete")
            return
        from lerobot_robot_ufactory.utils.webapp.web_preview import RecordingWebPreview
        camera_types = {name: camera["type"] for name, camera in raw["robot"].get("cameras", {}).items()}
        preview = RecordingWebPreview(cfg.web_preview, camera_types=camera_types)

        def publish_jpeg(name, jpeg, timestamp):
            header = json.dumps({"camera": name, "timestamp": timestamp,
                                 "session_id": session_id}).encode()
            images.send_bytes(struct.pack("!I", len(header)) + header + jpeg)

        preview.jpeg_sink = publish_jpeg
        preview.set_subscriptions(options.get("cameras", []))
        control.preview = preview
        if options.get("simulate"):
            _simulate(cfg, control, preview, folder)
        else:
            from lerobot_robot_ufactory.scripts.uf_lerobot_record import record
            cfg.resume = options["dataset_mode"] == "resume"
            cfg.play_sounds = False
            cfg.display_data = False
            cfg.web_preview.enabled = True

            control.prepare_dataset = lambda: prepare_web_dataset(project, cfg, raw, options)
            record(cfg, recording_control=control)
        stage = "Postprocessing complete" if control.state["phase"] == "postprocessing" else "Devices released"
        control.transition("finished", has_unsaved=False, joint_mode="all", precision_active=False, stage=stage)
    except BaseException as exc:
        traceback.print_exc()
        control.transition("error", error=str(exc), stage="Session failed; see console log")
    finally:
        if preview is not None:
            preview.stop()
        control.close()
        images.close()
        connection.close()
        log.close()


def prepare_web_dataset(project, cfg, raw, options):
    """Called under the shared recording lock, before device construction."""
    from lerobot_robot_ufactory.utils.webapp.recording_web_config import dataset_stamp, dataset_status
    root = Path(cfg.dataset.root).resolve()
    if dataset_stamp(root) != options["dataset_stamp"]:
        raise RuntimeError("Dataset changed since confirmation; launch again")
    if options["dataset_mode"] == "resume":
        status = dataset_status(project, raw)
        if not status["resumable"]:
            raise RuntimeError(status["reason"] or "Cannot resume dataset")
    elif root.exists():
        if options["dataset_mode"] != "rebuild":
            raise RuntimeError("Existing dataset requires resume/rebuild selection")
        allowed = (Path(project) / "datasets").resolve()
        if root == allowed or not root.is_relative_to(allowed):
            raise RuntimeError("Rebuild is restricted to a child of project/datasets")
        shutil.rmtree(root)
        print(f"Previous dataset removed; rebuilding at {root}", flush=True)


def prepare_web_postprocess(project, cfg, raw, options):
    """Recheck the selected dataset under the recording lock without rebuilding it."""
    from lerobot_robot_ufactory.utils.webapp.recording_web_config import dataset_stamp, postprocess_status

    root = (Path(project) / cfg.dataset.root).resolve()
    if dataset_stamp(root) != options["dataset_stamp"]:
        raise RuntimeError("Dataset changed before postprocessing; refresh and launch again")
    status = postprocess_status(project, raw)
    if not status["ready"]:
        raise RuntimeError(status["reason"] or "No raw episodes need postprocessing")


def _simulate(cfg, control, preview, folder):
    """UI acceptance mode: synthetic cameras and records in the session folder only."""
    import numpy as np

    from types import SimpleNamespace

    class Teleop:
        config = cfg.teleop
        _keyboard_j7_active = False
        _keyboard_j7_until = 0.0
        _joint7_only_active = False
        _joint_precision_active = False
        _xy_plane_mode = SimpleNamespace(active=False)
        def set_xy_plane_mode_key(self, pressed):
            if pressed:
                self._xy_plane_mode.active = not self._xy_plane_mode.active
        def set_gripper_keyboard_state(self, **keys):
            pass
        def keyboard_joint7_busy(self):
            return time.monotonic() < self._keyboard_j7_until
        def set_joint7_keyboard_key(self, direction, pressed, *, modifier="none"):
            if self.config.joint7_keyboard_hold_to_rotate:
                if not pressed:
                    self._keyboard_j7_until = 0.0
                    return False
                if self.config.joint7_keyboard_mode_enabled:
                    self._keyboard_j7_active = True
                    self._joint7_only_active = False
                    self._keyboard_j7_until = float("inf")
                    return True
                return False
            if pressed and self.config.joint7_keyboard_mode_enabled and not self.keyboard_joint7_busy():
                self._keyboard_j7_active = True
                self._joint7_only_active = False
                self._keyboard_j7_until = time.monotonic() + {"ctrl": 10.0, "shift": 60.0}.get(modifier, self.config.joint7_keyboard_angle_deg) / self.config.joint7_keyboard_speed_deg_s
                return True
            return False
        def set_joint7_mode_key(self, pressed):
            if pressed:
                if self._keyboard_j7_active:
                    self._keyboard_j7_active = False
                    self._keyboard_j7_until = 0.0
                    self._joint7_only_active = False
                else:
                    self._joint7_only_active = not self._joint7_only_active
        def set_joint_precision_mode_key(self, pressed):
            if pressed and self.config.joint_precision_mode_enabled:
                self._joint_precision_active = not self._joint_precision_active

    teleop = Teleop()
    control.bind(teleop)
    preview.start_encoder()
    count = 0
    camera_at = 0.0

    def frames():
        nonlocal camera_at
        if time.monotonic() < camera_at:
            return
        camera_at = time.monotonic() + 1 / max(cfg.web_preview.fps, 1)
        values = {}
        for i, name in enumerate(cfg.robot.cameras):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            frame[:, :, i % 3] = 70 + int(time.monotonic() * 20) % 180
            values[name] = frame
        preview.publish(values)

    print("SIMULATION: no robot/camera devices or configured datasets will be touched", flush=True)
    control.transition("ready", stage="SIMULATION / synthetic cameras")
    while not control.events["stop_recording"] and count < cfg.dataset.num_episodes:
        action = control.wait_action(frames)
        if action != "start":
            break
        control.events.update(exit_early=False, rerecord_episode=False, pause_recording=False)
        teleop._keyboard_j7_active = False
        teleop._keyboard_j7_until = 0.0
        teleop._joint7_only_active = False
        teleop._joint_precision_active = False
        teleop._xy_plane_mode.active = False
        control.transition("recording", frames=0, elapsed=0, stage="SIMULATION recording")
        started = time.monotonic()
        while not control.events["exit_early"]:
            control.watchdog()
            elapsed = time.monotonic() - started
            frames()
            control.progress(frames=int(elapsed * cfg.dataset.fps), elapsed=elapsed, has_unsaved=True)
            if elapsed >= cfg.dataset.episode_time_s:
                break
            time.sleep(0.02)
        control.release_keys()
        teleop._keyboard_j7_active = False
        teleop._keyboard_j7_until = 0.0
        teleop._joint7_only_active = False
        teleop._joint_precision_active = False
        teleop._xy_plane_mode.active = False
        decision = "discard" if control.events["rerecord_episode"] else "save"
        if control.events["pause_recording"]:
            control.transition("paused", has_unsaved=True, joint_mode="all", precision_active=False,
                               stage="Disconnected; save or discard")
            decision = control.wait_action(frames)
        if decision == "exit" or control.events["stop_recording"]:
            break
        if decision == "save":
            control.transition("saving", stage="SIMULATION saving")
            count += 1
            (folder / f"simulated_episode_{count:06d}.json").write_text(json.dumps(dict(control.state)))
            print(f"[Finish] Simulated save episode {count - 1}", flush=True)
        else:
            control.transition("resetting", stage="SIMULATION reset (no motion)")
        control.events.update(exit_early=False, rerecord_episode=False, pause_recording=False)
        control.transition("ready", saved=count, episode=count, frames=0, elapsed=0,
                           has_unsaved=False, joint_mode="all", precision_active=False,
                           stage="SIMULATION ready")
