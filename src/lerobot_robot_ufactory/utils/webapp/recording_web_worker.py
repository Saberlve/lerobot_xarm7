"""Isolated hardware worker (spawned only after an explicit web launch)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import struct
import sys
import time
import traceback
from uuid import uuid4

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
        from lerobot_robot_ufactory.utils.webapp.web_preview import RecordingWebPreview
        raw, cfg = validate_text((folder / "effective.yaml").read_text())
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
        control.transition("finished", has_unsaved=False, joint_mode="all", stage="Devices released")
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
        archive = root.parent / ".web-dataset-trash" / f"{root.name}-{uuid4().hex}"
        archive.parent.mkdir(parents=True, exist_ok=True)
        root.replace(archive)
        print(f"Previous dataset archived at {archive}", flush=True)


def _simulate(cfg, control, preview, folder):
    """UI acceptance mode: synthetic cameras and records in the session folder only."""
    import numpy as np

    class Teleop:
        config = cfg.teleop
        _joint7_only_active = False
        def set_gripper_keyboard_state(self, **keys):
            pass
        def set_joint7_mode_key(self, pressed):
            if pressed:
                self._joint7_only_active = not self._joint7_only_active

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
        teleop._joint7_only_active = False
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
        teleop._joint7_only_active = False
        decision = "discard" if control.events["rerecord_episode"] else "save"
        if control.events["pause_recording"]:
            control.transition("paused", has_unsaved=True, joint_mode="all", stage="Disconnected; save or discard")
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
                           has_unsaved=False, joint_mode="all", stage="SIMULATION ready")
