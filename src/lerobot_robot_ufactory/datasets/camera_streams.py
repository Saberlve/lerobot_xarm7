"""Keep native-rate camera intervals independent of the action/state row clock.

Tactile images use TacVerse's H.264/YUV420P representation. Mesh3DFlow is
persisted independently from uncompressed images, before any video encoding.
"""

import math
from types import SimpleNamespace

import numpy as np


def camera_recording_plan(robot, dataset_fps, *, use_videos=True):
    """Return prefixed camera rates, including multi-arm cameras."""
    if hasattr(robot, "robots"):
        return {
            name: item
            for child in robot.robots.values()
            for name, item in camera_recording_plan(
                child, dataset_fps, use_videos=use_videos
            ).items()
        }
    plan = {}
    for key, camera in getattr(robot, "cameras", {}).items():
        fps = getattr(camera, "fps", None)
        if fps is None:
            fps = getattr(getattr(camera, "config", None), "fps", None)
        if fps is None:
            continue  # Legacy third-party camera without a rate contract.
        if (
            isinstance(fps, bool)
            or not isinstance(fps, (int, float))
            or not math.isfinite(fps)
            or fps <= 0
        ):
            raise ValueError(f"Camera {key}: fps must be finite and positive")
        name = f"{getattr(robot, 'prefix', '')}{key}"
        tactile = hasattr(camera, "samples_between")
        plan[name] = {
            "fps": fps,
            "tactile": tactile,
            "interval": fps != dataset_fps,
            "storage": "video" if use_videos else "png",
            "codec": "h264" if tactile else None,
            "pix_fmt": "yuv420p" if use_videos else None,
            # TacVerse publishes the codec/pixel format, not its CRF. Use a
            # conservative quality setting rather than claiming an exact match.
            "crf": 18 if tactile else 30,
            "require_mesh": tactile
            and getattr(getattr(camera, "config", None), "motion_3d_output", None) == "Mesh3DFlow",
        }
    return plan


def apply_camera_storage_plan(features, plan):
    """Keep standard LeRobot video fields, including native-rate streams."""
    for name, item in plan.items():
        key = f"observation.images.{name}"
        if key in features:
            features[key]["dtype"] = "video" if item["storage"] == "video" else "image"


def initial_camera_samples(robot, camera_timing, start, stream_names):
    """Copy the exact pre-start representatives before history can be evicted.

    These samples seed the native stream but do not belong to an action window.
    Keep original sensor pixels (and displacement), not resized observation images.
    """
    if hasattr(robot, "robots"):
        return {
            name: sample
            for child in robot.robots.values()
            for name, sample in initial_camera_samples(
                child, camera_timing, start, stream_names
            ).items()
        }
    timings = {
        timing.get("camera_stream_name", timing.get("tactile_stream_name", key)): timing
        for key, timing in camera_timing.items()
    }
    result = {}
    for key, camera in getattr(robot, "cameras", {}).items():
        name = f"{getattr(robot, 'prefix', '')}{key}"
        timing = timings.get(name, {})
        capture_s = timing.get("capture_monotonic_s")
        if name not in stream_names or capture_s is None or capture_s > start:
            continue
        capture_ns = timing.get("capture_monotonic_ns")
        if capture_ns is None:
            capture_ns = round(capture_s * 1e9)
        source = camera if hasattr(camera, "sync_samples") else robot._rgb_sync_buffers[key]
        sample = next((sample for sample in source.sync_samples() if (
            sample.capture_monotonic_ns if sample.capture_monotonic_ns is not None
            else round(sample.capture_monotonic_s * 1e9)
        ) == capture_ns), None)
        if sample is None:
            raise RuntimeError(f"Initial representative frame is no longer available: {name}")
        if hasattr(sample, "frame_bgr"):
            result[name] = SimpleNamespace(
                frame_bgr=sample.frame_bgr.copy(),
                capture_monotonic_s=sample.capture_monotonic_s,
                capture_monotonic_ns=sample.capture_monotonic_ns,
                sensor_timestamp_s=sample.sensor_timestamp_s,
                marker_motion_3d=(None if sample.marker_motion_3d is None
                                  else sample.marker_motion_3d.copy()),
            )
        else:
            result[name] = _rgb_stream_sample(camera, sample)
    return result


def _rgb_stream_sample(camera, sample):
    mode = getattr(getattr(camera, "config", camera), "color_mode", "rgb")
    mode = str(getattr(mode, "value", mode)).lower()
    return SimpleNamespace(
        frame_bgr=np.array(sample.frame[:, :, ::-1] if mode == "rgb" else sample.frame,
                           copy=True, order="C"),
        capture_monotonic_s=sample.capture_monotonic_s,
        capture_monotonic_ns=sample.capture_monotonic_ns,
        sensor_timestamp_s=sample.timing.get("sensor_timestamp_s"),
        timing=sample.timing.copy(),
        marker_motion_3d=None,
    )


def camera_samples_between(robot, start, end, stream_names):
    if hasattr(robot, "robots"):
        return {
            name: samples
            for child in robot.robots.values()
            for name, samples in camera_samples_between(child, start, end, stream_names).items()
        }
    result = {}
    for key, camera in getattr(robot, "cameras", {}).items():
        name = f"{getattr(robot, 'prefix', '')}{key}"
        if name not in stream_names:
            continue
        if hasattr(camera, "samples_between"):
            result[name] = camera.samples_between(
                start, end, getattr(getattr(robot, "config", None), "sync_wait_ms", 0.0)
            )
        else:
            source = robot._rgb_sync_buffers[key]
            samples = source.samples_between(
                start, end, getattr(getattr(robot, "config", None), "sync_wait_ms", 0.0)
            )
            result[name] = tuple(
                _rgb_stream_sample(camera, sample)
                for sample in samples
            )
    return result


def encode_camera_interval(staged, rows, fps, vcodec, relative_root, *, crf=30):
    """Encode full-rate images; publish frame indexes before removing PNGs.

    Video presentation time is index / configured camera fps. Capture time is
    retained separately, so jitter and empty action windows never invent frames.
    """
    import json
    import os
    import shutil
    from fractions import Fraction

    import av
    from lerobot.datasets.video_utils import encode_video_frames

    # LeRobot's encoder expects frame-NNNNNN.png. The interval writer uses
    # frame_NNNNNN.png; hard links avoid copying the image payload.
    inputs = staged / ".encoding"
    inputs.mkdir(exist_ok=True)
    temporary = staged / "video.pending.mp4"
    video = staged / "video.mp4"
    try:
        for index in range(len(rows)):
            destination = inputs / f"frame-{index:06d}.png"
            if not destination.exists():
                os.link(staged / "frames" / f"frame_{index:06d}.png", destination)
        encode_video_frames(
            inputs,
            temporary,
            Fraction(str(fps)),
            vcodec=vcodec,
            pix_fmt="yuv420p",
            crf=crf,
            overwrite=True,
        )
        with av.open(str(temporary)) as container:
            stream = container.streams.video[0]
            rate = float(stream.average_rate)
            codec = stream.codec_context.name
            pix_fmt = stream.codec_context.format.name
            count = sum(1 for _ in container.decode(video=0))
        if (
            count != len(rows)
            or abs(rate - fps) > 1e-6
            or pix_fmt != "yuv420p"
            or (vcodec == "h264" and codec != "h264")
        ):
            raise RuntimeError(f"Native camera video failed validation: frames={count}, fps={rate}")
        os.replace(temporary, video)
        for index, row in enumerate(rows):
            row.update(
                frame_path=None,
                image_encoding=f"{vcodec}_yuv420p",
                video_path=(relative_root / "video.mp4").as_posix(),
                video_frame_index=index,
                video_timestamp_s=index / fps,
            )
        (staged / "video.json").write_text(
            json.dumps(
                {
                    "fps": fps,
                    "frame_count": len(rows),
                    "codec": vcodec,
                    "pix_fmt": pix_fmt,
                    "crf": crf,
                    "lossless": False,
                    "capture_clock": "samples.parquet",
                    "presentation_clock": "video_frame_index / camera_fps",
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        shutil.rmtree(inputs)
        temporary.unlink(missing_ok=True)
