#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from concurrent.futures import ThreadPoolExecutor
from typing import cast
from lerobot.utils.import_utils import make_device_from_device_class
from lerobot.cameras.utils import make_cameras_from_configs as lerobot_make_cameras_from_configs
from lerobot.cameras.camera import Camera
from lerobot.cameras.configs import CameraConfig


def connect_cameras(cameras: dict[str, Camera]) -> None:
    """Overlap RealSense warmup; keep other SDK initialization serialized."""
    if not cameras:
        return
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="uf-rgb-connect") as pool:
        pending = [pool.submit(camera.connect) for camera in cameras.values()
                   if camera.config.type == "intelrealsense"]
        for camera in cameras.values():
            if camera.config.type != "intelrealsense":
                camera.connect()
        # Join every worker before the caller starts readers or handles failure.
        for future in pending:
            future.result()
    if not all(camera.is_connected for camera in cameras.values()):
        raise ConnectionError("Could not connect to the cameras, check that all cameras are plugged-in.")


def make_cameras_from_configs(camera_configs: dict[str, CameraConfig]) -> dict[str, Camera]:
    xense_serials = [
        cfg.serial_number for cfg in camera_configs.values()
        if cfg.type == "photon"
    ]
    if len(xense_serials) != len(set(xense_serials)):
        raise ValueError("Each Xense Photon camera must have a distinct serial_number")
    lerobot_camera_configs = {
        key: cfg for key, cfg in camera_configs.items()
        if not cfg.type.startswith("uf::")
    }
    uf_camera_configs = {key: cfg for key, cfg in camera_configs.items() if cfg.type.startswith("uf::")}
    cameras = lerobot_make_cameras_from_configs(lerobot_camera_configs)
    
    for key, cfg in uf_camera_configs.items():
        if cfg.type == "uf::umi_camera":
            from .umi_camera import UmiCamera
            cameras[key] = UmiCamera(cfg)
        else:
            try:
                cameras[key] = cast(Camera, make_device_from_device_class(cfg))
            except Exception as e:
                raise ValueError(f"Error creating camera with config {cfg}: {e}") from e
    return {key: cameras[key] for key in camera_configs}
