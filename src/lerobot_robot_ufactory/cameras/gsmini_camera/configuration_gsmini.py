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

from dataclasses import dataclass

from lerobot.cameras.configs import CameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig


@CameraConfig.register_subclass("uf::gsmini_camera")
@dataclass
class GsminiCameraConfig(OpenCVCameraConfig):
    """OpenCV camera config for GelSight Mini tactile sensors.

    Behaves like a regular OpenCVCamera, plus the border crop used by the
    GelSight Mini SDK demos (`crop_and_resize`) so the recorded view matches
    the SDK live view. Multiple sensors are supported by adding one config
    entry per sensor, each pointing at its own device (prefer stable
    `/dev/v4l/by-id/...` paths over numeric indices).
    """

    border_fraction: float = 0.15

    def __post_init__(self) -> None:
        super().__post_init__()
        # Clamp to the same range as the GelSight Mini SDK.
        self.border_fraction = min(max(0.0, self.border_fraction), 0.49)
