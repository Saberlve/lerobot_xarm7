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

"""GelSight Mini tactile sensor as a LeRobot camera.

The Mini is a plain UVC camera, so connection, background-thread capture and
color handling are inherited from OpenCVCamera. The only addition is the
border crop from the GelSight Mini SDK, applied after OpenCVCamera's color
conversion/rotation, with the result resized back to the configured
width/height — matching what `GelSightMini.update()` produces in
third_party/tactile/GsminiSDK.
"""

import logging
from typing import Any

import cv2
from lerobot.cameras.configs import ColorMode
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from numpy.typing import NDArray

from .configuration_gsmini import GsminiCameraConfig

logger = logging.getLogger(__name__)


def _crop_and_resize(image: NDArray[Any], target_size, border_fraction: float) -> NDArray[Any]:
    """Crop a border fraction from each edge, then resize to target_size (w, h).

    Vendored from third_party/tactile/GsminiSDK/utilities/image_processing.py
    (crop_and_resize) to avoid the SDK's package-relative imports and its
    scipy dependency; behavior is identical.
    """
    border_fraction = min(max(0.0, border_fraction), 0.49)
    border_x = int(image.shape[0] * border_fraction)
    border_y = int(image.shape[1] * border_fraction)
    cropped = image[border_x : image.shape[0] - border_x, border_y : image.shape[1] - border_y]
    if target_size is not None:
        cropped = cv2.resize(cropped, target_size)
    return cropped


class GsminiCamera(OpenCVCamera):
    def __init__(self, config: GsminiCameraConfig):
        super().__init__(config)
        self.config: GsminiCameraConfig = config

    def _postprocess_image(
        self, image: NDArray[Any], color_mode: ColorMode | None = None
    ) -> NDArray[Any]:
        processed = super()._postprocess_image(image, color_mode)
        if self.config.border_fraction <= 0.0:
            return processed
        # After rotation, self.width/self.height describe the frame layout;
        # resize back to them so the reported feature shape stays accurate.
        return _crop_and_resize(processed, (self.width, self.height), self.config.border_fraction)
