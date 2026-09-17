"""Common interfaces for camera-like tactile sensors."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from lerobot.cameras.camera import Camera
from lerobot.cameras.configs import CameraConfig
from numpy.typing import NDArray


class TactileCameraConfig(CameraConfig, ABC):
    """Configuration contract for tactile sensors with deferred inference."""

    @abstractmethod
    def configure_deferred_processing(self) -> None:
        """Select lossless raw/rectified capture suitable for offline inference."""


class TactileCamera(Camera, ABC):
    """Sensor-independent interface used by the dataset recorder."""

    runtime_export_dir: Path | None = None

    @property
    @abstractmethod
    def deferred_feature_shapes(self) -> dict[str, tuple[int, ...]]:
        """Map feature suffixes to tensor shapes produced after recording."""

    @abstractmethod
    def runtime_manifest(self) -> dict[str, Any]:
        """Describe the captured runtime configuration."""

    @abstractmethod
    def compute_deferred_features(
        self, image_bgr: NDArray[np.uint8], runtime_dir: Path
    ) -> dict[str, NDArray[np.float32]]:
        """Compute deferred tactile features for one lossless image."""
