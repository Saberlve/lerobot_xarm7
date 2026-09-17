"""Configuration for one Xense Photon tactile image stream."""

from dataclasses import dataclass

from lerobot.cameras.configs import ColorMode

from ..base import TactileCameraConfig


@TactileCameraConfig.register_subclass("photon")
@dataclass(kw_only=True)
class XensePhotonCameraConfig(TactileCameraConfig):
    serial_number: str
    width: int = 400
    height: int = 700
    fps: int = 30
    color_mode: ColorMode = ColorMode.RGB
    output_type: str = "Rectify"
    config_path: str | None = None
    # Save the SDK Marker3DFlow output with the RGB image. The SDK documents
    # this as the 3D marker displacement field.
    disable_infer: bool = False
    save_marker_motion_3d: bool = True
    # Legacy save_marker_motion_3d enables the selected displacement output.
    # Mesh3DFlow and Marker3DFlow have distinct dataset feature names.
    motion_3d_output: str = "Marker3DFlow"
    infer_mode: str | None = None
    marker_rows: int = 35
    marker_cols: int = 20
    # Recent complete SDK samples retained for timestamp-bounded pairing by
    # the recorder. At 30 Hz, 30 samples retain roughly one second.
    sync_history_size: int = 30
    timeout_ms: int = 2000
    max_frame_age_ms: int = 1000

    def configure_deferred_processing(self) -> None:
        if self.output_type != "Rectify":
            raise ValueError("Offline tactile processing requires Rectify images")
        self.disable_infer = True
        self.save_marker_motion_3d = False

    def __post_init__(self) -> None:
        if not isinstance(self.serial_number, str) or not self.serial_number.strip():
            raise ValueError("serial_number must be a non-empty Xense sensor serial number")
        for name in (
            "width",
            "height",
            "fps",
            "marker_rows",
            "marker_cols",
            "sync_history_size",
            "timeout_ms",
            "max_frame_age_ms",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.output_type not in ("Rectify", "Raw", "Difference"):
            raise ValueError("output_type must be Rectify, Raw or Difference (BGR images)")
        if self.save_marker_motion_3d and self.disable_infer:
            raise ValueError("save_marker_motion_3d requires disable_infer=False")
        if self.motion_3d_output not in ("Marker3DFlow", "Mesh3DFlow"):
            raise ValueError("motion_3d_output must be Marker3DFlow or Mesh3DFlow")
        self.color_mode = ColorMode(self.color_mode)
