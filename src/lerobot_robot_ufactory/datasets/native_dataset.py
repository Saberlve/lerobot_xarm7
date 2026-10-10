"""Standard LeRobot v3 video layout with independent camera clocks.

Numeric rows keep dataset.fps. Native-rate videos are prepared transactionally
by the interval recorder, then published by upstream LeRobot's video writer.
Use this class when reading to select the exact causal representative frame.
"""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from .tactile_indices import tactile_range_fields


class NativeRateLeRobotDataset(LeRobotDataset):
    def camera_stream_plan(self):
        plan = getattr(self, "_camera_stream_plan", None)
        if plan is None:
            path = self.root / "meta/camera_streams.json"
            plan = json.loads(path.read_text())["cameras"] if path.is_file() else {}
            self._camera_stream_plan = plan
        return plan

    def save_episode(self, episode_data=None, parallel_encoding=True):
        buffer = self.episode_buffer if episode_data is None else episode_data
        self._validate_tactile_ranges(buffer)
        if self.camera_stream_plan():
            # Upstream's process workers bypass per-camera overrides and always
            # use dataset.fps. Keep encoding through this instance instead.
            if self.batch_encoding_size != 1:
                raise ValueError("Native camera video requires batch_encoding_size=1")
            parallel_encoding = False
        return super().save_episode(episode_data=episode_data, parallel_encoding=parallel_encoding)

    def _validate_tactile_ranges(self, buffer):
        fields = tactile_range_fields(self.features)
        if not fields:
            return
        root = Path(getattr(self, "_native_stream_root", self.root))
        episode = int(buffer["episode_index"])
        timing = pq.read_table(root / "timestamps" / f"episode_{episode:06d}.parquet").to_pylist()
        if len(timing) != int(buffer["size"]):
            raise ValueError("Tactile index timing rows differ from action rows")
        for key, camera in fields.items():
            index_path = (
                root / "tactile_streams" / camera / f"episode_{episode:06d}" / "samples.parquet"
            )
            samples = pq.read_table(index_path).to_pylist()
            if len(buffer[key]) != len(timing):
                raise ValueError(f"Missing tactile index rows: {camera}")
            for value, row in zip(buffer[key], timing, strict=True):
                value = np.asarray(value)
                interval = json.loads(row["camera_intervals_json"])[camera]
                expected = [interval["start_index"], interval["end_index"]]
                if value.dtype != np.int64 or value.shape != (2,) or value.tolist() != expected:
                    raise ValueError(
                        f"Tactile range must match the exact camera interval: {camera}"
                    )
                start, end = expected
                if not 0 <= start <= end <= len(samples):
                    raise ValueError(f"Tactile range out of bounds: {camera}")
            if any(
                not (sample.get("frame_path") or sample.get("video_path")) for sample in samples
            ):
                raise ValueError(f"Tactile index refers to missing image payloads: {camera}")
            require_mesh = self.camera_stream_plan().get(camera, {}).get("require_mesh", True)
            if require_mesh and any(
                not (sample.get("mesh_path") or sample.get("motion_path")) for sample in samples
            ):
                raise ValueError(f"Mesh index refers to missing lossless payloads: {camera}")

    def load_mesh_interval(self, row_index, camera):
        """Read the full-rate lossless mesh slice referenced by one action row."""
        keys = {name: key for key, name in tactile_range_fields(self.features).items()}
        require_mesh = self.camera_stream_plan().get(camera, {}).get("require_mesh", True)
        if camera not in keys or not require_mesh:
            raise KeyError(f"No indexed Mesh3DFlow stream: {camera}")
        key = keys[camera]
        self._ensure_hf_dataset_loaded()
        row = self.hf_dataset[row_index]
        episode = int(row["episode_index"])
        start, end = (int(value) for value in row[key])
        stream_root = self.root / "tactile_streams" / camera / f"episode_{episode:06d}"
        path = stream_root / "mesh3dflow.npy"
        if path.is_file():
            mesh = np.load(path, allow_pickle=False, mmap_mode="r")
            if not 0 <= start <= end <= len(mesh):
                raise ValueError("Mesh interval is outside the saved native array")
            return mesh[start:end]
        # Online inference saves each SDK array separately without a dtype cast.
        samples = pq.read_table(stream_root / "samples.parquet").to_pylist()
        if not 0 <= start <= end <= len(samples) or not samples:
            raise ValueError("Mesh interval is outside the saved native samples")
        arrays = [
            np.load(self.root / sample["motion_path"], allow_pickle=False)
            for sample in samples[start:end]
        ]
        if arrays:
            return np.stack(arrays)
        reference = np.load(self.root / samples[0]["motion_path"], allow_pickle=False)
        return np.empty((0, *reference.shape), dtype=reference.dtype)

    def _encode_temporary_episode_video(self, video_key, episode_index):
        name = video_key.removeprefix("observation.images.")
        item = self.camera_stream_plan().get(name)
        if item and item["storage"] == "video" and (item["interval"] or item["tactile"]):
            root = Path(getattr(self, "_native_stream_root", self.root))
            source = root / "tactile_streams" / name / f"episode_{episode_index:06d}" / "video.mp4"
            if not source.is_file():
                raise FileNotFoundError(
                    f"Native camera video must be prepared before dataset save: {source}"
                )
            # Upstream moves the file and removes its entire parent directory.
            # Never pass the transaction/mesh directory directly to that writer.
            directory = Path(tempfile.mkdtemp(prefix=".native-video-", dir=self.root))
            temporary = directory / "video.mp4"
            try:
                shutil.copyfile(source, temporary)
            except BaseException:
                shutil.rmtree(directory)
                raise
            return temporary
        return super()._encode_temporary_episode_video(video_key, episode_index)

    def _save_episode_video(self, video_key, episode_index, temp_path=None):
        metadata = super()._save_episode_video(video_key, episode_index, temp_path=temp_path)
        locations = getattr(self, "_native_video_locations", {})
        locations.setdefault(episode_index, {})[video_key] = metadata
        self._native_video_locations = locations
        return metadata

    def finish_native_episode(self, episode_index):
        """Point indexes at committed standard videos; clean new temporary copies."""
        from lerobot_robot_ufactory.datasets.episode_images import discard_episode_images

        locations = getattr(self, "_native_video_locations", {}).pop(episode_index, {})
        for key, metadata in locations.items():
            name = key.removeprefix("observation.images.")
            item = self.camera_stream_plan().get(name)
            if not item or not (item["interval"] or item["tactile"]):
                continue
            root = self.root / "tactile_streams" / name / f"episode_{episode_index:06d}"
            index_path = root / "samples.parquet"
            if not index_path.is_file():
                continue
            table = pq.read_table(index_path)
            rows = table.to_pylist()
            video_path = self.meta.video_path.format(
                video_key=key,
                chunk_index=metadata[f"videos/{key}/chunk_index"],
                file_index=metadata[f"videos/{key}/file_index"],
            )
            offset = metadata[f"videos/{key}/from_timestamp"]
            for row in rows:
                row["video_path"] = video_path
                row["video_timestamp_s"] = offset + row["video_frame_index"] / item["fps"]
            temporary = index_path.with_suffix(".parquet.tmp")
            import pyarrow as pa

            pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), temporary)
            temporary.replace(index_path)
            info_path = root / "video.json"
            info = json.loads(info_path.read_text())
            info.update(
                video_path=video_path,
                from_timestamp=offset,
                presentation_clock="from_timestamp + video_frame_index / camera_fps",
            )
            info_path.write_text(json.dumps(info, indent=2) + "\n")
            (root / "video.mp4").unlink(missing_ok=True)
        discard_episode_images(self, episode_index)

    def _query_videos(self, query_timestamps, ep_idx):
        plan = self.camera_stream_plan()
        native = {
            name: item
            for name, item in plan.items()
            if item["storage"] == "video" and (item["interval"] or item["tactile"])
        }
        if not native:
            return super()._query_videos(query_timestamps, ep_idx)
        cache = getattr(self, "_native_frame_cache", {})
        if ep_idx not in cache:
            path = self.root / "timestamps" / f"episode_{ep_idx:06d}.parquet"
            cache[ep_idx] = [
                json.loads(row["camera_intervals_json"]) for row in pq.read_table(path).to_pylist()
            ]
            # Bound retained per-episode indexes for long training runs.
            if len(cache) > 8:
                del cache[next(iter(cache))]
            self._native_frame_cache = cache
        timing = cache[ep_idx]
        mapped = {}
        for key, timestamps in query_timestamps.items():
            name = key.removeprefix("observation.images.")
            if name not in native:
                mapped[key] = timestamps
                continue
            mapped[key] = []
            for timestamp in timestamps:
                row_index = round(timestamp * self.fps)
                if not 0 <= row_index < len(timing):
                    raise IndexError(f"Camera query outside episode: {timestamp}")
                interval = timing[row_index][name]
                index = interval.get("representative_tactile_index")
                if index is None:
                    # Empty windows retain the most recent captured frame.
                    index = interval["end_index"] - 1
                if index < 0:
                    raise RuntimeError(f"No captured frame for {name} at action {row_index}")
                mapped[key].append(index / native[name]["fps"])
        return super()._query_videos(mapped, ep_idx)
