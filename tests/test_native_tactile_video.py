import hashlib
import json
from types import SimpleNamespace as NS

import av
import numpy as np
import pyarrow.parquet as pq
import pytest

from lerobot_robot_ufactory.scripts.uf_lerobot_record import EpisodeSynchronization
from lerobot_robot_ufactory.datasets.deferred_mesh import compute_episode_mesh
from lerobot_robot_ufactory.datasets.camera_streams import (
    apply_camera_storage_plan,
    camera_recording_plan,
)
from lerobot_robot_ufactory.datasets.tactile_indices import apply_mesh_index_plan, frame_tactile_ranges
from lerobot_robot_ufactory.datasets.native_dataset import NativeRateLeRobotDataset


def make_sample(t, index):
    image = np.random.default_rng(index).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    return NS(
        frame_bgr=image,
        capture_monotonic_s=t,
        capture_monotonic_ns=round(t * 1e9),
        sensor_timestamp_s=None,
        marker_motion_3d=None,
    )


def flow_for_image(image, dtype=np.float32):
    # Include adjacent float32 values and signed zero to catch quantization or
    # conversions that preserve approximate values but change their bits.
    values = np.arange(12, dtype=dtype).reshape(2, 2, 3) / dtype(7)
    values.flat[0] = dtype(-0.0)
    values.flat[1] = np.nextafter(dtype(image[0, 0, 0]), dtype(np.inf))
    return values


class Camera:
    fps = 60
    dtype = np.float32
    config = NS(motion_3d_output="Mesh3DFlow")
    deferred_feature_shapes = {"mesh_motion_3d": (2, 2, 3)}

    def samples_between(self, *args):
        return ()

    def compute_deferred_features(self, image, runtime_dir):
        return {"mesh_motion_3d": flow_for_image(image, self.dtype)}


def create_dataset(tmp_path, legacy=False, *, camera_fps=60, require_mesh=True, use_videos=True):
    camera = Camera()
    camera.fps = camera_fps
    camera.config = NS(motion_3d_output="Mesh3DFlow" if require_mesh else None)
    plan = camera_recording_plan(
        NS(cameras={"photon": camera, "rgb": NS(fps=15)}), 15, use_videos=use_videos
    )
    features = {
        "action": {"dtype": "float32", "shape": (1,), "names": ["joint"]},
        "observation.state": {"dtype": "float32", "shape": (1,), "names": ["joint"]},
        "observation.images.photon": {
            "dtype": "video",
            "shape": (64, 64, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.photon.mesh_motion_3d": {
            "dtype": "float32",
            "shape": (2, 2, 3),
            "names": None,
        },
    }
    if not require_mesh:
        features.pop("observation.photon.mesh_motion_3d")
    features["observation.images.rgb"] = dict(features["observation.images.photon"])
    apply_camera_storage_plan(features, plan)
    apply_mesh_index_plan(features, plan)
    if legacy:
        features["observation.photon.mesh3dflow_range"] = features.pop(
            "observation.photon.tactile_range"
        )
    dataset = NativeRateLeRobotDataset.create(
        "test/native-tactile",
        15,
        features,
        root=tmp_path / "data",
        use_videos=use_videos,
        vcodec="h264",
        video_backend="pyav",
    )
    dataset._camera_stream_plan = plan
    (dataset.root / "meta/camera_streams.json").write_text(
        json.dumps({"dataset_fps": 15, "cameras": plan})
    )
    return dataset


@pytest.mark.parametrize("camera_fps", [15, 60])
@pytest.mark.parametrize("use_videos", [False, True])
def test_full_tactile_images_without_mesh_have_ranges_and_separate_representatives(
    tmp_path, camera_fps, use_videos
):
    dataset = create_dataset(
        tmp_path, camera_fps=camera_fps, require_mesh=False, use_videos=use_videos
    )
    key = "observation.photon.tactile_range"
    plan = dataset.camera_stream_plan()["photon"]
    assert plan["tactile_range_targets"] == ["video" if use_videos else "image"]
    assert "observation.rgb.tactile_range" not in dataset.features
    samples = [make_sample(10 + i / camera_fps, i) for i in range(1, 4)]
    bounds = [10, 10 + 2.1 / camera_fps, 10 + 2.5 / camera_fps, 10 + 3.1 / camera_fps]
    ranges = [[0, 2], [2, 2], [2, 3]]
    representatives = [0, 0, 2]
    sync = EpisodeSynchronization(
        None,
        15,
        dataset_root=dataset.root,
        episode_index=0,
        tactile_stream_names=("photon",),
        stream_fps={"photon": camera_fps},
        video_streams={"photon": "h264"} if use_videos else {},
    )
    try:
        for index, subset in enumerate([samples[:2], [], samples[2:]]):
            representative = samples[representatives[index]]
            sync.add_frame(
                index,
                bounds[index + 1],
                None,
                action_send_start_s=bounds[index + 1],
                tactile_window_start_s=bounds[index],
                tactile_samples={"photon": tuple(subset)},
                camera_timing={
                    "photon": {
                        "capture_monotonic_s": representative.capture_monotonic_s,
                        "capture_monotonic_ns": representative.capture_monotonic_ns,
                    }
                },
            )
            frame = frame_tactile_ranges(dataset.features, sync)
            assert frame[key].dtype == np.int64
            assert frame[key].tolist() == ranges[index]
            image = representative.frame_bgr[:, :, ::-1].copy()
            dataset.add_frame(
                {
                    "action": np.array([index], np.float32),
                    "observation.state": np.array([index], np.float32),
                    "observation.images.photon": image,
                    "observation.images.rgb": image,
                    "task": "test",
                    **frame,
                }
            )
        sync.write(dataset.root, 0, defer_commit=True)
        dataset.save_episode()
        sync.commit()
        dataset.finish_native_episode(0)
        dataset.finalize()
        stream = dataset.root / "tactile_streams/photon/episode_000000"
        rows = pq.read_table(stream / "samples.parquet").to_pylist()
        assert len(rows) == 3
        assert all(row["mesh_path"] is None and row["motion_path"] is None for row in rows)
        assert not list(stream.rglob("*.npy"))
        timing = pq.read_table(dataset.root / "timestamps/episode_000000.parquet").to_pylist()
        assert [
            json.loads(row["camera_intervals_json"])["photon"]["representative_tactile_index"]
            for row in timing
        ] == representatives
        reopened = NativeRateLeRobotDataset(
            dataset.repo_id, root=dataset.root, video_backend="pyav"
        )
        if use_videos:
            with av.open(str(dataset.root / rows[0]["video_path"])) as container:
                assert container.streams.video[0].average_rate == camera_fps
                decoded = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
            assert len(decoded) == 3
        else:
            decoded = [sample.frame_bgr[:, :, ::-1] for sample in samples]
        for index, representative in enumerate(representatives):
            item = reopened[index]
            assert item[key].numpy().tolist() == ranges[index]
            actual = (
                (item["observation.images.photon"].numpy().transpose(1, 2, 0) * 255)
                .round()
                .astype(np.uint8)
            )
            np.testing.assert_array_equal(actual, decoded[representative])
        with pytest.raises(KeyError, match="No indexed Mesh3DFlow stream"):
            reopened.load_mesh_interval(1, "photon")
    finally:
        sync.discard()
        dataset.finalize()


@pytest.mark.parametrize("episodes", [1, 2])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("legacy", [False, True])
def test_standard_video_60fps_mesh_bit_exact_and_exact_row_lookup(
    tmp_path, episodes, dtype, legacy
):
    dataset = create_dataset(tmp_path, legacy=legacy)
    key = "observation.photon." + ("mesh3dflow_range" if legacy else "tactile_range")
    expected = []
    try:
        for episode in range(episodes):
            samples = [make_sample(10 + i / 60, episode * 10 + i) for i in range(1, 8)]
            sync = EpisodeSynchronization(
                None,
                15,
                dataset_root=dataset.root,
                episode_index=episode,
                tactile_stream_names=("photon",),
                stream_fps={"photon": 60},
                video_streams={"photon": "h264"},
                video_crf={"photon": 18},
                required_mesh_streams=("photon",),
            )
            try:
                # Three action rows, seven tactile frames, including an empty
                # window. Range references must cover all frames exactly once.
                for row_index, (start, end, subset) in enumerate(
                    [(10.0, 10.06, samples[:3]), (10.06, 10.13, samples[3:]), (10.13, 10.20, [])]
                ):
                    representative = subset[-1] if subset else samples[-1]
                    dataset.add_frame(
                        {
                            "action": np.array([row_index], np.float32),
                            "observation.state": np.array([row_index + 10], np.float32),
                            "observation.images.photon": representative.frame_bgr[
                                :, :, ::-1
                            ].copy(),
                            "observation.images.rgb": representative.frame_bgr[:, :, ::-1].copy(),
                            key: np.array([[0, 3], [3, 7], [7, 7]][row_index], dtype=np.int64),
                            "task": "test",
                        }
                    )
                    sync.add_frame(
                        row_index,
                        end,
                        None,
                        action_send_start_s=end,
                        tactile_window_start_s=start,
                        tactile_samples={"photon": tuple(subset)},
                        camera_timing={
                            "photon": {
                                "capture_monotonic_s": representative.capture_monotonic_s,
                                "capture_monotonic_ns": representative.capture_monotonic_ns,
                            }
                        },
                    )
                camera = Camera()
                camera.dtype = dtype
                compute_episode_mesh(
                    dataset, {"photon": camera}, tmp_path, episode, synchronization=sync
                )
                sync.write(dataset.root, episode, defer_commit=True)
                mesh = np.load(
                    dataset.root / f"tactile_streams/photon/episode_{episode:06d}/mesh3dflow.npy"
                )
                wanted = np.stack([flow_for_image(sample.frame_bgr, dtype) for sample in samples])
                assert mesh.dtype == wanted.dtype and mesh.tobytes() == wanted.tobytes()
                original = dataset.episode_buffer[key][0]
                dataset.episode_buffer[key][0] = np.array([0, 4], dtype=np.int64)
                with pytest.raises(ValueError, match="exact camera interval"):
                    dataset.save_episode()
                dataset.episode_buffer[key][0] = original
                dataset.save_episode()
                sync.commit()
                dataset.finish_native_episode(episode)
                expected.append(wanted)
            finally:
                sync.discard()
        dataset.finalize()
        assert dataset.meta.info["fps"] == 15
        info = dataset.meta.info["features"]["observation.images.photon"]["info"]
        assert info["video.fps"] == 60 and info["video.codec"] == "h264"
        assert info["video.pix_fmt"] == "yuv420p"
        videos = list((dataset.root / "videos").rglob("*.mp4"))
        total = 0
        for path in videos:
            with av.open(str(path)) as video:
                assert video.streams.video[0].average_rate == (60 if "photon" in str(path) else 15)
                total += len(list(video.decode(video=0)))
        assert total == 10 * episodes
        assert not list((dataset.root / "tactile_streams").rglob("*.mp4"))
        assert not list(dataset.root.rglob("*.png"))
        reopened = NativeRateLeRobotDataset(
            dataset.repo_id, root=dataset.root, video_backend="pyav"
        )
        for episode, wanted in enumerate(expected):
            stream = dataset.root / f"tactile_streams/photon/episode_{episode:06d}"
            rows = pq.read_table(stream / "samples.parquet").to_pylist()
            assert len(rows) == 7
            assert [row["mesh_frame_index"] for row in rows] == list(range(7))
            assert all((dataset.root / row["video_path"]).is_file() for row in rows)
            assert all((dataset.root / row["mesh_path"]).is_file() for row in rows)
            mesh_path = stream / "mesh3dflow.npy"
            manifest = json.loads((stream / "mesh3dflow.json").read_text())
            assert manifest["sha256"] == hashlib.sha256(mesh_path.read_bytes()).hexdigest()
            for row_index, native_index in enumerate([2, 6, 6]):
                absolute_row = episode * 3 + row_index
                item = reopened[absolute_row]
                assert "observation.photon.mesh_motion_3d" not in item
                indices = item[key].numpy()
                assert indices.dtype == np.int64
                assert indices.tolist() == [[0, 3], [3, 7], [7, 7]][row_index]
                saved = reopened.load_mesh_interval(absolute_row, "photon")
                assert saved.tobytes() == wanted[indices[0] : indices[1]].tobytes()
                assert saved.shape == wanted[indices[0] : indices[1]].shape
                with av.open(str(dataset.root / rows[native_index]["video_path"])) as container:
                    all_frames = list(container.decode(video=0))
                    frame_index = round(rows[native_index]["video_timestamp_s"] * 60)
                    decoded = all_frames[frame_index].to_ndarray(format="rgb24")
                actual = (
                    (item["observation.images.photon"].numpy().transpose(1, 2, 0) * 255)
                    .round()
                    .astype(np.uint8)
                )
                np.testing.assert_array_equal(actual, decoded)
        delta = NativeRateLeRobotDataset(
            dataset.repo_id,
            root=dataset.root,
            video_backend="pyav",
            delta_timestamps={"observation.images.photon": [-1 / 15, 0, 1 / 15]},
        )
        queried = delta[0]["observation.images.photon"]
        np.testing.assert_array_equal(queried[0].numpy(), queried[1].numpy())
        np.testing.assert_array_equal(
            queried[1].numpy(), reopened[0]["observation.images.photon"].numpy()
        )
        np.testing.assert_array_equal(
            queried[2].numpy(), reopened[1]["observation.images.photon"].numpy()
        )
    finally:
        dataset.finalize()


def test_mesh_failure_prevents_encoding_and_keeps_uncompressed_inputs(tmp_path, monkeypatch):
    from lerobot_robot_ufactory.datasets import camera_streams

    sync = EpisodeSynchronization(
        None,
        15,
        dataset_root=tmp_path,
        episode_index=0,
        tactile_stream_names=("photon",),
        stream_fps={"photon": 60},
        video_streams={"photon": "h264"},
        required_mesh_streams=("photon",),
    )
    try:
        sync.add_frame(
            0,
            1.1,
            None,
            action_send_start_s=1.1,
            tactile_window_start_s=1.0,
            tactile_samples={"photon": (make_sample(1.05, 1),)},
        )
        monkeypatch.setattr(
            camera_streams, "encode_camera_interval", lambda *a, **kw: pytest.fail("mesh missing")
        )
        with pytest.raises(RuntimeError, match="before lossless Mesh3DFlow"):
            sync.write(tmp_path, 0)
        assert list(sync.tactile_recorder.staging_root.rglob("*.png"))
        assert not list(tmp_path.rglob("*.mp4"))
    finally:
        sync.discard()


@pytest.mark.parametrize("invalid", ["integer", "nonfinite"])
def test_invalid_mesh_is_rejected_before_video_encoding(tmp_path, invalid):
    class InvalidCamera(Camera):
        def compute_deferred_features(self, image, runtime_dir):
            flow = flow_for_image(image)
            if invalid == "integer":
                flow = flow.astype(np.int64)
            else:
                flow.flat[0] = np.inf
            return {"mesh_motion_3d": flow}

    sync = EpisodeSynchronization(
        None,
        15,
        dataset_root=tmp_path,
        episode_index=0,
        tactile_stream_names=("photon",),
        stream_fps={"photon": 60},
        video_streams={"photon": "h264"},
        required_mesh_streams=("photon",),
    )
    try:
        sync.add_frame(
            0,
            1.1,
            None,
            action_send_start_s=1.1,
            tactile_window_start_s=1.0,
            tactile_samples={"photon": (make_sample(1.05, 1),)},
        )
        with pytest.raises(RuntimeError, match="Invalid native Mesh3DFlow"):
            sync.tactile_recorder.compute_mesh({"photon": InvalidCamera()}, tmp_path)
        assert list(sync.tactile_recorder.staging_root.rglob("*.png"))
        assert not list(tmp_path.rglob("*.mp4"))
    finally:
        sync.discard()


@pytest.mark.parametrize("fail_second_encoder", [False, True])
def test_original_pixels_reach_sdk_before_encoding_and_all_png_cleanup(
    tmp_path, monkeypatch, fail_second_encoder
):
    from lerobot_robot_ufactory.datasets.stream_recorder import TactileStreamRecorder
    from lerobot_robot_ufactory.datasets import camera_streams

    names = ("left", "right")
    samples = {
        name: [make_sample(1.01 + i / 60, offset + i) for i in range(2)]
        for name, offset in [("left", 100), ("right", 200)]
    }
    recorder = TactileStreamRecorder(
        tmp_path / "new_recording",
        0,
        names,
        stream_fps=dict.fromkeys(names, 60),
        video_streams=dict.fromkeys(names, "h264"),
        required_mesh_streams=names,
    )
    unrelated = recorder.dataset_root / "previous_episode_original.png"
    unrelated.write_bytes(b"existing data must remain unchanged")
    sdk_calls = []

    class OriginalFrameCamera(Camera):
        def __init__(self, name):
            self.name = name
            self.index = 0

        def compute_deferred_features(self, image, runtime_dir):
            assert not list(recorder.staging_root.rglob("*.mp4"))
            np.testing.assert_array_equal(image, samples[self.name][self.index].frame_bgr)
            sdk_calls.append((self.name, self.index))
            self.index += 1
            return {"mesh_motion_3d": flow_for_image(image, np.float64)}

    encode = camera_streams.encode_camera_interval
    failed_once = False

    def checked_encode(staged, rows, fps, codec, relative, **kwargs):
        nonlocal failed_once
        assert sdk_calls == [(name, i) for name in names for i in range(2)]
        assert len(list(recorder.staging_root.rglob("*.png"))) == 4
        expected = np.stack([flow_for_image(s.frame_bgr, np.float64) for s in samples[staged.name]])
        mesh = np.load(staged / "mesh3dflow.npy", allow_pickle=False)
        assert mesh.dtype == np.float64 and mesh.tobytes() == expected.tobytes()
        manifest = json.loads((staged / "mesh3dflow.json").read_text())
        assert manifest["roundtrip"] == "bit_exact"
        if fail_second_encoder and staged.name == "right" and not failed_once:
            failed_once = True
            raise RuntimeError("second camera encoding failed")
        return encode(staged, rows, fps, codec, relative, **kwargs)

    monkeypatch.setattr(camera_streams, "encode_camera_interval", checked_encode)
    try:
        for name in names:
            recorder.add_window(name, samples[name], 1.0, 1.1)
        recorder.compute_mesh({name: OriginalFrameCamera(name) for name in names}, tmp_path)
        if fail_second_encoder:
            with pytest.raises(RuntimeError, match="second camera encoding failed"):
                recorder.prepare(0)
            assert len(list(recorder.staging_root.rglob("*.png"))) == 4
        recorder.prepare(0)
        assert not list(recorder.staging_root.rglob("*.png"))
        recorder.finalize(0)
        assert len(list(recorder.dataset_root.rglob("*.mp4"))) == 2
        assert len(list(recorder.dataset_root.rglob("mesh3dflow.npy"))) == 2
        assert unrelated.read_bytes() == b"existing data must remain unchanged"
        assert list(recorder.dataset_root.rglob("*.png")) == [unrelated]
    finally:
        recorder.discard()


@pytest.mark.parametrize("require_mesh", [False, True])
@pytest.mark.parametrize("first_window_empty", [False, True])
def test_prestart_representative_survives_native_save_and_read(
    tmp_path, require_mesh, first_window_empty
):
    dataset = create_dataset(tmp_path, require_mesh=require_mesh)
    seed = make_sample(9.99, 100)
    fresh = [make_sample(10.02 + i / 60, i) for i in range(3)]
    sync = EpisodeSynchronization(
        None, 15, dataset_root=dataset.root, episode_index=0,
        tactile_stream_names=("photon",), stream_fps={"photon": 60},
        video_streams={"photon": "h264"},
        required_mesh_streams=("photon",) if require_mesh else (),
    )
    # Even a nonempty first window can use an older representative selected
    # against the state anchor. Preserve that exact image, not the window tail.
    first = [] if first_window_empty else fresh[:1]
    rest = fresh if first_window_empty else fresh[1:]
    cases = [(10., 10.04, first, seed), (10.04, 10.05, [], seed),
             (10.05, 10.12, rest, rest[-1])]
    # Keep all new sample times inside their actual windows.
    for i, sample in enumerate(rest):
        sample.capture_monotonic_s = 10.06 + i / 60
        sample.capture_monotonic_ns = round(sample.capture_monotonic_s * 1e9)
    expected_ranges = []
    expected_index = 1
    try:
        for i, (start, end, samples, representative) in enumerate(cases):
            sync.add_frame(
                i, end, None, action_send_start_s=end,
                tactile_window_start_s=start, tactile_samples={"photon": tuple(samples)},
                initial_samples={"photon": seed} if i == 0 else None,
                camera_timing={"photon": {
                    "capture_monotonic_s": representative.capture_monotonic_s,
                    "capture_monotonic_ns": representative.capture_monotonic_ns,
                }},
            )
            ranges = frame_tactile_ranges(dataset.features, sync)
            expected_ranges.append([expected_index, expected_index + len(samples)])
            expected_index += len(samples)
            assert ranges["observation.photon.tactile_range"].tolist() == expected_ranges[-1]
            dataset.add_frame({
                "action": np.array([i], np.float32),
                "observation.state": np.array([i], np.float32),
                "observation.images.photon": representative.frame_bgr[:, :, ::-1].copy(),
                "observation.images.rgb": representative.frame_bgr[:, :, ::-1].copy(),
                **ranges, "task": "pre-start representative",
            })
        if require_mesh:
            compute_episode_mesh(dataset, {"photon": Camera()}, tmp_path, 0, synchronization=sync)
        sync.write(dataset.root, 0, defer_commit=True)
        dataset.save_episode()
        sync.commit()
        dataset.finish_native_episode(0)
        dataset.finalize()
        reopened = NativeRateLeRobotDataset(dataset.repo_id, root=dataset.root, video_backend="pyav")
        stream = dataset.root / "tactile_streams/photon/episode_000000"
        rows = pq.read_table(stream / "samples.parquet").to_pylist()
        assert len(rows) == 4  # One initial representative, three window samples.
        assert rows[0]["capture_monotonic_ns"] == seed.capture_monotonic_ns
        intervals = [json.loads(row["camera_intervals_json"])["photon"] for row in sync.frames]
        assert [row["representative_tactile_index"] for row in intervals] == [0, 0, 3]
        with av.open(str(dataset.root / rows[0]["video_path"])) as container:
            frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
        for i, representative in enumerate([0, 0, 3]):
            actual = reopened[i]
            np.testing.assert_array_equal(
                (actual["observation.images.photon"].numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8),
                frames[representative],
            )
            assert actual["observation.photon.tactile_range"].tolist() == expected_ranges[i]
            if require_mesh:
                start, end = expected_ranges[i]
                full = np.load(stream / "mesh3dflow.npy")
                assert reopened.load_mesh_interval(i, "photon").tobytes() == full[start:end].tobytes()
        if require_mesh:
            assert full[0].tobytes() == flow_for_image(seed.frame_bgr).tobytes()
    finally:
        sync.discard()
        dataset.finalize()
