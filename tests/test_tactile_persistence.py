import json
import threading
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from lerobot_robot_ufactory.scripts.uf_lerobot_record import (
    EpisodeSynchronization,
    _EpisodeSynchronizationOwner,
)
from lerobot_robot_ufactory.tactile import persistence
from lerobot_robot_ufactory.tactile.persistence import (
    TactileBackpressureError,
    TactileStreamRecorder,
    cleanup_stale_tactile_staging,
)
from lerobot_robot_ufactory.tactile.photon.camera import XensePhotonSample


def _sample(timestamp_s, value, *, timestamp_ns=None):
    return XensePhotonSample(
        frame_bgr=np.full((3, 4, 3), value, dtype=np.uint8),
        marker_motion_3d=None,
        sensor_timestamp_s=1_700_000_000.0 + timestamp_s,
        capture_monotonic_s=timestamp_s,
        capture_monotonic_ns=timestamp_ns,
    )


def _sync_with_one_window(root: Path, episode_index: int = 0):
    sync = EpisodeSynchronization(
        None,
        fps=15,
        dataset_root=root,
        episode_index=episode_index,
        tactile_stream_names=("photon_left", "photon_right"),
    )
    sync.add_frame(
        0,
        10.0,
        10.0,
        action_send_start_s=10.02,
        action_send_end_s=10.021,
        action_send_start_ns=10_020_000_003,
        action_send_end_ns=10_021_000_007,
        tactile_window_start_s=9.99,
        tactile_window_start_ns=9_990_000_001,
        tactile_samples={
            "photon_left": (_sample(10.01, 1, timestamp_ns=10_010_000_003),),
            "photon_right": (_sample(10.015, 2, timestamp_ns=10_015_000_005),),
        },
    )
    return sync


def test_multi_stream_publish_rolls_back_when_second_stream_fails(tmp_path, monkeypatch):
    sync = _sync_with_one_window(tmp_path)
    original_replace = persistence.os.replace

    def fail_right_publish(source, destination):
        if Path(source).name == "photon_right" and Path(destination).name == "episode_000000":
            raise OSError("injected right publish failure")
        return original_replace(source, destination)

    monkeypatch.setattr(persistence.os, "replace", fail_right_publish)
    with pytest.raises(OSError, match="right publish"):
        sync.write(tmp_path, 0)
    sync.discard()

    assert not (tmp_path / "tactile_streams/photon_left/episode_000000").exists()
    assert not (tmp_path / "tactile_streams/photon_right/episode_000000").exists()
    assert not (tmp_path / "timestamps/episode_000000.parquet").exists()


def test_sidecar_publish_failure_rolls_back_tactile_streams(tmp_path, monkeypatch):
    sync = _sync_with_one_window(tmp_path)
    original_replace = persistence.os.replace
    failed_destination = tmp_path / "timestamps/episode_000000.parquet"

    def fail_sidecar_publish(source, destination):
        if Path(destination) == failed_destination:
            raise OSError("injected sidecar publish failure")
        return original_replace(source, destination)

    monkeypatch.setattr(persistence.os, "replace", fail_sidecar_publish)
    with pytest.raises(OSError, match="sidecar publish"):
        sync.write(tmp_path, 0)
    sync.discard()

    assert not (tmp_path / "tactile_streams/photon_left/episode_000000").exists()
    assert not (tmp_path / "tactile_streams/photon_right/episode_000000").exists()
    assert not failed_destination.exists()
    assert not (tmp_path / "timestamps/episode_000000_commit.json").exists()


def test_dataset_save_failure_rolls_back_deferred_aux_publish(tmp_path):
    sync = _sync_with_one_window(tmp_path)
    sync.write(tmp_path, 0, defer_commit=True)

    assert (tmp_path / "tactile_streams/photon_left/episode_000000").is_dir()
    assert (tmp_path / "timestamps/episode_000000.parquet").is_file()
    assert not (tmp_path / "timestamps/episode_000000_commit.json").exists()

    # This is the cleanup path used when LeRobotDataset.save_episode raises.
    sync.discard()
    assert not (tmp_path / "tactile_streams/photon_left/episode_000000").exists()
    assert not (tmp_path / "tactile_streams/photon_right/episode_000000").exists()
    assert not (tmp_path / "timestamps/episode_000000.parquet").exists()


def test_deferred_aux_publish_becomes_visible_only_after_commit(tmp_path):
    sync = _sync_with_one_window(tmp_path)
    marker = tmp_path / "timestamps/episode_000000_commit.json"
    sync.write(tmp_path, 0, defer_commit=True)
    assert not marker.exists()

    sync.commit()
    assert marker.is_file()
    assert json.loads(marker.read_text())["transaction_id"] == sync.tactile_recorder.transaction_id
    sync.discard()
    assert marker.is_file()


def test_commit_marker_failure_after_dataset_save_is_retried_during_cleanup(
    tmp_path, monkeypatch
):
    sync = _sync_with_one_window(tmp_path)
    marker = tmp_path / "timestamps/episode_000000_commit.json"
    sync.write(tmp_path, 0, defer_commit=True)
    original_replace = persistence.os.replace
    failed = False

    def fail_marker_once(source, destination):
        nonlocal failed
        if Path(destination) == marker and not failed:
            failed = True
            raise OSError("injected commit marker failure")
        return original_replace(source, destination)

    monkeypatch.setattr(persistence.os, "replace", fail_marker_once)
    with pytest.raises(OSError, match="commit marker"):
        sync.commit()

    sync.discard()
    assert marker.is_file()
    staging = tmp_path / "tactile_streams/.staging"
    assert not staging.exists() or not any(staging.iterdir())


def test_normal_episode_publish_is_complete_and_indexes_match(tmp_path):
    sync = _sync_with_one_window(tmp_path)
    sync.write(tmp_path, 0)

    left_path = tmp_path / "tactile_streams/photon_left/episode_000000/samples.parquet"
    right_path = tmp_path / "tactile_streams/photon_right/episode_000000/samples.parquet"
    timing_path = tmp_path / "timestamps/episode_000000.parquet"
    assert left_path.is_file()
    assert right_path.is_file()
    assert timing_path.is_file()
    assert (tmp_path / "timestamps/episode_000000_commit.json").is_file()

    mapping = json.loads(pq.read_table(timing_path).to_pylist()[0]["tactile_timing_json"])
    assert mapping["photon_left"]["end_index"] == len(pq.read_table(left_path))
    assert mapping["photon_right"]["end_index"] == len(pq.read_table(right_path))
    assert pq.read_table(left_path).to_pylist()[0]["capture_monotonic_ns"] == 10_010_000_003
    frame_row = pq.read_table(timing_path).to_pylist()[0]
    assert frame_row["action_send_start_ns"] == 10_020_000_003
    assert frame_row["action_send_end_ns"] == 10_021_000_007
    action_row = pq.read_table(
        tmp_path / "timestamps/episode_000000_actions.parquet"
    ).to_pylist()[0]
    assert action_row["action_send_start_ns"] == 10_020_000_003
    assert action_row["action_send_end_ns"] == 10_021_000_007
    status = sync.tactile_recorder.status()
    assert status["enqueued_count"] == status["written_count"] == 2
    assert status["writer_error_count"] == 0


def test_rerecord_replaces_existing_episode_without_old_raw_frames(tmp_path):
    first = _sync_with_one_window(tmp_path)
    first.write(tmp_path, 0)

    second = EpisodeSynchronization(
        None,
        fps=15,
        dataset_root=tmp_path,
        episode_index=0,
        tactile_stream_names=("photon_left", "photon_right"),
    )
    second.add_frame(
        0,
        20.0,
        20.0,
        action_send_start_s=20.02,
        action_send_end_s=20.021,
        tactile_window_start_s=19.99,
        tactile_samples={
            "photon_left": (_sample(20.01, 9),),
            "photon_right": (),
        },
    )
    second.write(tmp_path, 0)

    left_root = tmp_path / "tactile_streams/photon_left/episode_000000"
    right_root = tmp_path / "tactile_streams/photon_right/episode_000000"
    assert len(list((left_root / "frames").glob("*.png"))) == 1
    assert list((right_root / "frames").glob("*.png")) == []
    assert len(pq.read_table(left_root / "samples.parquet")) == 1
    assert len(pq.read_table(right_root / "samples.parquet")) == 0
    assert not list((tmp_path / "tactile_streams/.staging").glob("episode_*"))


def test_failed_rerecord_restores_previous_complete_episode(tmp_path, monkeypatch):
    original = _sync_with_one_window(tmp_path)
    original.write(tmp_path, 0)
    replacement = EpisodeSynchronization(
        None,
        fps=15,
        dataset_root=tmp_path,
        episode_index=0,
        tactile_stream_names=("photon_left", "photon_right"),
    )
    replacement.add_frame(
        0,
        20.0,
        20.0,
        action_send_start_s=20.02,
        action_send_end_s=20.021,
        tactile_window_start_s=19.99,
        tactile_samples={
            "photon_left": (_sample(20.01, 9),),
            "photon_right": (),
        },
    )
    original_replace = persistence.os.replace
    failed_destination = tmp_path / "timestamps/episode_000000.parquet"
    failed = False

    def fail_replacement_sidecar(source, destination):
        nonlocal failed
        if Path(destination) == failed_destination and not failed:
            failed = True
            assert not (tmp_path / "timestamps/episode_000000_commit.json").exists()
            raise OSError("injected replacement failure")
        return original_replace(source, destination)

    monkeypatch.setattr(persistence.os, "replace", fail_replacement_sidecar)
    with pytest.raises(OSError, match="replacement failure"):
        replacement.write(tmp_path, 0)
    replacement.discard()

    left_rows = pq.read_table(
        tmp_path / "tactile_streams/photon_left/episode_000000/samples.parquet"
    ).to_pylist()
    timing_rows = pq.read_table(failed_destination).to_pylist()
    assert left_rows[0]["capture_monotonic_ns"] == 10_010_000_003
    assert timing_rows[0]["action_send_start_ns"] == 10_020_000_003
    assert len(
        pq.read_table(
            tmp_path / "tactile_streams/photon_right/episode_000000/samples.parquet"
        )
    ) == 1
    assert (tmp_path / "timestamps/episode_000000_commit.json").is_file()


def test_writer_backpressure_is_bounded_observable_and_cleanup_is_complete(
    tmp_path, monkeypatch
):
    entered = threading.Event()
    release = threading.Event()

    def blocked_imwrite(_path, _frame):
        entered.set()
        assert release.wait(2)
        return True

    monkeypatch.setattr(persistence.cv2, "imwrite", blocked_imwrite)
    recorder = TactileStreamRecorder(
        tmp_path,
        2,
        ("photon_left",),
        queue_size=1,
        enqueue_timeout_s=0.05,
    )
    recorder.add_window("photon_left", [_sample(1.01, 1)], 1.0, 1.02)
    assert entered.wait(1)
    recorder.add_window("photon_left", [_sample(1.03, 2)], 1.02, 1.04)
    status = recorder.status()
    assert status["queue_depth"] == 1
    assert status["queue_capacity"] == 1
    assert status["queue_high_watermark"] == 1

    started = time.perf_counter()
    with pytest.raises(TactileBackpressureError, match="could not enqueue"):
        recorder.add_window("photon_left", [_sample(1.05, 3)], 1.04, 1.06)
    assert time.perf_counter() - started < 0.3
    assert recorder.status()["backpressure_error_count"] == 1

    release.set()
    recorder.discard()
    staging = tmp_path / "tactile_streams/.staging"
    assert not staging.exists() or not any(staging.iterdir())
    assert not (tmp_path / "tactile_streams/photon_left/episode_000002").exists()


def test_uncommitted_owner_discards_after_post_record_exception(tmp_path):
    sync = _sync_with_one_window(tmp_path, episode_index=7)
    with pytest.raises(RuntimeError, match="injected before save"):
        with _EpisodeSynchronizationOwner() as owner:
            owner.track(sync)
            raise RuntimeError("injected before save")

    staging = tmp_path / "tactile_streams/.staging"
    assert not staging.exists() or not any(staging.iterdir())
    assert not (tmp_path / "tactile_streams/photon_left/episode_000007").exists()


def test_startup_cleanup_removes_only_recorder_staging(tmp_path):
    stale = tmp_path / "tactile_streams/.staging/episode_000003_deadbeef"
    unrelated = tmp_path / "tactile_streams/.staging/not_a_recorder_transaction"
    stale.mkdir(parents=True)
    unrelated.mkdir()
    (stale / "partial.png").write_bytes(b"partial")

    cleanup_stale_tactile_staging(tmp_path)

    assert not stale.exists()
    assert unrelated.is_dir()


def test_representative_can_explicitly_reference_a_previous_window(tmp_path):
    capture_ns = 10_010_000_003
    sync = EpisodeSynchronization(
        None,
        fps=15,
        dataset_root=tmp_path,
        episode_index=0,
        tactile_stream_names=("photon_left",),
    )
    representative = {
        "photon_left": {
            "capture_monotonic_s": 10.01,
            "capture_monotonic_ns": capture_ns,
            "tactile_stream_name": "photon_left",
        }
    }
    sync.add_frame(
        0,
        10.0,
        10.0,
        camera_timing=representative,
        action_send_start_s=10.02,
        action_send_end_s=10.021,
        tactile_window_start_s=10.0,
        tactile_samples={"photon_left": (_sample(10.01, 1, timestamp_ns=capture_ns),)},
    )
    sync.add_frame(
        1,
        10.03,
        10.03,
        camera_timing=representative,
        action_send_start_s=10.04,
        action_send_end_s=10.041,
        tactile_window_start_s=10.02,
        tactile_samples={"photon_left": ()},
    )
    sync.write(tmp_path, 0)

    rows = pq.read_table(tmp_path / "timestamps/episode_000000.parquet").to_pylist()
    second = json.loads(rows[1]["tactile_timing_json"])["photon_left"]
    assert second["start_index"] == second["end_index"] == 1
    assert second["representative_capture_monotonic_ns"] == capture_ns
    assert second["representative_tactile_index"] == 0


def test_four_tactile_streams_publish_with_independent_continuous_indexes(tmp_path):
    streams = ("photon_left", "photon_right", "photon_left_2", "photon_right_2")
    windows = (
        {
            "photon_left": (_sample(10.005, 1), _sample(10.015, 2)),
            "photon_right": (_sample(10.01, 3),),
            "photon_left_2": (),
            "photon_right_2": (
                _sample(10.008, 4),
                _sample(10.012, 5),
                _sample(10.018, 6),
            ),
        },
        {
            "photon_left": (_sample(10.07, 7),),
            "photon_right": (_sample(10.065, 8), _sample(10.075, 9)),
            "photon_left_2": (_sample(10.07, 10),),
            "photon_right_2": (),
        },
    )
    sync = EpisodeSynchronization(
        None,
        fps=15,
        dataset_root=tmp_path,
        episode_index=0,
        tactile_stream_names=streams,
    )
    window_bounds = ((9.99, 10.02), (10.02, 10.08))
    for frame_index, samples in enumerate(windows):
        start_s, end_s = window_bounds[frame_index]
        sync.add_frame(
            frame_index,
            end_s - 0.02,
            end_s - 0.02,
            action_send_start_s=end_s,
            action_send_end_s=end_s + 0.001,
            tactile_window_start_s=start_s,
            tactile_samples=samples,
        )
    sync.write(tmp_path, 0)

    totals = {name: 0 for name in streams}
    for name in streams:
        episode_dir = tmp_path / f"tactile_streams/{name}/episode_000000"
        assert (episode_dir / "samples.parquet").is_file(), name
    rows = pq.read_table(tmp_path / "timestamps/episode_000000.parquet").to_pylist()
    assert len(rows) == 2
    for frame_index, samples in enumerate(windows):
        mapping = json.loads(rows[frame_index]["tactile_timing_json"])
        assert set(mapping) == set(streams)
        for name in streams:
            entry = mapping[name]
            expected_count = len(samples[name])
            assert entry["end_index"] - entry["start_index"] == expected_count
            assert entry["start_index"] == totals[name]
            totals[name] += expected_count
    for name in streams:
        table = pq.read_table(
            tmp_path / f"tactile_streams/{name}/episode_000000/samples.parquet"
        )
        assert len(table) == totals[name]
        assert list(table.to_pandas()["tactile_index"]) == list(range(totals[name]))
        frame_files = sorted(
            (tmp_path / f"tactile_streams/{name}/episode_000000/frames").glob("*.png")
        )
        assert len(frame_files) == totals[name]
    commit = json.loads(
        (tmp_path / "timestamps/episode_000000_commit.json").read_text()
    )
    assert commit["tactile_streams"] == list(streams)
    status = sync.tactile_recorder.status()
    assert status["enqueued_count"] == status["written_count"] == 10
    assert status["writer_error_count"] == 0


def test_four_stream_publish_rolls_back_atomically(tmp_path, monkeypatch):
    streams = ("photon_left", "photon_right", "photon_left_2", "photon_right_2")
    sync = EpisodeSynchronization(
        None,
        fps=15,
        dataset_root=tmp_path,
        episode_index=0,
        tactile_stream_names=streams,
    )
    sync.add_frame(
        0,
        10.0,
        10.0,
        action_send_start_s=10.02,
        action_send_end_s=10.021,
        tactile_window_start_s=9.99,
        tactile_samples={
            name: (_sample(10.01, index),) for index, name in enumerate(streams)
        },
    )
    original_replace = persistence.os.replace

    def fail_third_stream_publish(source, destination):
        if (
            Path(source).name == "photon_left_2"
            and Path(destination).name == "episode_000000"
        ):
            raise OSError("injected photon_left_2 publish failure")
        return original_replace(source, destination)

    monkeypatch.setattr(persistence.os, "replace", fail_third_stream_publish)
    with pytest.raises(OSError, match="photon_left_2 publish"):
        sync.write(tmp_path, 0)
    sync.discard()

    for name in streams:
        assert not (tmp_path / f"tactile_streams/{name}/episode_000000").exists()
    assert not list((tmp_path / "timestamps").glob("episode_000000*"))


def test_writer_threads_scale_with_stream_count(tmp_path):
    streams = ("photon_left", "photon_right", "photon_left_2", "photon_right_2")
    recorder = TactileStreamRecorder(tmp_path, 0, streams)
    try:
        assert len(recorder._threads) == 4
    finally:
        recorder.discard()
    two = TactileStreamRecorder(tmp_path, 1, streams[:2])
    try:
        assert len(two._threads) == 2
    finally:
        two.discard()
    with pytest.raises(ValueError):
        TactileStreamRecorder(tmp_path, 2, ("photon_left",), num_writer_threads=0)
