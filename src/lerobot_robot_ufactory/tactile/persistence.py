"""Lossless per-episode Photon streams kept outside LeRobot video features."""

from __future__ import annotations

import os
import json
import logging
import queue
import shutil
import threading
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np


logger = logging.getLogger(__name__)


class TactilePersistenceError(RuntimeError):
    """Base class for raw tactile persistence failures."""


class TactileBackpressureError(TactilePersistenceError):
    """The bounded writer queue could not accept a sample in time."""


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _rollback_transaction(items: list[dict[str, str]]) -> None:
    """Restore old destinations and move new files back into staging."""
    errors = []
    for item in reversed(items):
        source = Path(item["source"])
        destination = Path(item["destination"])
        backup = Path(item["backup"])
        try:
            if backup.exists():
                if destination.exists():
                    if source.exists():
                        _remove_path(destination)
                    else:
                        source.parent.mkdir(parents=True, exist_ok=True)
                        os.replace(destination, source)
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(backup, destination)
            elif not source.exists() and destination.exists():
                source.parent.mkdir(parents=True, exist_ok=True)
                os.replace(destination, source)
        except BaseException as exc:
            errors.append(exc)
    if errors:
        raise TactilePersistenceError("Failed to roll back tactile episode publish") from errors[0]


def cleanup_stale_tactile_staging(dataset_root: Path) -> None:
    """Recover or remove only this recorder's stale tactile staging trees."""
    dataset_root = Path(dataset_root).resolve()
    staging_parent = dataset_root / "tactile_streams" / ".staging"
    if not staging_parent.is_dir():
        return
    for staging_root in tuple(staging_parent.iterdir()):
        if not staging_root.is_dir() or not staging_root.name.startswith("episode_"):
            continue
        journal_path = staging_root / "transaction.json"
        safe_to_remove = True
        if journal_path.is_file():
            try:
                journal = json.loads(journal_path.read_text())
                items = journal["items"]
                for item in items:
                    for field in ("source", "destination", "backup"):
                        Path(item[field]).resolve().relative_to(dataset_root)
                marker_path = journal.get("commit_marker")
                committed = False
                if marker_path:
                    marker = Path(marker_path)
                    marker.resolve().relative_to(dataset_root)
                    if marker.is_file():
                        committed = (
                            json.loads(marker.read_text()).get("transaction_id")
                            == journal.get("transaction_id")
                        )
                if not committed and journal.get("dataset_saved") and marker_path:
                    marker_item = items[-1]
                    marker_source = Path(marker_item["source"])
                    marker_destination = Path(marker_item["destination"])
                    if marker_source.is_file() and all(
                        Path(item["destination"]).exists() for item in items[:-1]
                    ):
                        os.replace(marker_source, marker_destination)
                        committed = True
                if not committed:
                    _rollback_transaction(items)
            except BaseException:
                logger.exception("Could not recover stale tactile transaction %s", staging_root)
                # Never touch final episode paths when a journal cannot be trusted.
                safe_to_remove = False
        if safe_to_remove and staging_root.exists():
            shutil.rmtree(staging_root)
    try:
        staging_parent.rmdir()
    except OSError:
        pass


class TactileStreamRecorder:
    """Asynchronously persist variable-rate tactile samples and their clocks.

    Dataset rows refer to a half-open index range ``[start_index, end_index)``
    in each stream. Samples themselves are lossless PNGs plus a Parquet index;
    optional SDK displacement arrays are saved as ``.npy`` files.
    """

    _STOP = object()

    def __init__(
        self,
        dataset_root: Path,
        episode_index: int,
        stream_names: tuple[str, ...] | list[str],
        *,
        queue_size: int = 256,
        enqueue_timeout_s: float = 0.1,
        num_writer_threads: int | None = None,
    ) -> None:
        if episode_index < 0:
            raise ValueError("episode_index must be non-negative")
        if not isinstance(queue_size, int) or isinstance(queue_size, bool) or queue_size <= 0:
            raise ValueError("queue_size must be a positive integer")
        if not np.isfinite(enqueue_timeout_s) or enqueue_timeout_s <= 0:
            raise ValueError("enqueue_timeout_s must be finite and positive")
        names = tuple(stream_names)
        if len(set(names)) != len(names):
            raise ValueError("tactile stream names must be unique")
        for name in names:
            if not name or Path(name).name != name:
                raise ValueError(f"Unsafe tactile stream name: {name!r}")
        if num_writer_threads is None:
            # PNG encoding is CPU-bound; one worker per stream (capped) keeps
            # the writer ahead of ~60 fps per stream without changing the
            # on-disk format or the persistence semantics.
            num_writer_threads = max(1, min(4, len(names)))
        if (
            not isinstance(num_writer_threads, int)
            or isinstance(num_writer_threads, bool)
            or num_writer_threads <= 0
        ):
            raise ValueError("num_writer_threads must be a positive integer")

        self.dataset_root = Path(dataset_root)
        self.episode_index = int(episode_index)
        self.stream_names = names
        self._base = f"episode_{self.episode_index:06d}"
        self._transaction_id = uuid4().hex
        self._staging_root = (
            self.dataset_root
            / "tactile_streams"
            / ".staging"
            / f"{self._base}_{self._transaction_id}"
        )
        self._rows: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
        self._last_capture_s: dict[str, float | None] = {name: None for name in names}
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._enqueue_timeout_s = float(enqueue_timeout_s)
        self._exception: BaseException | None = None
        self._closed = False
        self._prepared = False
        self._published = False
        self._committed = False
        self._pending_journal_items: list[dict[str, str]] | None = None
        self._pending_marker_item: dict[str, str] | None = None
        self._dataset_saved = False
        self._metrics_lock = threading.Lock()
        self._queue_high_watermark = 0
        self._enqueued_count = 0
        self._written_count = 0
        self._writer_error_count = 0
        self._backpressure_error_count = 0
        self._write_latency_total_ms = 0.0
        self._write_latency_max_ms = 0.0
        self._pending_enqueued_ns: dict[int, int] = {}
        self._next_queue_item_id = 0
        self._warned_thresholds: set[int] = set()
        self._capture_index: dict[str, dict[int, int]] = {name: {} for name in names}
        for name in names:
            (self._staging_root / name / "frames").mkdir(parents=True, exist_ok=False)
        self._threads = [
            threading.Thread(
                target=self._run,
                name=f"tactile-writer-{self.episode_index:06d}-{worker_index}",
                daemon=True,
            )
            for worker_index in range(num_writer_threads)
        ]
        for thread in self._threads:
            thread.start()

    def add_window(
        self,
        stream_name: str,
        samples: tuple[Any, ...] | list[Any],
        start_monotonic_s: float,
        end_monotonic_s: float,
        *,
        start_monotonic_ns: int | None = None,
        end_monotonic_ns: int | None = None,
    ) -> dict[str, Any]:
        """Queue one action interval and return its dataset-row mapping."""
        self._raise_if_failed()
        if self._closed:
            raise RuntimeError("tactile stream recorder is closed")
        if stream_name not in self._rows:
            raise KeyError(f"Unknown tactile stream: {stream_name}")
        if not np.isfinite(start_monotonic_s) or not np.isfinite(end_monotonic_s):
            raise ValueError("tactile window bounds must be finite")
        if end_monotonic_s < start_monotonic_s:
            raise ValueError("tactile window end must be at or after its start")

        start_index = len(self._rows[stream_name])
        capture_times = []
        sensor_times = []
        previous = self._last_capture_s[stream_name]
        enqueue_deadline = time.perf_counter() + self._enqueue_timeout_s
        for sample in samples:
            capture_s = float(sample.capture_monotonic_s)
            if not start_monotonic_s < capture_s <= end_monotonic_s:
                raise AssertionError(
                    f"Tactile sample {capture_s:.9f} is outside "
                    f"({start_monotonic_s:.9f}, {end_monotonic_s:.9f}]"
                )
            if previous is not None and capture_s <= previous:
                raise AssertionError(
                    f"Tactile stream {stream_name} is not strictly monotonic"
                )
            index = len(self._rows[stream_name])
            frame_relative = (
                Path("tactile_streams")
                / stream_name
                / self._base
                / "frames"
                / f"frame_{index:06d}.png"
            )
            marker = getattr(sample, "marker_motion_3d", None)
            marker_relative = None
            if marker is not None:
                marker_relative = (
                    Path("tactile_streams")
                    / stream_name
                    / self._base
                    / "motion"
                    / f"motion_{index:06d}.npy"
                )
            sensor_timestamp_s = getattr(sample, "sensor_timestamp_s", None)
            device_to_host_offset_s = (
                None
                if sensor_timestamp_s is None
                else capture_s - float(sensor_timestamp_s)
            )
            row = {
                "tactile_index": index,
                "capture_monotonic_ns": int(
                    getattr(sample, "capture_monotonic_ns", None)
                    if getattr(sample, "capture_monotonic_ns", None) is not None
                    else round(capture_s * 1_000_000_000)
                ),
                "capture_monotonic_s": capture_s,
                "sensor_timestamp_s": (
                    None if sensor_timestamp_s is None else float(sensor_timestamp_s)
                ),
                "device_to_host_offset_s": device_to_host_offset_s,
                "frame_path": frame_relative.as_posix(),
                "motion_path": (
                    None if marker_relative is None else marker_relative.as_posix()
                ),
                "image_encoding": "png_bgr8",
            }
            self._rows[stream_name].append(row)
            item_id = self._next_queue_item_id
            self._next_queue_item_id += 1
            enqueued_ns = time.perf_counter_ns()
            with self._metrics_lock:
                self._pending_enqueued_ns[item_id] = enqueued_ns
            try:
                remaining_s = enqueue_deadline - time.perf_counter()
                if remaining_s <= 0:
                    raise queue.Full
                self._queue.put(
                    (item_id, enqueued_ns, stream_name, index, sample),
                    timeout=remaining_s,
                )
            except queue.Full as exc:
                self._rows[stream_name].pop()
                with self._metrics_lock:
                    self._pending_enqueued_ns.pop(item_id, None)
                    self._backpressure_error_count += 1
                raise TactileBackpressureError(
                    f"Raw tactile writer could not enqueue this window within "
                    f"{self._enqueue_timeout_s * 1_000:.1f} ms "
                    f"(depth={self._queue.qsize()}, capacity={self._queue.maxsize})"
                ) from exc
            with self._metrics_lock:
                self._enqueued_count += 1
                depth = self._queue.qsize()
                self._queue_high_watermark = max(self._queue_high_watermark, depth)
                occupancy = depth / self._queue.maxsize
                for threshold in (75, 90):
                    if occupancy >= threshold / 100 and threshold not in self._warned_thresholds:
                        self._warned_thresholds.add(threshold)
                        logger.warning(
                            "Raw tactile writer queue reached %d%%: depth=%d capacity=%d",
                            threshold,
                            depth,
                            self._queue.maxsize,
                        )
            self._capture_index[stream_name][row["capture_monotonic_ns"]] = index
            capture_times.append(capture_s)
            sensor_times.append(row["sensor_timestamp_s"])
            previous = capture_s

        self._last_capture_s[stream_name] = previous
        end_index = len(self._rows[stream_name])
        return {
            "window_start_ns": int(
                start_monotonic_ns
                if start_monotonic_ns is not None
                else round(start_monotonic_s * 1_000_000_000)
            ),
            "window_end_ns": int(
                end_monotonic_ns
                if end_monotonic_ns is not None
                else round(end_monotonic_s * 1_000_000_000)
            ),
            "start_index": start_index,
            "end_index": end_index,
            "frame_count": end_index - start_index,
            "capture_monotonic_ns": [
                self._rows[stream_name][index]["capture_monotonic_ns"]
                for index in range(start_index, end_index)
            ],
            "sensor_timestamp_s": sensor_times,
            "device_to_host_offset_s": [
                None
                if sensor_timestamp_s is None
                else capture_s - sensor_timestamp_s
                for capture_s, sensor_timestamp_s in zip(capture_times, sensor_times)
            ],
            "span_ms": (
                0.0
                if len(capture_times) < 2
                else (capture_times[-1] - capture_times[0]) * 1_000
            ),
        }

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is self._STOP:
                    return
                item_id, enqueued_ns, stream_name, index, sample = item
                write_started_ns = time.perf_counter_ns()
                stream_root = self._staging_root / stream_name
                frame_path = stream_root / "frames" / f"frame_{index:06d}.png"
                frame = np.asarray(sample.frame_bgr)
                if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
                    raise ValueError(
                        f"Invalid raw tactile frame for {stream_name}: "
                        f"dtype={frame.dtype}, shape={frame.shape}"
                    )
                if not cv2.imwrite(str(frame_path), frame):
                    raise OSError(f"Failed to write tactile frame: {frame_path}")
                marker = getattr(sample, "marker_motion_3d", None)
                if marker is not None:
                    motion_dir = stream_root / "motion"
                    motion_dir.mkdir(exist_ok=True)
                    np.save(motion_dir / f"motion_{index:06d}.npy", np.asarray(marker))
                latency_ms = (time.perf_counter_ns() - write_started_ns) / 1_000_000
                with self._metrics_lock:
                    self._written_count += 1
                    self._write_latency_total_ms += latency_ms
                    self._write_latency_max_ms = max(self._write_latency_max_ms, latency_ms)
            except BaseException as exc:
                with self._metrics_lock:
                    if self._exception is None:
                        self._exception = exc
                    self._writer_error_count += 1
            finally:
                if item is not self._STOP:
                    with self._metrics_lock:
                        self._pending_enqueued_ns.pop(item[0], None)
                self._queue.task_done()

    def _close_writer(self) -> None:
        if self._closed:
            self._raise_if_failed()
            return
        self._queue.join()
        for _ in self._threads:
            self._queue.put(self._STOP)
        self._queue.join()
        for thread in self._threads:
            thread.join()
        self._closed = True
        self._raise_if_failed()

    @property
    def staging_root(self) -> Path:
        return self._staging_root

    @property
    def transaction_id(self) -> str:
        return self._transaction_id

    def status(self) -> dict[str, Any]:
        now_ns = time.perf_counter_ns()
        with self._metrics_lock:
            oldest_ns = min(self._pending_enqueued_ns.values(), default=None)
            return {
                "queue_depth": self._queue.qsize(),
                "queue_capacity": self._queue.maxsize,
                "queue_high_watermark": self._queue_high_watermark,
                "enqueued_count": self._enqueued_count,
                "written_count": self._written_count,
                "writer_error_count": self._writer_error_count,
                "backpressure_error_count": self._backpressure_error_count,
                "oldest_pending_age_ms": (
                    None if oldest_ns is None else (now_ns - oldest_ns) / 1_000_000
                ),
                "mean_write_latency_ms": (
                    0.0
                    if self._written_count == 0
                    else self._write_latency_total_ms / self._written_count
                ),
                "max_write_latency_ms": self._write_latency_max_ms,
            }

    def representative_index(self, stream_name: str, capture_monotonic_ns: int) -> int | None:
        return self._capture_index[stream_name].get(int(capture_monotonic_ns))

    def prepare(self, episode_index: int) -> None:
        """Flush and validate all raw files while they are still private."""
        if int(episode_index) != self.episode_index:
            raise RuntimeError("Tactile stream episode mismatch")
        if self._published:
            return
        if self._prepared:
            return
        self._close_writer()

        import pyarrow as pa
        import pyarrow.parquet as pq

        schema = pa.schema(
            [
                ("tactile_index", pa.int64()),
                ("capture_monotonic_ns", pa.int64()),
                ("capture_monotonic_s", pa.float64()),
                ("sensor_timestamp_s", pa.float64()),
                ("device_to_host_offset_s", pa.float64()),
                ("frame_path", pa.string()),
                ("motion_path", pa.string()),
                ("image_encoding", pa.string()),
            ]
        )
        for name in self.stream_names:
            staged = self._staging_root / name
            for index, row in enumerate(self._rows[name]):
                if row["tactile_index"] != index:
                    raise TactilePersistenceError(f"Non-contiguous tactile index for {name}")
                if not (staged / "frames" / f"frame_{index:06d}.png").is_file():
                    raise TactilePersistenceError(f"Missing staged tactile frame for {name}:{index}")
                if row["motion_path"] is not None and not (
                    staged / "motion" / f"motion_{index:06d}.npy"
                ).is_file():
                    raise TactilePersistenceError(f"Missing staged tactile motion for {name}:{index}")
            index_path = staged / "samples.parquet"
            temporary = index_path.with_suffix(".parquet.tmp")
            table = pa.Table.from_pylist(self._rows[name], schema=schema)
            pq.write_table(table, temporary)
            os.replace(temporary, index_path)

        self._prepared = True

    def _tactile_publish_items(self) -> list[tuple[Path, Path]]:
        return [
            (
                self._staging_root / name,
                self.dataset_root / "tactile_streams" / name / self._base,
            )
            for name in self.stream_names
        ]

    def publish_episode(
        self,
        episode_index: int,
        extra_items: list[tuple[Path, Path]] | None = None,
        *,
        commit_marker: tuple[Path, Path] | None = None,
        defer_commit_marker: bool = False,
    ) -> None:
        """Publish one complete episode, rolling every destination back on failure."""
        if defer_commit_marker and commit_marker is None:
            raise ValueError("defer_commit_marker requires a commit marker")
        self.prepare(episode_index)
        publish_items = [*self._tactile_publish_items(), *(extra_items or [])]
        if commit_marker is not None:
            publish_items.append(commit_marker)
        backup_root = self._staging_root / ".rollback"
        backup_root.mkdir(exist_ok=False)
        journal_items = []
        for ordinal, (source, destination) in enumerate(publish_items):
            if not source.exists():
                raise TactilePersistenceError(f"Missing staged publish source: {source}")
            journal_items.append(
                {
                    "source": str(source.resolve()),
                    "destination": str(destination.resolve()),
                    "backup": str((backup_root / f"item_{ordinal:04d}").resolve()),
                }
            )
        journal = {
            "transaction_id": self._transaction_id,
            "episode_index": self.episode_index,
            "dataset_saved": False,
            "commit_marker": (
                None if commit_marker is None else str(commit_marker[1].resolve())
            ),
            "items": journal_items,
        }
        journal_path = self._staging_root / "transaction.json"
        journal_tmp = journal_path.with_suffix(".json.tmp")
        journal_tmp.write_text(json.dumps(journal, indent=2) + "\n")
        os.replace(journal_tmp, journal_path)

        try:
            marker_item = journal_items[-1] if commit_marker is not None else None
            if marker_item is not None:
                marker_destination = Path(marker_item["destination"])
                marker_backup = Path(marker_item["backup"])
                marker_destination.parent.mkdir(parents=True, exist_ok=True)
                if marker_destination.exists():
                    # Removing the previous marker first makes partial replacement
                    # logically invisible to readers that honor the commit marker.
                    os.replace(marker_destination, marker_backup)
            for item in journal_items:
                if defer_commit_marker and item is marker_item:
                    continue
                source = Path(item["source"])
                destination = Path(item["destination"])
                backup = Path(item["backup"])
                destination.parent.mkdir(parents=True, exist_ok=True)
                if item is not marker_item and destination.exists():
                    os.replace(destination, backup)
                os.replace(source, destination)
            published_items = (
                journal_items[:-1]
                if defer_commit_marker and marker_item is not None
                else journal_items
            )
            if not all(Path(item["destination"]).exists() for item in published_items):
                raise TactilePersistenceError("Published tactile episode failed validation")
        except BaseException as publish_exc:
            try:
                _rollback_transaction(journal_items)
            except BaseException as rollback_exc:
                raise TactilePersistenceError(
                    "Tactile publish failed and rollback was incomplete"
                ) from rollback_exc
            raise publish_exc

        self._published = True
        if defer_commit_marker:
            self._pending_journal_items = journal_items
            self._pending_marker_item = marker_item
            return
        self._committed = True
        if self._staging_root.exists():
            shutil.rmtree(self._staging_root)

    def confirm_publish(self, *, dataset_saved: bool = False) -> None:
        """Make a prepared episode visible by publishing its commit marker last."""
        if self._committed:
            return
        if not self._published or self._pending_marker_item is None:
            raise TactilePersistenceError("No deferred tactile publish to confirm")
        if dataset_saved and not self._dataset_saved:
            self._dataset_saved = True
            journal_path = self._staging_root / "transaction.json"
            journal = json.loads(journal_path.read_text())
            journal["dataset_saved"] = True
            temporary = journal_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(journal, indent=2) + "\n")
            os.replace(temporary, journal_path)
        marker_source = Path(self._pending_marker_item["source"])
        marker_destination = Path(self._pending_marker_item["destination"])
        if marker_destination.exists():
            raise TactilePersistenceError(
                f"Unexpected commit marker before confirmation: {marker_destination}"
            )
        os.replace(marker_source, marker_destination)
        if not marker_destination.is_file():
            raise TactilePersistenceError("Tactile commit marker publish failed validation")
        self._committed = True
        self._pending_journal_items = None
        self._pending_marker_item = None
        if self._staging_root.exists():
            shutil.rmtree(self._staging_root)

    def finalize(self, episode_index: int) -> None:
        """Backward-compatible tactile-only transactional publish."""
        self.publish_episode(episode_index)

    def discard(self) -> None:
        """Remove an unsaved episode's staged raw tactile data."""
        if self._committed:
            return
        remove_staging = True
        try:
            if self._published and self._pending_journal_items is not None:
                if self._dataset_saved:
                    try:
                        self.confirm_publish()
                    except BaseException:
                        remove_staging = False
                        logger.exception(
                            "Dataset saved but tactile commit marker is still pending; "
                            "staging retained for startup recovery"
                        )
                    return
                _rollback_transaction(self._pending_journal_items)
                self._published = False
            else:
                self._close_writer()
        finally:
            if remove_staging and self._staging_root.exists():
                shutil.rmtree(self._staging_root)

    def _raise_if_failed(self) -> None:
        if self._exception is not None:
            raise RuntimeError("Raw tactile stream writer failed") from self._exception
