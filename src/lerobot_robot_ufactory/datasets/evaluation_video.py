"""Best-effort dual RGB recording from existing immutable acquisition buffers.

No device reads, encoding, frame copies or queue operations run on the policy
thread. Each view has a background pipe writer and a low-priority, single-thread
FFmpeg encoder. The acquisition history is the bounded latest-frame mailbox.
"""
from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
import select
import shutil
import subprocess
import threading
import time


class EvaluationVideo:
    FPS = 15

    def __init__(self, robot, keys, directory):
        self.robot, self.keys = robot, tuple(keys)
        self.directory = Path(directory)
        self.owns_directory = False
        self.stop = threading.Event()
        self.abort = threading.Event()
        self.workers = []
        self.processes = []
        self.views = []
        self.started = None
        self.ended = None
        self.error = None
        self.manifest_lock = threading.Lock()

    def _latest(self, key):
        # Never call async_read: the existing acquisition thread owns that API.
        samples = self.robot._rgb_sync_buffers[key].sync_samples()
        if not samples:
            return None
        return samples[-1]

    def start(self):
        self.directory.mkdir(parents=True, exist_ok=False)
        self.owns_directory = True
        if len(self.keys) != 2 or len(set(self.keys)) != 2:
            raise ValueError('评测录像需要两个不同的 RGB 相机')
        ffmpeg = shutil.which('ffmpeg')
        if not ffmpeg:
            raise RuntimeError('评测录像需要 ffmpeg（含 libx264）')
        deadline = time.monotonic() + 2
        initial = []
        for key in self.keys:
            sample = self._latest(key)
            while sample is None and time.monotonic() < deadline:
                time.sleep(.01)
                sample = self._latest(key)
            if sample is None:
                raise RuntimeError(f'相机 {key} 尚无可录制帧')
            initial.append(sample)
        try:
            for index, (key, sample) in enumerate(zip(self.keys, initial), 1):
                height, width, channels = sample.frame.shape
                if channels != 3 or sample.frame.dtype.name != 'uint8':
                    raise ValueError('录像要求 uint8 三通道 RGB 图像')
                camera = self.robot.cameras[key]
                mode = getattr(getattr(camera, 'config', None), 'color_mode', 'rgb')
                pixel_format = 'bgr24' if 'bgr' in str(mode).lower() else 'rgb24'
                filename = f'view_{index}.mp4'
                view = dict(key=key, file=filename, width=width, height=height,
                            frames=0, repeated_frames=0, skipped_slots=0, error=None)
                self.views.append(view)
                log = (self.directory / f'view_{index}.ffmpeg.log').open('wb')
                try:
                    process = subprocess.Popen([
                        ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin',
                        '-f', 'rawvideo', '-pixel_format', pixel_format,
                        '-video_size', f'{width}x{height}', '-framerate', str(self.FPS),
                        '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'ultrafast',
                        '-crf', '23', '-threads', '1', '-filter_threads', '1',
                        '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-pix_fmt', 'yuv420p',
                        '-g', str(self.FPS), '-movflags', '+frag_keyframe+empty_moov+default_base_moof',
                        '-n', str(self.directory / filename),
                    ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log, bufsize=0)
                finally:
                    log.close()
                self.processes.append(process)
                os.set_blocking(process.stdin.fileno(), False)
                try:
                    os.setpriority(os.PRIO_PROCESS, process.pid, 10)
                except (OSError, AttributeError):
                    pass
            self.started = time.monotonic()
            self._manifest('recording')
            for key, sample, view, process in zip(self.keys, initial, self.views, self.processes):
                worker = threading.Thread(target=self._encode, args=(key, sample, view, process),
                                          name=f'eval-video-{key}', daemon=True)
                self.workers.append(worker)
                worker.start()
        except Exception:
            self.close()
            raise

    def _write(self, process, frame):
        import numpy as np
        data = memoryview(np.ascontiguousarray(frame)).cast('B')
        fd = process.stdin.fileno()
        while data:
            if self.abort.is_set():
                raise TimeoutError('录像编码收尾超时')
            _, writable, _ = select.select([], [fd], [], .05)
            if writable:
                try:
                    count = os.write(fd, data)
                    data = data[count:]
                except BlockingIOError:
                    continue

    def _encode(self, key, previous, view, process):
        try:
            while not self.abort.is_set():
                slot = view['frames']
                if self.stop.is_set():
                    target = max(1, math.ceil((self.ended - self.started) * self.FPS))
                    if slot >= target:
                        break
                    sample = previous
                else:
                    if self.stop.wait(max(0, self.started + slot / self.FPS - time.monotonic())):
                        continue
                    current_slot = int((time.monotonic() - self.started) * self.FPS)
                    # Slow encoders skip source samples and repeat the previous
                    # image in elapsed slots, preserving both views' common clock.
                    sample = previous if slot < current_slot else (self._latest(key) or previous)
                    if sample.frame.shape != previous.frame.shape:
                        raise ValueError('录像过程中相机分辨率发生变化')
                self._write(process, sample.frame)
                view['frames'] += 1
                if slot > 0 and sample.capture_monotonic_s == previous.capture_monotonic_s:
                    view['repeated_frames'] += 1
                    if not self.stop.is_set() and slot < current_slot:
                        view['skipped_slots'] += 1
                previous = sample
        except Exception as exc:
            view['error'] = str(exc)
            logging.warning('评测录像 %s 失败，评测继续：%s', key, exc)
        finally:
            process.stdin.close()
            try:
                code = process.wait(timeout=2)
                if code and not view['error']:
                    view['error'] = f'ffmpeg 退出码 {code}，请查看编码日志'
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                view['error'] = 'ffmpeg 保存超时'
            if view['error']:
                try:
                    self._manifest('error')
                except OSError:
                    logging.exception('无法保存录像错误状态')

    def _manifest(self, status):
        data = dict(status=status, fps=self.FPS, views=self.views, error=self.error,
                    started_monotonic_s=self.started, ended_monotonic_s=self.ended,
                    timing='共同主机时间轴；缓存 RGB 帧，非硬件同步；掉帧重复上一帧')
        with self.manifest_lock:
            temp = self.directory / '.recording.json.tmp'
            temp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
            temp.replace(self.directory / 'recording.json')

    def close(self):
        self.ended = self.ended or time.monotonic()
        self.stop.set()
        deadline = time.monotonic() + 3
        for worker in self.workers:
            worker.join(max(0, deadline - time.monotonic()))
        self.abort.set()
        for process in self.processes:
            if process.poll() is None:
                process.kill()
        for worker in self.workers:
            worker.join(1)
        for process in self.processes:
            process.wait(timeout=2)
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
        for view in self.views:
            view['duration_s'] = view['frames'] / self.FPS
        if self.owns_directory:
            try:
                self._manifest('error' if self.error or any(v['error'] for v in self.views) else 'saved')
            except OSError as exc:
                logging.warning('无法保存录像信息：%s', exc)


def start_evaluation_video(robot, cfg):
    """Off by default; the console can override YAML per trial without rewriting it."""
    override = os.environ.get('VLARMORY_RECORD_VIDEO')
    enabled = getattr(cfg, 'record_video', False) if override is None else override == '1'
    if not enabled:
        return None
    directory = os.environ.get('VLARMORY_VIDEO_DIR') or getattr(cfg, 'video_dir', '')
    if not directory:
        directory = str(Path('videos') / f'eval-{time.time_ns()}')
    recording = EvaluationVideo(robot, cfg.camera_keys, directory)
    try:
        recording.start()
        logging.info('双视角评测录像已开启（15 FPS）：%s', directory)
        return recording
    except Exception as exc:
        recording.error = str(exc)
        try:
            recording.close()
        except Exception:
            logging.exception('录像清理失败')
        logging.warning('无法开启评测录像，评测继续：%s', exc)
        return None
