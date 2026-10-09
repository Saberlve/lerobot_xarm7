"""Synthetic preview benchmark, never connects hardware or changes datasets."""
import json
import threading
import time

import numpy as np

from lerobot_robot_ufactory.utils.webapp.web_preview import RecordingWebPreview, WebPreviewConfig


def benchmark(seconds=3):
    rng = np.random.default_rng(0)
    frames = {f"rgb_{i}": rng.integers(0, 256, (480, 640, 3), dtype=np.uint8) for i in range(3)}
    frames.update({f"photon_{i}": rng.integers(0, 256, (700, 400, 3), dtype=np.uint8) for i in range(4)})
    results = []
    for label, cameras in (("off", []), ("rgb", list(frames)[:3]), ("rgb_and_four_photon", list(frames))):
        preview = RecordingWebPreview(WebPreviewConfig())
        preview.set_subscriptions(cameras)
        preview.start_encoder()
        stop = threading.Event()
        lateness = []
        def control_ticks():
            deadline = time.perf_counter()
            while not stop.is_set():
                lateness.append(max(0, time.perf_counter() - deadline) * 1000)
                deadline += 1 / 30
                stop.wait(max(0, deadline - time.perf_counter()))
        thread = threading.Thread(target=control_ticks)
        thread.start()
        calls = []
        deadline = start = time.perf_counter()
        while time.perf_counter() - start < seconds:
            before = time.perf_counter()
            preview.publish(frames)
            calls.append((time.perf_counter() - before) * 1000)
            deadline += 1 / 15
            time.sleep(max(0, deadline - time.perf_counter()))
        stop.set()
        thread.join()
        stats = preview.timing_stats()
        preview.stop()
        results.append({"mode": label, "record_ticks": len(calls), "control_ticks": len(lateness),
                        "publish_p95_ms": round(float(np.percentile(calls, 95)), 4),
                        "control_lateness_p95_ms": round(float(np.percentile(lateness, 95)), 4),
                        "encoded_frames": stats["preview_encoded_frames"]})
    return results


if __name__ == "__main__":
    print(json.dumps({"kind": "synthetic_no_hardware", "seconds_per_mode": 3,
                      "results": benchmark()}, indent=2))
