"""Read-only encoder stream for the unified GELLO tuning page."""

import threading
import time

import numpy as np

from ..hardware.transport import XL330Transport


class EncoderMonitor:
    """One serial owner; HTTP clients only copy the latest completed sample.

    Uses PING/READ only. Never enables torque, changes mode or writes current.
    Failed connections are closed and retried, preserving a labelled last sample.
    """

    def __init__(self, profile, *, rate_hz=None, transport_factory=XL330Transport):
        self.profile = profile
        self.rate_hz = profile.rate_hz if rate_hz is None else rate_hz
        if not np.isfinite(self.rate_hz) or not 1 <= self.rate_hz <= 200:
            raise ValueError("Read rate must be finite and between 1 and 200 Hz")
        self.transport_factory = transport_factory
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self._sample = None
        self._status = "connecting"
        self._error = None
        self._sequence = 0
        self._motors = []

    def snapshot(self):
        with self._lock:
            age = None if self._sample is None else time.monotonic() - self._sample["stamp"]
            status = self._status
            if status == "connected" and age > max(0.5, 3 / self.rate_hz):
                status = "stale"
            return {
                "status": status,
                "error": self._error,
                "age_ms": None if age is None else round(age * 1000, 1),
                "sequence": self._sequence,
                "sample": self._sample,
                "motors": self._motors,
            }

    def start(self):
        self._thread = threading.Thread(target=self._run, name="gello-encoder-reader", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                raise RuntimeError("Encoder reader did not stop within 5 seconds")

    def _run(self):
        while not self._stop.is_set():
            transport = self.transport_factory(self.profile)
            try:
                with self._lock:
                    self._status = "connecting"
                transport.open()
                with self._lock:
                    self._motors = transport.info
                previous_stamp = None
                next_read = time.monotonic()
                while not self._stop.is_set():
                    state = transport.state()
                    position = np.asarray(state["position"], dtype=float)
                    velocity = np.asarray(state["velocity"], dtype=float)
                    stamp = float(state["stamp"])
                    if (
                        position.shape != (len(self.profile.all_ids),)
                        or velocity.shape != position.shape
                        or not np.isfinite(position).all()
                        or not np.isfinite(velocity).all()
                        or not np.isfinite(stamp)
                    ):
                        raise ValueError("Invalid encoder sample")
                    model_q, _ = self.profile.model_state(position, velocity)
                    sample = {
                        "stamp": stamp,
                        "encoder_rad": position.tolist(),
                        "model_q_rad": model_q.tolist(),
                        "temperature_c": list(state["temperature_c"]),
                        "voltage_v": list(state["voltage_v"]),
                        "read_hz": None if previous_stamp is None else 1 / max(stamp - previous_stamp, 1e-9),
                    }
                    with self._lock:
                        self._sample = sample
                        self._sequence += 1
                        self._status = "connected"
                        self._error = None
                    previous_stamp = stamp
                    next_read += 1 / self.rate_hz
                    now = time.monotonic()
                    if next_read < now:
                        next_read = now
                    self._stop.wait(max(0, next_read - now))
            except Exception as exc:
                with self._lock:
                    self._status = "error"
                    self._error = str(exc)
            finally:
                transport.close()
            self._stop.wait(2)
        with self._lock:
            self._status = "stopped"

