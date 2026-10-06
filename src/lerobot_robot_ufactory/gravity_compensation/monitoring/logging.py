"""Drain diagnostics outside the serial/control worker."""

import json
import logging
import threading
import time
from pathlib import Path


class GravityLog:
    def __init__(self, runtime, directory):
        self.runtime = runtime
        directory = Path(directory).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{runtime.profile.name}_teleop_{time.time_ns()}.jsonl"
        self._stream = self.path.open("x", buffering=1)
        self._stop = threading.Event()
        self._error = None
        self._thread = threading.Thread(target=self._run, name="gello-gravity-log", daemon=True)

    def start(self):
        try:
            self._thread.start()
        except BaseException:
            self._stream.close()
            raise

    def _drain(self):
        for record in self.runtime.drain_records():
            self._stream.write(json.dumps(record) + "\n")

    def _run(self):
        try:
            while not self._stop.wait(0.1):
                self._drain()
            # Stop the control worker before stopping this writer, preserving final records.
            self._drain()
        except BaseException as exc:
            self._error = exc
            self.runtime.request_stop()
            logging.exception("GELLO diagnostic logging failed; stopping compensation")
        finally:
            try:
                self._stream.close()
            except BaseException as exc:
                self._error = self._error or exc
                self.runtime.request_stop()

    def raise_if_failed(self):
        if self._error is not None:
            raise RuntimeError(f"GELLO logging failed: {self._error}") from self._error

    def stop(self):
        self._stop.set()
        if self._thread.ident is not None:
            self._thread.join(timeout=3)
            if self._thread.is_alive():
                raise RuntimeError("GELLO diagnostic logger did not stop")
        else:
            self._stream.close()
        self.raise_if_failed()
