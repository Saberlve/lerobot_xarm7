"""One recording owner per Unix user, shared by CLI and web workers."""
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
import os
import threading

_guard = threading.RLock()
_depth = 0


@contextmanager
def recording_lock():
    import fcntl

    global _depth
    with _guard:
        if _depth:
            _depth += 1
            try:
                yield
            finally:
                _depth -= 1
            return
        path = Path(f"/tmp/uf-lerobot-record-{os.getuid()}.lock")
        with path.open("a+") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Another CLI/web recording session owns the devices") from exc
            _depth = 1
            try:
                yield
            finally:
                _depth = 0
                fcntl.flock(stream, fcntl.LOCK_UN)


def exclusive_recording(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with recording_lock():
            return function(*args, **kwargs)
    return wrapped
