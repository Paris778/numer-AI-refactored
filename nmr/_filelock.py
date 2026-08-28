"""Small cross-platform inter-process advisory file lock (stdlib only).

The champion read-compare-write (``nmr.registry.promote_if_better``) must be
serialized across processes: without a lock, N concurrent writers can all read
"no champion", compare against the same stale value, and the last write wins —
not the best value. This module provides the exclusive lock used around that
critical section.

Windows uses ``msvcrt.locking`` (byte-range, non-blocking, polled); POSIX uses
``fcntl.flock`` (``LOCK_EX | LOCK_NB``, polled). The lock file
(``<root>/champion.json.lock``) is never removed — unlinking a lock file is a
classic race (a waiter holds an fd to the unlinked inode while a third process
creates a fresh file and locks it). It is a zero-byte-ish marker beside the
pointer it guards.
"""

from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path

__all__ = ["FileLockTimeout", "file_lock"]

# Documented acquisition timeout (seconds): a writer that cannot acquire the
# lock within this window fails loud with a clear error rather than hanging or
# silently proceeding unsynchronized.
LOCK_TIMEOUT_DEFAULT = 30.0
_POLL_INTERVAL = 0.05


class FileLockTimeout(TimeoutError):
    """Raised when the advisory lock cannot be acquired within the timeout."""


@contextlib.contextmanager
def file_lock(
    path: str | Path, *, timeout: float = LOCK_TIMEOUT_DEFAULT
):
    """Acquire an exclusive advisory lock at ``path`` for the enclosed block.

    Blocks up to ``timeout`` seconds, polling non-blocking acquisition; on
    expiry raises :class:`FileLockTimeout` with clear text. The lock is always
    released (and the fd closed) on exit, including exceptions.
    """
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        # msvcrt.locking locks a byte range starting at the CURRENT file
        # position; ensure the file holds at least one byte so the range is
        # valid and rewind to position 0 (os.write advanced the cursor).
        if os.fstat(fd).st_size == 0:
            os.write(fd, b"\x00")
            os.fsync(fd)
        os.lseek(fd, 0, os.SEEK_SET)
        deadline = time.monotonic() + timeout
        while True:
            try:
                _try_lock(fd)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise FileLockTimeout(
                        f"timed out after {timeout:.0f}s waiting for the "
                        f"advisory lock at {lock_path} — another writer holds "
                        "it (the champion read-compare-write is single-writer)"
                    ) from None
                time.sleep(_POLL_INTERVAL)
        try:
            yield
        finally:
            _unlock(fd)
    finally:
        os.close(fd)


def _try_lock(fd: int) -> None:
    """Non-blocking exclusive lock; raises OSError on contention."""
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(fd: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)
