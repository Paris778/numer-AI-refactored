"""Atomic file writes: temp file in the target directory + fsync + os.replace.

This is the single implementation of the repo's atomic-write contract
(AGENTS.md §9). Every registry JSON write, artifact payload/manifest write,
OOF parquet write, and neutralization-cache write goes through these helpers.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically (temp + fsync + os.replace)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=target.parent, prefix=f"{target.name}.tmp.", suffix=".part"
    )
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, target)
    finally:
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def atomic_write_text(
    path: str | Path, text: str, *, encoding: str = "utf-8"
) -> None:
    """Write ``text`` to ``path`` atomically (UTF-8 by default)."""
    atomic_write_bytes(path, text.encode(encoding))
