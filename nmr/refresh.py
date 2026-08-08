"""Round-aware Numerai dataset refresh policy — pure logic, no I/O, no numerapi.

``refresh_data.py`` performs all downloads and file I/O; this module only
decides *what* to do, given facts the script has already gathered (round
numbers, era ranges, available versions). Deterministic: same inputs, same
outputs.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

__all__ = [
    "CURRENT_DATA_VERSION",
    "detect_newer_version",
]

# The data version this repo's pipeline targets. Drift-guarded by a test
# asserting equality with configs/first_model.yaml's data.version.
CURRENT_DATA_VERSION = "v5.2"

_VERSION_RE = re.compile(r"^v(\d+)\.(\d+)$")


def _parse_version(v: str) -> tuple[int, int]:
    """Parse ``v<major>.<minor>`` into integers, or raise ``ValueError``."""
    match = _VERSION_RE.match(v)
    if match is None:
        raise ValueError(
            f"Unrecognized dataset version {v!r}: expected 'v<major>.<minor>' "
            "(e.g. 'v5.2'); patch components are not supported"
        )
    return int(match.group(1)), int(match.group(2))


def detect_newer_version(available: Sequence[str], current: str) -> str | None:
    """Return the numerically-greatest version in ``available`` that strictly
    exceeds ``current``, or ``None``. Malformed entries raise (fail loudly —
    a strange filename in the API listing is a real signal)."""
    current_parsed = _parse_version(current)
    newest: tuple[tuple[int, int], str] | None = None
    for item in available:
        parsed = _parse_version(item)
        if parsed > current_parsed and (newest is None or parsed > newest[0]):
            newest = (parsed, item)
    return newest[1] if newest is not None else None
