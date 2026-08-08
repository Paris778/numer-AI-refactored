"""Pure logic for the round-aware Numerai data refresh (nmr/refresh.py)."""

from __future__ import annotations

import pytest

from nmr.config import REPO_ROOT, load_config
from nmr.refresh import (
    CURRENT_DATA_VERSION,
    _parse_version,
    detect_newer_version,
)


def test_current_version_is_v5_2() -> None:
    assert CURRENT_DATA_VERSION == "v5.2"


def test_parse_version_valid() -> None:
    assert _parse_version("v5.2") == (5, 2)
    assert _parse_version("v5.10") == (5, 10)  # multi-digit minor
    assert _parse_version("v6.0") == (6, 0)
    assert _parse_version("v0.0") == (0, 0)


@pytest.mark.parametrize(
    "bad",
    ["5.2", "v5", "v5.2.1", "vX.2", "v5.a", "", "v5.2 "],
)
def test_parse_version_malformed_raises(bad: str) -> None:
    with pytest.raises(ValueError):
        _parse_version(bad)


def test_detect_newer_version_none_cases() -> None:
    assert detect_newer_version([], "v5.2") is None
    assert detect_newer_version(["v5.2"], "v5.2") is None
    assert detect_newer_version(["v5.0", "v5.1"], "v5.2") is None


def test_detect_newer_version_finds_newest() -> None:
    assert detect_newer_version(["v5.3"], "v5.2") == "v5.3"
    assert detect_newer_version(["v6.0"], "v5.2") == "v6.0"
    # multi-digit regression: v5.10 > v5.3 numerically, not lexicographically
    assert detect_newer_version(["v5.10"], "v5.3") == "v5.10"
    assert (
        detect_newer_version(["v4.9", "v5.3", "v5.2", "v5.10"], "v5.2")
        == "v5.10"
    )


def test_detect_newer_version_malformed_raises() -> None:
    with pytest.raises(ValueError):
        detect_newer_version(["v5.2", "garbage"], "v5.2")


def test_drift_guard_current_version_matches_canonical_config() -> None:
    cfg_path = REPO_ROOT / "configs" / "first_model.yaml"
    if not cfg_path.exists():
        pytest.skip("configs/first_model.yaml absent in this checkout")
    cfg = load_config(cfg_path)
    assert CURRENT_DATA_VERSION == cfg.data.version
