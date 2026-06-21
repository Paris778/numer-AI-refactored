"""Shared pytest fixtures for the nmr test suite."""

from __future__ import annotations

from pathlib import Path

import pytest
from nmr.config import REPO_ROOT


@pytest.fixture
def example_config_path() -> Path:
    """Path to the checked-in example experiment config."""
    return REPO_ROOT / "configs" / "example.yaml"
