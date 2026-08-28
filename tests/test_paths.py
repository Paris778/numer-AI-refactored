import pytest

from nmr import paths
from nmr.config import REPO_ROOT


def test_experiment_layout():
    assert paths.EXPERIMENTS_ROOT == REPO_ROOT / "experiments"
    assert paths.experiment_dir("ender-xgb-v1") == REPO_ROOT / "experiments" / "ender-xgb-v1"
    assert paths.run_dir("ender-xgb-v1", "ab" * 32) == (
        REPO_ROOT / "experiments" / "ender-xgb-v1" / "runs" / ("ab" * 32)
    )
    assert paths.run_json_path("f", "a" * 64).name == "run.json"
    assert paths.export_dir("f", "partial", "a" * 64) == (
        REPO_ROOT / "experiments" / "f" / "exports" / "partial" / ("a" * 64)
    )
    assert paths.export_json_path("f", "full", "a" * 64).name == "export.json"
    assert paths.current_pointer_path("f").name == "current.json"
    assert paths.champion_path() == REPO_ROOT / "experiments" / "champion.json"
    assert paths.shared_cache_dir() == REPO_ROOT / "artifacts" / "cache"
    assert paths.shared_reports_dir() == REPO_ROOT / "artifacts" / "reports"

def test_validate_slug_rejects_bad():
    for bad in ("UPPER", "has space", "semi;colon", ""):
        with pytest.raises(ValueError):
            paths.validate_slug(bad)
    assert paths.validate_slug("ender-xgb_v1") == "ender-xgb_v1"


def test_export_dir_rejects_non_hex_run_id():
    """SECONDARY 2: export_dir enforces the 64-hex run-id rule when non-empty —
    a corrupt pointer must never resolve an unexpected path."""
    with pytest.raises(ValueError, match="64-char"):
        paths.export_dir("fam-a", "full", "not-a-run-id")
    with pytest.raises(ValueError, match="64-char"):
        paths.export_dir("fam-a", "full", "..")
    with pytest.raises(ValueError, match="64-char"):
        paths.export_dir("fam-a", "full", "AB" * 32)  # uppercase not allowed
    # Valid 64-hex run ids pass.
    assert paths.export_dir("fam-a", "full", "a" * 64) == (
        REPO_ROOT / "experiments" / "fam-a" / "exports" / "full" / ("a" * 64)
    )


def test_current_pointer_path_resolves_directly():
    """SECONDARY 2: current_pointer_path builds the pointer path directly
    (never through export_dir, whose run_id validation would reject the empty
    form) — the pointer still resolves at the canonical location."""
    assert paths.current_pointer_path("fam-a") == (
        REPO_ROOT / "experiments" / "fam-a" / "exports" / "full" / "current.json"
    )
    assert paths.current_pointer_path("fam-a").name == "current.json"
