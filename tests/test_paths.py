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
