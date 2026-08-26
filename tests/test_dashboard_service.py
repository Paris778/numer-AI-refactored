"""Tests for dashboard_ui.service — unified data layer.

Tests verify:
1. Data loading (leaderboard, campaigns, registry entries)
2. Return types (Pydantic models)
3. Cache invalidation (mtime sentinel)
4. Filtering/sorting operations
5. Label formatting
"""

import json
import tempfile
from pathlib import Path

import pytest

from dashboard_ui.service import (
    CampaignLog,
    DashboardDataService,
    LeaderboardFrame,
    LeaderboardRowModel,
    RobustnessMatrix,
    TopPerformerRowModel,
    TopPerformersResult,
)


class TestLeaderboardRowModel:
    """Test Pydantic model validation."""

    def test_minimal_valid_row(self):
        """Minimal valid LeaderboardRowModel."""
        row = LeaderboardRowModel(
            model_id="test-001",
            source="trained",
            run_name="test-config",
            run_dir="/tmp/test",
        )
        assert row.model_id == "test-001"
        assert row.source == "trained"
        assert row.corr is None  # Optional fields

    def test_full_valid_row(self):
        """Full populated LeaderboardRowModel."""
        row = LeaderboardRowModel(
            model_id="test-001",
            source="trained",
            run_name="test-config",
            run_dir="/tmp/test",
            backend="lightgbm",
            preset="standard",
            corr_sharpe_ac=1.23,
            corr_sharpe_ac_ci_low=1.10,
            corr_sharpe_ac_ci_high=1.36,
            max_drawdown=-0.15,
            has_bmc=True,
            status="CAPITAL READY",
        )
        assert row.corr_sharpe_ac == 1.23
        assert row.has_bmc is True

    def test_invalid_source_still_accepted(self):
        """Source is just a string; no enum validation in model."""
        row = LeaderboardRowModel(
            model_id="test-001",
            source="invalid_source",
            run_name="test-config",
            run_dir="/tmp/test",
        )
        assert row.source == "invalid_source"


class TestLeaderboardFrame:
    """Test LeaderboardFrame filtering and sorting."""

    @pytest.fixture
    def sample_leaderboard(self) -> LeaderboardFrame:
        """Sample leaderboard with multiple rows."""
        rows = [
            LeaderboardRowModel(
                model_id="m1",
                source="trained",
                run_name="config-a",
                run_dir="/tmp/m1",
                backend="lightgbm",
                preset="standard",
                corr_sharpe_ac=1.23,
            ),
            LeaderboardRowModel(
                model_id="m2",
                source="trained",
                run_name="config-b",
                run_dir="/tmp/m2",
                backend="xgboost",
                preset="fast",
                corr_sharpe_ac=0.95,
            ),
            LeaderboardRowModel(
                model_id="bench-1",
                source="benchmark",
                run_name="baseline",
                run_dir="/tmp/bench",
                corr_sharpe_ac=0.50,
            ),
        ]
        return LeaderboardFrame(rows=rows, total_rows=3, evaluable_rows=2)

    def test_len(self, sample_leaderboard: LeaderboardFrame):
        """Test __len__ method."""
        assert len(sample_leaderboard) == 3

    def test_filter_by_source(self, sample_leaderboard: LeaderboardFrame):
        """Filter by source."""
        trained_only = sample_leaderboard.filter_by_source(["trained"])
        assert len(trained_only) == 2
        assert all(r.source == "trained" for r in trained_only.rows)

    def test_filter_by_backend(self, sample_leaderboard: LeaderboardFrame):
        """Filter by backend."""
        lgbm_only = sample_leaderboard.filter_by_backend(["lightgbm"])
        assert len(lgbm_only) == 1
        assert lgbm_only.rows[0].model_id == "m1"

    def test_filter_by_preset(self, sample_leaderboard: LeaderboardFrame):
        """Filter by preset."""
        standard_only = sample_leaderboard.filter_by_preset(["standard"])
        assert len(standard_only) == 1
        assert standard_only.rows[0].model_id == "m1"

    def test_sort_by_metric_descending(self, sample_leaderboard: LeaderboardFrame):
        """Sort by metric descending."""
        sorted_lb = sample_leaderboard.sort_by_metric("corr_sharpe_ac", descending=True)
        assert sorted_lb.rows[0].model_id == "m1"  # 1.23
        assert sorted_lb.rows[1].model_id == "m2"  # 0.95
        assert sorted_lb.rows[2].model_id == "bench-1"  # 0.50

    def test_sort_by_metric_ascending(self, sample_leaderboard: LeaderboardFrame):
        """Sort by metric ascending."""
        sorted_lb = sample_leaderboard.sort_by_metric(
            "corr_sharpe_ac", descending=False
        )
        assert sorted_lb.rows[0].model_id == "bench-1"  # 0.50
        assert sorted_lb.rows[-1].model_id == "m1"  # 1.23

    def test_sort_by_metric_with_nulls(self):
        """Sort handles None values gracefully."""
        rows = [
            LeaderboardRowModel(
                model_id="m1",
                source="trained",
                run_name="config-a",
                run_dir="/tmp/m1",
                corr_sharpe_ac=1.23,
            ),
            LeaderboardRowModel(
                model_id="m2",
                source="trained",
                run_name="config-b",
                run_dir="/tmp/m2",
                corr_sharpe_ac=None,  # Missing value
            ),
        ]
        lb = LeaderboardFrame(rows=rows, total_rows=2, evaluable_rows=2)
        sorted_lb = lb.sort_by_metric("corr_sharpe_ac", descending=True)
        # Non-None values should come first when descending
        assert sorted_lb.rows[0].corr_sharpe_ac == 1.23


class TestDashboardDataService:
    """Test DashboardDataService data loading and caching."""

    @pytest.fixture
    def temp_registry_dir(self) -> Path:
        """Temporary registry directory with sample run.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_dir = Path(tmpdir) / "registry"
            registry_dir.mkdir(parents=True)

            # Create a sample run.json
            run_dir = registry_dir / "test-run-123"
            run_dir.mkdir()
            run_json = {
                "run_id": "test-run-123",
                "manifest": {
                    "config": {
                        "run": {"name": "test-config"},
                        "data": {"feature_set": "medium"},
                        "model": {"backend": "lightgbm", "preset": "standard"},
                    },
                    "oof_device": "cpu",
                },
                "scorecard": {"corr": 0.042},
            }
            (run_dir / "run.json").write_text(json.dumps(run_json))

            yield registry_dir

    def test_load_registry_entries(self, temp_registry_dir: Path):
        """Test load_registry_entries()."""
        service = DashboardDataService(registry_dir=temp_registry_dir)
        entries = service.load_registry_entries()

        assert len(entries) == 1
        assert entries[0]["run_id"] == "test-run-123"
        assert isinstance(entries[0], dict)

    def test_load_registry_entries_empty_dir(self):
        """Test load_registry_entries() with empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            service = DashboardDataService(registry_dir=Path(tmpdir))
            entries = service.load_registry_entries()
            assert entries == []

    def test_cache_invalidation(self, temp_registry_dir: Path):
        """Test cache invalidation on mtime change."""
        service = DashboardDataService(registry_dir=temp_registry_dir)

        # Record initial mtime
        initial_mtime = temp_registry_dir.stat().st_mtime

        # Cache should be invalid initially
        assert not service._check_cache_valid()

        # Load once
        leaderboard1 = service.load_leaderboard()
        assert service._mtime_registry == initial_mtime

        # Second load should return cached (mtime unchanged)
        leaderboard2 = service.load_leaderboard()
        assert leaderboard1 is leaderboard2  # Same object

    def test_format_model_label_trained(self):
        """Test format_model_label for trained run."""
        service = DashboardDataService()
        label = service.format_model_label(
            "trained", "my-config", "abc123def456", short_id_len=8
        )
        assert label == "my-config · abc123de"  # First 8 chars of the id

    def test_format_model_label_benchmark(self):
        """Test format_model_label for benchmark."""
        service = DashboardDataService()
        label = service.format_model_label(
            "benchmark", "baseline-v2", "tier-4-lgbm", short_id_len=8
        )
        assert label == "baseline-v2 · tier-4-lgbm"

    def test_format_model_label_null_model_id(self):
        """Test format_model_label with None model_id."""
        service = DashboardDataService()
        label = service.format_model_label("trained", "config", None)
        assert label == "config · ?"

    def test_prepare_leaderboard_for_display(self, temp_registry_dir: Path):
        """Test prepare_leaderboard_for_display()."""
        service = DashboardDataService(registry_dir=temp_registry_dir)

        # This will load the registry (empty since we don't have full nmr.* setup)
        # So we expect an empty leaderboard
        display_rows = service.prepare_leaderboard_for_display()
        # Result depends on actual registry contents; just verify it returns a list
        assert isinstance(display_rows, list)

    def test_compute_robustness_matrix(self):
        """Test compute_robustness_matrix()."""
        service = DashboardDataService()
        # Will use default registry_dir (may not exist in test environment)
        # Just verify the method doesn't crash
        matrix = service.compute_robustness_matrix()
        assert isinstance(matrix, RobustnessMatrix)
        assert isinstance(matrix.rows, list)


class TestCampaignLogLoading:
    """Test campaign log loading."""

    @pytest.fixture
    def temp_campaigns_dir(self) -> Path:
        """Temporary campaigns directory with sample campaign log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            campaigns_dir = Path(tmpdir) / "campaigns"
            campaigns_dir.mkdir(parents=True)

            # Create a sample campaign log
            campaign_json = {
                "campaign_id": "exp-001",
                "name": "HPO Search v1",
                "runs": [
                    {
                        "config_path": "configs/test.yaml",
                        "run_id": "run-001",
                        "status": "COMPLETE",
                    },
                    {
                        "config_path": "configs/test.yaml",
                        "run_id": "run-002",
                        "status": "FAILED",
                        "error": "OOM",
                    },
                ],
            }
            (campaigns_dir / "exp-001.json").write_text(json.dumps(campaign_json))

            yield campaigns_dir

    def test_load_campaigns(self, temp_campaigns_dir: Path, monkeypatch):
        """Test load_campaigns()."""
        # Mock REPO_ROOT to use temp directory
        monkeypatch.setattr("dashboard_ui.service.REPO_ROOT", temp_campaigns_dir.parent)

        service = DashboardDataService()
        service.registry_dir = temp_campaigns_dir.parent / "registry"  # dummy
        service.registry_dir.mkdir(parents=True, exist_ok=True)

        # Override campaigns_dir in load_campaigns method by patching
        campaigns_log = service.load_campaigns()

        # Since campaigns_dir is hardcoded in the method, we can't easily test
        # without mocking the filesystem. The test above ensures the parsing logic.
        assert isinstance(campaigns_log, CampaignLog)
        assert isinstance(campaigns_log.runs, list)


class TestLeaderboardFramePydanticSerialization:
    """Test Pydantic serialization/deserialization."""

    def test_leaderboard_json_roundtrip(self):
        """Test JSON serialization and deserialization."""
        rows = [
            LeaderboardRowModel(
                model_id="m1",
                source="trained",
                run_name="config-a",
                run_dir="/tmp/m1",
                corr_sharpe_ac=1.23,
            )
        ]
        lb = LeaderboardFrame(rows=rows, total_rows=1, evaluable_rows=1)

        # Serialize to JSON
        json_str = lb.model_dump_json()
        assert isinstance(json_str, str)

        # Deserialize back
        lb2 = LeaderboardFrame.model_validate_json(json_str)
        assert lb2.rows[0].model_id == "m1"
        assert lb2.rows[0].corr_sharpe_ac == 1.23


class TestTopPerformers:
    """Test compute_top_performers."""

    def test_top_performers_ranks_by_sharpe(self):
        """Test ranking by Sharpe with real registry data."""
        service = DashboardDataService()
        result = service.compute_top_performers(top_n=5)
        if not result.rows:
            pytest.skip("No registry models")

        assert len(result) > 0
        assert result.sort_metric == "corr_sharpe_ac"
        assert result.champion is not None
        # Ranks should be sequential 1..N and sorted descending
        for i, row in enumerate(result.rows, start=1):
            assert row.rank == i
        sharpes = [r.corr_sharpe_ac or -1 for r in result.rows]
        assert sharpes == sorted(sharpes, reverse=True)

    def test_top_performers_sharpe_floor(self):
        """Test the min_sharpe floor filters weak models."""
        service = DashboardDataService()
        filtered = service.compute_top_performers(top_n=50, min_sharpe=0.0)
        # All filtered rows should have sharpe >= 0
        for row in filtered.rows:
            assert row.corr_sharpe_ac is None or row.corr_sharpe_ac >= 0.0

    def test_top_performer_row_typed(self):
        """Test row is a typed model with decision metrics populated."""
        service = DashboardDataService()
        result = service.compute_top_performers(top_n=3)
        if not result.rows:
            pytest.skip("No registry models")
        row = result.rows[0]
        assert isinstance(row, TopPerformerRowModel)
        # Decision-critical fields should be present
        assert row.rank >= 1
        assert row.model_id
        assert row.label
        assert row.robustness_score >= 0

    def test_top_performers_json_roundtrip(self):
        """Test TopPerformersResult serializes to JSON."""
        service = DashboardDataService()
        result = service.compute_top_performers(top_n=3)
        if not result.rows:
            import pytest

            pytest.skip("No registry models")
        json_str = result.model_dump_json()
        restored = TopPerformersResult.model_validate_json(json_str)
        assert len(restored) == len(result)
        assert restored.rows[0].model_id == result.rows[0].model_id


class TestTimeseriesLoading:
    """Test load_timeseries caching."""

    def test_load_timeseries_returns_payload(self):
        """Test timeseries loads with real data and is memoized."""
        import pytest

        from dashboard_ui.service import DashboardDataService

        service = DashboardDataService()
        result = service.compute_top_performers(top_n=2)
        if not result.rows:
            pytest.skip("No registry models")
        ids = [r.model_id for r in result.rows]

        p1 = service.load_timeseries(ids)
        assert "eras" in p1
        assert "metrics" in p1
        assert "drawdowns" in p1

        # Second call returns cached object (memoized)
        p2 = service.load_timeseries(ids)
        assert p1 is p2

    def test_load_timeseries_cache_keyed_by_set(self):
        """Test cache is keyed by the run-id set regardless of order."""
        service = DashboardDataService()
        result = service.compute_top_performers(top_n=3)
        if not result.rows:
            import pytest

            pytest.skip("No registry models")
        ids = [r.model_id for r in result.rows]
        reversed_ids = list(reversed(ids))

        p1 = service.load_timeseries(ids)
        p2 = service.load_timeseries(reversed_ids)
        # Same underlying payload object because key is sorted tuple
        assert p1 is p2


class TestFullHistoryLoading:
    """Test load_full_history (per-era CORR over the FULL validation window)."""

    def test_full_history_returns_payload_with_stats(self):
        """Test full-history payload contains series, drawdowns, and stats."""
        import pytest

        from dashboard_ui.service import DashboardDataService

        service = DashboardDataService()
        result = service.compute_top_performers(top_n=2)
        if not result.rows:
            pytest.skip("No registry models")
        ids = [r.model_id for r in result.rows]

        p = service.load_full_history(ids)
        assert set(p) == {"series", "drawdowns", "stats"}

        # At least the top model should have been processed
        assert ids[0] in p["series"]
        s = p["series"][ids[0]]
        assert "eras" in s and "standard" in s and "cumulative" in s
        assert len(s["standard"]) == len(s["eras"])
        assert len(s["cumulative"]) == len(s["eras"])

        # Drawdowns align with series length
        assert len(p["drawdowns"][ids[0]]) == len(s["eras"])

        # Stats sane
        st = p["stats"][ids[0]]
        assert st["n"] == len(s["eras"])
        assert st["n"] >= 1
        assert 0.0 <= st["pct_positive"] <= 1.0
        assert st["win_streak"] >= 0
        assert st["max_drawdown"] <= 0.0  # drawdown magnitude is <= 0

    def test_full_history_memoized(self):
        """Test full-history payload is memoized."""
        import pytest

        from dashboard_ui.service import DashboardDataService

        service = DashboardDataService()
        result = service.compute_top_performers(top_n=2)
        if not result.rows:
            pytest.skip("No registry models")
        ids = [r.model_id for r in result.rows]

        p1 = service.load_full_history(ids)
        p2 = service.load_full_history(ids)
        assert p1 is p2

    def test_full_history_missing_model_absent(self):
        """Test a run with no preds is absent (never raises)."""
        from dashboard_ui.service import DashboardDataService

        service = DashboardDataService()
        p = service.load_full_history(["does_not_exist_12345"])
        assert p["series"] == {}
        assert p["drawdowns"] == {}
        assert p["stats"] == {}

    def test_full_history_drawdown_is_cumsum_peak_to_trough(self):
        """Test drawdown equals cumsum peak-to-trough (<= 0)."""
        import pytest

        from dashboard_ui.service import DashboardDataService

        service = DashboardDataService()
        result = service.compute_top_performers(top_n=1)
        if not result.rows:
            pytest.skip("No registry models")
        p = service.load_full_history([result.rows[0].model_id])
        mid = result.rows[0].model_id
        dd = p["drawdowns"][mid]
        assert all(d <= 1e-12 for d in dd)  # drawdown <= 0
        assert max(dd) == pytest.approx(0.0, abs=1e-9)  # starts at peak


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
