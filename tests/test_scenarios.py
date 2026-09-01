"""Tests for nmr.scenarios module."""

import pytest

from nmr.scenarios import (
    AllocationConstraints,
    AllocationScenario,
    ModelAllocation,
    ScenarioEngine,
)


class TestAllocationConstraints:
    """Test AllocationConstraints."""

    def test_default_constraints(self):
        """Test default constraint values."""
        c = AllocationConstraints()
        assert c.min_sharpe_threshold == 0.5
        assert c.max_single_weight == 0.25
        assert c.min_models == 3
        assert c.max_models == 10


class TestModelAllocation:
    """Test ModelAllocation."""

    def test_allocation_creation(self):
        """Test ModelAllocation creation."""
        a = ModelAllocation(
            model_id="m1",
            weight=0.25,
            corr_sharpe_ac=1.2,
            conviction="high",
            reason="Top Sharpe",
        )
        assert a.model_id == "m1"
        assert a.weight == 0.25


class TestAllocationScenario:
    """Test AllocationScenario."""

    def test_empty_scenario(self):
        """Test empty scenario."""
        s = AllocationScenario(scenario_id="empty", name="Empty")
        assert s.num_models == 0
        assert s.total_weight == 0.0
        assert not s.is_valid

    def test_valid_scenario(self):
        """Test valid scenario."""
        s = AllocationScenario(
            scenario_id="test",
            name="Test",
            allocations=[
                ModelAllocation("m1", 0.25, 1.0, "high", "reason"),
                ModelAllocation("m2", 0.25, 0.9, "high", "reason"),
                ModelAllocation("m3", 0.25, 0.8, "medium", "reason"),
                ModelAllocation("m4", 0.25, 0.7, "medium", "reason"),
            ],
            portfolio_volatility=0.1,
            portfolio_max_drawdown=0.1,
        )
        assert s.num_models == 4
        assert s.is_valid


class TestScenarioEngine:
    """Test ScenarioEngine."""

    @pytest.fixture
    def base_models(self):
        """Sample base models."""
        eras = [f"{era:04d}" for era in range(1, 9)]
        return [
            {
                "model_id": f"m{index}",
                "corr_sharpe_ac": sharpe,
                "max_drawdown": drawdown,
                "payout_by_era": {
                    era: 0.01 * ((-1) ** (era_index + index)) + 0.002 * index
                    for era_index, era in enumerate(eras)
                },
            }
            for index, (sharpe, drawdown) in enumerate(
                [(1.2, 0.05), (1.0, 0.08), (0.8, 0.12), (0.6, 0.15)],
                start=1,
            )
        ]

    @pytest.fixture
    def engine(self, base_models):
        """ScenarioEngine fixture."""
        return ScenarioEngine(base_models)

    def test_apply_constraints(self, engine, base_models):
        """Test constraint-based scenario."""
        constraints = AllocationConstraints(
            min_sharpe_threshold=0.7,
            max_single_weight=0.34,
            max_combined_weight=0.68,
        )
        scenario = engine.scenario_apply_constraints(base_models, constraints)

        assert len(scenario.allocations) == 3
        assert all(a.corr_sharpe_ac >= 0.7 for a in scenario.allocations)

    def test_remove_model(self, engine, base_models):
        """Test remove model scenario."""
        baseline = engine.scenario_apply_constraints(
            base_models, AllocationConstraints()
        )
        original = tuple((a.model_id, a.weight) for a in baseline.allocations)
        original_metrics = (
            baseline.portfolio_sharpe,
            baseline.portfolio_volatility,
            baseline.portfolio_max_drawdown,
        )
        scenario = engine.scenario_remove_model(baseline, "m1", rebalance=True)

        assert "m1" not in [a.model_id for a in scenario.allocations]
        assert scenario.sharpe_vs_baseline is not None
        assert tuple((a.model_id, a.weight) for a in baseline.allocations) == original
        assert (
            baseline.portfolio_sharpe,
            baseline.portfolio_volatility,
            baseline.portfolio_max_drawdown,
        ) == original_metrics

    def test_reweight_model(self, engine, base_models):
        """Test reweight model scenario."""
        baseline = engine.scenario_apply_constraints(
            base_models, AllocationConstraints()
        )
        original = tuple((a.model_id, a.weight) for a in baseline.allocations)
        scenario = engine.scenario_reweight_model(baseline, "m1", 0.35)

        m1 = next(a for a in scenario.allocations if a.model_id == "m1")
        assert m1.weight == 0.35
        assert tuple((a.model_id, a.weight) for a in baseline.allocations) == original

    def test_regime_robust(self, engine, base_models):
        """Test regime-robust scenario."""
        scenario = engine.scenario_regime_robust(base_models)

        assert len(scenario.allocations) > 0
        assert scenario.name == "Regime-Robust Portfolio"
        assert [allocation.model_id for allocation in scenario.allocations] == [
            "m1",
            "m2",
        ]

    def test_portfolio_metrics_computed(self, engine, base_models):
        """Test portfolio metrics computation."""
        scenario = engine.scenario_apply_constraints(
            base_models, AllocationConstraints()
        )

        assert scenario.portfolio_sharpe is not None
        assert scenario.portfolio_max_drawdown is not None
        assert scenario.portfolio_volatility is not None
        assert scenario.metrics_reason is None

    def test_metrics_are_unavailable_without_observed_payout_series(self, base_models):
        models = [
            {key: value for key, value in model.items() if key != "payout_by_era"}
            for model in base_models
        ]
        scenario = ScenarioEngine(models).scenario_apply_constraints(
            models, AllocationConstraints()
        )

        assert scenario.portfolio_sharpe is None
        assert scenario.portfolio_volatility is None
        assert scenario.portfolio_max_drawdown is None
        assert scenario.metrics_reason == "observed_payout_series_unavailable"

    def test_degenerate_model_correlation_is_unavailable_not_nan(self):
        models = [
            {
                "model_id": "m1",
                "corr_sharpe_ac": 1.0,
                "payout_by_era": {"0001": 0.1, "0002": 0.1},
            },
            {
                "model_id": "m2",
                "corr_sharpe_ac": 1.0,
                "payout_by_era": {"0001": 0.2, "0002": 0.2},
            },
        ]
        scenario = AllocationScenario(
            "degenerate",
            "Degenerate",
            allocations=(
                ModelAllocation("m1", 0.5, 1.0, "high", "test"),
                ModelAllocation("m2", 0.5, 1.0, "high", "test"),
            ),
            constraints=AllocationConstraints(min_models=2, max_single_weight=0.5),
        )

        result = ScenarioEngine(models)._compute_portfolio_metrics(scenario)

        assert result.avg_correlation is None
        assert result.metrics_reason is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
