"""Model metadata and explainers.

Provides human-readable descriptions, architectural context, and performance
profile for each model in the registry. Used for hover cards, drill-downs,
and scenario explanations.
"""

from dataclasses import dataclass

# ============================================================================
# MODEL EXPLAINER SYSTEM
# ============================================================================


@dataclass
class ModelProfile:
    """Human-readable profile for a trained model."""

    model_id: str
    run_name: str
    architecture: str  # "tree", "linear", "ensemble", "hybrid"
    universe: str  # "small", "medium", "large", "derived"
    target: str  # "20d", "60d"

    # Explainer text (what this model does)
    description: str  # One-liner (hover card)
    long_description: str  # Full paragraph (drill-down)

    # Key characteristics (for filtering/discovery)
    tags: list[str]  # ["gc-robust", "stable", "high-ic", "low-corr", "new"]

    # Performance context
    context: str  # e.g., "Trained on 2026-08-18, 657 era validation"

    # Risk/return profile
    risk_profile: str  # "conservative", "moderate", "aggressive"
    diversification: str  # "isolated", "moderate", "highly-correlated"

    # Research notes (why we built this)
    research_intent: str  # "Maximize MMC", "Stress-test GC", "Isolated feature group"

    @property
    def summary(self) -> str:
        """Short summary for display."""
        return f"{self.run_name} ({self.architecture.upper()})"


# ============================================================================
# CURRENT MODEL REGISTRY WITH EXPLAINERS
# ============================================================================
# These are sample profiles. In production, load from database/JSON.


def get_model_profile(model_id: str) -> ModelProfile | None:
    """Lookup model profile by ID.

    First tries the static catalog, then falls back to a dynamically generated
    profile from registry metadata (run.json). This ensures every model in the
    registry has a human-readable explainer.
    """
    profiles = _build_model_catalog()
    static = next((p for p in profiles if p.model_id == model_id), None)
    if static:
        return static

    # Try dynamic generation from registry
    dynamic = _build_dynamic_profile(model_id)
    if dynamic:
        return dynamic

    return None


def _build_dynamic_profile(model_id: str) -> ModelProfile | None:
    """Build a ModelProfile from registry run.json metadata.

    Args:
        model_id: The model/run ID to look up.

    Returns:
        ModelProfile or None if registry data cannot be loaded.
    """
    try:
        from dashboard_ui.service import DashboardDataService

        service = DashboardDataService()
        entries = service.load_registry_entries()
        entry = next((e for e in entries if e.get("run_id") == model_id), None)
        if not entry:
            return None

        manifest = entry.get("manifest", {})
        config = manifest.get("config", {})
        run_cfg = config.get("run", {})
        data_cfg = config.get("data", {})
        model_cfg = config.get("model", {})

        run_name = run_cfg.get("name", model_id[:12])
        backend = model_cfg.get("backend", "unknown")
        preset = model_cfg.get("preset", "standard")
        feature_set = data_cfg.get("feature_set", "medium")
        horizon = data_cfg.get("horizon", "20d")

        # Map backend to architecture family
        architecture = _backend_to_architecture(backend)

        # Derive risk/diversification tags from config
        tags = [backend, preset]
        if feature_set in ["small", "tiny"]:
            tags.append("decorrelated")
        if preset in ["gc_robust", "robust"]:
            tags.append("gc-robust")
        if backend in ["ensemble", "ridge"]:
            tags.append("ensemble")

        # Risk profile heuristic
        risk_profile = _risk_profile_from_config(architecture, feature_set, preset)
        diversification = _diversification_from_feature_set(feature_set)

        description = (
            f"{backend.upper()} model on {feature_set} features, "
            f"optimized for {horizon} horizon ({preset} preset)."
        )
        long_description = (
            f"Trained with the '{run_name}' configuration. Uses {backend} "
            f"backend with {preset} preset over the {feature_set} feature universe "
            f"({_feature_count(feature_set)} features). Targets the {horizon} "
            f"return horizon. {architecture.capitalize()}-based architecture."
        )

        return ModelProfile(
            model_id=model_id,
            run_name=run_name,
            architecture=architecture,
            universe=feature_set,
            target=horizon,
            description=description,
            long_description=long_description,
            tags=tags,
            context=f"Registry entry loaded for {model_id[:12]}...",
            risk_profile=risk_profile,
            diversification=diversification,
            research_intent=f"Explore {backend} performance on {feature_set} features",
        )
    except Exception:
        return None


def _backend_to_architecture(backend: str) -> str:
    """Map model backend to architecture family."""
    tree_backends = {"lightgbm", "xgboost", "catboost", "tree"}
    linear_backends = {"ridge", "linear", "elasticnet"}
    ensemble_backends = {"ensemble", "stack"}
    if backend in tree_backends:
        return "tree"
    if backend in linear_backends:
        return "linear"
    if backend in ensemble_backends:
        return "ensemble"
    return "hybrid"


def _feature_count(feature_set: str) -> int:
    """Approximate feature counts by feature set name."""
    counts = {
        "small": 42,
        "medium": 780,
        "large": 3555,
        "v4": 1050,
        "v5": 3555,
    }
    return counts.get(feature_set.lower(), 780)


def _risk_profile_from_config(architecture: str, feature_set: str, preset: str) -> str:
    """Heuristic risk profile from config."""
    if architecture == "ensemble" or preset in ["robust", "gc_robust"]:
        return "conservative"
    if feature_set in ["small", "tiny"]:
        return "moderate"
    return "moderate"


def _diversification_from_feature_set(feature_set: str) -> str:
    """Heuristic diversification label from feature set."""
    if feature_set in ["small", "tiny"]:
        return "isolated"
    if feature_set in ["large", "v5"]:
        return "highly-correlated"
    return "moderate"


def _build_model_catalog() -> list[ModelProfile]:
    """Build catalog of all models with explainers."""
    return [
        ModelProfile(
            model_id="lgbm_standard_v1",
            run_name="LightGBM Standard v1",
            architecture="tree",
            universe="medium",
            target="20d",
            description="Baseline LightGBM on medium feature set; highest IC reliability",
            long_description=(
                "Trained on 657 validation eras with medium feature universe (780 features). "
                "Standard LightGBM parameters tuned for Sharpe maximization. Serves as "
                "portfolio anchor due to low era-to-era instability. Excellent for "
                "diversification because it captures different signal than ensemble models."
            ),
            tags=["stable", "high-ic", "low-corr-to-ensemble", "anchor-model"],
            context="Trained 2026-08-18, Validated eras 0575-1231 (657 eras)",
            risk_profile="conservative",
            diversification="moderate",
            research_intent="Build reliable portfolio anchor with consistent performance",
        ),
        ModelProfile(
            model_id="xgb_deep_v1",
            run_name="XGBoost Deep v1",
            architecture="tree",
            universe="medium",
            target="60d",
            description="Deeper XGBoost tree ensemble optimized for 60-day Sharpe",
            long_description=(
                "60-day target focuses on lower-frequency regime changes. Deep trees (depth 12) "
                "capture complex interactions in medium-universe features. Adds 0.35 correlation "
                "to lgbm_standard_v1 (beneficial diversification). Best for capturing seasonal effects."
            ),
            tags=["gc-robust", "low-corr", "60d-target", "new"],
            context="Trained 2026-08-20, Validated eras 0575-1231",
            risk_profile="moderate",
            diversification="low-correlation",
            research_intent="Diversify lgbm_standard via longer-horizon signal",
        ),
        ModelProfile(
            model_id="ridge_ensemble_v2",
            run_name="Ridge Ensemble v2",
            architecture="linear",
            universe="medium",
            target="20d",
            description="Linear ridge regression with rank-gaussianized ensemble averaging",
            long_description=(
                "Blends 5 tree-based models (lgbm/xgb variants) with per-era rank-gaussianization. "
                "Linear pooling reduces overfitting from any single tree model. High correlation to "
                "individual trees (~0.75 avg) but smoother drawdown profile. Excellent for "
                "regime-robustness due to averaging effect."
            ),
            tags=["ensemble", "smooth-dd", "regime-robust"],
            context="Trained 2026-08-21, Validated eras 0575-1231",
            risk_profile="conservative",
            diversification="highly-correlated-to-trees",
            research_intent="Reduce overfitting via ensemble averaging; improve Sharpe smoothness",
        ),
        ModelProfile(
            model_id="catboost_gc_v1",
            run_name="CatBoost GC-Robust v1",
            architecture="tree",
            universe="medium",
            target="20d",
            description="CatBoost with explicit GC robustness tuning",
            long_description=(
                "Ordered boosting targets GC immunity via categorical feature prioritization. "
                "Trained with in-fold GC penalties (neutralizes GC signals in loss). Results in "
                "lower IC (0.032 vs 0.042 baseline) but 35% lower GMC impact. Use when GC stress "
                "is primary concern over absolute IC."
            ),
            tags=["gc-immune", "defensive", "gmac-robust"],
            context="Trained 2026-08-19, Validated eras 0575-1231",
            risk_profile="conservative",
            diversification="moderate",
            research_intent="Protect against game-mechanic changes via GC robustness",
        ),
        ModelProfile(
            model_id="isolation_small_v3",
            run_name="Isolation Small Feature Set v3",
            architecture="tree",
            universe="small",
            target="20d",
            description="Minimal feature set (42 features) isolated from full universe",
            long_description=(
                "Tests whether a tiny, hand-curated feature set can compete. Uses only "
                "highest-stability features from pre-modelling study. Lower IC (0.038) but "
                "extreme diversification: -0.15 correlation to medium-universe models! "
                "Captures orthogonal signal; excellent for decorrelation."
            ),
            tags=["decorrelated", "stability-screened", "ultra-low-dd"],
            context="Trained 2026-08-17, Validated eras 0575-1231",
            risk_profile="moderate",
            diversification="extremely-isolated",
            research_intent="Find maximally-decorrelated signal via feature constraint",
        ),
        ModelProfile(
            model_id="lstm_sequential_v1",
            run_name="LSTM Sequential v1",
            architecture="hybrid",
            universe="medium",
            target="60d",
            description="LSTM sequence model capturing era-to-era momentum",
            long_description=(
                "Experimental: LSTM on 20-era sequences predicts next-era residuals. "
                "Tests whether temporal dependencies matter (they may not given randomness). "
                "High variance model; use cautiously. Interesting for understanding if "
                "any momentum/mean-reversion is hidden in our features."
            ),
            tags=["experimental", "high-variance", "momentum"],
            context="Trained 2026-08-22, Validated eras 0575-1231",
            risk_profile="aggressive",
            diversification="high-variance",
            research_intent="Test if temporal sequences add value vs random cross-section",
        ),
    ]


def get_all_profiles() -> list[ModelProfile]:
    """Get all model profiles."""
    return _build_model_catalog()


def get_profiles_by_tag(tag: str) -> list[ModelProfile]:
    """Filter profiles by tag."""
    return [p for p in _build_model_catalog() if tag in p.tags]


def get_profiles_by_architecture(arch: str) -> list[ModelProfile]:
    """Filter profiles by architecture."""
    return [p for p in _build_model_catalog() if p.architecture == arch]


__all__ = [
    "ModelProfile",
    "get_model_profile",
    "get_all_profiles",
    "get_profiles_by_tag",
    "get_profiles_by_architecture",
]
