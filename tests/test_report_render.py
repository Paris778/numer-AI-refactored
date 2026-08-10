"""Renderer tests: deterministic Markdown from synthetic dump content."""

from __future__ import annotations

import json

import render_dataset_report


def _fixture() -> dict:
    return {
        "manifest": {
            "data_version": "v5.3",
            "feature_set": "small",
            "feature_count": 2,
            "target_count": 1,
            "era_ranges": {"train": "0001..0002", "validation": "0003..0004"},
            "refresh_date": "2026-08-08",
        },
        "overview": {
            "splits": {
                "train": {"n_rows": 20, "n_eras": 2, "min_era": "0001", "max_era": "0002"},
                "validation": {"n_rows": 20, "n_eras": 2, "min_era": "0003", "max_era": "0004"},
            },
            "n_features": 2,
            "targets": ["target"],
            "feature_sets": {"small": 2, "medium": 3, "all": 3},
        },
        "era_structure_rows": [
            {"era": "0001", "n_rows": 10, "n_ids": 10, "gap": False},
            {"era": "0002", "n_rows": 10, "n_ids": 10, "gap": False},
        ],
        "targets": {
            "target": {
                "n_eras_present": 4,
                "missing_rate": 0.0,
                "pooled_mean": 0.0,
                "pooled_std": 1.0,
            }
        },
        "target_corr_rows": [
            {"target_a": "target", "target_b": "target", "mean_corr": 1.0, "n_eras": 4}
        ],
        "feature_summary_rows": [
            {"feature": "f1", "pooled_mean": 0.1, "pooled_std": 1.0, "missing_rate": 0.0}
        ],
        "ic_screen_rows": [
            {
                "feature": "f1",
                "target": "target",
                "mean_corr": 0.05,
                "mean_spearman": 0.06,
                "n_eras": 4,
                "stable": True,
                "nonlinear": False,
            }
        ],
        "split_ic_rows": [
            {
                "feature": "f1",
                "train_mean_ic": 0.05,
                "train_n_eras": 2,
                "val_mean_ic": 0.02,
                "val_n_eras": 2,
                "delta_ic": -0.03,
            }
        ],
        "drift_rows": [
            {
                "feature": "f1",
                "psi": 0.05,
                "w1": 0.12,
                "auc_roc": 0.55,
                "n_train": 20,
                "n_val": 20,
                "drifted": False,
            }
        ],
        "meta_ortho_rows": [
            {
                "feature": "f1",
                "meta": "numerai_meta_model",
                "corr_meta": 0.001,
                "corr_target": 0.05,
                "n_eras": 86,
                "orthogonal": True,
            }
        ],
        "redundancy_rows": [
            {"feature_set": "small", "n_features": 1, "mean_abs_corr": 0.1, "n_pairs": 0}
        ],
        "regime": {
            "regime_thresholds": {"q1": -0.01, "q3": 0.01},
            "crash_eras": ["0001"],
            "hot_eras": ["0004"],
            "ic_persistence": {"mean": 0.5, "std": 0.1, "n_adjacent": 3},
        },
        "era_signal_rows": [
            {
                "era": "0001",
                "mean_ic": -0.02,
                "regime": "low",
                "crash": True,
                "hot": False,
                "degenerate": False,
            },
            {
                "era": "0002",
                "mean_ic": 0.0,
                "regime": "unlabeled",
                "crash": False,
                "hot": False,
                "degenerate": True,
            },
        ],
        "benchmark_rows": [
            {"benchmark": "benchmark_small", "mean_corr": 0.03, "n_eras": 4}
        ],
        "fne_profile": [
            {"signal": "benchmark_small", "proportion": 0.0, "mean_ic": 0.03, "n_eras": 4},
            {"signal": "benchmark_small", "proportion": 1.0, "mean_ic": 0.001, "n_eras": 4},
        ],
        "corr_summary": {"mean_abs_corr": 0.2, "min_eigenvalue": 0.5, "top_pairs": []},
        "set_membership": {"sets": {"small": {"n_features": 2}}},
    }


def test_render_report_deterministic() -> None:
    md1 = render_dataset_report.render_report(**{**_fixture()})
    md2 = render_dataset_report.render_report(**{**_fixture()})
    assert md1 == md2


def test_render_report_structure() -> None:
    md = render_dataset_report.render_report(**{**_fixture()})
    assert md.startswith("# Dataset Analysis")
    for header in ["## 1. Dataset Overview", "## 2. Era Structure", "## 3. Targets",
                   "## 4. Features", "## 5. Regimes & Signal Dynamics",
                   "## 6. Benchmarks & Meta-Model", "## 7. Modeling Implications",
                   "### 4.3 Cross-Split Drift (PSI + W1 + Adversarial AUC)",
                   "### 4.4 Signal by Split", "### 4.5 Set Redundancy",
                   "unlabeled", "min eigenvalue", "orthogonal=True"]:
        assert header in md
    # schema blocks precede tables; takeaways present
    assert "**Schema:**" in md
    assert "Key takeaways" in md


def test_render_report_escapes_markdown_special_chars() -> None:
    fx = _fixture()
    fx["feature_summary_rows"] = [
        {"feature": "feat|name", "pooled_mean": 0.1, "pooled_std": 1.0, "missing_rate": 0.0}
    ]
    md = render_dataset_report.render_report(**fx)
    assert "feat\\|name" in md  # escaped pipe inside table cell


def test_main_rejects_version_mismatch(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps({"data_version": "v5.4", "feature_count": 2}), encoding="utf-8"
    )
    rc = render_dataset_report.main(
        ["--dumps-dir", str(tmp_path), "--output", str(tmp_path / "out.md")]
    )
    assert rc == 1
    assert not (tmp_path / "out.md").exists()


def test_render_report_meta_watermark() -> None:
    md = render_dataset_report.render_report(**{**_fixture()})
    assert "SMALL SAMPLE: 86 ERAS — HIGH SAMPLING VARIANCE" in md
    # the watermark sits directly above the meta-model table section
    assert md.index("SMALL SAMPLE") < md.index("corr_meta")


def test_render_report_campaign_section() -> None:
    kwargs = {**_fixture()}
    kwargs["campaign_rows"] = [
        {
            "variant": "lgbm_v2", "status": "recorded", "backend": "lightgbm",
            "device": "cpu", "n_features": 3, "mean_ic": 0.012,
            "ic_ci_lo": 0.004, "ic_ci_hi": 0.020, "ic_sharpe": 0.7,
            "max_drawdown": -0.15, "fne100": 0.003, "fne100_ci_lo": 0.0,
            "fne100_ci_hi": 0.006, "n_eras": 640,
        }
    ]
    kwargs["pairwise_rows"] = [
        {
            "pair": "lgbm_v2 vs lgbm_v3", "backend": "lightgbm",
            "mean_diff": -0.003, "ci_low": -0.007, "ci_high": 0.001,
            "n_eras": 640, "error": None,
        }
    ]
    md = render_dataset_report.render_report(**kwargs)
    assert "### 7.1 Feature Campaign" in md
    assert "### 7.2 Paired Screen Verdicts" in md
    assert "lgbm_v2" in md
    assert "fne100" in md
    assert "lgbm_v2 vs lgbm_v3" in md
