"""Integration tests for analyze_dataset.py on tiny synthetic data."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

import analyze_dataset


@pytest.fixture
def fake_data(tmp_path: Path) -> Path:
    """A minimal v5.3 data dir: features.json + tiny train/validation parquets."""
    d = tmp_path / "data" / "v5.3"
    d.mkdir(parents=True)
    (d / "features.json").write_text(
        json.dumps(
            {
                "feature_sets": {
                    "small": ["f_alpha", "f_beta"],
                    "medium": ["f_alpha", "f_beta", "f_gamma"],
                    "all": ["f_alpha", "f_beta", "f_gamma"],
                },
                "targets": ["target_alpha_20", "target_beta_60", "target"],
            }
        ),
        encoding="utf-8",
    )
    rows = []
    for e in range(4):
        era = f"{e + 1:04d}"
        for i in range(10):
            rows.append(
                {
                    "era": era,
                    "id": f"{era}_{i}",
                    "f_alpha": float(i),
                    "f_beta": float(i % 3),
                    "f_gamma": float(10 - i),
                    "target": float(i),
                    "target_alpha_20": float(i),
                    "target_beta_60": float(9 - i),
                }
            )
    train = pl.DataFrame(rows[:20])
    valid = pl.DataFrame(rows[20:])
    train.write_parquet(d / "train.parquet")
    valid.write_parquet(d / "validation.parquet")
    bench_rows = [
        {"era": "0003", "id": f"0003_{i}", "bench_a": float(i), "data_type": "validation"}
        for i in range(10)
    ] + [
        {"era": "0004", "id": f"0004_{i}", "bench_a": float(i), "data_type": "validation"}
        for i in range(10)
    ]
    pl.DataFrame(bench_rows).write_parquet(
        d / "validation_benchmark_models.parquet"
    )
    (tmp_path / "data" / "numerai_era_data.csv").write_text(
        "date,dataset,start_era,end_era,round_id\n"
        "2026-08-08,train,0001,0002,\n"
        "2026-08-08,validation,0003,0004,\n"
        "2026-08-08,live,X,X,1300.0\n",
        encoding="utf-8",
    )
    return d.parent


def test_resolve_reference_targets_default_picks_primary_and_first_60() -> None:
    targets = ["target", "target_cyrusd_20", "target_agnes_60", "target_cyrusd_60"]
    assert analyze_dataset._resolve_reference_targets(targets, None, False) == [
        "target",
        "target_agnes_60",
    ]


def test_resolve_reference_targets_missing_primary_raises() -> None:
    targets = ["target_cyrusd_20", "target_agnes_60"]  # primary 'target' gone
    with pytest.raises(ValueError, match="'target'"):
        analyze_dataset._resolve_reference_targets(targets, None, False)


def test_resolve_reference_targets_missing_60_raises() -> None:
    targets = ["target", "target_cyrusd_20"]  # no distinct 60-day target
    with pytest.raises(ValueError, match="_60"):
        analyze_dataset._resolve_reference_targets(targets, None, False)


def test_resolve_reference_targets_explicit_and_all_flag() -> None:
    targets = ["target", "target_agnes_60"]
    assert analyze_dataset._resolve_reference_targets(
        targets, ["target_cyrusd_20"], False
    ) == ["target_cyrusd_20"]
    assert analyze_dataset._resolve_reference_targets(targets, None, True) == targets


def test_analyze_writes_all_dumps(tmp_path: Path, fake_data: Path) -> None:
    out = tmp_path / "dumps"
    rc = analyze_dataset.main(
        [
            "--data-dir", str(fake_data),
            "--output-dir", str(out),
            "--features", "small",
            "--max-eras", "3",
        ]
    )
    assert rc == 0
    expected = [
        "overview.json",
        "era_structure.parquet",
        "targets.json",
        "target_corr.parquet",
        "feature_summary.parquet",
        "feature_ic_screen.parquet",
        "feature_ic_by_era.parquet",
        "feature_ic_by_split.parquet",
        "feature_corr_medium.parquet",
        "feature_corr_medium_matrix.parquet",
        "feature_corr_all_summary.json",
        "feature_set_redundancy.json",
        "feature_drift_psi.parquet",
        "feature_drift_profile.parquet",
        "derived_feature_sets.json",
        "set_membership.json",
        "regimes.json",
        "era_signal.parquet",
        "benchmarks.json",
        "neutralized_ic.json",
        "meta_ortho.parquet",
        "manifest.json",
    ]
    for name in expected:
        assert (out / name).exists(), name
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["data_version"] == "v5.3"
    assert manifest["feature_count"] == 2
    assert "generated_at" in manifest
    overview = json.loads((out / "overview.json").read_text(encoding="utf-8"))
    assert set(overview["splits"]) == {"train", "validation"}
    # benchmarks.json exists even without a benchmark parquet
    assert "benchmarks" in json.loads(
        (out / "benchmarks.json").read_text(encoding="utf-8")
    )
    # FNE profile: bench_a neutralized against the medium set (11-point grid)
    fne = json.loads((out / "neutralized_ic.json").read_text(encoding="utf-8"))
    assert len(fne["profile"]) == 11
    assert {r["signal"] for r in fne["profile"]} == {"bench_a"}
    assert {r["proportion"] for r in fne["profile"]} == set(
        round(i / 10, 1) for i in range(11)
    )
    # redundancy artifact is a JSON array of per-set summaries
    redundancy = json.loads(
        (out / "feature_set_redundancy.json").read_text(encoding="utf-8")
    )
    assert {"all", "medium", "small"} <= {r["feature_set"] for r in redundancy}
    # split profile exists with the expected columns
    split_ic = pl.read_parquet(out / "feature_ic_by_split.parquet")
    assert set(split_ic.columns) == {
        "feature",
        "train_mean_ic",
        "train_n_eras",
        "val_mean_ic",
        "val_n_eras",
        "delta_ic",
    }


def test_analyze_only_targets_runs_subset(tmp_path: Path, fake_data: Path) -> None:
    out = tmp_path / "only_targets"
    rc = analyze_dataset.main(
        [
            "--data-dir", str(fake_data),
            "--output-dir", str(out),
            "--features", "small",
            "--only", "targets",
        ]
    )
    assert rc == 0
    assert (out / "overview.json").exists()  # dep of manifest
    assert (out / "targets.json").exists()
    assert (out / "target_corr.parquet").exists()
    assert (out / "manifest.json").exists()
    assert not (out / "feature_ic_screen.parquet").exists()
    assert not (out / "feature_ic_by_era.parquet").exists()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stages_run"] == ["overview", "targets", "manifest"]
    assert "hardware" in manifest


def test_analyze_only_regimes_includes_dependency(tmp_path: Path, fake_data: Path) -> None:
    out = tmp_path / "only_regimes"
    rc = analyze_dataset.main(
        [
            "--data-dir", str(fake_data),
            "--output-dir", str(out),
            "--features", "small",
            "--only", "regimes",
        ]
    )
    assert rc == 0
    assert (out / "feature_ic_by_era.parquet").exists()  # auto-included dep
    assert (out / "regimes.json").exists()
    assert (out / "era_signal.parquet").exists()
    assert not (out / "feature_summary.parquet").exists()


def test_analyze_skip_excludes_stage(tmp_path: Path, fake_data: Path) -> None:
    out = tmp_path / "skip_corr"
    rc = analyze_dataset.main(
        [
            "--data-dir", str(fake_data),
            "--output-dir", str(out),
            "--features", "small",
            "--skip", "corr_all",
        ]
    )
    assert rc == 0
    assert not (out / "feature_corr_all_summary.json").exists()
    assert (out / "feature_ic_screen.parquet").exists()
    assert (out / "regimes.json").exists()


def test_analyze_unknown_stage_returns_error(tmp_path: Path, fake_data: Path) -> None:
    out = tmp_path / "bad"
    rc = analyze_dataset.main(
        [
            "--data-dir", str(fake_data),
            "--output-dir", str(out),
            "--only", "bogus",
        ]
    )
    assert rc == 1
    assert not (out / "manifest.json").exists()


def test_analyze_only_and_skip_conflict(tmp_path: Path, fake_data: Path) -> None:
    out = tmp_path / "conflict"
    rc = analyze_dataset.main(
        [
            "--data-dir", str(fake_data),
            "--output-dir", str(out),
            "--only", "targets",
            "--skip", "overview",
        ]
    )
    assert rc == 1


def test_analyze_deterministic_dumps(tmp_path: Path, fake_data: Path) -> None:
    out1 = tmp_path / "d1"
    out2 = tmp_path / "d2"
    analyze_dataset.main(
        ["--data-dir", str(fake_data), "--output-dir", str(out1), "--features", "small"]
    )
    analyze_dataset.main(
        ["--data-dir", str(fake_data), "--output-dir", str(out2), "--features", "small"]
    )
    for name in [
        "era_structure.parquet",
        "feature_summary.parquet",
        "feature_ic_screen.parquet",
        "feature_ic_by_era.parquet",
        "feature_ic_by_split.parquet",
        "feature_corr_medium.parquet",
        "feature_corr_medium_matrix.parquet",
        "feature_drift_psi.parquet",
        "feature_drift_profile.parquet",
        "derived_feature_sets.json",
        "meta_ortho.parquet",
        "era_signal.parquet",
    ]:
        assert (out1 / name).read_bytes() == (out2 / name).read_bytes(), name


def test_derived_sets_stage_content(tmp_path: Path, fake_data: Path) -> None:
    out = tmp_path / "dumps"
    rc = analyze_dataset.main(
        [
            "--data-dir", str(fake_data),
            "--output-dir", str(out),
            "--features", "small",
            "--max-eras", "3",
        ]
    )
    assert rc == 0
    payload = json.loads(
        (out / "derived_feature_sets.json").read_text(encoding="utf-8")
    )
    sets = payload["feature_sets"]
    assert set(sets) == {
        "screen_stable",
        "screen_nonlinear",
        "screen_linear_or_nonlinear",
        "screen_drift_filtered",
    }
    all_feats = {"f_alpha", "f_beta", "f_gamma"}
    for name, values in sets.items():
        assert sorted(values) == values, name
        assert set(values) <= all_feats, name
    # drift-filtered is a subset of linear-or-nonlinear (drift rows only exist
    # for medium features; features without a drift row are kept)
    assert set(sets["screen_drift_filtered"]) <= set(
        sets["screen_linear_or_nonlinear"]
    )
    # linear-or-nonlinear is exactly the union of the two flags
    assert set(sets["screen_linear_or_nonlinear"]) == set(
        sets["screen_stable"]
    ) | set(sets["screen_nonlinear"])


def test_derived_sets_missing_inputs_returns_error(
    tmp_path: Path, fake_data: Path
) -> None:
    out = tmp_path / "dumps"
    with pytest.raises(RuntimeError, match="screens"):
        analyze_dataset.main(
            [
                "--data-dir", str(fake_data),
                "--output-dir", str(out),
                "--only", "derived_sets",
            ]
        )
    assert not (out / "derived_feature_sets.json").exists()
