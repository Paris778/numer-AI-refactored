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
    (tmp_path / "data" / "numerai_era_data.csv").write_text(
        "date,dataset,start_era,end_era,round_id\n"
        "2026-08-08,train,0001,0002,\n"
        "2026-08-08,validation,0003,0004,\n"
        "2026-08-08,live,X,X,1300.0\n",
        encoding="utf-8",
    )
    return d.parent


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
        "feature_corr_medium.parquet",
        "feature_corr_all_summary.json",
        "set_membership.json",
        "regimes.json",
        "era_signal.parquet",
        "benchmarks.json",
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
        "feature_corr_medium.parquet",
        "era_signal.parquet",
    ]:
        assert (out1 / name).read_bytes() == (out2 / name).read_bytes(), name
