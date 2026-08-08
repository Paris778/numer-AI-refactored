"""Integration tests for refresh_data.py with a mocked NumerAPI."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

import refresh_data


class FakeNapi:
    """Records download_dataset calls; serves fixed API responses."""

    def __init__(
        self,
        *,
        round_num: int | None = 1294,
        datasets: list[str] | None = None,
        fail_on: str | None = None,
    ) -> None:
        self.round_num = round_num
        self.datasets = datasets or [
            f"v5.3/{name}"
            for name in (
                "features.json",
                "train.parquet",
                "validation.parquet",
                "live.parquet",
                "train_benchmark_models.parquet",
                "validation_benchmark_models.parquet",
                "live_benchmark_models.parquet",
                "meta_model.parquet",
                "live_example_preds.parquet",
                "live_example_preds.csv",
                "validation_example_preds.parquet",
                "validation_example_preds.csv",
            )
        ]
        self.fail_on = fail_on
        self.downloads: list[tuple[str, str]] = []

    def get_current_round(self) -> int | None:
        return self.round_num

    def list_datasets(self) -> list[str]:
        return list(self.datasets)

    def download_dataset(self, filename: str, dest_path: str | Path) -> None:
        self.downloads.append((filename, str(dest_path)))
        if self.fail_on is not None and self.fail_on in filename:
            raise ConnectionError(f"simulated failure on {filename}")
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if filename.endswith("features.json"):
            dest.write_text(
                json.dumps(
                    {
                        "feature_sets": {"small": ["f1", "f2"], "medium": ["f1", "f2"]},
                        "targets": ["target"],
                    }
                ),
                encoding="utf-8",
            )
        elif filename.endswith(".csv"):
            dest.write_text("id,era\nn1,0001\n", encoding="utf-8")
        else:
            is_live = filename.endswith("live.parquet")
            pl.DataFrame(
                {
                    "era": ["X", "X"] if is_live else ["0001", "0002"],
                    "id": ["n1", "n2"],
                    "target": [0.0, 1.0],
                }
            ).write_parquet(dest)


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    return d


@pytest.fixture(autouse=True)
def _fake_napi(monkeypatch: pytest.MonkeyPatch) -> FakeNapi:
    fake = FakeNapi()

    class _Napi:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def get_current_round(self) -> int | None:
            return fake.get_current_round()

        def list_datasets(self) -> list[str]:
            return fake.list_datasets()

        def download_dataset(self, filename: str, dest_path: str | Path) -> None:
            fake.download_dataset(filename, dest_path)

    monkeypatch.setattr(refresh_data.numerapi, "NumerAPI", _Napi)
    return fake


def _era_csv(data_dir: Path) -> Path:
    return data_dir / "numerai_era_data.csv"


def test_dry_run_writes_nothing(data_dir: Path, _fake_napi: FakeNapi) -> None:
    rc = refresh_data.main(
        ["--data-dir", str(data_dir), "--era-csv", str(_era_csv(data_dir)), "--dry-run"]
    )
    assert rc == 0
    assert _fake_napi.downloads == []
    assert not _era_csv(data_dir).exists()


def test_fresh_refresh_writes_manifest(data_dir: Path, _fake_napi: FakeNapi) -> None:
    rc = refresh_data.main(
        ["--data-dir", str(data_dir), "--era-csv", str(_era_csv(data_dir))]
    )
    assert rc == 0
    downloaded = {Path(f).name for f, _ in _fake_napi.downloads}
    assert "live.parquet" in downloaded
    assert "validation.parquet" in downloaded
    assert "features.json" in downloaded
    csv_text = _era_csv(data_dir).read_text(encoding="utf-8")
    assert "live" in csv_text and "1294" in csv_text
    assert "X" in csv_text  # live era serialization


def test_no_refresh_when_up_to_date(data_dir: Path, _fake_napi: FakeNapi) -> None:
    rc = refresh_data.main(
        ["--data-dir", str(data_dir), "--era-csv", str(_era_csv(data_dir))]
    )
    assert rc == 0
    n_downloads = len(_fake_napi.downloads)
    rc = refresh_data.main(
        ["--data-dir", str(data_dir), "--era-csv", str(_era_csv(data_dir))]
    )
    assert rc == 0
    # second run: round unchanged, all files present -> no new downloads
    assert len(_fake_napi.downloads) == n_downloads


def test_failed_download_does_not_write_csv(
    data_dir: Path, _fake_napi: FakeNapi
) -> None:
    _fake_napi.fail_on = "validation.parquet"
    with pytest.raises(ConnectionError):
        refresh_data.main(
            ["--data-dir", str(data_dir), "--era-csv", str(_era_csv(data_dir))]
        )
    assert not _era_csv(data_dir).exists()


def test_none_round_aborts(data_dir: Path, _fake_napi: FakeNapi) -> None:
    _fake_napi.round_num = None
    rc = refresh_data.main(
        ["--data-dir", str(data_dir), "--era-csv", str(_era_csv(data_dir))]
    )
    assert rc == 1
    assert _fake_napi.downloads == []


def test_check_only_newer_version_exit_3(data_dir: Path, _fake_napi: FakeNapi) -> None:
    _fake_napi.datasets.append("v5.4/live.parquet")
    rc = refresh_data.main(
        ["--data-dir", str(data_dir), "--era-csv", str(_era_csv(data_dir)), "--check-only"]
    )
    assert rc == 3
    assert _fake_napi.downloads == []


def test_check_only_all_current_exit_0(data_dir: Path, _fake_napi: FakeNapi) -> None:
    rc = refresh_data.main(
        ["--data-dir", str(data_dir), "--era-csv", str(_era_csv(data_dir))]
    )
    assert rc == 0
    rc = refresh_data.main(
        ["--data-dir", str(data_dir), "--era-csv", str(_era_csv(data_dir)), "--check-only"]
    )
    assert rc == 0


def test_strict_newer_version_aborts(data_dir: Path, _fake_napi: FakeNapi) -> None:
    _fake_napi.datasets.append("v5.4/live.parquet")
    rc = refresh_data.main(
        ["--data-dir", str(data_dir), "--era-csv", str(_era_csv(data_dir)), "--strict"]
    )
    assert rc == 3
    assert _fake_napi.downloads == []


def test_live_only_skips_expanding(data_dir: Path, _fake_napi: FakeNapi) -> None:
    rc = refresh_data.main(
        [
            "--data-dir", str(data_dir),
            "--era-csv", str(_era_csv(data_dir)),
            "--live-only",
        ]
    )
    assert rc == 0
    downloaded = {Path(f).name for f, _ in _fake_napi.downloads}
    assert "live.parquet" in downloaded
    assert "validation.parquet" not in downloaded


def test_csv_round_trip_matches_legacy_format(data_dir: Path, _fake_napi: FakeNapi) -> None:
    import pandas as pd

    refresh_data.main(
        ["--data-dir", str(data_dir), "--era-csv", str(_era_csv(data_dir))]
    )
    df = pd.read_csv(_era_csv(data_dir))
    live = df[df["dataset"] == "live"].iloc[0]
    assert live["round_id"] == 1294.0  # legacy float format
    assert live["start_era"] == "X" and live["end_era"] == "X"
    train = df[df["dataset"] == "train"].iloc[0]
    assert pd.isna(train["round_id"])  # empty serialization
    assert train["start_era"] == "0001" and train["end_era"] == "0002"
