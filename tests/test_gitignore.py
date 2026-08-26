import subprocess

from nmr.config import REPO_ROOT

TRACKED: list[str] = [
    "experiments/demo/README.md",
    "experiments/demo/base_config.yaml",
    "experiments/demo/meta.json",
    "experiments/demo/runs/abc/run.json",
    "experiments/demo/exports/full/abc/export.json",
    "experiments/demo/exports/partial/abc/scorecard.json",
    "experiments/demo/exports/full/current.json",
    "experiments/champion.json",
]
IGNORED: list[str] = [
    "experiments/demo/runs/abc/oof.parquet",
    "experiments/demo/runs/abc/validation_preds.parquet",
    "experiments/demo/runs/abc/predict.pkl",
    "experiments/demo/exports/full/abc/predict.pkl",
]


def _check_ignore(path: str) -> bool:
    out = subprocess.run(
        ["git", "check-ignore", "-q", path], cwd=REPO_ROOT, capture_output=True
    )
    return out.returncode == 0


def test_gitignore_classification() -> None:
    for p in TRACKED:
        assert not _check_ignore(p), f"{p} must be trackable"
    for p in IGNORED:
        assert _check_ignore(p), f"{p} must be ignored"
    assert not _check_ignore("experiments/.gitkeep")
