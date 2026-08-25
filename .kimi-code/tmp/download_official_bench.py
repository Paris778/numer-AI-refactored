"""One-off: download official v5.3 validation_benchmark_models.parquet.

Loads credentials from the repo .env via python-dotenv (same pattern the
notebooks use). Never prints secret values. Writes to a temp path so the
existing file is untouched until the merge step.
"""
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # repo .env -> os.environ

from numerapi import NumerAPI  # noqa: E402

out = Path("data/v5.3/validation_benchmark_models_official.parquet")
napi = NumerAPI()
print("downloading v5.3/validation_benchmark_models.parquet ...")
napi.download_dataset(
    "v5.3/validation_benchmark_models.parquet",
    str(out),
)
print("downloaded to", out, "size=", out.stat().st_size if out.exists() else "MISSING")
sys.exit(0)
