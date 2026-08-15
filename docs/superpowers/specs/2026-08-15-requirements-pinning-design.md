# Requirements.txt Full Direct-Dependency Pinning — Design Spec

- **Date:** 2026-08-15
- **Status:** Design approved (Approach A locked). Implementation complete — all four plan tasks executed and reviewed; changeset staged for user authorization to commit.
- **Scope:** Pin every unpinned direct dependency in `requirements.txt` to the exact
  versions in the verified venv (the versions behind 651 green tests), and prove the pin
  set installs and passes from a clean environment.
- **Target systems:** `requirements.txt` only, plus SSOT documentation
  (`CONTRIBUTING.md`, `README.md`, `AGENTS.md`). No `nmr/` code, no test code, no CI
  workflow changes.

## 1. Context & Problem

`requirements.txt` is partially pinned: `numerapi`, `cloudpickle`, `numerai-tools`,
`optuna`, `catboost`, `streamlit`, `plotly`, `cupy-cuda12x`, and the `nvidia-*` wheels
carry `==` pins, but the core compute engine — `numpy`, `scipy`, `pandas`, `polars`,
`lightgbm`, `xgboost`, `scikit-learn`, `pyarrow`, `pyyaml` — and the test/notebook
tooling (`pytest`, `pytest-cov`, `matplotlib`, `ipykernel`, `tqdm`) are unpinned.

This breaks the repo's determinism doctrine in two concrete ways:

1. **Silent model drift.** A fresh `pip install -r requirements.txt` (CI included) can
   resolve a newer lightgbm/xgboost minor release with different tree-split heuristics or
   thread defaults, changing OOF predictions and invalidating historical scorecard
   comparisons and `run_id` lineage.
2. **API/ABI breakage.** Polars changes APIs across minors; numpy 1.x→2.x-style
   transitions break compiled C-extensions (scipy, lightgbm, cupy) when wheels mismatch.

Verified versions recorded 2026-08-15 from the venv that passed the full 651-test suite
and the benchmark smoke gate — these are the versions to lock in.

## 2. Locked Decisions

1. **Approach A (exact-pin direct deps in place).** Rejected: full `pip freeze`
   (Approach C — the venv is shared with the legacy `../numer-AI/` repo and would leak
   legacy-only transitive packages into this manifest) and lockfile tooling
   (Approach B — uv/pip-tools adds tooling creep for zero incremental benefit once direct
   deps are pinned; tree-split and metric behavior is driven entirely by direct deps).
2. **Pin set = all 14 unpinned direct deps**, including test/notebook tooling (CI runs
   pytest from this file; a pytest major bump can break the suite on its own).
3. **Existing pins untouched.** `numerapi==2.22.0`, `cloudpickle==3.1.1`,
   `numerai-tools==0.5.3`, `optuna==4.9.0`, `catboost==1.2.10`, `streamlit==1.61.1`,
   `plotly==6.6.0`, `cupy-cuda12x==14.1.1`, and all eight `nvidia-*` wheels stay exactly
   as-is.
4. **File order preserved.** In-place edits only — the diff is 14 line changes.
5. **Transitive deps remain unpinned.** Accepted residual risk, consistent with the
   repo's direct-dep manifest style; upgrading a pin is a deliberate act (see §6).

## 3. Change Spec

### 3.1 `requirements.txt`

Exact final content (only the 14 unpinned lines change):

```text
numerapi==2.22.0
pytest==9.0.2
pytest-cov==7.1.0
pandas==2.3.3
polars==1.41.2
pyarrow==23.0.1
matplotlib==3.10.8
lightgbm==4.6.0
xgboost==3.2.0
scikit-learn==1.8.0
scipy==1.17.1
numpy==2.4.1
pyyaml==6.0.3
cloudpickle==3.1.1
numerai-tools==0.5.3
optuna==4.9.0
catboost==1.2.10
streamlit==1.61.1
plotly==6.6.0
ipykernel==7.1.0
tqdm==4.67.1
# GPU acceleration (user-granted deps): cupy rankdata for the analysis
# pipeline; nvidia-* wheels provide the CUDA runtime DLLs cupy needs on
# Windows. Optional at runtime — everything degrades to scipy without them.
cupy-cuda12x==14.1.1
nvidia-cublas-cu12==12.9.2.10
nvidia-cuda-nvrtc-cu12==12.9.86
nvidia-cuda-runtime-cu12==12.9.79
nvidia-cufft-cu12==11.4.1.4
nvidia-curand-cu12==10.3.10.19
nvidia-cusolver-cu12==11.7.5.82
nvidia-cusparse-cu12==12.5.10.65
nvidia-nvjitlink-cu12==12.9.86
```

Rationale for each pin value: exact version installed in the verified venv at
`C:/dev/numer-AI-refactored/.venv` on 2026-08-15 (measured via
`./.venv/Scripts/python -m pip list --format=freeze`), behind the green full suite
(651 passed) and benchmark smoke.

### 3.2 SSOT documentation (same changeset)

- **`CONTRIBUTING.md`** (§ setup): add a short "Dependency pinning policy" note —
  all direct dependencies are exact-pinned to the verified venv; upgrading a pin is a
  deliberate edit (change the `==` version, reinstall via
  `.\.venv\Scripts\python -m pip install -r requirements.txt`, re-run the gates), never a
  casual `pip install -U`.
- **`README.md`** (project-tree comment): `requirements.txt  # runtime + dev dependencies`
  → append `(all exact-pinned)`.
- **`AGENTS.md`**: extend the dependency-prohibition line (§3) with "all direct
  dependencies are exact-pinned in `requirements.txt`". No test-count changes (no tests
  added or removed — count stays 651).

## 4. Verification Plan

Clean-room proof, executed inside the repo root (no installs outside the working
directory):

1. Create throwaway venv: `./.venv/Scripts/python -m venv .venv-verify`
2. Install the pin set:
   `./.venv-verify/Scripts/python -m pip install -r requirements.txt`
3. Dependency integrity: `./.venv-verify/Scripts/python -m pip check`
4. Fast gate from the clean venv: `./.venv-verify/Scripts/python -m pytest -q`
   — must be 651 passed, 0 failed (real-data tests use the present `data/v5.3/` assets).
5. Confirm key versions resolve to the pins via `pip list --format=freeze`.
6. Delete `.venv-verify/` (created by this task; safe to remove). If any step fails,
   keep the venv for forensics, report the failing step, and only delete after the gate
   is green.

CI (`.github/workflows/ci.yml`, ubuntu, Python 3.12) re-proves cross-platform on the next
push: fresh `pip install -r requirements.txt` + `pytest -q`.

## 5. Non-Goals

- No lockfile tooling (uv/pip-tools), no hashed requirements, no transitive freeze.
- No CI workflow changes, no Python version pinning beyond CI's existing 3.12.
- No reordering or reformatting of `requirements.txt` beyond the 14 in-place pin edits.
- No changes to `nmr/`, tests, or `.gitignore`.

## 6. Risks & Mitigations

- **Shared legacy venv drift:** `../numer-AI/` shares the venv; a future install there
  could drift versions away from the pins. Mitigation: the documented deliberate-upgrade
  path in `CONTRIBUTING.md`; re-verify via the clean-room protocol when any pin changes.
- **Ubuntu wheel availability:** all pinned versions are mainstream releases with manylinux
  wheels for cp312; if any were unavailable, CI fails loudly on push — the intended
  behavior.
- **Residual transitive float:** accepted (Locked Decision 5).
- **GPU wheels on CI:** `cupy-cuda12x` and `nvidia-*` install fine on Linux runners and
  degrade to scipy fallback without a GPU — already the status quo, unchanged by this
  work.
