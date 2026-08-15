# Requirements.txt Full Direct-Dependency Pinning — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Exact-pin all 14 unpinned direct dependencies in `requirements.txt` to the verified-venv versions and prove the pin set installs and passes the full suite from a clean environment.

**Architecture:** No code changes — this is a manifest + documentation change. The pin set locks the versions behind the green 651-test suite (measured 2026-08-15); reproducibility is proven by a throwaway clean venv (`.venv-verify/`) that installs from the pinned manifest, passes `pip check`, and runs the full fast gate. CI re-proves cross-platform on the next push.

**Tech Stack:** pip (24.x, Python 3.12 venv), pytest, Git Bash on Windows.

**Spec:** `docs/superpowers/specs/2026-08-15-requirements-pinning-design.md` — the authoritative contract for pin values, doc edits, and verification protocol.

## Global Constraints

- Pin values must exactly match the spec §3.1 inventory — 14 pins: `pytest==9.0.2`, `pytest-cov==7.1.0`, `pandas==2.3.3`, `polars==1.41.2`, `pyarrow==23.0.1`, `matplotlib==3.10.8`, `lightgbm==4.6.0`, `xgboost==3.2.0`, `scikit-learn==1.8.0`, `scipy==1.17.1`, `numpy==2.4.1`, `pyyaml==6.0.3`, `ipykernel==7.1.0`, `tqdm==4.67.1`.
- Existing pins untouched: `numerapi==2.22.0`, `cloudpickle==3.1.1`, `numerai-tools==0.5.3`, `optuna==4.9.0`, `catboost==1.2.10`, `streamlit==1.61.1`, `plotly==6.6.0`, `cupy-cuda12x==14.1.1`, and all eight `nvidia-*` lines — byte-identical to current file.
- File order preserved; diff must be exactly 14 changed lines (14 insertions / 14 deletions).
- All pip commands via `./.venv/Scripts/python -m pip` or `./.venv-verify/Scripts/python -m pip` — NEVER the `Scripts/pip` shim (it targets the legacy repo's venv).
- Run pytest from the repo root (`pytest.ini` sets `pythonpath = .` relative to cwd).
- **No git commits or pushes without explicit user authorization** (standing instruction). When a task reaches its commit step: stage with `git add`, report the staged state, and wait for the user's word.
- All commands execute inside `C:/dev/numer-AI-refactored`; `.venv-verify/` must be deleted after a green gate (kept only for forensics on failure).
- No CI workflow changes, no `.gitignore` changes, no `nmr/` or `tests/` changes.

---

### Task 1: Pin the 14 direct dependencies in `requirements.txt`

**Files:**
- Modify: `requirements.txt` (whole file, exact content below)

**Interfaces:**
- Consumes: nothing (first task).
- Produces: a fully pinned `requirements.txt` consumed by Task 3's clean-room install and CI.

- [ ] **Step 1: Replace `requirements.txt` with the exact pinned content**

Overwrite the file with exactly this (matches spec §3.1; identical to the current file except the 14 `==` pins):

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

- [ ] **Step 2: Verify the diff is exactly the 14 pin lines**

Run:

```bash
git diff --numstat requirements.txt
```

Expected: `14	14	requirements.txt` (14 insertions, 14 deletions, nothing else).

- [ ] **Step 3: Verify no unpinned requirement line remains**

Run:

```bash
grep -nE '^[a-zA-Z0-9._-]+$' requirements.txt
```

Expected: no output (every package line carries `==x.y.z`; comment and blank lines are excluded by the character class).

- [ ] **Step 4: Verify the pins match the verified venv (dry-run resolution)**

Run:

```bash
./.venv/Scripts/python -m pip install --dry-run -r requirements.txt 2>&1 | tail -5
```

Expected: only `Requirement already satisfied` lines and a final `Would install` section that is empty or absent — no package should need downloading or changing.

- [ ] **Step 5: Stage and report**

```bash
git add requirements.txt
```

Report the numstat/dry-run outputs. Do NOT commit until the user authorizes it.

---

### Task 2: SSOT documentation updates

**Files:**
- Modify: `CONTRIBUTING.md` (after the venv-install block, line ~23)
- Modify: `README.md:67` (project-tree comment)
- Modify: `AGENTS.md:82` (dependency-prohibition bullet)

**Interfaces:**
- Consumes: nothing from Task 1 (docs-only; the pin values referenced come from the spec inventory).
- Produces: doc statements that downstream reviewers grep to confirm the pinning policy exists.

- [ ] **Step 1: Add the pinning-policy note to `CONTRIBUTING.md`**

In `CONTRIBUTING.md`, replace:

```
### Making changes
```

with:

```
### Dependency pinning policy

All direct dependencies in `requirements.txt` are exact-pinned (`==x.y.z`) to the
versions in the verified venv (2026-08-15: the versions behind the green 651-test
suite). Upgrading a pin is a deliberate act, never a casual `pip install -U`:

1. Edit the pin in `requirements.txt`.
2. Reinstall: `.\.venv\Scripts\python -m pip install -r requirements.txt`
3. Re-run the full suite and the pre-sign-off gate.

### Making changes
```

- [ ] **Step 2: Update the `README.md` tree comment**

In `README.md`, replace:

```
├── requirements.txt           # runtime + dev dependencies
```

with:

```
├── requirements.txt           # runtime + dev dependencies (all exact-pinned)
```

- [ ] **Step 3: Extend the `AGENTS.md` dependency-prohibition bullet**

In `AGENTS.md`, replace the tail of the line-82 bullet:

```
cupy + NVIDIA runtime wheels (analysis rankdata — imported only in `nmr/_gpu.py`; optional at runtime, automatic scipy fallback; §8).
```

with:

```
cupy + NVIDIA runtime wheels (analysis rankdata — imported only in `nmr/_gpu.py`; optional at runtime, automatic scipy fallback; §8). All direct dependencies are exact-pinned in `requirements.txt`; upgrading a pin is a deliberate act (see `CONTRIBUTING.md`).
```

- [ ] **Step 4: Verify the AGENTS.md size budget**

Run:

```bash
wc -c AGENTS.md
```

Expected: ≤ 32768 bytes (the hard 32 KB budget — a few hundred bytes of growth is fine; if it exceeds, trim elsewhere in AGENTS.md, never skip the edit).

- [ ] **Step 5: Verify all three edits landed and no stale "unpinned" claim remains**

Run:

```bash
grep -n "exact-pinned" CONTRIBUTING.md README.md AGENTS.md
grep -rniE "unpinned" CONTRIBUTING.md README.md AGENTS.md ARCHITECTURE.md || echo "no stale claims"
```

Expected: three `exact-pinned` hits (one per file), and `no stale claims`.

- [ ] **Step 6: Stage and report**

```bash
git add CONTRIBUTING.md README.md AGENTS.md
```

Report the grep/wc outputs. Do NOT commit until the user authorizes it.

---

### Task 3: Clean-room verification (`.venv-verify/` install + full fast gate)

**Files:**
- Create (transient, deleted on success): `.venv-verify/`, `.venv-verify-install.log`

**Interfaces:**
- Consumes: the pinned `requirements.txt` from Task 1; the present `data/v5.3/` assets and `pytest.ini` from the repo.
- Produces: evidence (log excerpts + pytest count) that the pinned manifest installs from clean and passes all 651 tests.

- [ ] **Step 1: Create the throwaway venv**

```bash
./.venv/Scripts/python -m venv .venv-verify
```

- [ ] **Step 2: Install from the pinned manifest (background, poll the log)**

The install pulls multi-hundred-MB wheels (lightgbm, xgboost, catboost, cupy, scipy) and can take 5–20 minutes. Start it detached and poll:

```bash
nohup ./.venv-verify/Scripts/python -m pip install -r requirements.txt > .venv-verify-install.log 2>&1 &
```

Poll every ~60s with:

```bash
tail -3 .venv-verify-install.log
```

Expected completion: the last line is `Successfully installed ...` (a long package list) and no `error:` lines. If it errors, keep `.venv-verify/` for forensics and report the log tail.

- [ ] **Step 3: Dependency integrity check**

```bash
./.venv-verify/Scripts/python -m pip check
```

Expected: `No broken requirements found.`

- [ ] **Step 4: Spot-check that the resolved versions equal the pins**

```bash
./.venv-verify/Scripts/python -m pip list --format=freeze | grep -E '^(numpy|scipy|pandas|polars|pyarrow|lightgbm|xgboost|scikit-learn|PyYAML|pytest|pytest-cov|matplotlib|ipykernel|tqdm)=='
```

Expected output exactly:

```
numpy==2.4.1
scipy==1.17.1
pandas==2.3.3
polars==1.41.2
pyarrow==23.0.1
lightgbm==4.6.0
xgboost==3.2.0
scikit-learn==1.8.0
PyYAML==6.0.3
matplotlib==3.10.8
pytest==9.0.2
pytest-cov==7.1.0
ipykernel==7.1.0
tqdm==4.67.1
```

(Order may vary; values must match.)

- [ ] **Step 5: Full fast gate from the clean venv**

From the repo root:

```bash
./.venv-verify/Scripts/python -m pytest -q
```

Expected: `651 passed` (runtime ~100s). If it overflows the foreground timeout, it moves to background — poll with `TaskOutput`/`tail` until the summary line appears. Real-data tests exercise the present `data/v5.3/` assets.

- [ ] **Step 6: Delete the throwaway venv (green path only)**

```bash
rm -rf .venv-verify .venv-verify-install.log
```

Do this ONLY after Step 5 is green. If anything failed, keep both for forensics and report which step failed.

- [ ] **Step 7: Report**

Report: install log tail, `pip check` output, version spot-check, pytest count. No commit — nothing from this task is tracked (verify with `git status --short` that it shows only the Task 1/Task 2 files).

---

### Task 4: Final gate on the real venv + delivery report

**Files:**
- None modified (verification only).

**Interfaces:**
- Consumes: Tasks 1–3 complete and green.
- Produces: the delivery report for user review (AGENTS.md §4 format).

- [ ] **Step 1: Confirm the real venv is unchanged and green**

```bash
./.venv/Scripts/python -m pytest -q
```

Expected: `651 passed` (the real venv already matched the pins — this is a regression sanity, not a re-install).

- [ ] **Step 2: Confirm the working tree contains only the intended changes**

```bash
git status --short
git diff --numstat
```

Expected: modified `requirements.txt` (14/14), `CONTRIBUTING.md`, `README.md`, `AGENTS.md`; the spec file `docs/superpowers/specs/2026-08-15-requirements-pinning-design.md` and this plan are already committed/untracked per their own authorization state; no `.venv-verify` residue, no other files.

- [ ] **Step 3: Delivery report + commit/push request**

Report per AGENTS.md §4: summary, affected files, verification evidence (clean-room install + 651 green twice + pip check), risks (residual transitive float, CI proof deferred to next push). Then ask the user for authorization to commit and push the changeset.

---

## Self-Review Notes

- **Spec coverage:** §3.1 pins → Task 1; §3.2 docs → Task 2; §4 verification → Task 3 (+ Task 4 re-gate); §5 non-goals → Global Constraints; §6 risks → Task 4 report. No gaps.
- **Pin values** in Task 1 Step 1 and Task 3 Step 4 were cross-checked against the spec inventory and the measured `pip list --format=freeze` output.
- **No code interfaces** in this plan (manifest + docs only), so no cross-task type consistency risks.
