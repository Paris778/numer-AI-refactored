# Contributing to numer-AI-refactored

## Coding Standards

All code must follow the eight non-negotiable principles in [`AGENTS.md`](AGENTS.md) Section 2. Read them before contributing. In brief: `nmr/` is the only tested boundary, determinism is sacred, custom metrics require oracle parity tests, leakage is a correctness bug, fail early and loudly, no magic values, TDD, and no loose ends.

---

## Development Workflow

### Before you start

1. Read [`AGENTS.md`](AGENTS.md) — the authoritative source for principles, invariants, and hazards.
2. Read [`ARCHITECTURE.md`](ARCHITECTURE.md) — pipeline topology, formulas, and module specs.
3. Skim [docs/DOCS_README.md](docs/DOCS_README.md) — the Numerai domain knowledge base (canonical laws, scoring, purge/embargo conventions, tiered reading paths). For domain intuition, [docs/02-strategy/strategy-bible.md](docs/02-strategy/strategy-bible.md) and the [docs/05-notebooks/](docs/05-notebooks/) tutorials; before touching any metric or evaluation code, read [docs/06-evaluation/evaluation-suite-bible.md](docs/06-evaluation/evaluation-suite-bible.md) — the evaluation spec of record.
4. Activate the venv and install dependencies:

   ```powershell
   .\.venv\Scripts\Activate.ps1                                # Windows PowerShell
   .\.venv\Scripts\python -m pip install -r requirements.txt
   ```

   No install step for the package itself — `pythonpath = .` in [pytest.ini](pytest.ini) makes `nmr` importable from the repo root.

5. Ensure the `data/v5.2/` parquet assets are present (see [`README.md`](README.md#data-assets)) — real-data tests and the benchmark runner require them.

### Making changes

1. **Write tests first.** Follow the TDD cycle: failing test → smallest fix → verify → refactor.
2. Keep changes minimal — no unrelated refactors or cosmetic tweaks.
3. **Chase down every loose end.** Deleted functions lose their tests; renamed symbols get every call site updated **including `nmr/__init__.py` imports and `__all__`**; changed metric formulas get their parity tests updated; changed behavior gets docs updated in the same commit (`AGENTS.md`, `ARCHITECTURE.md`, `docs/06-evaluation/evaluation-suite-bible.md` for metric semantics).
4. Run the verification suite — see [Testing](#testing--verification).

### ⚠️ Critical footguns

- **New scorecard/instrumentation fields can break determinism tests.** Anything containing wall-clock time or absolute paths must be excluded from `canonical_scorecards_bytes()` and run-id payloads, or cross-process determinism tests will fail intermittently.
- **Real-v5.2 test fixtures: establish era overlap before limiting rows.** Join/filter validation, meta-model, and benchmark frames by shared eras *first*, then window/limit — otherwise fixtures flake with `NonVacuityError` or empty joins (benchmark train parquet has no rows for the first ~30 train eras).
- **Run pytest from the repo root.** `pythonpath = .` is relative; running from a subdirectory breaks `import nmr`.
- **`standard` / `deep` presets train for hours.** Use the `fast` preset, small feature set, or truncated era windows in tests; never let a test depend on a long training run.
- **There is no ruff/mypy config.** pytest is the only automated gate — do not invent lint commands in docs or CI, and do not add tooling as a side effect of another task.

---

## Testing & Verification

Run the full suite after every change (488 tests; from the repo root):

```powershell
.\.venv\Scripts\python -m pytest -q
```

CI (`.github/workflows/ci.yml`) runs `pytest -q` on Python 3.12 for every push/PR; real-data tests self-skip without `data/v5.2/`.

Useful targeted runs while iterating:

```powershell
.\.venv\Scripts\python -m pytest tests/test_parity.py tests/test_risk_parity.py -q   # oracle parity
.\.venv\Scripts\python -m pytest tests/test_benchmark_slice1.py -q                    # determinism hashes
.\.venv\Scripts\python -m pytest tests/test_runner.py tests/test_registry.py -q       # pipeline + registry
```

### Pre-sign-off gate

Before delivering completed work:

```powershell
.\.venv\Scripts\python -m pytest -q                      # full suite, zero failures
.\.venv\Scripts\python benchmark_runner.py --fast-mode --output artifacts/benchmark_scores_smoke.csv --labels-output artifacts/benchmark_test_era_labels_smoke.csv   # real-data smoke run (writes artifacts/*_smoke.csv)
```

A green unit run without the real-data smoke is not sufficient evidence for changes touching data loading, evaluation, scorecards, or the benchmark harness. Surface any pre-existing failures explicitly — never silently exclude them.

---

## Pull Request Expectations

- PRs must pass the full pytest suite.
- Include tests covering the happy path, error paths, degenerate inputs (zero-variance eras, <2 rows, non-finite values), and — for anything hashed or serialized — cross-process determinism.
- Keep commits focused; each commit tells a coherent story.
- **No secrets in commits.** `.env` (numerapi credentials) is `.gitignore`d. Never commit `data/v5.2/` parquet assets or regenerated `artifacts/` outputs unless the change is specifically about them.

---

## Code Review

Reviewers will check:

- Correctness and test coverage, including parity tests for any metric change
- Determinism: no timing/path contamination in hashed payloads; seeds threaded through config
- Leakage safety: purge invariants intact, no row-level CV, fold assertions untouched
- Boundary discipline: no business logic in scripts or notebooks
- **No loose ends:** `nmr/__init__.py` exports in sync, orphaned tests removed, docs updated in the same commit
- `AGENTS.md` / `ARCHITECTURE.md` accuracy (doc drift is a bug)

Be prepared to explain *why*, not just *what*.
