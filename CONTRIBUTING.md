# Contributing to numer-AI-refactored

## Coding Standards

All code must follow the eight non-negotiable principles in [`AGENTS.md`](AGENTS.md) Section 2. Read them before contributing.

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

   No install step for the package itself (see the pytest footgun under Critical footguns below).

5. Ensure the `data/v5.3/` parquet assets are present (see [`README.md`](README.md#data-assets)) — real-data tests and the benchmark runner require them.

### Dependency pinning policy

All direct dependencies in `requirements.txt` are exact-pinned (`==x.y.z`) to the
versions in the verified venv (2026-08-15: the versions behind the green 717-test
suite). Upgrading a pin is a deliberate act, never a casual `pip install -U`:

1. Edit the pin in `requirements.txt`.
2. Reinstall: `.\.venv\Scripts\python -m pip install -r requirements.txt`
3. Re-run the full suite and the pre-sign-off gate.

Dev tooling (ruff) is pinned separately in `requirements-dev.txt` (also exact-pinned);
it is never added to the runtime `requirements.txt`.

### Making changes

1. **Write tests first.** Follow the TDD cycle: failing test → smallest fix → verify → refactor.
2. Keep changes minimal — no unrelated refactors or cosmetic tweaks.
3. **Chase down every loose end.** Deleted functions lose their tests; renamed symbols get every call site updated **including `nmr/__init__.py` imports and `__all__`**; changed metric formulas get their parity tests updated; changed behavior gets its docs updated in the same commit (see the Self-Update Directive in [`AGENTS.md`](AGENTS.md)).
4. Run the verification suite — see [Testing](#testing--verification).

### ⚠️ Critical footguns

- **Never use `./.venv/Scripts/pip` — it is a shim into the legacy `../numer-AI/.venv`** (shared site-packages; installing through it touches the legacy repo's environment). Always `./.venv/Scripts/python -m pip install ...`. Verified 2026-08-09: `Scripts/pip --version` reports pip 24.0 from `C:\dev\numer-AI\.venv`, while `python -m pip` targets this repo's venv.
- **cupy needs the NVIDIA runtime DLLs on PATH on Windows.** The `cupy-cuda12x` wheel does not bundle them; the `nvidia-*-cu12` wheels (pinned in `requirements.txt`) ship them, and `nmr/_gpu.py` adds their `bin/` dirs to PATH at load. If cupy import fails with a `cublas` DLL error, check that those wheels are installed.
- **New scorecard/instrumentation fields can break determinism tests.** Anything containing wall-clock time or absolute paths must be excluded from `canonical_scorecards_bytes()` and run-id payloads, or cross-process determinism tests will fail intermittently.
- **Real-v5.3 test fixtures: establish era overlap before limiting rows.** See the era-overlap-before-limit rule and the benchmark-parquet gap in [`AGENTS.md`](AGENTS.md#8-critical-operational-hazards).
- **Run pytest from the repo root.** `pythonpath = .` is relative; running from a subdirectory breaks `import nmr`.
- **`standard` / `deep` presets train for hours.** Use the `fast` preset, small feature set, or truncated era windows in tests; never let a test depend on a long training run.
- **Ruff lint gate (adopted 2026-08-16).** `ruff check .` (config `ruff.toml`) is part of CI and the pre-sign-off gate. Install: `./.venv/Scripts/python -m pip install -r requirements-dev.txt` (never `Scripts/pip`). Fix findings, or add a scoped `# noqa: <code>` with a reason — never suppress a whole rule or edit `ruff.toml` casually.

---

## Testing & Verification

Run the full gate after every change (from the repo root):

```powershell
.\.venv\Scripts\python -m ruff check .   # lint gate (E/F/I/UP @120, ruff.toml)
.\.venv\Scripts\python -m pytest -q      # functional gate
```

CI (`.github/workflows/ci.yml`) runs `ruff check .` + `pytest -q` on Python 3.12 for every push/PR; real-data tests self-skip without `data/v5.3/`.

**End-of-session requirement:** at the end of a coding session (before stopping or handing off for review), re-run `ruff check .` and `pytest -q` on the final state and confirm both are clean. A session is not finished while the linter or test suite is dirty.

Useful targeted runs while iterating:

```powershell
.\.venv\Scripts\python -m pytest tests/test_parity.py tests/test_risk_parity.py -q   # oracle parity
.\.venv\Scripts\python -m pytest tests/test_benchmark_hierarchy.py -q              # benchmark determinism hashes
.\.venv\Scripts\python -m pytest tests/test_runner.py tests/test_registry.py -q       # pipeline + registry
```

### Pre-sign-off gate

Before delivering completed work:

```powershell
.\.venv\Scripts\python -m ruff check .                   # lint gate, zero findings
.\.venv\Scripts\python -m pytest -q                      # full suite, zero failures
.\.venv\Scripts\python benchmark_runner.py --fast-mode   # real-data smoke run (writes artifacts/reports/benchmark_hierarchy_scorecard_smoke.csv + benchmark_gate_report_smoke.csv)
```

A green unit run without the real-data smoke is not sufficient evidence for changes touching data loading, evaluation, scorecards, or the benchmark harness. Surface any pre-existing failures explicitly — never silently exclude them.

---

## Pull Request Expectations

- PRs must pass the full pytest suite.
- Include tests covering the happy path, error paths, degenerate inputs (zero-variance eras, <2 rows, non-finite values), and — for anything hashed or serialized — cross-process determinism.
- Keep commits focused; each commit tells a coherent story.
- **No secrets in commits.** `.env` (numerapi credentials) is `.gitignore`d. Never commit `data/v5.3/` parquet assets or regenerated `artifacts/` outputs unless the change is specifically about them.

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
