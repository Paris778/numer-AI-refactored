# Design Spec: Extensible Model Backend Registry

> Status: IMPLEMENTED - Tasks 1-8 accepted 2026-09-08; durable owner facts now
> live in `AGENTS.md`, `ARCHITECTURE.md`, `CODEBASE.md`, and
> `CONTRIBUTING.md`. This record remains as the detailed design provenance.
>
> Scope: add a deterministic backend protocol and explicit per-run registry,
> preserve the existing tree backend behavior, and add a protocol-test Ridge
> backend. The capital-evidence and promotion chain is frozen by this design.

## 1. Mission

The framework needs a flexible model-development layer without turning every
new model into a central `nmr/models.py` branch and without allowing arbitrary
research code to become capital evidence. The design separates three levels:

| Level | Capability | Requirement |
|---|---|---|
| 1 | Train and score any foreign model | Emit `(era, id, prediction)` and use `PredictionSet` |
| 2 | Run a model inside purged OOF/checkpoint infrastructure | Implement and explicitly register `BackendAdapter` |
| 3 | Promote or deploy a model | Satisfy the unchanged capital-evidence, artifact, and promotion contracts |

Level 1 is intentionally unlimited: PyTorch, TabPFN, sklearn, notebook
prototypes, external services, and custom ensembles do not enter `nmr/models.py`
or the built-in registry merely to be research-scored.

## 2. Non-Negotiable Constraints

1. `ModelOrchestrator` remains the owner of fold construction, leakage
   assertions, OOF sequencing, full-history roles, checkpoint lifecycle, RAM
   guards, device resolution/fallback, seed propagation, and chunked prediction.
2. The registry and adapters own only backend-specific behavior: parameter
   translation, model construction, fit/predict invocation, capability
   declaration, and canonical backend identity.
3. Existing LightGBM, XGBoost, and CatBoost behavior must be parity-preserved
   before Ridge is enabled in the runner.
4. No dynamic module discovery, entry-point loading, import-order registration,
   process-global mutable registry, or implicit dependency loading.
5. `CapitalContext`, `capital_evidence`, validation-window derivation, exact
   key coverage, promotion authorization, pointer repair, and submission
   acceptance do not change in this work.
6. GPU determinism is device-specific. CPU/GPU byte parity is neither required
   nor claimed.
7. New selected-backend code enters run and checkpoint identity. A changed
    selected adapter, selected registry entry, resolver, dependency identity,
    or backend version cannot reuse old fit artifacts; adding an unrelated
    backend to a registry does not invalidate another backend's checkpoints.
8. A Ridge or constant fixture is a protocol test, never alpha or capital
   evidence.
9. New core dependencies are prohibited. Ridge uses the already-pinned sklearn
   dependency; an external model requiring a new dependency remains Level 1
   until a deliberate dependency-policy change.

## 3. Support Boundaries

### 3.1 Level 1: foreign prediction artifacts

Foreign producers use the existing prediction contract directly:

```text
foreign training process
  -> parquet or Polars frame with era, id, prediction
  -> prediction_set_from_frame(..., fit_role="foreign")
  -> compose_predictions / evaluate_prediction_set(..., allow_research_stage=True)
```

No registry registration, `ModelOrchestrator` change, dependency installation,
or trainer import is required. Research provenance remains explicit. Capital
evaluation still requires a trusted `CapitalContext`; a foreign frame cannot
authorize promotion merely by producing a `MetricScorecard`.

### 3.2 Level 2: explicit first-class backends

A caller creates an isolated registry for one run or experiment:

```python
registry = BackendRegistry.with_builtins()
registry.register(MyCustomBackend())
runner = ExperimentRunner(config, backend_registry=registry)
```

`register()` rejects duplicate names, invalid identities, and incompatible
capabilities. Unknown names fail at runner construction or backend resolution.
The registry is scoped to the caller; there is no mutable process-global
registry. The lifecycle is explicit: `register()` calls happen first,
`snapshot()` creates an independent copy, and runner/orchestrator construction
seals that snapshot before any data load or fit. Registration after sealing
raises `RuntimeError`; mutations to the original registry do not affect the
snapshot. Adapter names are normalized to lowercase and validated against
`^[a-z][a-z0-9_]*$`; duplicate names fail regardless of registration order.
Registry identity is stable across registration order because names are sorted
before canonicalization.

A custom adapter is not automatically capital-eligible. It must still produce
the same authoritative validation evidence as any built-in backend, and the
promotion path must be given the matching explicit registry when custom model
serialization or refitting is required. The concrete API is:

```python
promote_full_version(
    run_id,
    family,
    *,
    backend_registry: BackendRegistry | None = None,
    ...,
)

rehearse_promotion(
    run_id,
    family,
    *,
    backend_registry: BackendRegistry | None = None,
    ...,
)
```

`ExperimentRunner`, `ModelOrchestrator`, both promotion APIs, and HPO accept
the same optional registry concept. A custom backend run refuses promotion,
rehearsal, deployment refitting, and pointer repair unless the caller supplies
an explicit registry whose selected adapter identity and dependency identity
match the persisted run manifest. The registry is rebuilt by operator code;
there is no import-path auto-discovery, plugin scan, or serialized executable
registry bundle. The built-in-only promotion CLIs fail clearly for a custom
backend and direct the operator to an API caller that constructs the registry.

If the adapter does not declare both `supports_full_history=True` and
`supports_deployment=True`, it is Level 2 research-only and promotion refuses
it. Pointer repair does not refit, but still requires the matching registry for
custom backends so the persisted adapter identity is explicitly resolved
before the immutable authorization receipt is accepted.

Level 2 adapters may depend only on packages already governed by the pinned
project dependency policy. Each adapter declares a deterministic
`dependency_identity` mapping of package name to exact version (or an explicit
stdlib marker). A source fingerprint without dependency identity is invalid.
Adding a new dependency or supporting a package version outside the project
policy is a separate dependency-policy change and keeps the model at Level 1
until that change is approved. Deployment additionally requires every declared
dependency to be available in the hosted runtime; otherwise the adapter must
declare `supports_deployment=False`.

### 3.3 Level 3: promotion-eligible backends

The promotion contract remains:

```text
custom or built-in backend
  -> exact purged OOF/validation protocol
  -> exact prediction and auxiliary key universe
  -> CapitalContext
  -> capital scorecard
  -> persisted capital_evidence
  -> promotion authorization
```

A direct `evaluate_model()` result, a `SweepResult`, or a research scorecard
from a custom adapter is never sufficient. Deployment additionally requires
the existing artifact hash, `(0, 1)` output, hosted-runtime, and submission
acceptance checks.

## 4. Backend Protocol

The backend layer is split by responsibility; none of these modules owns fold
construction or checkpoint lifecycle:

| File | Owns |
|---|---|
| `nmr/model_backend_protocol.py` | `BackendCapabilities`, `BackendIdentity`, and `BackendAdapter` protocol |
| `nmr/model_backend_registry.py` | `BackendRegistry`, built-in mapping, registration, sealing, and registry identity |
| `nmr/model_backend_lightgbm.py` | LightGBM adapter and LightGBM-only parameter translation |
| `nmr/model_backend_xgboost.py` | XGBoost adapter and XGBoost-only parameter translation |
| `nmr/model_backend_catboost.py` | CatBoost adapter and CatBoost-only parameter translation |
| `nmr/model_backend_ridge.py` | train-fold scaler, zero-variance handling, Ridge adapter, and serializable scaler/model state |
| `nmr/models.py` | `ModelOrchestrator` and compatibility dispatch into the registry |

The protocol lives in `nmr/model_backend_protocol.py`, separate from the
registry and from training orchestration.

```python
@dataclass(frozen=True)
class BackendCapabilities:
    supports_gpu: bool
    supports_full_history: bool
    supports_deployment: bool
    deployment_device: str  # currently "cpu" for all deployable adapters


@dataclass(frozen=True)
class BackendIdentity:
    schema_version: int
    name: str
    adapter_version: str
    implementation_fingerprint: str  # lowercase SHA-256 over source bytes
    resolved_params: Mapping[str, Any]
    device: str
    capabilities: BackendCapabilities
    dependency_identity: Mapping[str, str]


class BackendAdapter(Protocol):
    name: str
    adapter_version: str
    capabilities: BackendCapabilities

    def resolve_params(
        self, *, preset: str, params: Mapping[str, Any], n_features: int
    ) -> dict[str, Any]: ...

    def build_model(
        self, *, resolved_params: Mapping[str, Any], seed: int, device: str
    ) -> object: ...

    def fit(
        self,
        model: object,
        features: np.ndarray,
        target: np.ndarray,
        *,
        progress: Callable[[str], None] | None = None,
    ) -> object: ...

    def predict(self, model: object, features: np.ndarray) -> np.ndarray: ...

    def identity(
        self,
        *,
        resolved_params: Mapping[str, Any],
        device: str,
    ) -> BackendIdentity: ...
```

The protocol does not receive folds, checkpoint paths, `DataConfig`, RAM
thresholds, or validation frames. Those remain orchestration concerns.

`BackendCapabilities` is descriptive, not an alternate device policy. The
orchestrator validates requested devices against capabilities and retains the
existing auto GPU-first / CPU fallback and forced-GPU behavior.

The identity returned by an adapter is canonical JSON data and must include:

```text
backend name
adapter version
implementation/source fingerprint
resolved parameter payload
resolved device role
capability declaration
```

All mappings are sorted before hashing. Timing, absolute paths, environment
secrets, and progress output are excluded. Built-in adapters derive their
implementation fingerprint from the SHA-256 of their tracked adapter source
bytes. A custom adapter must provide a 64-character lowercase SHA-256 source
fingerprint; object addresses, module import order, and runtime paths are
invalid identity inputs.

`resolved_params`, dependency declarations, and registry metadata must be
recursively JSON-compatible: mappings have string keys and sorted canonical
serialization; sequences contain JSON-compatible values; NumPy scalar values
are normalized to native Python scalars; `allow_nan=False` is mandatory.
`Path`, estimator objects, functions, callbacks, object addresses, and other
unsupported values fail explicitly before fitting. `BackendIdentity.schema_version`
starts at `1` and changes whenever the identity schema or canonicalization
rules change.

## 5. Static Registry

The registry is a small resolver, not a second orchestrator:

```python
class BackendRegistry:
    @classmethod
    def with_builtins(cls) -> "BackendRegistry": ...

    def register(self, adapter: BackendAdapter) -> None: ...
    def resolve(self, name: str) -> BackendAdapter: ...
    def identity(self) -> Mapping[str, Any]: ...
    def snapshot(self) -> "BackendRegistry": ...
```

The built-in mapping is explicit and deterministic:

```python
_BACKENDS = {
    "lightgbm": LightGBMAdapter(),
    "xgboost": XGBoostAdapter(),
    "catboost": CatBoostAdapter(),
    "ridge": RidgeAdapter(),
}
```

Registration order never affects identity: registry identities sort backend
names. Duplicate names fail loudly. Unknown names fail loudly. `snapshot()`
freezes the mapping and identity for a run; a sealed registry rejects further
registration.

The default `ModelOrchestrator(config, seed=...)` path uses
`BackendRegistry.with_builtins()`. An explicit `backend_registry=` is accepted
by `ModelOrchestrator` and `ExperimentRunner` for Level 2 extensions. The
public runner API remains otherwise compatible.

`ModelConfig.backend` is a backend identifier, not the only closed set of
built-in names. Config parsing validates the identifier pattern
`^[a-z][a-z0-9_]*$`; construction against a registry resolves the name and
rejects unknown identifiers. The built-in names remain a published constant
for configuration help and tests, but are not the extension boundary.

Validation stages are explicit:

| Stage | Behavior |
|---|---|
| `load_config()` / direct `ModelConfig` construction | Validate only identifier syntax and ordinary model-field types; an unknown but syntactically valid backend is allowed for later registry resolution. |
| `ModelOrchestrator(config, backend_registry=...)` / `ExperimentRunner(config, backend_registry=...)` construction | Snapshot and seal the supplied registry, resolve `config.model.backend`, validate capabilities, and reject unknown names before data loading or fitting. Omitting the registry uses built-ins. |
| HPO | Built-in backends use the default built-in registry; a custom backend requires an explicit registry passed through the HPO entry point. HPO results remain research-only. |
| Benchmark runners | Built-in benchmark backends only; unknown/custom model identifiers fail before a benchmark cell starts. Benchmark control-plane `model_kind` values are not backend registry names. |
| Promotion/rehearsal/pointer repair | Built-in runs resolve through the built-in registry. Custom runs require the explicit matching registry argument described in §3.2 before any refit, artifact publication, or repair. |

Every path resolves the backend before its first fit. No path silently falls
back to a different backend because a name is unknown or a capability is
unsupported.

### 5.1 Public API ownership

The package-level public API exports exactly these registry/protocol symbols
through `nmr/__init__.py` and `__all__`:

```text
BackendAdapter
BackendCapabilities
BackendIdentity
BackendRegistry
```

Concrete built-in adapters (`LightGBMAdapter`, `XGBoostAdapter`,
`CatBoostAdapter`, `RidgeAdapter`) and implementation helpers remain internal
module symbols. Level 2 custom adapters implement the public protocol but do
not become package exports merely by registration.

## 6. Built-in Adapter Rules

### 6.1 Tree adapters

LightGBM, XGBoost, and CatBoost adapters initially wrap the existing branches
without changing behavior:

- the same preset merge and explicit parameter precedence;
- the same sampling floors and backend-specific parameter translation;
- the same seed and deterministic flags;
- the same device candidate ordering and forced-GPU failure behavior;
- the same progress markers;
- the same fitted model objects and prediction values on fixed CPU fixtures;
- the same model serialization and deployment closure behavior.

The first registry migration extracts only backend-specific branches or wraps
them behind the protocol. It does not move fold loops, full-history logic,
checkpoint writes, RAM guards, or device fallback into adapters.

Parity gates must prove: resolved parameters, fold boundaries, OOF keys, CPU
predictions, learned ensemble weights, scorecard digest, deployment
predictions, and checkpoint manifests are unchanged.

### 6.2 Ridge adapter

Ridge is a real first-class estimator, not an import from benchmark control
plane code. Its adapter owns or calls tested `nmr/` preprocessing:

1. fit a `StandardScaler` on the training fold only;
2. reject non-finite feature values with `ValueError` before fit or predict;
3. map zero-variance scales to a stable finite transform (`scale=1.0`);
4. fit deterministic `sklearn.linear_model.Ridge` with explicit `alpha`;
5. serialize the scaler-plus-model state as one fitted object;
6. predict with the training statistics only.

The default Ridge parameters are explicit and deterministic (`alpha=1.0`,
`fit_intercept=True`, `solver="lsqr"`). The adapter accepts only its declared
parameter keys and requires finite `alpha > 0`. Null/non-finite target rows
are filtered by the orchestrator's existing fit path before the adapter
receives targets. Ridge declares `supports_gpu=False`; GPU requests raise a
clear `ValueError` and never silently fall back or pretend to use CUDA.

Ridge full-history fitting uses the same orchestrator path as other backends,
including CPU deployment and checkpoint/RAM behavior. A Ridge deployment
configuration is allowed only after the serialization fidelity test passes.

### 6.3 Serialization and deployment contract

Deployment composition remains an orchestrator/deployment concern. The adapter
does **not** construct an alternate predictor closure. The existing shared
closure continues to own multiple-target prediction, per-era rank normalization,
weighted blending, neutralization, final `(0, 1)` ranking, and the pandas
submission contract. It consumes the adapter's fitted model state through the
adapter's ordinary `predict()` method.

Every adapter that declares `supports_full_history=True` must return a fitted
state that is cloudpickle-compatible. Every adapter that declares
`supports_deployment=True` must make its adapter instance plus fitted state
usable by the existing shared deployment closure and pass all of these checks:

1. serialize the existing shared deployment closure through `serialize_predict`
    path;
2. reload it in a fresh Python process with `load_predict`;
3. produce deterministic predictions before and after reload on a fixed frame;
4. avoid imports unavailable in the hosted prediction runtime;
5. use only declared, pinned dependencies;
6. pass the existing `(0, 1)` submission-output and hosted-runtime acceptance
    tests.

Built-in adapters use the existing deployment closure and model serialization
path; this registry change must not alter its closure-by-value behavior. Custom
adapters are not automatically embedded or imported by the hosted runtime. A
custom adapter may declare `supports_deployment=True` only when its adapter
implementation and fitted state are self-contained under cloudpickle, or are
available through the hosted runtime's approved imports, and its declared
dependencies are hosted-runtime available. The serialized predictor must not
require the registry object. Otherwise the adapter remains research-only
(`supports_deployment=False`) and cannot be promoted to a deployment artifact.

## 7. Orchestrator Integration

`ModelOrchestrator` remains the protocol owner. Its backend-specific calls are
replaced with registry dispatch at these narrow points:

- resolve canonical backend params;
- construct the backend model;
- invoke fit with the existing progress callback contract;
- predict through the adapter;
- obtain backend identity for checkpoint/run manifests.

The following methods and behaviors remain in `ModelOrchestrator`:

- `PurgedEraSplitter` fold construction and `_assert_fold_is_leakage_safe`;
- OOF fold sequencing and disjoint validation checks;
- full-history and deployment fit roles;
- checkpoint manifests and atomic fold writes;
- RAM thresholds and spawned full-history process;
- device request resolution and fallback ordering;
- era-batched prediction;
- seed propagation and progress logging orchestration.

Deployment composition also remains outside adapters: the existing shared
deployment closure consumes per-target adapter predictions and owns rank
normalization, weighted blending, neutralization, final ranking, and the
pandas submission contract. Adapters provide fitted states and ordinary
`predict()` behavior only; they do not introduce a second deployment-composer
API.

No model-specific CV loop, score implementation, or capital evaluation is
introduced.

## 8. Identity and Checkpoints

There are three deliberately separate identity layers:

1. **Selected backend identity** — backend name, adapter version, selected
    source fingerprint, selected params, selected device, capabilities, and
    dependency identity. This controls compatibility for the selected model.
2. **Shared repository code identity** — the source of the common
    orchestration/protocol/registry path. This controls compatibility for all
    backends when shared behavior changes.
3. **Full registry audit identity** — a sorted digest of all registered adapter
    identities, stored for provenance but not used as the selected-backend
    checkpoint key. Adding Ridge must not invalidate a LightGBM checkpoint.

The selected backend identity enters:

- `ExperimentRunner` run identity;
- OOF checkpoint manifests;
- validation/deploy checkpoint manifests;
- run manifest backend provenance;
- deployment metadata where the fitted adapter is serialized.

The run-id payload contains the shared repository code identity plus the
selected backend identity. It excludes unrelated adapter source files and the
full registry audit digest, so adding Ridge cannot rename an otherwise
unchanged LightGBM run. The run manifest records all three layers: shared code
identity, selected backend identity, and full registry audit identity.

The fit/checkpoint identity includes exactly: registry schema version,
selected backend name, selected adapter version, selected adapter source
fingerprint, selected resolved parameter identity, selected device role,
selected capability declaration, and selected dependency identity. An
unrelated backend added to the registry does **not** invalidate a LightGBM
checkpoint. The full sorted registry mapping digest is stored separately for
run-manifest audit and custom-registry provenance, but is not part of the
selected-backend checkpoint key unless the selected adapter entry changes.

`nmr/_oof.py::fitting_code_sha256()` becomes selected-backend-aware:

```python
fitting_code_sha256(
    *,
    backend_name: str,
    backend_source_files: Sequence[Path],
) -> str
```

It hashes the fixed shared orchestration files plus only the selected adapter
source file in sorted fixed order:

```text
nmr/model_backend_protocol.py
nmr/model_backend_registry.py
nmr/models.py
nmr/splitter.py
nmr/runner.py
nmr/model_backend_<selected_backend>.py
```

The checkpoint manifest stores the selected backend identity separately from
this selected fitting-code digest. A changed selected adapter, protocol,
selected resolver, selected registry entry, selected dependency identity, or
device role refuses checkpoint reuse. An unrelated adapter source file does
not. Existing manifests without the new identity fields are legacy and fail
closed rather than being silently upgraded.

Registry renames are identity changes. Built-in registration order is
irrelevant because identity is sorted by backend name. A custom adapter must
provide a stable source and dependency fingerprint; runtime object addresses
and absolute paths are forbidden.

## 9. Research Provenance

`SweepResult` remains a frozen research proxy with `is_capital=False`. It gains
explicit backend/protocol identity fields:

```text
backend
adapter_version
backend_identity
resolved_params_identity
proxy_metric
proxy_split
training_geometry
selection_bias
is_capital = false
```

`training_geometry` records the actual research geometry (split scheme, fold
count, purge eras, feature set, target, and horizon). HPO geometry may remain
held-out 80/20 and distinct from capital validation, but the distinction is
stored and rendered in result payloads. No `SweepResult` is accepted by
promotion, and no `MetricScorecard` is constructed merely to wrap a proxy.

## 10. Real-Data Foreign Acceptance

The existing foreign prediction acceptance test is extended, not replaced:
when the v5.3 assets and real meta/feature/target frames are available, a
foreign parquet must pass the research contract and the capital contract only
with a context derived from the trusted validation loader. Missing real assets
remain explicit skips in the receipt report; synthetic contract tests remain
mandatory in CI.

## 11. Testing Gates

The implementation is accepted only when these gates pass:

1. Existing LightGBM/XGBoost/CatBoost parameter-resolution parity.
2. Existing CPU OOF behavior parity.
3. Checkpoint refusal after adapter or registry identity changes.
4. Deterministic Ridge OOF across repeated processes.
5. Ridge zero-variance and null-target tests.
6. Ridge CPU-only enforcement.
7. Ridge full-history fit and reload.
8. Ridge deployment serialization if enabled in deployment configs.
9. Unknown backend and duplicate registration failures.
10. Foreign parquet scoring through the existing research/capital contract.
11. Existing capital scorecard digests unchanged.
12. `ruff check .`.
13. Full `pytest -q`.
14. Real-data receipt tests reported separately when skipped.
15. Custom promotion refuses a missing registry and mismatched adapter
    version, dependency identity, selected registry entry, or source
    fingerprint before fit, rehearsal, deployment, or pointer repair.

No Ridge training is used as capital evidence in this implementation cycle.
No Tier-4 gate, promotion, pointer repair, or submission contract changes.

## 12. Same-Change Documentation Contract

Changing backend configuration from a closed tuple to syntax-plus-registry
resolution is an intentional agent-contract change. The implementation change
must update these files in the same change set:

- `AGENTS.md`: closed-set/config-validation rule and model-backend toolkit row;
- `ARCHITECTURE.md`: model backend protocol, registry, identity layers,
    configuration stages, and selected checkpoint hash contract;
- `nmr/config.py`: `ModelConfig` validation/docstring and the built-in backend
    names constant semantics;
- `nmr/__init__.py`: the four public protocol/registry exports;
- configuration, package-API, model, checkpoint, and registry tests;
- `CODEBASE.md` or the relevant public API routing section if the repository
    map gains a new backend entry point.

The implementation must not leave the current `VALID_MODEL_BACKENDS` closed
set documented as the complete accepted backend universe after the registry
is introduced.

## 13. Explicit Non-Goals

This design does not add dynamic plugin discovery, cross-family stacking, new
HPO strategies, gate changes, dashboard work, upload automation, model-specific
CV loops, or model-specific scoring implementations.
