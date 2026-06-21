# numerapi Reference (Classic Tournament Only)

## Scope
This reference is intentionally scoped to Numerai Classic (tournament_id = 8).

## Purpose
`numerapi` provides API transport and orchestration for Classic workflows:
- dataset listing and download
- prediction and diagnostics upload
- account/model queries
- leaderboard and performance queries
- staking and wallet operations
- Numerai Compute model upload operations

## Classic-Relevant Modules
- `numerapi/base_api.py`: shared `Api` class used by `NumerAPI`
- `numerapi/numerapi.py`: `NumerAPI` specialization for Classic
- `numerapi/utils.py`: auth loading, parsing, resilient request/download helpers
- `numerapi/cli.py`: CLI commands (usable in Classic mode)

## Core Runtime Model
1. `Api.__init__` configures logging and token state.
2. `raw_query` sends GraphQL requests to `https://api-tournament.numer.ai`.
3. Upload flows use:
   - GraphQL auth query for a presigned URL
   - `requests.put` to upload bytes
   - GraphQL mutation to finalize the action and return an ID

## Authentication
- Header format: `Token <public_id>$<secret_key>`
- Token source priority:
  - constructor args
  - `NUMERAI_PUBLIC_ID` and `NUMERAI_SECRET_KEY`

## Classic API Surface

### Base `Api` methods used in Classic

#### Data
- `list_datasets(round_num: int | None = None) -> list[str]`
- `download_dataset(filename: str, dest_path: str | None = None, round_num: int | None = None) -> str`
- `set_global_data_dir(directory: str)`

#### Account and models
- `get_account() -> dict`
- `models_of_account(account) -> dict[str, str]`
- `get_models(tournament: int | None = None) -> dict`
- `modelid_to_modelname(model_id: str) -> str`

#### Rounds and pipeline timing
- `get_current_round(tournament: int | None = None) -> int | None`
- `check_round_open() -> bool`
- `check_new_round(hours: int = 12) -> bool`
- `pipeline_status(date: str | None = None) -> dict`

#### Profile/webhook settings
- `set_bio(model_id: str, bio: str) -> bool`
- `set_link(model_id: str, link_text: str, link: str) -> bool`
- `set_submission_webhook(model_id: str, webhook: str | None = None) -> bool`

#### Wallet and staking
- `wallet_transactions() -> list`
- `stake_change(nmr: float | str, action: str = "decrease", model_id: str = "") -> dict`
- `stake_drain(model_id: str | None = None) -> dict`
- `stake_decrease(nmr: float | str, model_id: str) -> dict`
- `stake_increase(nmr: float | str, model_id: str) -> dict`

#### Diagnostics
- `upload_diagnostics(file_path: str = "predictions.csv", tournament: int | None = None, model_id: str = "", df: pandas.DataFrame | None = None) -> str`
- `diagnostics(model_id: str, diagnostics_id: str | None = None) -> dict`

#### Submissions and retrieval
- `upload_predictions(file_path: str = "predictions.csv", model_id: str | None = None, df: pandas.DataFrame | None = None, data_datestamp: int | None = None, timeout=(10, 600)) -> str`
- `submission_ids(model_id: str)`
- `download_submission(submission_id: str | None = None, model_id: str = "", dest_path: str = "") -> str`

#### Historical metrics and leaderboards
- `round_model_performances_v2(model_id: str)`
- `intra_round_scores(model_id: str)`
- `round_model_performances(username: str) -> list[dict]` (deprecated)
- `get_account_leaderboard(limit: int = 50, offset: int = 0) -> list[dict]`

#### Numerai Compute model upload
- `model_upload(file_path: str, tournament: int | None = None, model_id: str | None = None, data_version: str | None = None, docker_image: str | None = None) -> str`
- `model_upload_data_versions() -> dict`
- `model_upload_docker_images() -> dict`

### `NumerAPI` methods (Classic specialization)
- `get_competitions(tournament=8)`
- `get_submission_filenames(tournament=None, round_num=None, model_id=None)`
- `get_leaderboard(limit=50, offset=0)`
- `stake_set(nmr, model_id: str)`
- `stake_get(modelname: str)`
- `public_user_profile(username: str)`
- `daily_model_performances(username: str)`

## Utility Functions (Classic workflows)
- `load_secrets()`
- `parse_datetime_string(string)`
- `parse_float_string(string)`
- `replace(dictionary, key, function)`
- `download_file(url, dest_path, show_progress_bars=True)`
- `post_with_err_handling(url, body, headers, timeout=None, retries=3, delay=1, backoff=2)`
- `is_valid_uuid(val)`

## CLI (Classic Mode)
Common Classic commands:
- `list-datasets`
- `download-dataset`
- `competitions`
- `current-round`
- `leaderboard`
- `submission-filenames`
- `check-new-round`
- `account`
- `models`
- `profile`
- `daily-model-performances`
- `transactions`
- `submit`
- `stake-get`
- `stake-drain`
- `stake-decrease`
- `stake-increase`

## Operational Notes
- `raw_query` raises `ValueError` when GraphQL returns errors.
- Upload path is always: auth query -> file PUT -> create mutation.
- DataFrame uploads serialize with `to_csv(index=False)`.
- `check_round_open` and `check_new_round` return `False` during inactive between-round windows.

## Known Classic Caveats
- `numerapi/__init__.py` uses `version("package-name")`, so `__version__` may resolve to `unknown`.
- `Api.models_of_account` uses GraphQL type text `Str!` in the query.
- `submission_ids` calls `utils.replace` on a list object, so `insertedAt` parsing may not be applied.
