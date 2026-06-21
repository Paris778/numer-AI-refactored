# NumerAPI Reference (Classic-Focused)

This is a practical merged reference for Numerai Classic API usage.

## Scope

Primary target: Classic tournament workflows (tournament id 8).

Also note:

- The CLI can point to Signals (11) and Crypto (12) when needed.

## Purpose

Use numerapi for orchestration tasks:

- list and download datasets
- upload submissions and diagnostics
- query account, models, round state, and leaderboard
- manage staking and wallet operations
- upload compute artifacts for hosted model execution

Use numerai_tools for local scoring math and validation utilities.

## Core Modules

- numerapi.base_api (shared Api class)
- numerapi.numerapi (NumerAPI Classic specialization)
- numerapi.utils (request, parsing, download helpers)
- numerapi.cli (command-line wrapper)

## Authentication

Token format:

Token PUBLIC_ID$SECRET_KEY

Token source precedence:

1. constructor arguments
2. environment variables NUMERAI_PUBLIC_ID and NUMERAI_SECRET_KEY

## Runtime Flow

Typical upload flow:

1. GraphQL call requests presigned upload URL
2. bytes are uploaded with HTTP PUT
3. mutation finalizes upload and returns submission identifier

## Core Classic API Surface

### Data and datasets

- list_datasets(round_num=None)
- download_dataset(filename, dest_path=None, round_num=None)
- set_global_data_dir(directory)

### Rounds and status

- get_current_round(tournament=None)
- check_round_open()
- check_new_round(hours=12)
- pipeline_status(date=None)

### Account and model metadata

- get_account()
- get_models(tournament=None)
- models_of_account(account)
- modelid_to_modelname(model_id)

### Submissions and diagnostics

- upload_predictions(file_path="predictions.csv", model_id=None, df=None, data_datestamp=None)
- upload_diagnostics(file_path="predictions.csv", tournament=None, model_id="", df=None)
- diagnostics(model_id, diagnostics_id=None)
- submission_ids(model_id)
- download_submission(submission_id=None, model_id="", dest_path="")

### Leaderboard and performance

- get_account_leaderboard(limit=50, offset=0)
- round_model_performances_v2(model_id)
- intra_round_scores(model_id)

### Staking and wallet

- wallet_transactions()
- stake_change(nmr, action="decrease", model_id="")
- stake_increase(nmr, model_id)
- stake_decrease(nmr, model_id)
- stake_drain(model_id=None)

### Profile utilities

- set_bio(model_id, bio)
- set_link(model_id, link_text, link)
- set_submission_webhook(model_id, webhook=None)

### Numerai Compute model upload

- model_upload(file_path, tournament=None, model_id=None, data_version=None, docker_image=None)
- model_upload_data_versions()
- model_upload_docker_images()

## NumerAPI Specialization Methods

Common additional convenience methods include:

- get_competitions(tournament=8)
- get_submission_filenames(tournament=None, round_num=None, model_id=None)
- get_leaderboard(limit=50, offset=0)
- public_user_profile(username)
- daily_model_performances(username)
- stake_get(modelname)
- stake_set(nmr, model_id)

## CLI Commands (Common)

- list-datasets
- download-dataset
- current-round
- check-new-round
- leaderboard
- account
- models
- submit
- submission-filenames
- stake-get
- stake-increase
- stake-decrease
- stake-drain
- transactions

## Utility Helpers

From numerapi.utils, commonly useful helpers include:

- download_file
- post_with_err_handling
- parse_datetime_string
- parse_float_string
- is_valid_uuid

## Submodule Inventory (Quick Index)

- numerapi.base_api.Api
- numerapi.cli
- numerapi.numerapi.NumerAPI
- numerapi.signalsapi.SignalsAPI
- numerapi.utils

## Practical Caveats

- GraphQL errors can surface as ValueError during raw query operations.
- Uploads from DataFrame generally serialize via CSV conversion.
- Some edge behavior and historical methods can differ by numerapi version.

Always verify behavior against installed package version when building production automation.
