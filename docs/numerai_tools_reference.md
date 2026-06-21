# numerai_tools Reference (Classic Tournament Focus)

## Scope
This reference keeps only Classic-relevant utilities.

## Purpose
`numerai_tools` is the local math and validation layer for Classic model development:
- ranking and normalization transforms
- canonical CORR/FNC/MMC-adjacent evaluation utilities
- neutralization and contribution calculations
- Numerai-style submission validation and cleaning for Classic IDs

## Classic-Relevant Modules
- `numerai_tools/scoring.py`: ranking, normalization, correlation, neutralization, contribution math
- `numerai_tools/submissions.py`: schema/value/id checks and cleaning used for Classic-style submissions

## Core Characteristics
- function-oriented (no classes)
- heavy use of pandas index alignment checks
- assertions enforce overlap, sorted indices, and non-NaN assumptions

## `scoring.py` (Classic-relevant surface)

### Constants and types
- `DEFAULT_MAX_FILTERED_INDEX_RATIO = 0.2`
- `RANK_METHOD_TYPE = Literal["average", "min", "max", "first", "dense"]`

### Index alignment and filtering
- `filter_sort_index(s1, s2, max_filtered_ratio=0.2)`
- `filter_sort_index_many(inputs, max_filtered_ratio=0.2)`
- `filter_sort_top_bottom(s, top_bottom)`
- `filter_sort_top_bottom_concat(s, top_bottom)`

### Ranking and normalization
- `rank_series(s, method="average")`
- `rank(s, method="average")`
- `tie_broken_rank(df)`
- `tie_kept_rank(s)`
- `min_max_normalize(s)`
- `variance_normalize(df)`
- `weight_normalize(s)`
- `center(s)`
- `standardize(df)`

### Correlation and validation
- `validate_indices(live_targets, predictions)`
- `correlation(live_targets, predictions)`
- `tie_broken_rank_correlation(target, predictions)`
- `spearman_correlation(target, predictions)`
- `pearson_correlation(target, predictions, top_bottom=None)`
- `sharpe_ratio(s)`

### Transformation and geometry
- `power(df, p)`
- `gaussian(df)`
- `orthogonalize(v, u)`
- `neutralize(df, neutralizers, proportion=1.0)`
- `one_hot_encode(df, columns, dtype=np.float64)`

### Canonical scoring chains
- `tie_kept_rank__gaussianize__pow_1_5(df)`
- `tie_kept_rank__gaussianize__neutralize__variance_normalize(df, neutralizers)`

### Classic model metrics
- `numerai_corr(predictions, targets, max_filtered_index_ratio=0.2, top_bottom=None, target_pow15=True)`
- `feature_neutral_corr(predictions, features, targets, top_bottom=None)`
- `correlation_contribution(predictions, meta_model, live_targets, top_bottom=None)`
- `max_feature_correlation(s, features, top_bottom=None)`

### Portfolio/weight utilities used in Classic research
- `stake_weight(predictions, stakes)`
- `generate_neutralized_weights(predictions, neutralizers, sample_weights, center_and_normalize=False)`
- `alpha(predictions, neutralizers, sample_weights, targets)`
- `meta_portfolio_contribution(predictions, stakes, neutralizers, sample_weights, targets)`

## `submissions.py` (Classic-only surface)

### Constants
- `NUMERAI_ALLOWED_ID_COLS = ["id"]`
- `NUMERAI_ALLOWED_PRED_COLS = ["prediction", "probability"]`

### Classic validators
- `_validate_headers(submission, expected_id_cols, expected_pred_cols)`
- `validate_headers_numerai(submission)`
- `validate_values(submission, prediction_col)`
- `_validate_ids(live_ids, submission, id_col, min_tickers)`
- `validate_ids_numerai(live_ids, submission, id_col)`
- `validate_submission_numerai(universe, submission)`

### Cleaning and remapping
- `remap_ids(data, ticker_map, src_id_col, dst_id_col)`
- `clean_submission(universe, submission, src_id_col, src_signal_col, dst_id_col=None, dst_signal_col=None, rank_and_fill=False)`

`clean_submission(..., rank_and_fill=True)` applies tie-kept ranking then fills missing values with `0.5`, matching common Numerai scoring prep.

## Practical Constraints
- overlap checks can assert if universes diverge too much
- most scoring methods require sorted indices and no NaNs
- `validate_values` enforces near-exclusive `(0, 1)` style predictions with non-zero variance

## Usage Boundary
- use `numerai_tools` for offline Classic scoring, neutralization, and pre-upload validation
- use `numerapi` for live API operations (download/upload/account/staking)
