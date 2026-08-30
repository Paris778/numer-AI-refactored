# Submissions and Rounds

> **Canonical scope:** Submission values, round lifecycle, schedules, late and queued submissions, and automation choices.

For hosted automation, see [Model Uploads](05-model-uploads.md). For credentials, see [API and MCP](11-api-and-mcp.md).

## Submission contract

A submission is a vector of floating-point predictions with one value for each live `id`.

- Values must be between `0` and `1`.
- `0` represents the lowest predicted return.
- `0.5` represents the average predicted return.
- `1` represents the highest predicted return.

Submissions are combined into the Stake-Weighted Meta Model (SWMM), which Numerai uses for trading. Numerai receives prediction values, not the code, data, features, or trained model that produced them. Model Uploads are the opt-in exception because Numerai must access and unpickle the uploaded model.

## Round lifecycle

Each submission is associated with a tournament **round**. A round has four stages:

1. **Open:** A new round starts and new live features are released.
2. **Close:** The submission window ends.
3. **Score:** The submission receives daily scores during the configured scoring period.
4. **Resolve:** Final scores and payouts are resolved.

A new round starts each day from Tuesday through Saturday. A round spans 64 business days, or about three months, so approximately 64 rounds overlap at a time.

Tuesday through Friday rounds are normally open for one hour. Saturday rounds remain open through Sunday. No scores are normally released on Sunday or Monday.

| Round start | Normal open time | Normal close time | Scores start | Approximate resolve |
| --- | --- | --- | --- | --- |
| Tuesday | Tue 12:00 UTC | Tue 13:00 UTC | Following Saturday | About 87 days later, Friday |
| Wednesday | Wed 12:00 UTC | Wed 13:00 UTC | Following Tuesday | About 89 days later, Monday |
| Thursday | Thu 12:00 UTC | Thu 13:00 UTC | Following Wednesday | About 89 days later, Tuesday |
| Friday | Fri 12:00 UTC | Fri 13:00 UTC | Following Thursday | About 89 days later, Wednesday |
| Saturday | Sat 12:00 UTC | Sun 14:00 UTC | Following Friday | About 89 days later, Thursday |

Actual times can vary by round. Numerai maintains a minimum one-hour submission window, and rounds do not open earlier than 12:00 UTC or close earlier than 13:00 UTC.

## Making a submission

Submit live predictions in every round in which you want a live track record. A minimal NumerAPI flow is:

```python
from numerapi import NumerAPI
import pandas as pd

napi = NumerAPI()
current_round = napi.get_current_round()

VERSION = "v5.3"
napi.download_dataset(f"{VERSION}/live_{current_round}.parquet")
live_data = pd.read_parquet(f"{VERSION}/live_{current_round}.parquet")
feature_cols = [column for column in live_data.columns if "feature" in column]
live_predictions = model.predict(live_data[feature_cols])

submission = pd.Series(
    live_predictions,
    index=live_data.index,
).to_frame("prediction")
submission.to_csv(f"prediction_{current_round}.csv")
napi.upload_predictions(
    f"prediction_{current_round}.csv",
    model_id="your-model-id",
)
```

Use the current dataset files and current API response when operating a live pipeline. The current round determines which submission is selected and which score configuration applies.

## Multiple submissions

You can upload multiple submissions during a submission window. Numerai selects only the latest valid submission for scoring and payouts.

## Late submissions

You can upload after a round's submission window closes, but the upload is considered late:

- It is still scored.
- It cannot be staked for that round.
- Its at-risk NMR is `0`.
- It does not affect the SWMM.
- It does not affect the payout factor for other users.

## Queued and delayed submissions

If a submission misses the current round's window, Numerai automatically queues it for the upcoming round. It becomes an on-time submission when the upcoming round opens.

Live IDs change between rounds. Numerai maps prediction IDs from the previous round to the latest round's IDs when processing queued or delayed predictions.

If a pipeline takes more than 24 hours, predictions generated for the previous round can be used for the current round through the delayed-submission flow.

## Automation choices

### Model Uploads

Model Uploads are the simplest hosted option. Upload a pickled prediction function and Numerai runs it daily. Numerai can access and unpickle the uploaded model. See [Model Uploads](05-model-uploads.md).

### Numerai CLI

Numerai CLI, formerly called Compute Heavy, is a self-hosted cloud solution for AWS, Azure, or Google Cloud. It is appropriate for a custom pipeline or a participant who wants to control the prediction infrastructure. See [Model Uploads](05-model-uploads.md#numerai-cli) for setup and deployment.

### Local server

A participant can run a local prediction service. Common patterns include a scheduled script such as cron or a webhook receiver such as ngrok. The participant is responsible for availability, monitoring, and support.
