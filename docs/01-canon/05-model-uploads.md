# Model Uploads and Numerai CLI

> **Canonical scope:** Hosted model execution, the `predict` contract, cloudpickle artifacts, execution states, runtime limits, and Numerai CLI.

Model Uploads are an optional automation method. Use [Submissions](04-submissions.md) for round mechanics and manual submission behavior.

## Model Uploads

Model Uploads are a free hosted way to automate daily submissions. Upload a `.pkl` file containing a trained model and a prediction function. Numerai runs the function and submits its predictions. This removes the participant's infrastructure and scheduling burden, but Numerai can access and unpickle the uploaded model.

### Requirements

Install the package versions supported by the Numerai Predict execution environment. Packages not listed by that environment are unavailable. Python 3.10 through 3.13 can be selected when uploading a model; the default is Python 3.13. See the [Numerai Predict requirements](https://github.com/numerai/numerai-predict/blob/master/py3.13/requirements.txt) for the current package list.

The hosted model cannot access the internet. The default execution machine has 1 CPU, 4 GB of RAM, and a runtime limit of up to 10 minutes, excluding queue time.

### Prediction function

Wrap the trained model in a function that accepts live features and returns a DataFrame with a `prediction` column:

```python
import pandas as pd


def predict(
    live_features: pd.DataFrame,
    live_benchmark_models: pd.DataFrame,
) -> pd.DataFrame:
    live_predictions = model.predict(live_features[feature_cols])
    submission = pd.Series(
        live_predictions,
        index=live_features.index,
    )
    return submission.to_frame("prediction")
```

The function signature must be compatible with the Model Uploads runtime. The benchmark-model argument is available to the function even when the model does not use it.

Cloudpickle serializes the function's local context. In the example, `model` and `feature_cols` are global objects referenced by `predict`; cloudpickle includes them in the artifact.

### Serialize and upload

Use `cloudpickle`, not the standard `pickle` module:

```python
import cloudpickle

payload = cloudpickle.dumps(predict)
with open("predict.pkl", "wb") as handle:
    handle.write(payload)
```

Upload `predict.pkl` from the Submissions page with the Python version used to create it. Numerai executes the model for the current round, validates the generated submission, and generates validation diagnostics.

### Daily execution states

A successful model cycles through:

1. **Pending:** Cloud resources are being provisioned.
2. **Running:** Numerai is executing the model.
3. **Validating:** The prediction has been generated and is being checked.
4. **Success:** The submission has been accepted.

Possible failure states are:

- **Error:** Numerai encountered an unexpected execution problem.
- **Failed:** The model failed to run. Check logs and upload a working model.

Common causes include a Python or dependency mismatch, an invalid submission, insufficient memory, or a timeout.

### Disable a Model Upload

Open the Model Upload control on the Submissions page, open the **Settings** tab, and select **Disable**.

### Model Upload FAQ

**Can I upload manual predictions after uploading a model?** No. Disable the Model Upload before using API submission again.

**Can I configure a webhook on a Model Upload?** No. Disable an existing compute configuration before uploading a model to avoid race conditions.

**Does Numerai have access to my trained model?** Yes. Numerai can access and unpickle the uploaded model. Use Numerai CLI or your own infrastructure if this is not acceptable.

**Does re-uploading a model change existing history?** No. A new `.pkl` changes predictions from that point forward. It does not reset scores, payouts, reputation, or round history. Create a new model to start from zero.

**Can I download my uploaded model?** Yes. On the Submissions page, use the cloud icon, open **Settings**, and select **Download**.

### Terms and responsibility

Numerai may disable a Model Upload for reasons including security, abuse, account inactivity, or poor performance. Participants are responsible for their pipeline and its performance. Numerai does not guarantee that an uploaded model will submit successfully every day and is not responsible for gains or losses caused by model performance.

## Numerai CLI

Numerai CLI, formerly called Compute Heavy, deploys automated prediction nodes to AWS, Azure, or Google Cloud. Use it when Numerai should trigger a model on infrastructure controlled by the participant.

### Requirements

- A Numerai account with at least one model.
- A paid AWS, Azure, or Google Cloud account.
- Python and Docker installed locally.
- A Numerai API key with the required scopes.

The prediction node runs in the participant's cloud account. The participant is responsible for infrastructure cost and maintenance.

### API key scopes

Create a separate key for the prediction node with:

- **View user info**, so the CLI can find models and verify the key.
- **Upload submissions and pickled models**, so it can register a webhook and upload predictions.
- **View historical submission info**, so `numerai node test` can verify the uploaded prediction.

Do not grant staking or deletion scopes to a prediction node. See [API and MCP](11-api-and-mcp.md) for credential handling.

### Install and configure

Follow the provider guide in the Numerai CLI repository. After installing the required Python and Docker versions:

```bash
python -m pip install --upgrade numerai-cli
numerai setup --provider aws
```

Replace `aws` with `azure` or `gcp` as appropriate. The setup command asks for the API key's Public ID and Secret Key and the selected cloud provider's credentials.

Numerai CLI stores credentials and infrastructure state under `~/.numerai/`. Protect and back up that directory. Never add it to a repository.

### Deploy and test

From the model project:

```bash
numerai node config --example tournament-python3
numerai node deploy
numerai node test
```

Run `numerai --help` or `numerai node --help` for options such as model name, tournament ID, instance size, and provider. Re-run `numerai node deploy` and `numerai node test` after changing prediction code or retraining the model.

### CLI troubleshooting

- Invalid keys: the key must include **View user info**. Run `numerai setup` again to replace saved credentials.
- Model not found: confirm that the model exists in the selected tournament and select the correct model name and tournament ID.
- Failed node test: inspect webhook and compute logs in the cloud provider, then rerun `numerai node test`.
- Lost local configuration: identify resources in the cloud-provider console before recreating or removing infrastructure.
