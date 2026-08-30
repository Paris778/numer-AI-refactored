# API, GraphQL, and Numerai MCP

> **Canonical scope:** API keys, scopes, credential handling, NumerAPI, GraphQL, Numerai MCP, and round score configuration.

Never commit API credentials. Use the least-privileged key that supports the integration.

## API keys

1. Sign in and open **Account Settings**.
2. Open **Automation** and select **Create API Key**.
3. Give the key a name identifying the integration.
4. Select only the required scopes.
5. Confirm with the account password and MFA code if enabled.
6. Copy the Public ID and Secret Key. The Secret Key is shown only once.

Use a separate key for each integration. If a key is lost or exposed, revoke it from **View API Keys** and create a replacement.

### Scopes

| Scope | Allows |
| --- | --- |
| Update account and model profile information | Editing account and model profiles. |
| Upload submissions and pickled models | Uploading predictions and models and configuring submission webhooks. |
| Download previous submissions and pickled models | Downloading prior submissions and uploaded models. |
| Make stakes | Managing legacy tournament stakes. |
| Manage web3 staking | Managing Atomic Blockchain Staking. |
| View historical submission info | Reading private submission history. |
| View user info | Reading private account details, balances, withdrawals, and model IDs. |
| Ability to delete your models and your account | Deleting models or the account. |

> **Security rule:** Do not grant staking or deletion scopes to submission automation unless they are required by that integration.

## Credential storage

Do not place credentials in source code or commit them to a repository. For deployed workloads, use the host or CI secret manager. If using a local `.env` file, add it to `.gitignore` and restrict its permissions.

NumerAPI reads `NUMERAI_PUBLIC_ID` and `NUMERAI_SECRET_KEY` automatically.

### Environment variables

macOS and Linux:

```bash
export NUMERAI_PUBLIC_ID="YOUR_PUBLIC_ID"
export NUMERAI_SECRET_KEY="YOUR_SECRET_KEY"
```

Windows PowerShell:

```powershell
$env:NUMERAI_PUBLIC_ID = "YOUR_PUBLIC_ID"
$env:NUMERAI_SECRET_KEY = "YOUR_SECRET_KEY"
```

## Numerai MCP

Numerai MCP connects supported AI agents to Numerai tools for tournament information and research assistance. It uses the stateless 2026-07-28 MCP protocol over Streamable HTTP.

Supported clients include:

- Codex CLI, recommended;
- Cursor; and
- Claude Code.

### Authenticate MCP

Create an MCP key in the **Automation** section of Account Settings with the scopes required for the intended operations:

- Upload submissions and pickled models;
- Download previous submissions and pickled models;
- View historical submission info; and
- View user information, including balances and withdrawal history.

Set `NUMERAI_MCP_AUTH` to the Public ID and Secret Key in this format:

```bash
export NUMERAI_MCP_AUTH="Token PUBLIC_ID\$SECRET_KEY"
```

The `$` between the Public ID and Secret Key must be escaped in shells that expand variables. Export the variable in the environment that starts the agent.

### Install MCP

#### Codex CLI

The one-line installer guides through creating an MCP key and configuring the environment. Follow its authorization prompt:

```bash
curl -sL https://numer.ai/install-mcp.sh | bash
```

To configure manually, add this to `~/.codex/config.toml`:

```toml
[mcp_servers.numerai]
url = "https://api-tournament.numer.ai/mcp"

[mcp_servers.numerai.env_http_headers]
Authorization = "NUMERAI_MCP_AUTH"
```

#### Cursor

Add this entry to the `mcpServers` object in `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "numerai": {
      "url": "https://api-tournament.numer.ai/mcp",
      "headers": {
        "Authorization": "${env:NUMERAI_MCP_AUTH}"
      }
    }
  }
}
```

#### Claude Code

Export `NUMERAI_MCP_AUTH` before running:

```bash
claude mcp add --transport http numerai \
  https://api-tournament.numer.ai/mcp \
  --header "Authorization: ${NUMERAI_MCP_AUTH}"
```

### Use MCP

After installation, an agent can request:

- tournament information, such as leaderboards, model performance, and the current round;
- research assistance, such as creating and uploading models or checking submissions; and
- GraphQL schema inspection, request construction, and execution through the generic GraphQL tool.

## NumerAPI

NumerAPI is the official Python client and lightweight command-line interface for the Numerai API:

```bash
python -m pip install --upgrade numerapi
```

Once the environment variables are set:

```python
from numerapi import NumerAPI

napi = NumerAPI()
models = napi.get_models()
print(models)
```

Use the tournament-specific clients for Signals and Crypto:

```python
from numerapi import CryptoAPI, SignalsAPI

signals_api = SignalsAPI()
crypto_api = CryptoAPI()
```

The package also installs the `numerapi` command:

```bash
numerapi models
numerapi current-round
numerapi --help
```

See the NumerAPI reference for all methods and commands.

## GraphQL API

Applications in other languages can call the GraphQL API directly. The Authorization header contains the Public ID and Secret Key separated by a literal `$`:

```bash
NUMERAI_API_TOKEN="${NUMERAI_PUBLIC_ID}\$${NUMERAI_SECRET_KEY}"
curl https://api-tournament.numer.ai/ \
  --header "Authorization: Token ${NUMERAI_API_TOKEN}" \
  --header "Content-Type: application/json" \
  --data '{"query":"query { account { username } }"}'
```

The selected scopes must authorize every protected field or mutation in the request.

## Round score configuration

The public `Round.roundScoreConfigs` field returns score configurations attached to a round. It includes selected payout snapshots and non-payout snapshots.

`isPayout` is derived from the immutable per-round multiplier snapshot. It is true when at least one of `minMultiplier`, `maxMultiplier`, or `defaultMultiplier` is nonzero. Clients can use `isPayout` when displaying payout settings, including historical rounds whose reusable score definition later changed. No API key is required for this field.

Example query:

```graphql
query CurrentRoundPayoutConfig {
  rounds(tournament: 8, status: OPEN, limit: 1) {
    number
    roundScoreConfigs {
      name
      version
      displayName
      totalScoreDays
      returnsLagDays
      dataDelayDays
      scoringStart
      scoringEnd
      isPayout
      minMultiplier
      maxMultiplier
      defaultMultiplier
    }
  }
}
```

### Round score fields

| Field | Meaning |
| --- | --- |
| `id`, `scoreConfigId` | IDs of the per-round record and reusable score definition. |
| `name`, `version` | Stable score identity and definition version. |
| `displayName` | Score label used in API and UI output. |
| `roundNumberStart`, `roundNumberEnd` | Inclusive round range for the score definition; the end can be null. |
| `totalScoreDays` | Number of scoring days used for the final score. |
| `returnsLagDays`, `dataDelayDays` | Return lag and source-data delay. |
| `universe` | Data universe when the score is universe-specific. |
| `isCanonScore` | Whether the score is a canonical comparison score. |
| `isPayout` | Whether this per-round snapshot is selected for payout. |
| `scoringStart`, `scoringEnd` | Scoring window for this round. |
| `minMultiplier`, `maxMultiplier`, `defaultMultiplier` | Per-round multiplier range and default. |
| `clipThreshold`, `stakeThreshold` | Per-round payout clipping and stake thresholds. |
| `payoutFactor` | Optional per-score payout factor. |

Internal target specifications are intentionally not exposed.

Every payout configuration has its own IDs, name, version, and display name. Preserve that identity: Alpha is not CORR, MPC is not MMC, and an unfamiliar payout score must not be substituted for a familiar score because multipliers look similar. Aggregates iterate over selected configurations and apply each configuration's multiplier.

`RoundDetails.payoutMultipliers` similarly returns every selected display name and default multiplier without grouping by CORR or MMC family.

Profile and submission-history rows expose `payoutMultipliers`, an identity-preserving list containing the round-score-config ID, score-config ID, name, version, display name, and weekly multiplier. New clients should use this exact list.

Legacy corr/MMC/TC scalar multiplier fields remain deprecated until T-469. They project canonical weekly values but must not be used to identify which score a multiplier belongs to.

The current V2 payout mode is `Model.v2Stake.takeProfit`. The legacy `Model.currentPayoutSelection`, `V3UserProfile.stakeInfo`, and `SignalsLeaderboard.payoutSelection` fields remain deprecated until T-469. New clients must not adopt them.

## Troubleshooting

- Secret Key missing: it cannot be displayed again. Revoke the key and create a replacement.
- Unauthorized operation: confirm both key components are present and the key has the required scope. Scopes cannot be added to an existing key; create a replacement.
- MCP cannot authenticate: confirm `NUMERAI_MCP_AUTH` is exported in the agent's startup environment and contains the literal `$` separator.
- Key exposure: revoke it immediately, rotate credentials wherever they are used, and review integration logs.
