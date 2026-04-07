# NRFI Picks

ML pipeline to predict whether a run scores in the first inning of MLB games (YRFI / NRFI).

## How it works

1. **LR** — logistic regression retrained daily on all historical data
2. **NN** — neural network incrementally trained on yesterday's results, persisted to S3
3. Both models score today's games; consensus picks (both agree) are highest conviction
4. Picks are emailed via AWS SES and optionally written to S3 JSON

Threshold `<0.455 / >0.545` is selected daily via 5-fold CV, targeting ~20% coverage at ~55% accuracy.

## Running locally

```bash
python daily_picks.py
```

Requires AWS credentials with access to `s3://nrfi-store` and a configured SES sender.

## Key files

| File | Purpose |
|------|---------|
| `daily_picks.py` | Daily inference — retrain LR, update NN, fetch today's games, send email |
| `lambda_function.py` | AWS Lambda — post-game data collector, saves to `s3://nrfi-store/YYYY/MM/DD.txt` |
| `data/NRFI Dataset Cleaned v2.csv` | Primary dataset, 6,232 games May 2021–Sept 2023 |
| `scripts/run_models.py` | Model training, CV evaluation, threshold analysis |

## Environment variables

| Variable | Description |
|----------|-------------|
| `NRFI_DATA_PATH` | S3 URI or local path to training CSV (default: local `data/` path) |
| `NRFI_NN_MODEL_PATH` | S3 URI for persisted Keras model |
| `NRFI_SNS_TOPIC_ARN` | SNS topic for plain-text pick delivery |
| `NRFI_SES_FROM` | Verified SES sender address |
| `NRFI_SES_TO` | Comma-separated recipient addresses |
| `NRFI_OUTPUT_BUCKET` | S3 bucket for picks JSON output |
| `ODDS_API_KEY` | The Odds API key for live sportsbook odds |

