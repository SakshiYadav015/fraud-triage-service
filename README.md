# Razorpay Merchant Risk Desk — Fraud Triage Service

## Overview

A Python-based fraud triage service that prepares transaction data, trains a fraud classification model, retrieves relevant fraud policies, and generates structured case notes for risk review.

## Key Features

- Transaction data cleaning and duplicate removal
- Feature engineering for transaction risk signals
- Fraud classification using Logistic Regression and Random Forest
- Class imbalance evaluation and decision-threshold tuning
- TF-IDF and cosine-similarity policy retrieval
- Structured case-note generation with validation, retries, and safe defaults
- FastAPI endpoints for transaction scoring, case review, health checks, and metrics
- Request logging to `service.log`

## Project Structure

- `generate_data.py` — transaction data generation
- `transactions_raw.csv` — raw transaction dataset
- `prepare.py` — data cleaning and feature engineering
- `train.py` — model training, evaluation, and threshold selection
- `fraud_policies.json` — risk policy definitions
- `retrieval_queries.json` — policy retrieval evaluation queries
- `retrieval.py` — policy retrieval
- `mock_llm.py` — deterministic mock language-model responses
- `case_scenarios.json` — case-note test scenarios
- `casenote.py` — case-note validation and guardrails
- `app.py` — FastAPI application
- `smoke_test.py` — basic project smoke tests
- `service.log` — application request logs

## Requirements

- Python
- pandas
- scikit-learn
- FastAPI
- Uvicorn
- Pydantic

Install dependencies from the project directory:

```bash
python -m pip install -r requirements.txt
```

## Run the Service

Start the API from the project directory:

```bash
python -m uvicorn app:app --reload
```

Open the interactive API documentation at:

`http://127.0.0.1:8000/docs`

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check service health |
| POST | `/score` | Score a transaction and return a risk decision |
| POST | `/case` | Retrieve a policy and generate a structured case note |
| GET | `/metrics` | View in-memory request and decision counters |

## Model Evaluation

The project evaluates Logistic Regression and Random Forest models using precision, recall, F1-score, confusion matrices, and cross-validation.

The selected Random Forest configuration from grid search was:

- `max_depth`: `None`
- `min_samples_leaf`: `5`

The validation threshold-selection process selected `0.4` to satisfy the chosen recall constraint. At this threshold, the recorded validation results were:

- Precision: approximately `0.2819`
- Recall: approximately `0.7260`
- F1-score: approximately `0.4061`

These are validation results, not a guarantee of production performance. Fraud decisions should be reviewed in the context of false positives, missed fraud, and operational costs.

## Policy Retrieval Evaluation

The recorded retrieval test results were:

- In-scope top-1 hits: `8/8`
- Out-of-scope queries rejected: `2/2`
- Minimum similarity threshold: `0.45`

## Testing

Run the smoke tests:

```bash
python smoke_test.py
```

Check Python syntax:

```bash
python -m py_compile app.py smoke_test.py
```

## Notes and Limitations

- The language-model integration uses a deterministic mock implementation.
- Similarity scores are retrieval signals, not calibrated fraud probabilities.
- In-memory metrics reset when the application process restarts.
- This project is a prototype for educational evaluation and should not be treated as a production fraud decision system without further testing, monitoring, security review, and validation.
