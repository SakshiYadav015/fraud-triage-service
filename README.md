# Razorpay Merchant Risk Desk — Fraud Triage Service

## Overview
A Python-based fraud triage service for scoring transaction risk, retrieving relevant fraud policies, and generating case notes.
The service exposes FastAPI endpoints for transaction scoring, case creation, health checks, and operational metrics.
It combines a trained fraud classification model, policy retrieval, and a mock case-note generator.

## Reproduce
Run these commands from the project root in a clean clone:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python smoke_test.py
python -m py_compile app.py smoke_test.py
python -m uvicorn app:app --reload
```

In a second terminal, run the API checks against `http://127.0.0.1:8000`. The service can be stopped with `Ctrl+C`.

## Part A results
Transaction data preparation includes data cleaning and feature engineering. The smoke test confirms that the prepared data pipeline runs successfully.

## Part B results

The selected Random Forest configuration uses `max_depth=None` and `min_samples_leaf=5`, with a mean 5-fold cross-validation F1 score of 0.4642. At the selected threshold of `0.4`, validation precision is approximately `0.2663`, recall is `0.7260`, and F1 score is `0.3897`. The threshold was selected to prioritize fraud recall while meeting the target recall constraint. The test set was not used for model selection.

## Part C results
Policy retrieval evaluation reports 8/8 in-scope top-1 hits and 2/2 out-of-scope queries rejected. The configured minimum retrieval score is `0.45`. The smoke test also confirms that the example policy query retrieves P1 with a score of approximately `0.5501`.

## Part D results

The FastAPI service provides `/health`, `/score`, `/case`, and `/metrics` endpoints. Request logging and in-memory operational metrics support basic monitoring. The smoke test passed all 11 checks across the happy, escalation, and edge categories.

### Smoke test resultss

| ID | Category | Expected result | Status | Result |
|---|---|---|---|---|
| Q1 | Happy | Health endpoint returns OK | 200 | PASS |
| Q2 | Happy | Benign transaction is not flagged | 200 | PASS |
| Q3 | Happy | Risky transaction is flagged | 200 | PASS |
| Q4 | Escalation | Large payment is held for review | 200 | PASS |
| Q5 | Escalation | Invalid amount is rejected | 400 | PASS |
| Q6 | Escalation | Missing required field is rejected | 422 | PASS |
| Q7 | Escalation | Policy P2 is retrieved after two attempts | 200 | PASS |
| Q8 | Edge | Safe default is applied after three attempts | 200 | PASS |
| Q9 | Edge | Case-note service failure is handled safely | 503 | PASS |
| Q10 | Edge | Benign case returns no case note | 200 | PASS |
| Q11 | Edge | No policy match results in review | 200 | PASS |

Category summary:
- Happy: 3/3 passed
- Escalation: 4/4 passed
- Edge: 4/4 passed
- **Overall: 11/11 passed**

### Operational metrics

The `/metrics` endpoint returned the following results after the smoke test:

```json
{
  "requests_total": 10,
  "success_total": 7,
  "error_total": 3,
  "success_rate": 0.7,
  "error_rate": 0.3,
  "human_review_total": 5,
  "human_review_rate": 0.5,
  "avg_latency_ms": 39.18
}
```

The average latency above is from this test run; it may vary between runs. The health check is a GET request, so it is not included in the POST request counters.

### Request log examples

Representative entries from `service.log`:

```text
2026-10-21 16:42:40,385 INFO method=POST endpoint=/score status=200 latency_ms=53.48 txn_id=T800001 fraud_probability=0.0284
2026-10-21 16:42:40,529 WARNING method=POST endpoint=/score status=400 latency_ms=1.21 txn_id=T800004 fraud_probability=n/a
2026-10-21 16:42:40,733 ERROR method=POST endpoint=/case status=503 latency_ms=47.67 txn_id=T900099 error=ConnectionError
```

The logs demonstrate INFO logging for successful requests, WARNING logging for client errors, and ERROR logging for server-side failures.


## Limitations
- The case-note generator uses a mock LLM and does not provide real model-generated reasoning.
- Retrieval similarity scores are not calibrated probabilities and may produce false matches.
- In-memory metrics reset when the service restarts.
- Model performance depends on the quality and representativeness of the available training data.
- The service is a prototype and requires additional security, load, and production-readiness testing before deployment.

