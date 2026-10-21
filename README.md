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
The selected Random Forest configuration uses `max_depth=None` and `min_samples_leaf=5`. At threshold `0.4`, the reported precision is approximately `0.2819`, recall `0.7260`, and F1 score `0.4061`.

## Part C results
Policy retrieval evaluation reports 8/8 in-scope top-1 hits and 2/2 out-of-scope queries rejected. The configured minimum retrieval score is `0.45`. The smoke test also confirms that the example policy query retrieves P1 with a score of approximately `0.5501`.

## Part D results
The FastAPI service provides `/health`, `/score`, `/case`, and `/metrics` endpoints. Request logging and in-memory operational metrics support basic monitoring. The smoke tests and Python syntax check pass.

## Limitations
- The case-note generator uses a mock LLM and does not provide real model-generated reasoning.
- Retrieval similarity scores are not calibrated probabilities and may produce false matches.
- In-memory metrics reset when the service restarts.
- Model performance depends on the quality and representativeness of the available training data.
- The service is a prototype and requires additional security, load, and production-readiness testing before deployment.

