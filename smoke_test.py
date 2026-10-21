
"""Smoke tests for the Fraud Triage Service."""

import json
from pathlib import Path

import pandas as pd
import requests

from prepare import add_features, clean_transactions
from retrieval import retrieve, MIN_SCORE

ROOT = Path(__file__).resolve().parent
BASE_URL = "http://127.0.0.1:8000"


def main():
    print("Running fraud-triage-service smoke tests...")

    required_files = [
        "transactions_raw.csv", "fraud_policies.json",
        "retrieval_queries.json", "case_scenarios.json",
        "prepare.py", "train.py", "retrieval.py",
        "casenote.py", "app.py", "requirements.txt",
        "README.md", "service.log",
    ]
    missing = [name for name in required_files if not (ROOT / name).exists()]
    assert not missing, f"Missing required files: {missing}"
    print("PASS: Required project files exist")

    raw = pd.read_csv(ROOT / "transactions_raw.csv")
    cleaned = clean_transactions(raw.copy())
    featured = add_features(cleaned.copy())
    assert len(cleaned) > 0
    assert {"is_night", "high_value"}.issubset(featured.columns)
    print("PASS: Data cleaning and feature engineering")

    policy_id, policy_text, score = retrieve(
        "card payment above INR 50000 from a new device"
    )
    assert policy_id == "P1", f"Expected P1, got {policy_id}"
    assert score >= MIN_SCORE
    assert policy_text
    print(f"PASS: Policy retrieval (P1, score={score:.4f})")

    with open(ROOT / "case_scenarios.json", encoding="utf-8") as file:
        scenarios = json.load(file)
    assert isinstance(scenarios, list) and scenarios
    print(f"PASS: Case scenarios JSON ({len(scenarios)} scenarios)")

    health = requests.get(f"{BASE_URL}/health", timeout=10)
    assert health.status_code == 200, health.text
    print("PASS: GET /health")

    txn = {
        "txn_id": "T900001",
        "merchant_category": "Electronics",
        "city": "Bengaluru",
        "payment_mode": "Card",
        "amount_inr": 60000,
        "hour_of_day": 23,
        "txn_count_last_24h": 2,
        "avg_amount_30d_inr": 5000,
        "distance_from_home_km": 35,
        "is_new_device": 1,
        "account_age_days": 200,
    }

    response = requests.post(f"{BASE_URL}/score", json=txn, timeout=120)
    assert response.status_code == 200, response.text
    assert "fraud_probability" in response.json()
    print("PASS: POST /score")

    bad_txn = {**txn, "amount_inr": -1}
    response = requests.post(f"{BASE_URL}/score", json=bad_txn, timeout=30)
    assert response.status_code == 400, response.text
    print("PASS: POST /score validation")

    unknown_txn = {**txn, "city": "Unknown City"}
    response = requests.post(f"{BASE_URL}/score", json=unknown_txn, timeout=30)
    assert response.status_code == 400, response.text
    print("PASS: POST /score invalid category")

    case_payload = {
        **txn,
        "txn_id": "T900002",
        "question": "more than 6 UPI payments in 24 hours velocity alert",
        "payment_mode": "UPI",
        "txn_count_last_24h": 7,
    }
    response = requests.post(f"{BASE_URL}/case", json=case_payload, timeout=120)
    assert response.status_code == 200, response.text
    case_result = response.json()
    assert "attempts" in case_result and "status" in case_result
    print("PASS: POST /case velocity alert")

    safe_payload = {
        **txn,
        "txn_id": "T900005",
        "question": "electronics merchant payment on an account younger than 30 days",
        "account_age_days": 10,
    }
    response = requests.post(f"{BASE_URL}/case", json=safe_payload, timeout=120)
    assert response.status_code == 200, response.text
    print("PASS: POST /case young account")

    oos_payload = {
        **txn,
        "txn_id": "T900099",
        "question": "What is on the cafeteria menu today?",
    }
    response = requests.post(f"{BASE_URL}/case", json=oos_payload, timeout=120)
    assert response.status_code == 200, response.text
    print("PASS: POST /case out-of-scope question")

    metrics = requests.get(f"{BASE_URL}/metrics", timeout=10)
    assert metrics.status_code == 200, metrics.text
    assert "requests_total" in metrics.json()
    print("PASS: GET /metrics")

    print("\nSmoke tests completed.")


if __name__ == "__main__":
    main()
