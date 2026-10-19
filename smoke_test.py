
"""Basic smoke tests for the Fraud Triage Service."""

import json
from pathlib import Path

import pandas as pd

from prepare import add_features, clean_transactions
from retrieval import retrieve, MIN_SCORE
from casenote import run_case


ROOT = Path(__file__).resolve().parent


def main():
    print("Running fraud-triage-service smoke tests...")

    # 1. Required project files
    required_files = [
        "transactions_raw.csv",
        "fraud_policies.json",
        "retrieval_queries.json",
        "case_scenarios.json",
        "prepare.py",
        "train.py",
        "retrieval.py",
        "casenote.py",
        "app.py",
        "requirements.txt",
        "README.md",
        "service.log",
    ]
    missing = [name for name in required_files if not (ROOT / name).exists()]
    assert not missing, f"Missing required files: {missing}"
    print("PASS: Required project files exist")

    # 2. Data preparation
    raw = pd.read_csv(ROOT / "transactions_raw.csv")
    cleaned = clean_transactions(raw.copy())
    featured = add_features(cleaned.copy())

    assert len(cleaned) > 0, "Cleaning produced no rows"
    assert "is_night" in featured.columns, "Missing is_night feature"
    assert "high_value" in featured.columns, "Missing high_value feature"
    print("PASS: Data cleaning and feature engineering")

    # 3. Retrieval
    policy_id, policy_text, score = retrieve(
        "card payment above INR 50000 from a new device"
    )
    assert policy_id == "P1", f"Expected P1, got {policy_id}"
    assert score >= MIN_SCORE, f"Score below threshold: {score}"
    assert policy_text, "Retrieved policy text is empty"
    print(f"PASS: Policy retrieval (P1, score={score:.4f})")

    # 4. Case scenarios file format
    with open(ROOT / "case_scenarios.json", encoding="utf-8") as file:
        scenarios = json.load(file)
    assert isinstance(scenarios, list) and scenarios, (
        "case_scenarios.json must contain a non-empty list"
    )
    print(f"PASS: Case scenarios JSON ({len(scenarios)} scenarios)")

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
