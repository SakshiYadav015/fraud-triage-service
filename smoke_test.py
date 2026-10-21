
"""Part D acceptance smoke tests for Fraud Triage Service."""

import json
import time
from pathlib import Path

import requests

BASE_URL = "http://127.0.0.1:8000"
ROOT = Path(__file__).resolve().parent


BENIGN = {
    "txn_id": "T800001",
    "merchant_category": "Grocery",
    "city": "Pune",
    "payment_mode": "UPI",
    "amount_inr": 1500,
    "hour_of_day": 13,
    "txn_count_last_24h": 0,
    "avg_amount_30d_inr": 1600,
    "distance_from_home_km": 3.0,
    "is_new_device": 0,
    "account_age_days": 900,
}

RISKY = {
    "txn_id": "T800002",
    "merchant_category": "Electronics",
    "city": "Mumbai",
    "payment_mode": "Card",
    "amount_inr": 45000,
    "hour_of_day": 2,
    "txn_count_last_24h": 9,
    "avg_amount_30d_inr": 3000,
    "distance_from_home_km": 60.0,
    "is_new_device": 1,
    "account_age_days": 40,
}

BIG = {
    **BENIGN,
    "txn_id": "T800003",
    "merchant_category": "Utilities",
    "payment_mode": "NetBanking",
    "amount_inr": 150000,
    "avg_amount_30d_inr": 250000,
    "distance_from_home_km": 0.0,
    "is_new_device": 1,
    "account_age_days": 2200,
}


def wait_for_service():
    """Retry health checks for up to 120 seconds."""
    deadline = time.monotonic() + 120

    while time.monotonic() < deadline:
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=3)
            if response.status_code == 200:
                if response.json().get("status") == "ok":
                    return
        except requests.RequestException:
            pass

        time.sleep(1)

    raise SystemExit(
        "Service did not become healthy within 120 seconds. "
        "Start it using: uvicorn app:app --port 8000"
    )


def main():
    print("Waiting for Fraud Triage Service...")
    wait_for_service()
    print("Service is healthy. Running Q1-Q11...\n")

    big_probability_ok = lambda d: (
        isinstance(d.get("fraud_probability"), (int, float))
        and 0.17 <= d["fraud_probability"] <= 0.26
    )

    invalid_amount = {**BENIGN, "txn_id": "T800004", "amount_inr": -5}
    missing_hour = {**BENIGN, "txn_id": "T800005"}
    missing_hour.pop("hour_of_day")

    velocity_case = {
        **RISKY,
        "txn_id": "T900002",
        "question": "more than 6 UPI payments in 24 hours velocity alert",
    }
    safe_default_case = {
        **RISKY,
        "txn_id": "T900005",
        "question": (
            "electronics merchant payment on an account younger than 30 days"
        ),
    }
    connection_error_case = {
        **RISKY,
        "txn_id": "T900099",
        "question": "card payment above INR 50000 from a new device",
    }
    benign_case = {
        **BENIGN,
        "txn_id": "T800001",
        "question": "card payment above INR 50000 from a new device",
    }
    no_policy_case = {
        **RISKY,
        "txn_id": "T800006",
        "question": "what is the cafeteria menu today",
    }

    tests = [
        {
            "id": "Q1", "category": "happy", "method": "GET",
            "path": "/health", "payload": None, "status": 200,
            "expected": "200; status=ok",
            "check": lambda d: d.get("status") == "ok",
        },
        {
            "id": "Q2", "category": "happy", "method": "POST",
            "path": "/score", "payload": BENIGN, "status": 200,
            "expected": "200; not flagged; approve",
            "check": lambda d: (
                d.get("flagged") is False
                and d.get("decision") == "approve"
                and d.get("human_review") is False
            ),
        },
        {
            "id": "Q3", "category": "happy", "method": "POST",
            "path": "/score", "payload": RISKY, "status": 200,
            "expected": "200; flagged; hold_for_review",
            "check": lambda d: (
                d.get("flagged") is True
                and d.get("decision") == "hold_for_review"
                and d.get("human_review") is True
            ),
        },
        {
            "id": "Q4", "category": "escalation", "method": "POST",
            "path": "/score", "payload": BIG, "status": 200,
            "expected": "200; unflagged big payment held; probability 0.17-0.26",
            "check": lambda d: (
                d.get("flagged") is False
                and d.get("decision") == "hold_for_review"
                and d.get("human_review") is True
                and big_probability_ok(d)
            ),
        },
        {
            "id": "Q5", "category": "escalation", "method": "POST",
            "path": "/score", "payload": invalid_amount, "status": 400,
            "expected": "400; field=amount_inr",
            "check": lambda d: (
                d.get("field") == "amount_inr" and "error" in d
            ),
        },
        {
            "id": "Q6", "category": "escalation", "method": "POST",
            "path": "/score", "payload": missing_hour, "status": 422,
            "expected": "422; missing required hour_of_day",
            "check": lambda d: bool(d),
        },
        {
            "id": "Q7", "category": "escalation", "method": "POST",
            "path": "/case", "payload": velocity_case, "status": 200,
            "expected": "200; policy P2; attempts=2; no safe default",
            "check": lambda d: (
                d.get("flagged") is True
                and isinstance(d.get("case_note"), dict)
                and d["case_note"].get("policy_id") == "P2"
                and d.get("attempts") == 2
                and d.get("safe_default") is False
                and d.get("human_review") is True
            ),
        },
        {
            "id": "Q8", "category": "edge", "method": "POST",
            "path": "/case", "payload": safe_default_case, "status": 200,
            "expected": "200; safe default; attempts=3; hold_for_review",
            "check": lambda d: (
                d.get("attempts") == 3
                and d.get("safe_default") is True
                and d.get("decision") == "hold_for_review"
                and d.get("human_review") is True
            ),
        },
        {
            "id": "Q9", "category": "edge", "method": "POST",
            "path": "/case", "payload": connection_error_case, "status": 503,
            "expected": "503; case-note service error returned safely",
            "check": lambda d: "error" in d,
        },
        {
            "id": "Q10", "category": "edge", "method": "POST",
            "path": "/case", "payload": benign_case, "status": 200,
            "expected": "200; no case note; attempts=0; approve",
            "check": lambda d: (
                d.get("case_note") is None
                and d.get("attempts") == 0
                and d.get("decision") == "approve"
                and d.get("human_review") is False
            ),
        },
        {
            "id": "Q11", "category": "edge", "method": "POST",
            "path": "/case", "payload": no_policy_case, "status": 200,
            "expected": "200; no policy match; held for review",
            "check": lambda d: (
                d.get("case_note") is None
                and d.get("attempts") == 0
                and d.get("decision") == "hold_for_review"
                and d.get("human_review") is True
            ),
        },
    ]

    results = []

    for test in tests:
        try:
            response = requests.request(
                method=test["method"],
                url=f'{BASE_URL}{test["path"]}',
                json=test["payload"],
                timeout=120,
            )
            try:
                body = response.json()
            except ValueError:
                body = {}

            passed = (
                response.status_code == test["status"]
                and test["check"](body)
            )
            observed = str(response.status_code)

        except (requests.RequestException, ValueError) as exc:
            passed = False
            observed = f"ERROR: {exc}"

        results.append({
            "id": test["id"],
            "category": test["category"],
            "expected": test["expected"],
            "observed": observed,
            "passed": passed,
        })

    print("ID  CATEGORY      EXPECTED / CHECK                  OBSERVED  RESULT")
    print("-" * 90)

    for item in results:
        result = "PASS" if item["passed"] else "FAIL"
        print(
            f'{item["id"]:<4}'
            f'{item["category"]:<14}'
            f'{item["expected"]:<36}'
            f'{item["observed"]:<10}'
            f'{result}'
        )

    print("\nCategory summary:")
    for category in ("happy", "escalation", "edge"):
        group = [r for r in results if r["category"] == category]
        passed_count = sum(r["passed"] for r in group)
        print(f"{category}: {passed_count}/{len(group)} passed")

    print("\nGET /metrics:")
    metrics_response = requests.get(f"{BASE_URL}/metrics", timeout=10)
    print(json.dumps(metrics_response.json(), indent=2))

    metrics = metrics_response.json()
    expected_metrics = {
        "requests_total": 10,
        "success_total": 7,
        "error_total": 3,
        "human_review_total": 5,
    }
    metrics_ok = all(
        metrics.get(key) == value
        for key, value in expected_metrics.items()
    )
    metrics_ok = (
        metrics_ok
        and metrics.get("success_rate") == 0.7
        and metrics.get("error_rate") == 0.3
        and metrics.get("human_review_rate") == 0.5
        and metrics.get("avg_latency_ms", 0) > 0
    )

    print(f"\nMetrics acceptance: {'PASS' if metrics_ok else 'FAIL'}")

    all_passed = all(r["passed"] for r in results) and metrics_ok
    print(
        f'Overall: {sum(r["passed"] for r in results)}/11 requests passed'
    )
    print("SMOKE TEST RESULT:", "PASS" if all_passed else "FAIL")

    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()