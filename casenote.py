
import json

import mock_llm
from retrieval import retrieve, MIN_SCORE


CASE_NOTE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "priority": {
            "type": "string",
            "enum": ["low", "medium", "high", "critical"],
        },
        "recommended_team": {
            "type": "string",
            "enum": [
                "fraud_ops",
                "customer_care",
                "payments",
                "account_security",
                "card_risk",
                "upi_risk",
                "merchant_ops",
            ],
        },
        "recommended_action": {
            "type": "string",
            "enum": [
                "approve",
                "hold_for_review",
                "block_and_notify",
            ],
        },
        "policy_id": {
            "type": "string",
            "enum": ["P1", "P2", "P3", "P4", "P5", "P6"],
        },
    },
    "required": [
        "summary",
        "priority",
        "recommended_team",
        "recommended_action",
        "policy_id",
    ],
}


def build_prompt(
    txn_id,
    fraud_probability,
    amount_inr,
    policy_id,
    policy_text,
):
    """Build a constrained prompt for the mock LLM."""

    return f"""ROLE:
You are a payment fraud operations case-note assistant.

TASK:
Create one concise case note for the transaction using the supplied policy.

CONSTRAINTS:
Use only the supplied policy text to guide the recommendation.
Do not invent facts about the transaction.
Return exactly one valid JSON object and nothing else.
Every required field must be present and have the correct type.
Use only the allowed enum values specified in the output format.

OUTPUT FORMAT:
Return a JSON object with these five keys:
- summary: string
- priority: one of low, medium, high, critical
- recommended_team: one of fraud_ops, customer_care, payments, account_security
- recommended_action: one of approve, hold_for_review, block_and_notify
- policy_id: one of P1, P2, P3, P4, P5, P6

TRANSACTION DETAILS:
TXN_ID: {txn_id}
FRAUD_PROBABILITY: {fraud_probability}
AMOUNT_INR: {amount_inr}
POLICY_ID: {policy_id}

POLICY TEXT:
{policy_text}
"""


def validate(text, retrieved_policy_id):
    """Return (parsed_note, gate_number); gate 0 means valid."""

    # Gate 1: response must be a JSON object.
    try:
        note = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None, 1

    if not isinstance(note, dict):
        return None, 1

    # Gate 2: required keys must exist and all values must be strings.
    required = CASE_NOTE_SCHEMA["required"]
    properties = CASE_NOTE_SCHEMA["properties"]

    if any(key not in note for key in required):
        return None, 2

    if any(not isinstance(note[key], str) for key in required):
        return None, 2

    # Gate 3: enum values and retrieved policy ID must be valid.
    for key in required:
        allowed_values = properties[key].get("enum")
        if allowed_values and note[key] not in allowed_values:
            return None, 3

    if note["policy_id"] != retrieved_policy_id:
        return None, 3

    return note, 0


def run_case(
    txn_id,
    fraud_probability,
    amount_inr,
    question,
    max_attempts=3,
):
    """Retrieve a policy, generate a note, validate it, and retry if needed."""

    retrieved_policy_id, policy_text, score = retrieve(question)

    if score < MIN_SCORE:
        return {
            "status": "no_policy_match",
            "attempts": 0,
            "failures": [],
            "case_note": None,
            "safe_default": False,
            "guardrail_override": False,
            "retrieved_policy_id": retrieved_policy_id,
            "retrieval_score": score,
        }

    prompt = build_prompt(
        txn_id,
        fraud_probability,
        amount_inr,
        retrieved_policy_id,
        policy_text,
    )

    failures = []
    case_note = None
    attempts = 0

    for _ in range(max_attempts):
        attempts += 1

        # ConnectionError intentionally propagates to the caller.
        response = mock_llm.generate(prompt)
        case_note, gate = validate(response, retrieved_policy_id)

        if gate == 0:
            break

        failures.append(gate)

    safe_default = case_note is None

    if safe_default:
        case_note = {
            "summary": (
                "Automated case note unavailable; manual review required."
            ),
            "priority": "high",
            "recommended_team": "customer_care",
            "recommended_action": "hold_for_review",
            "policy_id": retrieved_policy_id,
        }

    guardrail_override = False

    if (
        case_note["recommended_action"] == "approve"
        and fraud_probability >= 0.5
    ):
        case_note["recommended_action"] = "hold_for_review"
        guardrail_override = True

    return {
        "status": "ok",
        "attempts": attempts,
        "failures": failures,
        "case_note": case_note,
        "safe_default": safe_default,
        "guardrail_override": guardrail_override,
        "retrieved_policy_id": retrieved_policy_id,
        "retrieval_score": score,
    }


def main():
    """Run all supplied scenarios and print a compact results table."""

    with open("case_scenarios.json", "r", encoding="utf-8") as file:
        scenarios = json.load(file)

    mock_llm.reset()

    print("\nCASE NOTE TEST RESULTS")
    print("-" * 112)
    print(
        f"{'Scenario':<10} {'Policy':<9} {'Attempts':<9} "
        f"{'Failures':<14} {'Status':<18} {'Safe default':<13} "
        f"{'Final action':<20} {'Override':<8}"
    )
    print("-" * 112)

    for scenario in scenarios:
        result = run_case(
            txn_id=scenario["txn_id"],
            fraud_probability=scenario["fraud_probability"],
            amount_inr=scenario["amount_inr"],
            question=scenario["question"],
        )

        note = result["case_note"]
        final_action = (
            note["recommended_action"] if note else "none"
        )

        print(
            f"{scenario['txn_id']:<10} "
            f"{str(result['retrieved_policy_id']):<9} "
            f"{result['attempts']:<9} "
            f"{str(result['failures']):<14} "
            f"{result['status']:<18} "
            f"{str(result['safe_default']):<13} "
            f"{final_action:<20} "
            f"{str(result['guardrail_override']):<8}"
        )

    print("-" * 112)
    print(f"Scenarios processed: {len(scenarios)}")
    print("Expected mock LLM calls: 19")


if __name__ == "__main__":
    main()
