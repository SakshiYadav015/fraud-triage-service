
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


MIN_SCORE = 0.45


def load_policies():
    """Load the six fraud policies from the project JSON file."""
    path = Path(__file__).with_name("fraud_policies.json")

    with path.open("r", encoding="utf-8") as file:
        policies = json.load(file)

    return policies


POLICIES = load_policies()

# Fit default TF-IDF settings on policy text only.
POLICY_TEXTS = [policy["text"] for policy in POLICIES]

VECTORIZER = TfidfVectorizer()
POLICY_MATRIX = VECTORIZER.fit_transform(POLICY_TEXTS)


def retrieve(query):
    """
    Return (policy_id, policy_text, score) for the most
    similar policy. MIN_SCORE is applied by the caller.
    """
    query_vector = VECTORIZER.transform([query])
    scores = cosine_similarity(query_vector, POLICY_MATRIX)[0]

    best_index = int(scores.argmax())
    best_policy = POLICIES[best_index]

    return (
        best_policy["policy_id"],
        best_policy["text"],
        float(scores[best_index]),
    )


def main():
    path = Path(__file__).with_name("retrieval_queries.json")

    with path.open("r", encoding="utf-8") as file:
        queries = json.load(file)

    print("\nPOLICY RETRIEVAL RESULTS")
    print("-" * 100)
    print(
        f"{'Query':<58} {'Expected':<12} "
        f"{'Retrieved':<18} {'Score':>7}"
    )
    print("-" * 100)

    hits = 0
    in_scope = 0
    out_of_scope_rejected = 0

    for item in queries:
        expected = item["expected_policy"]
        policy_id, _, score = retrieve(item["query"])

        if score < MIN_SCORE:
            displayed_policy = "no_policy_match"
        else:
            displayed_policy = policy_id

        if expected is not None:
            in_scope += 1
            if displayed_policy == expected:
                hits += 1
        elif displayed_policy == "no_policy_match":
            out_of_scope_rejected += 1

        query_display = item["query"][:56]

        print(
            f"{query_display:<58} "
            f"{str(expected):<12} "
            f"{displayed_policy:<18} "
            f"{score:>7.4f}"
        )

    print("-" * 100)
    print(f"In-scope top-1 hits: {hits}/{in_scope}")
    print(f"Out-of-scope queries rejected: {out_of_scope_rejected}/2")
    print(f"MIN_SCORE: {MIN_SCORE:.2f}")


if __name__ == "__main__":
    main()
