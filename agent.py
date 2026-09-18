import re
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity


# ==================================================
# Load Golden Set
# ==================================================

golden = pd.read_csv(
    "data/golden_set.csv"
)


# ==================================================
# Intent Classifier
# ==================================================

intent_model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        )
    )
])

intent_model.fit(
    golden["customer_message"],
    golden["intent"]
)


# ==================================================
# Historical Data
# ==================================================

historical = pd.read_csv(
    "data/retrieval_labeled.csv"
)


# ==================================================
# Retrieval Index
# ==================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_features=50000,
    sublinear_tf=True
)

historical_matrix = vectorizer.fit_transform(
    historical["customer_message_clean"]
)


# ==================================================
# Clean Response
# ==================================================

def clean_response(response):

    response = str(response)

    response = re.sub(
        r"@\w+",
        "",
        response
    )

    response = re.sub(
        r"https?://\S+",
        "",
        response
    )

    response = re.sub(
        r"\^[A-Z]{1,4}\b",
        "",
        response
    )

    response = re.sub(
        r"\s+",
        " ",
        response
    )

    return response.strip()


# ==================================================
# Agent
# ==================================================

def run_agent(query):

    # ----------------------------------------------
    # Intent
    # ----------------------------------------------

    intent = intent_model.predict(
        [query]
    )[0]

    probabilities = intent_model.predict_proba(
        [query]
    )[0]

    confidence = probabilities.max()


    # ----------------------------------------------
    # Candidate filtering
    # ----------------------------------------------

    candidates = historical[
        historical["pseudo_intent"] == intent
    ]


    # ----------------------------------------------
    # Retrieval
    # ----------------------------------------------

    query_vector = vectorizer.transform(
        [query]
    )

    candidate_indices = candidates.index

    candidate_matrix = historical_matrix[
        candidate_indices
    ]

    scores = cosine_similarity(
        query_vector,
        candidate_matrix
    ).flatten()

    top_position = scores.argmax()

    best_similarity = scores[top_position]

    best_row = candidates.iloc[
        top_position
    ]


    # ----------------------------------------------
    # Escalation
    # ----------------------------------------------
    reasons = []

    # Low confidence in predicted intent
    if confidence < 0.30:
        reasons.append(
            "low intent confidence"
        )

    # Weak historical evidence
    if best_similarity < 0.35:
        reasons.append(
            "weak historical evidence"
        )

    # Issues that may require account-specific information
    if intent in [
        "account_login",
        "payment_billing"
    ]:
        reasons.append(
            "issue may require account-specific information"
        )

    # Technical issues need stronger evidence
    if (
        intent == "product_technical"
        and best_similarity < 0.50
    ):
        reasons.append(
            "technical issue lacks strong historical evidence"
        )

    if reasons:

        decision = "ESCALATE"

        reason = "; ".join(reasons)

    else:

        decision = "AUTO_HANDLE"

        reason = (
            "intent confidence and historical evidence "
            "meet the automation criteria"
        )


    # ----------------------------------------------
    # Draft response
    # ----------------------------------------------

    response = clean_response(
        best_row["brand_response"]
    )

    response = (
        "Thanks for reaching out. "
        + response
    )


    return {
        "intent": intent,
        "confidence": confidence,
        "similarity": best_similarity,
        "decision": decision,
        "reason": reason,
        "response": response,
        "evidence_customer": (
            best_row["customer_message_clean"]
        ),
        "evidence_response": clean_response(
            best_row["brand_response"]
        )
    }


# ==================================================
# Test
# ==================================================

query = (
    "My package hasn't arrived yet "
    "and the delivery is late"
)

result = run_agent(query)


print("\n" + "=" * 70)
print("AI CUSTOMER SUPPORT AGENT")
print("=" * 70)

print("\nCustomer:")
print(query)

print("\nIntent:")
print(result["intent"])

print(
    "\nIntent confidence:",
    round(result["confidence"], 4)
)

print(
    "\nRetrieval similarity:",
    round(result["similarity"], 4)
)

print("\nDecision:")
print(result["decision"])

print("\nReason:")
print(result["reason"])

print("\nDraft response:")
print(result["response"])

print("\nHistorical evidence:")
print(result["evidence_customer"])

print("\nHistorical response:")
print(result["evidence_response"])