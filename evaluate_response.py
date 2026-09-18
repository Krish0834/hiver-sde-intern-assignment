import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


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
# Run Agent
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

    if len(candidates) == 0:

        return {
            "intent": intent,
            "confidence": confidence,
            "similarity": 0,
            "decision": "ESCALATE",
            "reason": "no historical evidence",
            "response": "",
            "evidence_customer": "",
            "evidence_response": ""
        }


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

    if confidence < 0.30:

        reasons.append(
            "low intent confidence"
        )

    if best_similarity < 0.35:

        reasons.append(
            "weak historical evidence"
        )

    if intent in [
        "account_login",
        "payment_billing"
    ]:

        reasons.append(
            "issue may require account-specific information"
        )

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
    # Response
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
        "evidence_customer":
            best_row["customer_message_clean"],
        "evidence_response":
            clean_response(
                best_row["brand_response"]
            )
    }


# ==================================================
# Response Quality Evaluation
# ==================================================

def evaluate_response(
    customer_message,
    generated_response,
    evidence_customer,
    evidence_response
):

    generated_response = clean_response(
        generated_response
    )

    evidence_response = clean_response(
        evidence_response
    )

    # ----------------------------------------------
    # Relevance
    # ----------------------------------------------

    customer_words = set(
        re.findall(
            r"\b[a-zA-Z]{4,}\b",
            customer_message.lower()
        )
    )

    response_words = set(
        re.findall(
            r"\b[a-zA-Z]{4,}\b",
            generated_response.lower()
        )
    )

    if customer_words:

        relevance_overlap = (
            customer_words
            .intersection(response_words)
        )

        relevance = min(
            len(relevance_overlap)
            / len(customer_words),
            1.0
        )

    else:

        relevance = 0


    # ----------------------------------------------
    # Groundedness
    # ----------------------------------------------

    evidence_words = set(
        re.findall(
            r"\b[a-zA-Z]{4,}\b",
            evidence_response.lower()
        )
    )

    if response_words:

        grounded_overlap = (
            response_words
            .intersection(evidence_words)
        )

        groundedness = min(
            len(grounded_overlap)
            / len(response_words),
            1.0
        )

    else:

        groundedness = 0


    # ----------------------------------------------
    # Helpfulness
    # ----------------------------------------------

    helpful_phrases = [
        "please",
        "let us know",
        "contact",
        "check",
        "try",
        "wait",
        "reach out",
        "follow up",
        "update",
        "tomorrow"
    ]

    helpful_count = 0

    for phrase in helpful_phrases:

        if phrase in generated_response.lower():

            helpful_count += 1


    helpfulness = min(
        helpful_count / 3,
        1.0
    )


    # ----------------------------------------------
    # Evidence Similarity
    # ----------------------------------------------

    evidence_vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
    )

    try:

        vectors = evidence_vectorizer.fit_transform([
            generated_response,
            evidence_response
        ])

        evidence_similarity = cosine_similarity(
            vectors[0:1],
            vectors[1:2]
        )[0][0]

    except:

        evidence_similarity = 0


    # ----------------------------------------------
    # Overall Score
    # ----------------------------------------------

    overall = (
        0.30 * relevance
        + 0.35 * groundedness
        + 0.20 * helpfulness
        + 0.15 * evidence_similarity
    )


    return {
        "relevance": relevance,
        "groundedness": groundedness,
        "helpfulness": helpfulness,
        "evidence_similarity":
            evidence_similarity,
        "overall": overall
    }


# ==================================================
# Evaluate Golden Set
# ==================================================

results = []


print("\nRunning agent evaluation...")


for index, row in golden.iterrows():

    customer_message = row[
        "customer_message"
    ]

    true_intent = row[
        "intent"
    ]


    # Run actual agent
    result = run_agent(
        customer_message
    )


    # Evaluate response
    quality = evaluate_response(
        customer_message,
        result["response"],
        result["evidence_customer"],
        result["evidence_response"]
    )


    results.append({

        "id":
            row["id"],

        "customer_message":
            customer_message,

        "true_intent":
            true_intent,

        "predicted_intent":
            result["intent"],

        "intent_confidence":
            result["confidence"],

        "retrieval_similarity":
            result["similarity"],

        "decision":
            result["decision"],

        "reason":
            result["reason"],

        "generated_response":
            result["response"],

        "evidence_customer":
            result["evidence_customer"],

        "evidence_response":
            result["evidence_response"],

        "relevance":
            quality["relevance"],

        "groundedness":
            quality["groundedness"],

        "helpfulness":
            quality["helpfulness"],

        "evidence_similarity":
            quality["evidence_similarity"],

        "overall_response_score":
            quality["overall"]
    })


# ==================================================
# Create DataFrame
# ==================================================

results_df = pd.DataFrame(
    results
)


# ==================================================
# Summary
# ==================================================

print("\n" + "=" * 70)
print("AGENT RESPONSE QUALITY EVALUATION")
print("=" * 70)


print(
    "\nExamples evaluated:",
    len(results_df)
)


print(
    "\nAverage relevance:",
    round(
        results_df["relevance"].mean(),
        3
    )
)


print(
    "Average groundedness:",
    round(
        results_df["groundedness"].mean(),
        3
    )
)


print(
    "Average helpfulness:",
    round(
        results_df["helpfulness"].mean(),
        3
    )
)


print(
    "Average evidence similarity:",
    round(
        results_df[
            "evidence_similarity"
        ].mean(),
        3
    )
)


print(
    "Average overall response score:",
    round(
        results_df[
            "overall_response_score"
        ].mean(),
        3
    )
)


# ==================================================
# Decision Statistics
# ==================================================

print("\nDecision distribution:")

print(
    results_df[
        "decision"
    ].value_counts()
)


# ==================================================
# Save Results
# ==================================================

results_df.to_csv(
    "data/response_quality_results.csv",
    index=False
)


print(
    "\nSaved:"
    " data/response_quality_results.csv"
)


# ==================================================
# Show Best Examples
# ==================================================

print("\nTop 5 responses:")

top_examples = results_df.sort_values(
    "overall_response_score",
    ascending=False
).head(5)


for _, row in top_examples.iterrows():

    print("\n" + "-" * 70)

    print(
        "Customer:",
        row["customer_message"]
    )

    print(
        "Intent:",
        row["predicted_intent"]
    )

    print(
        "Response:",
        row["generated_response"]
    )

    print(
        "Score:",
        round(
            row["overall_response_score"],
            3
        )
    )


# ==================================================
# Show Worst Examples
# ==================================================

print("\nBottom 5 responses:")

worst_examples = results_df.sort_values(
    "overall_response_score",
    ascending=True
).head(5)


for _, row in worst_examples.iterrows():

    print("\n" + "-" * 70)

    print(
        "Customer:",
        row["customer_message"]
    )

    print(
        "Intent:",
        row["predicted_intent"]
    )

    print(
        "Response:",
        row["generated_response"]
    )

    print(
        "Evidence:",
        row["evidence_response"]
    )

    print(
        "Score:",
        round(
            row["overall_response_score"],
            3
        )
    )