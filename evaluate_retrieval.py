import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==================================================
# Load data
# ==================================================

golden = pd.read_csv(
    "data/golden_set.csv"
)

historical = pd.read_csv(
    "data/retrieval_data.csv"
)


print("Golden examples:", len(golden))
print("Historical examples:", len(historical))


# ==================================================
# Normalize text for duplicate detection
# ==================================================

def normalize_text(text):

    return (
        str(text)
        .lower()
        .strip()
        .replace("’", "'")
    )


golden["normalized_message"] = (
    golden["customer_message"]
    .apply(normalize_text)
)

historical["normalized_message"] = (
    historical["customer_message_clean"]
    .apply(normalize_text)
)


# ==================================================
# Build TF-IDF index
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
# Evaluate
# ==================================================

results = []

exact_exclusions = 0


for _, row in golden.iterrows():

    query = row["customer_message"]

    normalized_query = row["normalized_message"]


    # ------------------------------------------------
    # Find historical rows with exactly the same text
    # ------------------------------------------------

    excluded_mask = (
        historical["normalized_message"]
        == normalized_query
    )

    exact_exclusions += excluded_mask.sum()


    # ------------------------------------------------
    # Calculate similarity
    # ------------------------------------------------

    query_vector = vectorizer.transform(
        [query]
    )

    scores = cosine_similarity(
        query_vector,
        historical_matrix
    ).flatten()


    # ------------------------------------------------
    # Exclude exact duplicate messages
    # ------------------------------------------------

    scores[excluded_mask.values] = -1


    # ------------------------------------------------
    # Get top 5
    # ------------------------------------------------

    top_indices = scores.argsort()[::-1][:5]

    top_scores = scores[top_indices]


    results.append({
        "id": row["id"],
        "intent": row["intent"],
        "query": query,
        "top1_similarity": top_scores[0],
        "top3_average": top_scores[:3].mean(),
        "top5_average": top_scores[:5].mean()
    })


results_df = pd.DataFrame(results)


# ==================================================
# Results
# ==================================================

print("\n" + "=" * 60)
print("LEAKAGE-SAFE RETRIEVAL EVALUATION")
print("=" * 60)

print(
    "\nExact duplicate rows excluded:",
    exact_exclusions
)


print(
    "\nTop-1 similarity:",
    round(
        results_df["top1_similarity"].mean(),
        4
    )
)


print(
    "\nTop-3 average similarity:",
    round(
        results_df["top3_average"].mean(),
        4
    )
)


print(
    "\nTop-5 average similarity:",
    round(
        results_df["top5_average"].mean(),
        4
    )
)


print("\nTop-1 similarity distribution:")

print(
    results_df["top1_similarity"].describe()
)


# ==================================================
# Lowest retrieval examples
# ==================================================

print("\n" + "=" * 60)
print("LOWEST RETRIEVALS")
print("=" * 60)


worst = results_df.sort_values(
    "top1_similarity"
).head(10)


for _, row in worst.iterrows():

    print("\nID:", row["id"])

    print(
        "Intent:",
        row["intent"]
    )

    print(
        "Query:",
        row["query"]
    )

    print(
        "Top-1 similarity:",
        round(
            row["top1_similarity"],
            4
        )
    )


# ==================================================
# Save
# ==================================================

results_df.to_csv(
    "data/retrieval_evaluation.csv",
    index=False
)


print(
    "\nSaved to: data/retrieval_evaluation.csv"
)