import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity


# ==================================================
# Load data
# ==================================================

golden = pd.read_csv(
    "data/golden_set.csv"
)

historical = pd.read_csv(
    "data/retrieval_labeled.csv"
)

print("Golden examples:", len(golden))
print("Historical examples:", len(historical))


# ==================================================
# Intent model
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


# ==================================================
# 5-fold out-of-fold predictions
# ==================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


predictions = cross_val_predict(
    intent_model,
    golden["customer_message"],
    golden["intent"],
    cv=cv,
    method="predict"
)


probabilities = cross_val_predict(
    intent_model,
    golden["customer_message"],
    golden["intent"],
    cv=cv,
    method="predict_proba"
)


confidence = probabilities.max(axis=1)


# ==================================================
# Base classifier metrics
# ==================================================

accuracy = accuracy_score(
    golden["intent"],
    predictions
)

macro_f1 = f1_score(
    golden["intent"],
    predictions,
    average="macro",
    zero_division=0
)


print("\n" + "=" * 80)
print("LEAKAGE-SAFE AGENT EVALUATION")
print("=" * 80)

print(
    "\nOut-of-fold Accuracy:",
    round(accuracy, 4)
)

print(
    "Out-of-fold Macro F1:",
    round(macro_f1, 4)
)


# ==================================================
# Retrieval index
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
# Normalize text for leakage prevention
# ==================================================

def normalize_text(text):

    return (
        str(text)
        .lower()
        .strip()
        .replace("’", "'")
    )


golden_messages = (
    golden["customer_message"]
    .apply(normalize_text)
)

historical_messages = (
    historical["customer_message_clean"]
    .apply(normalize_text)
)


# ==================================================
# Retrieval similarities
# ==================================================

retrieval_similarities = []


for i, row in golden.iterrows():

    query = row["customer_message"]

    query_normalized = golden_messages.iloc[i]


    # ----------------------------------------------
    # Predict intent from OUT-OF-FOLD prediction
    # ----------------------------------------------

    predicted_intent = predictions[i]


    # ----------------------------------------------
    # Filter historical examples by predicted intent
    # ----------------------------------------------

    candidates = historical[
        historical["pseudo_intent"]
        == predicted_intent
    ]


    # ----------------------------------------------
    # Remove exact duplicate query messages
    # ----------------------------------------------

    candidates = candidates[
        candidates["customer_message_clean"]
        .apply(normalize_text)
        != query_normalized
    ]


    if len(candidates) == 0:

        retrieval_similarities.append(0)

        continue


    # ----------------------------------------------
    # Similarity
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


    best_similarity = scores.max()

    retrieval_similarities.append(
        best_similarity
    )


# ==================================================
# Threshold evaluation
# ==================================================

thresholds = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60
]


rows = []


for threshold in thresholds:

    decisions = []

    auto_correct = []

    for i in range(len(golden)):

        intent_confidence = confidence[i]

        retrieval_score = (
            retrieval_similarities[i]
        )


        # ------------------------------------------
        # Decision
        # ------------------------------------------

        if (
            intent_confidence >= threshold
            and retrieval_score >= 0.35
        ):

            decision = "AUTO_HANDLE"

        else:

            decision = "ESCALATE"


        decisions.append(decision)


        # ------------------------------------------
        # Correctness of auto-handled cases
        # ------------------------------------------

        if decision == "AUTO_HANDLE":

            auto_correct.append(
                predictions[i]
                == golden["intent"].iloc[i]
            )


    auto_count = decisions.count(
        "AUTO_HANDLE"
    )

    escalate_count = decisions.count(
        "ESCALATE"
    )


    if auto_count > 0:

        auto_accuracy = (
            sum(auto_correct)
            / auto_count
        )

    else:

        auto_accuracy = 0


    rows.append({

        "threshold": threshold,

        "accuracy": accuracy,

        "macro_f1": macro_f1,

        "auto_handle": auto_count,

        "escalate": escalate_count,

        "auto_rate": auto_count / len(golden),

        "auto_accuracy": auto_accuracy

    })


# ==================================================
# Display
# ==================================================

results = pd.DataFrame(rows)


print(
    "\n" + "=" * 80
)

print(
    "LEAKAGE-SAFE CONFIDENCE THRESHOLD EVALUATION"
)

print(
    "=" * 80
)


print(
    results.to_string(
        index=False,
        formatters={
            "accuracy": "{:.3f}".format,
            "macro_f1": "{:.3f}".format,
            "auto_rate": "{:.3f}".format,
            "auto_accuracy": "{:.3f}".format
        }
    )
)


# ==================================================
# Save
# ==================================================

results.to_csv(
    "data/agent_threshold_evaluation.csv",
    index=False
)


print(
    "\nSaved to: "
    "data/agent_threshold_evaluation.csv"
)