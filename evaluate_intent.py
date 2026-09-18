import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)


# --------------------------------------------------
# Load Golden Set
# --------------------------------------------------

df = pd.read_csv(
    "data/golden_set.csv"
)

X = df["customer_message"]
y = df["intent"]


print("Golden set:", len(df))
print("Intents:", y.nunique())


# --------------------------------------------------
# Model
# --------------------------------------------------

model = Pipeline([
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


# --------------------------------------------------
# 5-Fold Stratified Cross Validation
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


predictions = cross_val_predict(
    model,
    X,
    y,
    cv=cv
)


# --------------------------------------------------
# Metrics
# --------------------------------------------------

accuracy = accuracy_score(
    y,
    predictions
)

macro_f1 = f1_score(
    y,
    predictions,
    average="macro",
    zero_division=0
)


print("\n" + "=" * 60)
print("5-FOLD CROSS-VALIDATION RESULTS")
print("=" * 60)

print(
    "Accuracy:",
    round(accuracy, 4)
)

print(
    "Macro F1:",
    round(macro_f1, 4)
)


print("\nClassification Report:")

print(
    classification_report(
        y,
        predictions,
        zero_division=0
    )
)