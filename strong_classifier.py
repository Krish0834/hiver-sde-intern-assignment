import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report


# --------------------------------------------------
# Load data
# --------------------------------------------------

FILE = "data/golden_set.csv"

df = pd.read_csv(FILE)

X = df["customer_message"]
y = df["intent"]


# --------------------------------------------------
# Train / Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# TF-IDF + Linear SVM
# --------------------------------------------------

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LinearSVC(
            class_weight="balanced"
        )
    )
])


# --------------------------------------------------
# Train
# --------------------------------------------------

model.fit(X_train, y_train)


# --------------------------------------------------
# Predict
# --------------------------------------------------

predictions = model.predict(X_test)


# --------------------------------------------------
# Evaluate
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    predictions
)

macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)


print("=" * 55)
print("STRONG CLASSIFIER: TF-IDF + LINEAR SVM")
print("=" * 55)

print("Accuracy:", round(accuracy, 4))
print("Macro F1:", round(macro_f1, 4))

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)