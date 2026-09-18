import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report


# --------------------------------------------------
# Load golden set
# --------------------------------------------------

FILE = "data/golden_set.csv"

df = pd.read_csv(FILE)

X = df["customer_message"]
y = df["intent"]


print("Dataset size:", len(df))
print("Number of intents:", y.nunique())


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


print("\nTraining examples:", len(X_train))
print("Test examples:", len(X_test))


# --------------------------------------------------
# Baseline 1: Majority Classifier
# --------------------------------------------------

majority_class = y_train.value_counts().idxmax()

majority_predictions = [majority_class] * len(y_test)

majority_accuracy = accuracy_score(
    y_test,
    majority_predictions
)

majority_f1 = f1_score(
    y_test,
    majority_predictions,
    average="macro",
    zero_division=0
)

print("\n" + "=" * 50)
print("BASELINE 1: MAJORITY CLASSIFIER")
print("=" * 50)

print("Majority class:", majority_class)
print("Accuracy:", round(majority_accuracy, 4))
print("Macro F1:", round(majority_f1, 4))


# --------------------------------------------------
# Baseline 2: TF-IDF + Logistic Regression
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


model.fit(X_train, y_train)

predictions = model.predict(X_test)


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


print("\n" + "=" * 50)
print("BASELINE 2: TF-IDF + LOGISTIC REGRESSION")
print("=" * 50)

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