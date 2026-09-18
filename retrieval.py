import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# Load historical Amazon conversations
# --------------------------------------------------

FILE = "data/retrieval_data.csv"

df = pd.read_csv(FILE)

print("Historical conversations:", len(df))


# --------------------------------------------------
# Build TF-IDF index
# --------------------------------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_features=50000
)

matrix = vectorizer.fit_transform(
    df["customer_message_clean"]
)


# --------------------------------------------------
# Retrieval function
# --------------------------------------------------

def retrieve_similar_messages(
    query,
    top_k=5
):

    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector,
        matrix
    ).flatten()

    top_indices = scores.argsort()[-top_k:][::-1]

    results = df.iloc[top_indices].copy()

    results["similarity"] = scores[top_indices]

    return results


# --------------------------------------------------
# Test
# --------------------------------------------------

query = "My package hasn't arrived yet and the delivery is late"

results = retrieve_similar_messages(
    query,
    top_k=5
)


print("\n" + "=" * 70)
print("QUERY")
print("=" * 70)

print(query)


print("\n" + "=" * 70)
print("TOP SIMILAR HISTORICAL CONVERSATIONS")
print("=" * 70)


for _, row in results.iterrows():

    print("\nSimilarity:",
          round(row["similarity"], 4))

    print("Customer:",
          row["customer_message_clean"])

    print("Amazon:",
          row["brand_response"])