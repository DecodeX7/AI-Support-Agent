"""
03_discover_amazonhelp_intents.py

Unsupervised intent discovery for AmazonHelp.

Run from:
    D:\My Work\AI Support Agent

Command:
    python 03_discover_amazonhelp_intents.py

Input:
    data/processed/amazonhelp_conversations_clean.csv

Outputs:
    data/processed/amazonhelp_cluster_profiles.csv
    data/processed/amazonhelp_cluster_samples.csv
    data/processed/amazonhelp_intent_discovery.json

Method:
    TF-IDF -> MiniBatchKMeans -> representative examples

IMPORTANT:
Clusters are NOT final intents. They are discovery groups. We will
inspect their top terms and representative examples, then manually
merge/name them into the final 8-12 intent taxonomy.
"""

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans


# ============================================================
# CONFIG
# ============================================================

INPUT = Path("data/processed/amazonhelp_conversations_clean.csv")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

N_CLUSTERS = 20
MAX_FEATURES = 12000
MIN_DF = 5
MAX_DF = 0.85
SAMPLE_PER_CLUSTER = 12
RANDOM_STATE = 42


# ============================================================
# LOAD
# ============================================================

if not INPUT.exists():
    raise FileNotFoundError(
        f"Could not find {INPUT}. "
        "Make sure you are running this from the project root."
    )

print("Loading AmazonHelp conversations...")
df = pd.read_csv(INPUT)

required = {"conversation_id", "customer_turns", "company_turns"}

missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")


# ============================================================
# BUILD TEXT FOR DISCOVERY
# ============================================================

# Prefer the complete customer side. This is more informative than
# relying on only the first customer tweet.
if "customer_messages_clean" in df.columns:
    text_col = "customer_messages_clean"
elif "customer_text_for_intent" in df.columns:
    text_col = "customer_text_for_intent"
elif "first_customer_message_clean" in df.columns:
    text_col = "first_customer_message_clean"
else:
    raise ValueError(
        "No customer text column found in the processed dataset."
    )

df["discovery_text"] = (
    df[text_col]
    .fillna("")
    .astype(str)
)


def clean_for_tfidf(text):
    text = text.lower()

    # Remove Twitter URLs / mentions already represented as tokens.
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\buser\b", " ", text)

    # Keep words and useful apostrophes.
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


df["tfidf_text"] = df["discovery_text"].map(clean_for_tfidf)

# Remove empty conversations.
df = df[df["tfidf_text"].str.len() >= 10].copy()

print(f"Conversations used for clustering: {len(df):,}")


# ============================================================
# TF-IDF
# ============================================================

print("Building TF-IDF matrix...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=MIN_DF,
    max_df=MAX_DF,
    max_features=MAX_FEATURES,
    sublinear_tf=True,
)

X = vectorizer.fit_transform(df["tfidf_text"])

print(f"TF-IDF shape: {X.shape}")


# ============================================================
# CLUSTERING
# ============================================================

print(f"Clustering into {N_CLUSTERS} discovery groups...")

model = MiniBatchKMeans(
    n_clusters=N_CLUSTERS,
    random_state=RANDOM_STATE,
    batch_size=2048,
    n_init=5,
    max_iter=100,
)

labels = model.fit_predict(X)

df["cluster_id"] = labels


# ============================================================
# TOP TERMS FOR EACH CLUSTER
# ============================================================

terms = np.array(vectorizer.get_feature_names_out())

profiles = []

for cluster_id in range(N_CLUSTERS):

    member_mask = labels == cluster_id
    count = int(member_mask.sum())

    # Cluster center gives a simple way to identify its strongest terms.
    center = model.cluster_centers_[cluster_id]
    top_indices = center.argsort()[::-1][:15]
    top_terms = terms[top_indices]

    cluster_df = df.loc[member_mask]

    profiles.append({
        "cluster_id": cluster_id,
        "conversation_count": count,
        "percentage": round(count / len(df) * 100, 2),
        "top_terms": ", ".join(top_terms),
        "median_turns": float(
            cluster_df["customer_turns"].median()
        ),
        "avg_customer_turns": round(
            float(cluster_df["customer_turns"].mean()), 2
        ),
        "avg_company_turns": round(
            float(cluster_df["company_turns"].mean()), 2
        ),
    })


profiles_df = pd.DataFrame(profiles).sort_values(
    "conversation_count",
    ascending=False
)

profiles_df.to_csv(
    OUT / "amazonhelp_cluster_profiles.csv",
    index=False
)


# ============================================================
# REPRESENTATIVE EXAMPLES
# ============================================================

print("Selecting representative examples...")

# Distance to each cluster center.
# For 100k rows, calculate one cluster at a time.
sample_rows = []

for cluster_id in range(N_CLUSTERS):

    member_indices = np.where(labels == cluster_id)[0]

    if len(member_indices) == 0:
        continue

    cluster_matrix = X[member_indices]

    # Squared Euclidean distance to cluster center.
    center = model.cluster_centers_[cluster_id]

    distances = np.asarray(
        cluster_matrix.multiply(cluster_matrix).sum(axis=1)
    ).ravel()

    # ||x-c||² = ||x||² + ||c||² - 2x.c
    center_norm = np.dot(center, center)

    dot = cluster_matrix.dot(center)
    dot = np.asarray(dot).ravel()

    distances = distances + center_norm - 2 * dot

    order = np.argsort(distances)[:SAMPLE_PER_CLUSTER]

    for rank, local_idx in enumerate(order, start=1):

        original_idx = member_indices[local_idx]
        row = df.iloc[original_idx]

        sample_rows.append({
            "cluster_id": cluster_id,
            "representative_rank": rank,
            "conversation_id": row["conversation_id"],
            "customer_turns": row["customer_turns"],
            "company_turns": row["company_turns"],
            "total_turns": row.get(
                "total_turns",
                row["customer_turns"] + row["company_turns"]
            ),
            "customer_text": row["discovery_text"],
            "company_response": row.get(
                "company_responses_clean",
                ""
            ),
        })


samples_df = pd.DataFrame(sample_rows)

samples_df.to_csv(
    OUT / "amazonhelp_cluster_samples.csv",
    index=False
)


# ============================================================
# HUMAN REVIEW SHEET
# ============================================================

review = profiles_df.copy()

review["proposed_intent_name"] = ""
review["merge_with_cluster"] = ""
review["keep_as_separate_intent"] = ""
review["notes"] = ""

review.to_csv(
    OUT / "amazonhelp_cluster_review_sheet.csv",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

summary = {
    "input_file": str(INPUT),
    "conversations_clustered": int(len(df)),
    "n_clusters": N_CLUSTERS,
    "tfidf_features": int(X.shape[1]),
    "method": "TF-IDF with unigrams+bigrams + MiniBatchKMeans",
    "outputs": [
        "amazonhelp_cluster_profiles.csv",
        "amazonhelp_cluster_samples.csv",
        "amazonhelp_cluster_review_sheet.csv",
    ],
    "warning": (
        "Clusters are discovery groups, not final intent labels. "
        "Review representative examples before defining the final taxonomy."
    ),
}

with open(
    OUT / "amazonhelp_intent_discovery.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(summary, f, indent=2)


# ============================================================
# PRINT USEFUL OUTPUT
# ============================================================

print("\n==========================================")
print("AmazonHelp intent discovery complete")
print("==========================================")

print(f"Conversations clustered : {len(df):,}")
print(f"Clusters                : {N_CLUSTERS}")
print(f"TF-IDF features         : {X.shape[1]:,}")

print("\nCluster overview:")
print(
    profiles_df[
        [
            "cluster_id",
            "conversation_count",
            "percentage",
            "top_terms",
        ]
    ].to_string(index=False)
)

print("\nGenerated files:")
print("  data/processed/amazonhelp_cluster_profiles.csv")
print("  data/processed/amazonhelp_cluster_samples.csv")
print("  data/processed/amazonhelp_cluster_review_sheet.csv")
print("  data/processed/amazonhelp_intent_discovery.json")

print("\nNEXT STEP:")
print(
    "Open cluster_profiles.csv and cluster_samples.csv. "
    "We will inspect the 20 clusters and merge/name them into "
    "the final 8-12 AmazonHelp intents."
)
