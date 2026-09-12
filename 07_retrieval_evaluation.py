import json
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# =========================================================
# AmazonHelp RAG Retrieval Evaluation
# Step 07 - ROBUST CSV ENCODING VERSION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

VECTOR_DIR = BASE_DIR / "data" / "vectorstore" / "amazonhelp_faiss"
INDEX_FILE = VECTOR_DIR / "index.faiss"
METADATA_FILE = VECTOR_DIR / "metadata.csv"

GOLDEN_FILE = BASE_DIR / "data" / "processed" / "amazonhelp_golden_final.csv"

OUTPUT_DIR = BASE_DIR / "data" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_FILE = OUTPUT_DIR / "retrieval_evaluation_results.csv"
SUMMARY_FILE = OUTPUT_DIR / "retrieval_evaluation_summary.json"

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K_VALUES = [1, 3, 5, 10]


def read_csv_robust(path):
    """Read CSVs containing Twitter text with mixed Windows/Unicode bytes."""
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1"]
    errors = []

    for encoding in encodings:
        try:
            df = pd.read_csv(path, encoding=encoding)
            print(f"Read {path.name} using encoding: {encoding}")
            return df.fillna("")
        except (UnicodeDecodeError, UnicodeError) as exc:
            errors.append(f"{encoding}: {exc}")

    raise UnicodeError(
        f"Could not decode CSV: {path}\n" + "\n".join(errors)
    )


def clean(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def main():
    print("=" * 70)
    print("AmazonHelp RAG Retrieval Evaluation")
    print("=" * 70)

    for path in [INDEX_FILE, METADATA_FILE, GOLDEN_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found:\n{path}")

    # -----------------------------------------------------
    # 1. Load vector store
    # -----------------------------------------------------
    print("\nLoading FAISS index...")
    index = faiss.read_index(str(INDEX_FILE))

    print("Loading metadata...")
    metadata = read_csv_robust(METADATA_FILE)

    print(f"RAG documents: {len(metadata):,}")
    print(f"FAISS vectors: {index.ntotal:,}")

    if index.ntotal != len(metadata):
        raise ValueError(
            f"FAISS/metadata mismatch: "
            f"{index.ntotal} vectors vs {len(metadata)} metadata rows."
        )

    # -----------------------------------------------------
    # 2. Load golden set
    # -----------------------------------------------------
    print("\nLoading golden evaluation set...")
    golden = read_csv_robust(GOLDEN_FILE)

    print(f"Golden examples: {len(golden):,}")

    required = [
        "golden_id",
        "golden_intent",
        "conversation_id",
        "first_customer_message",
        "customer_messages",
    ]

    missing = [c for c in required if c not in golden.columns]

    if missing:
        raise ValueError(
            f"Golden file is missing required columns: {missing}\n"
            f"Available columns: {list(golden.columns)}"
        )

    # The verified human-review label is golden_intent.
    intent_column = "golden_intent"

    # Fail early rather than silently producing meaningless 0% metrics.
    non_empty_intents = golden[intent_column].astype(str).str.strip().ne("").sum()

    if non_empty_intents != len(golden):
        raise ValueError(
            f"golden_intent contains {len(golden) - non_empty_intents} "
            f"empty values out of {len(golden)} rows. "
            "The evaluation cannot continue safely."
        )

    print("\nVerified intent labels:")
    print(golden[intent_column].value_counts().to_string())

    # -----------------------------------------------------
    # 3. Load embedding model
    # -----------------------------------------------------
    print(f"\nLoading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # -----------------------------------------------------
    # 4. Evaluate retrieval
    # -----------------------------------------------------
    print("\nRunning retrieval evaluation...")
    print("Query      = customer_messages")
    print("Gold label = golden_intent")
    print("K values   = 1, 3, 5, 10")

    rows = []
    latencies = []

    for count, (_, row) in enumerate(golden.iterrows(), start=1):

        query = clean(row["customer_messages"])

        if not query:
            query = clean(row["first_customer_message"])

        gold_intent = clean(row[intent_column])

        if not query:
            print(f"WARNING: {row['golden_id']} has empty customer query; skipping.")
            continue

        start = time.perf_counter()

        query_embedding = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = index.search(
            query_embedding,
            max(TOP_K_VALUES),
        )

        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)

        retrieved_indices = indices[0]
        retrieved_scores = scores[0]

        retrieved_intents = []
        retrieved_rag_ids = []

        for idx in retrieved_indices:
            idx = int(idx)

            if idx < 0:
                continue

            retrieved_rag_ids.append(clean(metadata.iloc[idx]["rag_id"]))
            retrieved_intents.append(clean(metadata.iloc[idx]["intent_hint"]))

        first_match_rank = None

        for rank, intent in enumerate(retrieved_intents, start=1):
            if intent == gold_intent:
                first_match_rank = rank
                break

        result = {
            "golden_id": clean(row["golden_id"]),
            "conversation_id": clean(row["conversation_id"]),
            "query": query,
            "gold_intent": gold_intent,
            "top1_intent": retrieved_intents[0] if retrieved_intents else "",
            "top1_score": (
                round(float(retrieved_scores[0]), 4)
                if len(retrieved_scores)
                else 0.0
            ),
            "first_matching_intent_rank": (
                first_match_rank if first_match_rank is not None else 0
            ),
            "top1_match": int(
                bool(retrieved_intents)
                and retrieved_intents[0] == gold_intent
            ),
            "mrr": (
                1.0 / first_match_rank
                if first_match_rank is not None
                else 0.0
            ),
            "latency_ms": round(latency_ms, 2),
        }

        for k in TOP_K_VALUES:
            result[f"recall_at_{k}"] = int(
                gold_intent in retrieved_intents[:k]
            )

        for rank, (rag_id, intent) in enumerate(
            zip(retrieved_rag_ids, retrieved_intents),
            start=1,
        ):
            result[f"top{rank}_rag_id"] = rag_id
            result[f"top{rank}_intent"] = intent
            result[f"top{rank}_score"] = round(
                float(retrieved_scores[rank - 1]),
                4,
            )

        rows.append(result)

        if count % 25 == 0:
            print(f"Processed {count}/{len(golden)} golden examples...")

    results = pd.DataFrame(rows)

    if results.empty:
        raise ValueError("No examples were evaluated.")

    # -----------------------------------------------------
    # 5. Overall metrics
    # -----------------------------------------------------
    metrics = {
        "evaluated_examples": int(len(results)),
        "rag_documents": int(len(metadata)),
        "embedding_model": MODEL_NAME,
        "top1_accuracy": round(float(results["top1_match"].mean()), 4),
        "recall_at_3": round(float(results["recall_at_3"].mean()), 4),
        "recall_at_5": round(float(results["recall_at_5"].mean()), 4),
        "recall_at_10": round(float(results["recall_at_10"].mean()), 4),
        "mrr": round(float(results["mrr"].mean()), 4),
        "mean_latency_ms": round(float(np.mean(latencies)), 2),
        "median_latency_ms": round(float(np.median(latencies)), 2),
    }

    # -----------------------------------------------------
    # 6. Intent-level breakdown
    # -----------------------------------------------------
    intent_breakdown = {}

    for intent, group in results.groupby("gold_intent"):
        intent_breakdown[intent] = {
            "examples": int(len(group)),
            "top1_accuracy": round(float(group["top1_match"].mean()), 4),
            "recall_at_5": round(float(group["recall_at_5"].mean()), 4),
            "mrr": round(float(group["mrr"].mean()), 4),
        }

    # -----------------------------------------------------
    # 7. Save
    # -----------------------------------------------------
    results.to_csv(RESULTS_FILE, index=False)

    summary = {
        "task": "AmazonHelp semantic retrieval evaluation",
        "method": (
            "SentenceTransformer all-MiniLM-L6-v2 + FAISS "
            "IndexFlatIP with normalized embeddings"
        ),
        "query_field": "customer_messages",
        "gold_label_field": "golden_intent",
        "metrics": metrics,
        "intent_breakdown": intent_breakdown,
        "evaluation_note": (
            "Recall@K and MRR measure whether at least one retrieved "
            "historical example has the same intent as the verified "
            "golden example. This is an intent-consistency proxy, "
            "not a manually annotated document-relevance benchmark."
        ),
    }

    SUMMARY_FILE.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    # -----------------------------------------------------
    # 8. Print
    # -----------------------------------------------------
    print("\n" + "=" * 70)
    print("RETRIEVAL EVALUATION COMPLETE")
    print("=" * 70)

    print(f"Evaluated examples: {metrics['evaluated_examples']}")
    print(f"RAG documents:      {metrics['rag_documents']}")
    print(f"Top-1 accuracy:     {metrics['top1_accuracy']:.2%}")
    print(f"Recall@3:           {metrics['recall_at_3']:.2%}")
    print(f"Recall@5:           {metrics['recall_at_5']:.2%}")
    print(f"Recall@10:          {metrics['recall_at_10']:.2%}")
    print(f"MRR:                {metrics['mrr']:.4f}")
    print(f"Mean latency:       {metrics['mean_latency_ms']:.2f} ms")
    print(f"Median latency:     {metrics['median_latency_ms']:.2f} ms")

    print("\nIntent breakdown:")
    for intent, values in intent_breakdown.items():
        print(
            f"  {intent:22s} "
            f"n={values['examples']:3d}  "
            f"Top1={values['top1_accuracy']:.1%}  "
            f"R@5={values['recall_at_5']:.1%}"
        )

    print("\nSaved:")
    print(RESULTS_FILE)
    print(SUMMARY_FILE)

    print("=" * 70)


if __name__ == "__main__":
    main()
