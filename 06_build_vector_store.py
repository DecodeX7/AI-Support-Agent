import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# =========================================================
# AmazonHelp RAG Vector Store
# Step 06
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "data" / "processed" / "amazonhelp_rag_corpus.csv"
VECTOR_DIR = BASE_DIR / "data" / "vectorstore" / "amazonhelp_faiss"

INDEX_FILE = VECTOR_DIR / "index.faiss"
METADATA_FILE = VECTOR_DIR / "metadata.csv"
CONFIG_FILE = VECTOR_DIR / "config.json"

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64


def main():
    print("=" * 65)
    print("AmazonHelp RAG Vector Store Builder")
    print("=" * 65)

    # -----------------------------------------------------
    # 1. Check input
    # -----------------------------------------------------
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"\nRAG corpus not found:\n{INPUT_FILE}\n\n"
            "Please run 05_create_rag_corpus.py first."
        )

    VECTOR_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # 2. Load RAG corpus
    # -----------------------------------------------------
    print("\nLoading corpus:")
    print(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE).fillna("")

    print(f"Documents loaded: {len(df):,}")

    # IMPORTANT:
    # The actual RAG corpus columns are:
    # rag_id, conversation_id, intent_hint,
    # customer_problem, historical_response,
    # document_text, source, golden_overlap
    required_columns = [
        "conversation_id",
        "intent_hint",
        "customer_problem",
        "historical_response",
        "document_text",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"\nMissing required columns: {missing_columns}\n"
            f"Available columns: {list(df.columns)}"
        )

    # -----------------------------------------------------
    # 3. Prepare text for embedding
    # -----------------------------------------------------
    # document_text is already prepared by
    # 05_create_rag_corpus.py and contains the historical
    # customer problem + support response.
    #
    # We use it directly for semantic retrieval.
    # -----------------------------------------------------

    df["embedding_text"] = df["document_text"].astype(str)

    # -----------------------------------------------------
    # 4. Load embedding model
    # -----------------------------------------------------
    print(f"\nLoading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    # -----------------------------------------------------
    # 5. Generate embeddings
    # -----------------------------------------------------
    print("\nGenerating embeddings...")

    embeddings = model.encode(
        df["embedding_text"].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    print(f"\nEmbedding shape: {embeddings.shape}")

    # -----------------------------------------------------
    # 6. Build FAISS index
    # -----------------------------------------------------
    #
    # Since embeddings are normalized:
    # Inner Product == Cosine Similarity
    #
    # IndexFlatIP performs exact similarity search.
    # With 5,000 documents this is fast and sufficient.
    # -----------------------------------------------------

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    print(f"FAISS vectors stored: {index.ntotal:,}")

    # -----------------------------------------------------
    # 7. Prepare metadata
    # -----------------------------------------------------
    #
    # Keep the original RAG fields so that the retriever
    # can later return useful historical examples.
    # -----------------------------------------------------

    metadata_columns = [
        "rag_id",
        "conversation_id",
        "intent_hint",
        "customer_problem",
        "historical_response",
        "document_text",
        "source",
        "golden_overlap",
    ]

    metadata = df[metadata_columns].copy()

    # Ensure rag_id exists and is aligned with FAISS index.
    if "rag_id" not in metadata.columns:
        metadata.insert(0, "rag_id", range(len(metadata)))

    # -----------------------------------------------------
    # 8. Save vector store
    # -----------------------------------------------------

    print("\nSaving vector store...")

    faiss.write_index(
        index,
        str(INDEX_FILE)
    )

    metadata.to_csv(
        METADATA_FILE,
        index=False
    )

    config = {
        "brand": "AmazonHelp",
        "embedding_model": MODEL_NAME,
        "similarity": "cosine",
        "faiss_index": "IndexFlatIP",
        "documents": int(len(metadata)),
        "embedding_dimension": int(dimension),
        "source_file": str(INPUT_FILE),
        "text_column": "document_text",
        "golden_overlap_checked": True,
    }

    CONFIG_FILE.write_text(
        json.dumps(
            config,
            indent=2
        ),
        encoding="utf-8"
    )

    # -----------------------------------------------------
    # 9. Final report
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("VECTOR STORE CREATED SUCCESSFULLY")
    print("=" * 65)

    print(f"Index:      {INDEX_FILE}")
    print(f"Metadata:   {METADATA_FILE}")
    print(f"Config:     {CONFIG_FILE}")
    print(f"Vectors:    {index.ntotal:,}")
    print(f"Dimension:  {dimension}")
    print(f"Model:      {MODEL_NAME}")

    print("=" * 65)


if __name__ == "__main__":
    main()
