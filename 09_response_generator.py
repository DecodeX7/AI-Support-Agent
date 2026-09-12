import json
import os
from pathlib import Path

import faiss
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# =========================================================
# AmazonHelp Grounded Response Generator
# Step 09
# =========================================================
#
# Purpose:
#   Given a customer message:
#       1. classify intent
#       2. retrieve historical AmazonHelp examples
#       3. build a grounded response context
#
# This version supports:
#   - Ollama (recommended for local/no API cost)
#   - OpenAI-compatible API if OPENAI_API_KEY is configured
#   - deterministic fallback response if no LLM is configured
#
# The actual LangGraph orchestration comes later.
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

VECTOR_DIR = BASE_DIR / "data" / "vectorstore" / "amazonhelp_faiss"
MODEL_DIR = BASE_DIR / "data" / "models" / "intent_classifier"

INDEX_FILE = VECTOR_DIR / "index.faiss"
METADATA_FILE = VECTOR_DIR / "metadata.csv"
CLASSIFIER_FILE = MODEL_DIR / "amazonhelp_intent_classifier.joblib"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5


def read_csv_robust(path):
    for encoding in ["utf-8-sig", "utf-8", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, encoding=encoding).fillna("")
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise UnicodeError(f"Could not decode CSV: {path}")


def clean(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def load_components():
    for path in [INDEX_FILE, METADATA_FILE, CLASSIFIER_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found:\n{path}")

    index = faiss.read_index(str(INDEX_FILE))
    metadata = read_csv_robust(METADATA_FILE)
    classifier = joblib.load(CLASSIFIER_FILE)
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    if index.ntotal != len(metadata):
        raise ValueError(
            f"Vector/metadata mismatch: {index.ntotal} vs {len(metadata)}"
        )

    return index, metadata, classifier, embedder


def classify_intent(classifier, text):
    prediction = classifier.predict([text])[0]
    probabilities = classifier.predict_proba([text])[0]

    best_index = int(np.argmax(probabilities))
    confidence = float(probabilities[best_index])

    return str(prediction), confidence


def retrieve(index, metadata, embedder, query, top_k=TOP_K):
    query_embedding = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        idx = int(idx)

        if idx < 0:
            continue

        row = metadata.iloc[idx]

        results.append(
            {
                "rag_id": clean(row["rag_id"]),
                "intent_hint": clean(row["intent_hint"]),
                "customer_problem": clean(row["customer_problem"]),
                "historical_response": clean(row["historical_response"]),
                "similarity": round(float(score), 4),
            }
        )

    return results


def build_context(retrieved):
    if not retrieved:
        return "No historical support examples were retrieved."

    blocks = []

    for i, item in enumerate(retrieved, start=1):
        blocks.append(
            f"""Historical Example {i}
Intent: {item['intent_hint']}
Customer problem: {item['customer_problem']}
Historical AmazonHelp response: {item['historical_response']}
Similarity: {item['similarity']}
"""
        )

    return "\n".join(blocks)


def deterministic_fallback(intent):
    """Safe fallback when no LLM is configured."""
    messages = {
        "delivery_shipping": (
            "I’m sorry you’re having trouble with your delivery. "
            "Please share your order details so the support team can "
            "check the delivery status and help you further."
        ),
        "payment": (
            "I’m sorry you’re facing a payment issue. "
            "Please share the order or payment details so the support "
            "team can check what happened and assist you."
        ),
        "refund": (
            "I’m sorry about the refund issue. "
            "Please share your order details so the support team can "
            "check the refund status and help you further."
        ),
        "return_replacement": (
            "I can help with the return or replacement issue. "
            "Please share your order details so the support team can "
            "check the available options."
        ),
        "product_issue": (
            "I’m sorry there is an issue with the product. "
            "Please share your order details and a brief description "
            "of the problem so the support team can assist you."
        ),
        "account_access": (
            "I’m sorry you’re having trouble accessing your account. "
            "Please share the issue you are seeing so the support team "
            "can help you regain access."
        ),
        "prime_membership": (
            "I can help with your Prime membership concern. "
            "Please share the relevant account or membership details "
            "so the support team can check this for you."
        ),
        "technical_app": (
            "I’m sorry you’re facing a technical issue. "
            "Please tell us what happens in the app or website, "
            "including any error message you see."
        ),
        "order_cancellation": (
            "I can help with the order cancellation issue. "
            "Please share your order details so the support team can "
            "check the cancellation status."
        ),
        "security_fraud": (
            "I’m sorry you’re dealing with a security or fraud concern. "
            "Please avoid sharing passwords or one-time codes. "
            "The support team should review the account activity securely."
        ),
        "general_support": (
            "I’d be happy to help. Please share your order or account "
            "details and describe the issue so the support team can "
            "look into it."
        ),
    }

    return messages.get(
        intent,
        "I’m sorry you’re facing an issue. Please share more details so we can help you."
    )


def generate_with_ollama(prompt):
    try:
        import requests
    except ImportError:
        return None

    model_name = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                },
            },
            timeout=60,
        )

        response.raise_for_status()
        data = response.json()

        return clean(data.get("response", "")) or None

    except Exception:
        return None


def generate_response(customer_message, intent, confidence, retrieved):
    context = build_context(retrieved)

    prompt = f"""
You are an Amazon customer support assistant.

Customer message:
{customer_message}

Predicted intent:
{intent}

Classifier confidence:
{confidence:.3f}

Historical AmazonHelp support examples:
{context}

Write one concise, professional customer-support reply.

Rules:
1. Use the historical examples only as guidance.
2. Do not invent order status, refund amounts, dates, tracking numbers,
   policies, links, or actions that are not supported by the context.
3. If required information is missing, ask the customer for it.
4. Do not claim that you performed an action unless the context supports it.
5. Never ask for passwords, OTPs, card PINs, or other sensitive credentials.
6. Keep the reply natural and helpful.
7. Do not mention that you are using RAG, embeddings, a classifier,
   historical examples, or internal systems.
8. Return only the customer-facing reply.
""".strip()

    # Prefer a local Ollama model if the user has one configured.
    ollama_response = generate_with_ollama(prompt)

    if ollama_response:
        return ollama_response, "ollama"

    # Safe deterministic fallback means the complete pipeline can still
    # be demonstrated without an LLM API key.
    return deterministic_fallback(intent), "deterministic_fallback"


def support_agent(customer_message):
    index, metadata, classifier, embedder = load_components()

    intent, confidence = classify_intent(
        classifier,
        customer_message,
    )

    retrieved = retrieve(
        index,
        metadata,
        embedder,
        customer_message,
        TOP_K,
    )

    response, generator = generate_response(
        customer_message,
        intent,
        confidence,
        retrieved,
    )

    return {
        "customer_message": customer_message,
        "intent": intent,
        "confidence": round(confidence, 4),
        "generator": generator,
        "response": response,
        "retrieved_examples": retrieved,
    }


def main():
    print("=" * 72)
    print("AmazonHelp Grounded Response Generator - Step 09")
    print("=" * 72)

    index, metadata, classifier, embedder = load_components()

    print(f"Vector documents: {index.ntotal:,}")
    print(f"Classifier classes: {len(classifier.classes_)}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Top-K retrieval: {TOP_K}")

    print("\nComponents loaded successfully.")

    print("\nInteractive demo")
    print("Type a customer message.")
    print("Type 'exit' to stop.")

    while True:
        try:
            customer_message = input("\nCustomer: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if customer_message.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        if not customer_message:
            continue

        result = support_agent(customer_message)

        print("\n--- Agent Result ---")
        print(f"Intent: {result['intent']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Generator: {result['generator']}")

        print("\nRetrieved historical examples:")
        for item in result["retrieved_examples"]:
            print(
                f"  [{item['similarity']:.4f}] "
                f"{item['intent_hint']}: "
                f"{item['customer_problem'][:100]}"
            )

        print("\nAgent response:")
        print(result["response"])


if __name__ == "__main__":
    main()
