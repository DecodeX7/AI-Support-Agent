import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline


# =========================================================
# AmazonHelp Intent Classifier
# Step 08
# =========================================================
#
# Training source:
#   amazonhelp_rag_corpus.csv
#
# Evaluation source:
#   amazonhelp_golden_final.csv
#
# IMPORTANT:
# The golden set is NEVER used for training.
# The RAG corpus already excludes the 200 golden conversations.
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

RAG_FILE = BASE_DIR / "data" / "processed" / "amazonhelp_rag_corpus.csv"
GOLDEN_FILE = BASE_DIR / "data" / "processed" / "amazonhelp_golden_final.csv"

MODEL_DIR = BASE_DIR / "data" / "models" / "intent_classifier"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "amazonhelp_intent_classifier.joblib"
METRICS_FILE = MODEL_DIR / "intent_classifier_metrics.json"
PREDICTIONS_FILE = MODEL_DIR / "golden_intent_predictions.csv"
CONFUSION_FILE = MODEL_DIR / "confusion_matrix.csv"

RANDOM_STATE = 42


# ---------------------------------------------------------
# Robust CSV reader
# ---------------------------------------------------------

def read_csv_robust(path):
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1"]

    for encoding in encodings:
        try:
            df = pd.read_csv(path, encoding=encoding)
            print(f"Read {path.name} using encoding: {encoding}")
            return df.fillna("")
        except (UnicodeDecodeError, UnicodeError):
            continue

    raise UnicodeError(f"Could not decode CSV: {path}")


def clean(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def main():
    print("=" * 72)
    print("AmazonHelp Intent Classifier - Step 08")
    print("=" * 72)

    # -----------------------------------------------------
    # 1. Validate files
    # -----------------------------------------------------

    if not RAG_FILE.exists():
        raise FileNotFoundError(f"Training file not found:\n{RAG_FILE}")

    if not GOLDEN_FILE.exists():
        raise FileNotFoundError(f"Golden file not found:\n{GOLDEN_FILE}")

    # -----------------------------------------------------
    # 2. Load training data
    # -----------------------------------------------------

    print("\nLoading training data...")
    train_df = read_csv_robust(RAG_FILE)

    required_train = [
        "conversation_id",
        "intent_hint",
        "customer_problem",
    ]

    missing = [c for c in required_train if c not in train_df.columns]

    if missing:
        raise ValueError(
            f"Training file is missing columns: {missing}\n"
            f"Available columns: {list(train_df.columns)}"
        )

    # Use the customer's problem rather than the historical company
    # response. This prevents the classifier from learning response
    # wording instead of customer intent.
    train_df["text"] = train_df["customer_problem"].map(clean)
    train_df["label"] = train_df["intent_hint"].map(clean)

    train_df = train_df[
        (train_df["text"] != "")
        & (train_df["label"] != "")
    ].copy()

    # Remove accidental duplicate conversation IDs if any.
    train_df = train_df.drop_duplicates(
        subset=["conversation_id"],
        keep="first",
    )

    print(f"Training examples: {len(train_df):,}")
    print("\nTraining intent distribution:")
    print(train_df["label"].value_counts().to_string())

    # -----------------------------------------------------
    # 3. Load golden evaluation set
    # -----------------------------------------------------

    print("\nLoading golden evaluation set...")
    golden_df = read_csv_robust(GOLDEN_FILE)

    required_golden = [
        "golden_id",
        "golden_intent",
        "conversation_id",
        "first_customer_message",
        "customer_messages",
    ]

    missing = [c for c in required_golden if c not in golden_df.columns]

    if missing:
        raise ValueError(
            f"Golden file is missing columns: {missing}\n"
            f"Available columns: {list(golden_df.columns)}"
        )

    golden_df["gold_intent"] = golden_df["golden_intent"].map(clean)
    golden_df["text"] = golden_df["customer_messages"].map(clean)

    golden_df.loc[
        golden_df["text"] == "",
        "text"
    ] = golden_df.loc[
        golden_df["text"] == "",
        "first_customer_message"
    ].map(clean)

    golden_df = golden_df[
        (golden_df["gold_intent"] != "")
        & (golden_df["text"] != "")
    ].copy()

    print(f"Golden evaluation examples: {len(golden_df):,}")

    # -----------------------------------------------------
    # 4. Explicit golden leakage check
    # -----------------------------------------------------

    training_ids = set(
        train_df["conversation_id"].astype(str).str.strip()
    )

    golden_ids = set(
        golden_df["conversation_id"].astype(str).str.strip()
    )

    overlap = training_ids.intersection(golden_ids)

    print(f"Training/golden conversation overlap: {len(overlap)}")

    if overlap:
        raise ValueError(
            "DATA LEAKAGE DETECTED: golden conversations appear in "
            "the training data. Stop and fix the training source."
        )

    # -----------------------------------------------------
    # 5. Build TF-IDF + Logistic Regression pipeline
    # -----------------------------------------------------

    print("\nBuilding classifier...")
    print("Model: TF-IDF (1-2 grams) + Logistic Regression")

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    sublinear_tf=True,
                    max_features=30000,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1500,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    X_train = train_df["text"].values
    y_train = train_df["label"].values

    print("\nTraining classifier...")
    pipeline.fit(X_train, y_train)

    # -----------------------------------------------------
    # 6. Evaluate on the untouched golden set
    # -----------------------------------------------------

    print("\nEvaluating on verified golden set...")

    X_gold = golden_df["text"].values
    y_gold = golden_df["gold_intent"].values

    predictions = pipeline.predict(X_gold)
    probabilities = pipeline.predict_proba(X_gold)

    confidence = probabilities.max(axis=1)

    labels = sorted(
        set(y_gold).union(set(pipeline.classes_))
    )

    accuracy = accuracy_score(y_gold, predictions)
    macro_f1 = f1_score(
        y_gold,
        predictions,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        y_gold,
        predictions,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y_gold,
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_gold,
        predictions,
        labels=labels,
    )

    # -----------------------------------------------------
    # 7. Save predictions for failure analysis
    # -----------------------------------------------------

    prediction_df = golden_df[
        [
            "golden_id",
            "conversation_id",
            "text",
            "gold_intent",
        ]
    ].copy()

    prediction_df["predicted_intent"] = predictions
    prediction_df["confidence"] = np.round(confidence, 4)
    prediction_df["correct"] = (
        prediction_df["gold_intent"]
        == prediction_df["predicted_intent"]
    )

    prediction_df.to_csv(
        PREDICTIONS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    confusion_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels,
    )

    confusion_df.to_csv(
        CONFUSION_FILE,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------
    # 8. Per-intent metrics
    # -----------------------------------------------------

    per_intent = {}

    for label in labels:
        values = report.get(label)

        if not isinstance(values, dict):
            continue

        per_intent[label] = {
            "precision": round(float(values["precision"]), 4),
            "recall": round(float(values["recall"]), 4),
            "f1": round(float(values["f1-score"]), 4),
            "support": int(values["support"]),
        }

    metrics = {
        "training_examples": int(len(train_df)),
        "golden_examples": int(len(golden_df)),
        "training_golden_overlap": int(len(overlap)),
        "classes": [str(x) for x in pipeline.classes_],
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "mean_confidence": round(float(confidence.mean()), 4),
        "low_confidence_rate_below_0_60": round(
            float((confidence < 0.60).mean()),
            4,
        ),
        "low_confidence_rate_below_0_70": round(
            float((confidence < 0.70).mean()),
            4,
        ),
        "per_intent": per_intent,
    }

    # -----------------------------------------------------
    # 9. Save model + metrics
    # -----------------------------------------------------

    joblib.dump(
        pipeline,
        MODEL_FILE,
    )

    METRICS_FILE.write_text(
        json.dumps(
            {
                "task": "AmazonHelp customer intent classification",
                "method": (
                    "TF-IDF unigrams+bigrams + Logistic Regression "
                    "with balanced class weights"
                ),
                "evaluation": metrics,
                "leakage_protection": (
                    "The verified 200-example golden set was held out. "
                    "Training used the 5,000-document RAG corpus, which "
                    "was constructed with the golden conversations excluded."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # -----------------------------------------------------
    # 10. Print final report
    # -----------------------------------------------------

    print("\n" + "=" * 72)
    print("INTENT CLASSIFIER EVALUATION COMPLETE")
    print("=" * 72)

    print(f"Training examples:      {metrics['training_examples']:,}")
    print(f"Golden examples:        {metrics['golden_examples']:,}")
    print(f"Train/golden overlap:   {metrics['training_golden_overlap']}")
    print(f"Intent classes:         {len(metrics['classes'])}")
    print(f"Accuracy:               {metrics['accuracy']:.2%}")
    print(f"Macro F1:               {metrics['macro_f1']:.4f}")
    print(f"Weighted F1:            {metrics['weighted_f1']:.4f}")
    print(f"Mean confidence:        {metrics['mean_confidence']:.4f}")
    print(
        f"Confidence < 0.60:     "
        f"{metrics['low_confidence_rate_below_0_60']:.2%}"
    )
    print(
        f"Confidence < 0.70:     "
        f"{metrics['low_confidence_rate_below_0_70']:.2%}"
    )

    print("\nPer-intent performance:")
    for intent, values in per_intent.items():
        print(
            f"  {intent:22s} "
            f"P={values['precision']:.1%}  "
            f"R={values['recall']:.1%}  "
            f"F1={values['f1']:.1%}  "
            f"n={values['support']}"
        )

    print("\nSaved:")
    print(MODEL_FILE)
    print(METRICS_FILE)
    print(PREDICTIONS_FILE)
    print(CONFUSION_FILE)

    print("=" * 72)


if __name__ == "__main__":
    main()
