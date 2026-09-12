import os
import json
import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GOLDEN_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "amazonhelp_golden_final.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "data",
    "models",
    "escalation_classifier"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "escalation"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# ROBUST CSV READER
# ============================================================

def read_csv_robust(path):

    for encoding in [
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin1"
    ]:
        try:
            return pd.read_csv(
                path,
                encoding=encoding
            )
        except UnicodeDecodeError:
            continue

    raise RuntimeError(
        f"Could not decode CSV: {path}"
    )


# ============================================================
# LOAD GOLDEN DATA
# ============================================================

print("=" * 72)
print("AmazonHelp Escalation Classifier - Step 10.2")
print("=" * 72)

golden = read_csv_robust(
    GOLDEN_PATH
)

print(
    f"Golden examples: {len(golden)}"
)


# ============================================================
# PREPARE TEXT
# ============================================================

def get_message(row):

    message = row.get(
        "customer_messages",
        ""
    )

    if pd.isna(message) or not str(message).strip():

        message = row.get(
            "first_customer_message",
            ""
        )

    return str(message)


golden["message"] = golden.apply(
    get_message,
    axis=1
)


# ============================================================
# TARGET
# ============================================================

def parse_bool(value):

    if isinstance(value, str):

        return value.strip().lower() in [
            "true",
            "1",
            "yes",
            "y"
        ]

    return bool(value)


golden["escalation_label"] = (
    golden["escalation_required"]
    .apply(parse_bool)
    .astype(int)
)


print("\nEscalation distribution:")
print(
    golden["escalation_label"]
    .value_counts()
    .sort_index()
    .rename(
        index={
            0: "No escalation",
            1: "Escalation"
        }
    )
)


# ============================================================
# BUILD TEXT FEATURES
# ============================================================

texts = golden["message"].fillna("").astype(str)


# Word features
word_vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=1,
    max_df=0.98,
    sublinear_tf=True,
    max_features=12000
)


# Character features
char_vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=1,
    max_features=15000,
    sublinear_tf=True
)


X_word = word_vectorizer.fit_transform(
    texts
)

X_char = char_vectorizer.fit_transform(
    texts
)


# ============================================================
# EXPLICIT RISK FEATURES
# ============================================================

SECURITY_TERMS = [
    "hacked",
    "hack",
    "fraud",
    "fraudulent",
    "unauthorized",
    "stolen",
    "identity theft",
    "not my purchase",
    "unknown charge",
    "don't recognize",
    "do not recognize",
    "someone accessed",
    "someone changed my password",
    "someone changed my email",
]

HIGH_RISK_TERMS = [
    "lawyer",
    "lawsuit",
    "legal action",
    "police",
    "chargeback",
    "scam",
    "stolen",
]


def risk_features(text):

    text = text.lower()

    security = int(
        any(
            term in text
            for term in SECURITY_TERMS
        )
    )

    high_risk = int(
        any(
            term in text
            for term in HIGH_RISK_TERMS
        )
    )

    question = int(
        "?" in text
    )

    urgency = int(
        any(
            word in text
            for word in [
                "urgent",
                "immediately",
                "asap",
                "emergency"
            ]
        )
    )

    return [
        security,
        high_risk,
        question,
        urgency
    ]


risk_matrix = np.array(
    [
        risk_features(text)
        for text in texts
    ],
    dtype=float
)

X_risk = csr_matrix(
    risk_matrix
)


# ============================================================
# COMBINE FEATURES
# ============================================================

X = hstack([
    X_word,
    X_char,
    X_risk
]).tocsr()

y = golden[
    "escalation_label"
].values


print("\nFeature matrix:")
print(
    f"Shape: {X.shape}"
)


# ============================================================
# CROSS VALIDATION
# ============================================================

print("\n")
print("=" * 72)
print("5-FOLD STRATIFIED CROSS-VALIDATION")
print("=" * 72)


skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


fold_results = []
all_true = []
all_pred = []
all_prob = []

oof_predictions = np.zeros(
    len(golden)
)

oof_probabilities = np.zeros(
    len(golden)
)


for fold, (train_idx, test_idx) in enumerate(
    skf.split(X, y),
    start=1
):

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        C=1.0,
        solver="liblinear",
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    fold_results.append({
        "fold": fold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "test_examples": len(test_idx)
    })

    oof_predictions[test_idx] = predictions
    oof_probabilities[test_idx] = probabilities

    all_true.extend(
        y_test
    )

    all_pred.extend(
        predictions
    )

    all_prob.extend(
        probabilities
    )

    print(
        f"Fold {fold}: "
        f"Accuracy={accuracy:.2%}, "
        f"Precision={precision:.2%}, "
        f"Recall={recall:.2%}, "
        f"F1={f1:.2%}"
    )


# ============================================================
# OVERALL OOF METRICS
# ============================================================

overall_accuracy = accuracy_score(
    y,
    oof_predictions
)

overall_precision = precision_score(
    y,
    oof_predictions,
    zero_division=0
)

overall_recall = recall_score(
    y,
    oof_predictions,
    zero_division=0
)

overall_f1 = f1_score(
    y,
    oof_predictions,
    zero_division=0
)


print("\n")
print("=" * 72)
print("OVERALL OUT-OF-FOLD RESULTS")
print("=" * 72)

print(
    f"Accuracy : {overall_accuracy:.2%}"
)

print(
    f"Precision: {overall_precision:.2%}"
)

print(
    f"Recall   : {overall_recall:.2%}"
)

print(
    f"F1       : {overall_f1:.2%}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y,
    oof_predictions
)

tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix")
print("----------------")
print(f"TN: {tn}")
print(f"FP: {fp}")
print(f"FN: {fn}")
print(f"TP: {tp}")


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

print("\n")
print("=" * 72)
print("DECISION THRESHOLD ANALYSIS")
print("=" * 72)

threshold_results = []

for threshold in np.arange(
    0.20,
    0.81,
    0.05
):

    predictions = (
        oof_probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y,
        predictions
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    threshold_results.append({
        "threshold": round(
            float(threshold),
            2
        ),
        "predicted_escalations":
            int(predictions.sum()),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })


threshold_df = pd.DataFrame(
    threshold_results
)

print(
    threshold_df.to_string(
        index=False,
        formatters={
            "accuracy":
                lambda x: f"{x:.2%}",
            "precision":
                lambda x: f"{x:.2%}",
            "recall":
                lambda x: f"{x:.2%}",
            "f1":
                lambda x: f"{x:.2%}"
        }
    )
)


# ============================================================
# BEST THRESHOLD
# ============================================================

best_threshold_row = threshold_df.loc[
    threshold_df["f1"].idxmax()
]

best_threshold = float(
    best_threshold_row["threshold"]
)

best_predictions = (
    oof_probabilities >= best_threshold
).astype(int)


best_accuracy = accuracy_score(
    y,
    best_predictions
)

best_precision = precision_score(
    y,
    best_predictions,
    zero_division=0
)

best_recall = recall_score(
    y,
    best_predictions,
    zero_division=0
)

best_f1 = f1_score(
    y,
    best_predictions,
    zero_division=0
)


print("\n")
print("=" * 72)
print("BEST THRESHOLD")
print("=" * 72)

print(
    f"Threshold : {best_threshold:.2f}"
)

print(
    f"Accuracy  : {best_accuracy:.2%}"
)

print(
    f"Precision : {best_precision:.2%}"
)

print(
    f"Recall    : {best_recall:.2%}"
)

print(
    f"F1        : {best_f1:.2%}"
)


# ============================================================
# FINAL MODEL ON ALL GOLDEN DATA
# ============================================================

print("\nTraining final escalation classifier...")

final_model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced",
    C=1.0,
    solver="liblinear",
    random_state=42
)

final_model.fit(
    X,
    y
)


# ============================================================
# SAVE MODEL COMPONENTS
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "amazonhelp_escalation_classifier.joblib"
)

word_vectorizer_path = os.path.join(
    MODEL_DIR,
    "word_vectorizer.joblib"
)

char_vectorizer_path = os.path.join(
    MODEL_DIR,
    "char_vectorizer.joblib"
)

joblib.dump(
    final_model,
    model_path
)

joblib.dump(
    word_vectorizer,
    word_vectorizer_path
)

joblib.dump(
    char_vectorizer,
    char_vectorizer_path
)


# ============================================================
# SAVE OOF PREDICTIONS
# ============================================================

prediction_df = golden[
    [
        "golden_id",
        "message",
        "golden_intent",
        "escalation_label"
    ]
].copy()

prediction_df[
    "oof_probability"
] = oof_probabilities

prediction_df[
    "oof_prediction"
] = best_predictions

prediction_df[
    "correct"
] = (
    prediction_df[
        "escalation_label"
    ]
    ==
    prediction_df[
        "oof_prediction"
    ]
)


predictions_path = os.path.join(
    OUTPUT_DIR,
    "escalation_classifier_oof_predictions.csv"
)

prediction_df.to_csv(
    predictions_path,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# SAVE THRESHOLD RESULTS
# ============================================================

threshold_path = os.path.join(
    OUTPUT_DIR,
    "escalation_classifier_thresholds.csv"
)

threshold_df.to_csv(
    threshold_path,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {
    "examples": len(golden),
    "positive_examples": int(y.sum()),
    "negative_examples": int((y == 0).sum()),

    "cross_validation": {
        "folds": 5,
        "accuracy": round(
            best_accuracy,
            4
        ),
        "precision": round(
            best_precision,
            4
        ),
        "recall": round(
            best_recall,
            4
        ),
        "f1": round(
            best_f1,
            4
        )
    },

    "selected_threshold":
        best_threshold,

    "confusion_matrix": {
        "tn": int(
            (
                (y == 0) &
                (best_predictions == 0)
            ).sum()
        ),
        "fp": int(
            (
                (y == 0) &
                (best_predictions == 1)
            ).sum()
        ),
        "fn": int(
            (
                (y == 1) &
                (best_predictions == 0)
            ).sum()
        ),
        "tp": int(
            (
                (y == 1) &
                (best_predictions == 1)
            ).sum()
        )
    },

    "model":
        "TF-IDF word + character features + explicit risk features + Logistic Regression",

    "evaluation_note":
        "Metrics are out-of-fold predictions from 5-fold stratified cross-validation."
}


metrics_path = os.path.join(
    OUTPUT_DIR,
    "escalation_classifier_metrics.json"
)

with open(
    metrics_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metrics,
        f,
        indent=2
    )


# ============================================================
# SAVE POLICY
# ============================================================

policy = {
    "model":
        "amazonhelp_escalation_classifier",
    "threshold":
        best_threshold,
    "features": [
        "word_tfidf_1_2_grams",
        "character_tfidf_3_5_grams",
        "security_risk_features",
        "high_risk_features",
        "question_feature",
        "urgency_feature"
    ],
    "decision":
        "escalate_when_probability_is_at_or_above_threshold"
}

policy_path = os.path.join(
    OUTPUT_DIR,
    "escalation_classifier_policy.json"
)

with open(
    policy_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        policy,
        f,
        indent=2
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 72)
print("FILES SAVED")
print("=" * 72)

print(model_path)
print(word_vectorizer_path)
print(char_vectorizer_path)
print(predictions_path)
print(threshold_path)
print(metrics_path)
print(policy_path)

print("\n")
print("=" * 72)
print("Step 10.2 COMPLETE")
print("=" * 72)