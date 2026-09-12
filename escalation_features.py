import numpy as np
from scipy.sparse import hstack, csr_matrix


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

    question = int("?" in text)

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


def build_escalation_features(
    messages,
    word_vectorizer,
    char_vectorizer
):

    word_features = word_vectorizer.transform(
        messages
    )

    char_features = char_vectorizer.transform(
        messages
    )

    risk_matrix = np.array(
        [
            risk_features(message)
            for message in messages
        ],
        dtype=float
    )

    risk_features_sparse = csr_matrix(
        risk_matrix
    )

    return hstack([
        word_features,
        char_features,
        risk_features_sparse
    ]).tocsr()