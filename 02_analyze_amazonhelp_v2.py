"""
02_analyze_amazonhelp_v2.py

Run from the root of the existing "AI Support Agent" project:

    python 02_analyze_amazonhelp_v2.py

Input:
    data/processed/amazonhelp_conversations_clean.csv

Outputs:
    data/processed/amazonhelp_intent_candidates_v2.csv
    data/processed/amazonhelp_intent_distribution_v2.csv
    data/processed/amazonhelp_rag_candidates_v2.csv
    data/processed/amazonhelp_review_candidates_v2.csv
    data/processed/amazonhelp_analysis_v2.json

IMPORTANT:
Candidate intents are heuristic labels used for data discovery only.
They are NOT ground truth and must be manually validated before
building the final golden evaluation set.
"""

from pathlib import Path
import json
import re
import pandas as pd


INPUT = Path("data/processed/amazonhelp_conversations_clean.csv")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. PHRASE-BASED INTENT RULES
# ============================================================
# These are intentionally broader than V1 and operate on the
# COMPLETE customer conversation, not just the first tweet.

RULES = {
    "delivery_issue": [
        r"\bwhere\s+(?:is|are)\s+(?:my|the)\s+(?:order|package|parcel)\b",
        r"\b(?:order|package|parcel)\s+(?:hasn't|has not|still hasn't|still has not)\s+(?:arrived|come)\b",
        r"\b(?:order|package|parcel)\s+(?:is\s+)?(?:late|delayed)\b",
        r"\bmarked\s+as\s+delivered\b",
        r"\bshows?\s+(?:as\s+)?delivered\b",
        r"\b(?:not|never)\s+(?:received|arrived)\b",
        r"\bstill\s+waiting\s+(?:for|on)\b",
        r"\bdelivery\s+(?:date|status|estimate)\b",
        r"\bshipping\s+(?:information|status|update)\b",
        r"\btracking\s+(?:number|information|status)\b",
        r"\bwhen\s+will\s+(?:my|the)\s+(?:order|package|parcel)\s+(?:arrive|come)\b",
        r"\b(?:order|package|parcel)\s+tracking\b",
    ],

    "payment_issue": [
        r"\bpayment\s+(?:failed|declined|error|problem|issue)\b",
        r"\b(?:can't|cannot|couldn't|could not)\s+(?:make|complete)\s+(?:a\s+)?payment\b",
        r"\b(?:card|payment\s+method)\s+(?:was|is|has\s+been)\s+(?:declined|rejected)\b",
        r"\bpayment\s+method\b",
        r"\bcredit\s+card\b",
        r"\bdebit\s+card\b",
        r"\bcharged\s+(?:twice|two\s+times|multiple\s+times)\b",
        r"\bdouble\s+charged\b",
        r"\bpayment\s+error\b",
        r"\bbilling\s+(?:issue|problem|error)\b",
    ],

    "refund_issue": [
        r"\brefund\b",
        r"\bmoney\s+(?:hasn't|has\s+not)\s+(?:been\s+)?(?:returned|refunded)\b",
        r"\bwhen\s+(?:will|should)\s+i\s+(?:get|receive)\s+(?:my\s+)?refund\b",
        r"\brefund\s+(?:hasn't|has\s+not)\s+(?:arrived|come|shown)\b",
        r"\brefund\s+(?:status|date)\b",
        r"\bwaiting\s+for\s+(?:my\s+)?refund\b",
    ],

    "return_replacement": [
        r"\b(?:want|need|how)\s+(?:to\s+)?return\b",
        r"\breturn\s+(?:an?\s+)?(?:item|order|product)\b",
        r"\breturn\s+label\b",
        r"\breturn\s+pickup\b",
        r"\breplacement\b",
        r"\breplace\s+(?:my|the|an?)\s+(?:item|order|product)\b",
        r"\bexchange\s+(?:an?\s+)?(?:item|product)\b",
    ],

    "product_issue": [
        r"\b(?:item|product)\s+(?:is\s+)?(?:damaged|broken|defective)\b",
        r"\bdamaged\s+(?:item|product|package)\b",
        r"\bdefective\s+(?:item|product)\b",
        r"\bwrong\s+(?:item|product)\b",
        r"\breceived\s+(?:the\s+)?wrong\s+(?:item|product)\b",
        r"\bcounterfeit\b",
        r"\bfake\s+(?:product|item)\b",
        r"\bproduct\s+quality\b",
    ],

    "account_issue": [
        r"\bcan't\s+(?:log|sign)\s+in\b",
        r"\bcannot\s+(?:log|sign)\s+in\b",
        r"\bcan't\s+access\s+(?:my\s+)?account\b",
        r"\bcannot\s+access\s+(?:my\s+)?account\b",
        r"\baccount\s+(?:locked|blocked|disabled|suspended)\b",
        r"\blogin\s+(?:problem|issue|error)\b",
        r"\bsign\s+in\s+(?:problem|issue|error)\b",
        r"\bforgot\s+(?:my\s+)?password\b",
        r"\bpassword\s+(?:reset|problem|issue)\b",
        r"\baccount\s+access\b",
    ],

    "membership_issue": [
        r"\bprime\s+(?:membership|subscription|trial)\b",
        r"\bprime\b",
        r"\bmembership\b",
        r"\bsubscription\b",
        r"\bsubscription\s+(?:renewed|renewal|cancel)\b",
        r"\bautomatically\s+renewed\b",
        r"\bfree\s+trial\b",
    ],

    "technical_issue": [
        r"\b(?:amazon\s+)?app\s+(?:is\s+)?(?:not\s+working|broken|crashing|crashes)\b",
        r"\bwebsite\s+(?:is\s+)?(?:not\s+working|broken|down)\b",
        r"\bsite\s+(?:is\s+)?(?:not\s+working|broken|down)\b",
        r"\b(?:app|website|site)\s+(?:error|issue|problem)\b",
        r"\bkindle\b",
        r"\balexa\b",
        r"\b(?:app|website)\s+crash(?:es|ed)?\b",
        r"\bnot\s+working\b",
        r"\bdoesn't\s+work\b",
        r"\bwon't\s+work\b",
        r"\btechnical\s+(?:issue|problem)\b",
        r"\bsync(?:ing)?\b",
    ],

    "cancellation_issue": [
        r"\bcancel(?:l|led|lation)?\b",
        r"\bcanceled\b",
        r"\bcancellation\b",
        r"\bwant\s+to\s+cancel\b",
        r"\bwhy\s+was\s+(?:my\s+)?order\s+cancelled\b",
    ],

    "security_fraud": [
        r"\bfraud\b",
        r"\bscam\b",
        r"\bphishing\b",
        r"\bhacked\b",
        r"\bhack(?:ed)?\s+account\b",
        r"\bunauthori[sz]ed\b",
        r"\bsuspicious\s+(?:charge|activity|email|message)\b",
        r"\bnot\s+my\s+(?:charge|purchase|order)\b",
        r"\bsomeone\s+(?:used|charged)\s+(?:my\s+)?(?:card|account)\b",
        r"\bidentity\s+theft\b",
        r"\bfake\s+(?:amazon\s+)?(?:email|message|website)\b",
    ],

    "other_general": [],
}


# ============================================================
# 2. HELPERS
# ============================================================

def norm(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = text.replace("&amp;", " and ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def score(text):
    scores = {}

    for intent, patterns in RULES.items():
        if not patterns:
            continue

        hits = 0
        for pattern in patterns:
            if re.search(pattern, text, flags=re.I):
                hits += 1

        if hits:
            scores[intent] = hits

    if not scores:
        return "other_general", 0, ""

    # Security gets priority because it is a high-risk category.
    if "security_fraud" in scores:
        best = "security_fraud"
    else:
        best = max(scores, key=scores.get)

    matches = ";".join(
        f"{k}:{v}"
        for k, v in sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    )

    return best, scores[best], matches


def has_any(text, patterns):
    return any(re.search(p, text, flags=re.I) for p in patterns)


def actionable_reply(text):
    patterns = [
        r"\bplease\b", r"\bgo to\b", r"\bclick\b", r"\bselect\b",
        r"\bcheck\b", r"\bcontact\b", r"\bcall\b", r"\breturn\b",
        r"\brefund\b", r"\breplace\b", r"\btrack\b", r"\bverify\b",
        r"\breset\b", r"\bupdate\b", r"\bfollow\b", r"\btry\b",
        r"\bsteps?\b", r"\binstructions?\b", r"\bhelp page\b",
    ]
    return has_any(text, patterns)


def generic_reply(text):
    patterns = [
        r"\bdm\b",
        r"\bdirect message\b",
        r"\bsend us a note\b",
        r"\breach out to us\b",
        r"\bcontact us\b",
        r"\bwe(?:'ll| will)\s+follow up\b",
        r"\bteam will be in touch\b",
        r"\bhere to help\b",
    ]
    return has_any(text, patterns)


# ============================================================
# 3. LOAD FULL CONVERSATIONS
# ============================================================

if not INPUT.exists():
    raise FileNotFoundError(
        f"Input not found: {INPUT}\n"
        "Run 01_extract_amazonhelp.py first."
    )

df = pd.read_csv(INPUT)

required = {
    "conversation_id",
    "total_turns",
    "customer_turns",
    "company_turns",
    "first_customer_message_clean",
    "company_responses_clean",
}

missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")


# ============================================================
# 4. USE THE WHOLE CUSTOMER SIDE
# ============================================================

# V1 only looked at first_customer_message_clean.
# V2 combines all customer messages when available.
if "customer_messages_clean" in df.columns:
    df["customer_text_for_intent"] = (
        df["customer_messages_clean"]
        .fillna("")
        .astype(str)
        .map(norm)
    )
else:
    df["customer_text_for_intent"] = (
        df["first_customer_message_clean"]
        .fillna("")
        .astype(str)
        .map(norm)
    )


# ============================================================
# 5. CANDIDATE INTENTS
# ============================================================

result = df["customer_text_for_intent"].map(score)

df["candidate_intent"] = result.map(lambda x: x[0])
df["intent_rule_score"] = result.map(lambda x: x[1])
df["intent_matches"] = result.map(lambda x: x[2])

df["company_reply_text"] = (
    df["company_responses_clean"]
    .fillna("")
    .astype(str)
    .map(norm)
)

df["actionable_reply_heuristic"] = (
    df["company_reply_text"].map(actionable_reply)
)

df["generic_reply_heuristic"] = (
    df["company_reply_text"].map(generic_reply)
)


# ============================================================
# 6. DATA QUALITY SIGNALS
# ============================================================

df["customer_text_length"] = (
    df["customer_text_for_intent"].str.len()
)

df["clear_intent_candidate"] = (
    (df["intent_rule_score"] >= 1)
    & (df["customer_text_length"] >= 10)
)

df["ambiguous_candidate"] = (
    df["intent_matches"].str.count(";") >= 1
)

# Good historical RAG candidates:
# - customer + company content exists
# - a candidate intent exists
# - response contains an action/instruction
# - avoid giant threads and purely generic routing replies
df["strong_rag_candidate"] = (
    (df["customer_turns"] > 0)
    & (df["company_turns"] > 0)
    & df["clear_intent_candidate"]
    & df["actionable_reply_heuristic"]
    & ~df["generic_reply_heuristic"]
    & (df["total_turns"] <= 12)
)


# ============================================================
# 7. GOLDEN-SET CANDIDATE QUALITY
# ============================================================

# We want the future golden set to contain diverse examples.
# These flags help us select easy/medium/hard cases.

df["difficulty"] = "medium"

df.loc[
    (df["intent_rule_score"] >= 2)
    & ~df["ambiguous_candidate"],
    "difficulty"
] = "easy"

df.loc[
    df["ambiguous_candidate"]
    | (df["intent_rule_score"] == 1)
    | (df["customer_text_length"] < 40),
    "difficulty"
] = "hard"

ESCALATION_PATTERNS = [
    r"\bfraud\b", r"\bscam\b", r"\bphishing\b",
    r"\bunauthori[sz]ed\b", r"\bhacked\b",
    r"\bsuspicious\b", r"\bnot\s+my\s+(?:charge|purchase)\b",
    r"\bcounterfeit\b", r"\bfake\s+(?:product|item)\b",
]

# has_any() expects one text string at a time, so apply it row-by-row.
df["escalation_candidate"] = df["customer_text_for_intent"].apply(
    lambda text: has_any(text, ESCALATION_PATTERNS)
)

# Candidate rows suitable for human golden-set review.
df["golden_review_candidate"] = (
    df["customer_turns"].gt(0)
    & df["company_turns"].gt(0)
    & df["clear_intent_candidate"]
    & df["total_turns"].le(12)
    & df["customer_text_length"].between(15, 800)
)


# ============================================================
# 8. SAVE INTENT CANDIDATES
# ============================================================

candidate_cols = [
    "conversation_id",
    "total_turns",
    "customer_turns",
    "company_turns",
    "customer_text_for_intent",
    "company_reply_text",
    "candidate_intent",
    "intent_rule_score",
    "intent_matches",
    "difficulty",
    "escalation_candidate",
    "actionable_reply_heuristic",
    "generic_reply_heuristic",
    "strong_rag_candidate",
    "golden_review_candidate",
]

df[candidate_cols].to_csv(
    OUT / "amazonhelp_intent_candidates_v2.csv",
    index=False
)


# ============================================================
# 9. INTENT DISTRIBUTION
# ============================================================

dist = (
    df["candidate_intent"]
    .value_counts()
    .rename_axis("candidate_intent")
    .reset_index(name="conversation_count")
)

dist["percentage"] = (
    dist["conversation_count"] / len(df) * 100
).round(2)

dist.to_csv(
    OUT / "amazonhelp_intent_distribution_v2.csv",
    index=False
)


# ============================================================
# 10. RAG CANDIDATES
# ============================================================

rag_cols = [
    "conversation_id",
    "candidate_intent",
    "total_turns",
    "customer_turns",
    "company_turns",
    "customer_text_for_intent",
    "company_reply_text",
    "difficulty",
]

df.loc[
    df["strong_rag_candidate"],
    rag_cols
].to_csv(
    OUT / "amazonhelp_rag_candidates_v2.csv",
    index=False
)


# ============================================================
# 11. GOLDEN REVIEW CANDIDATES
# ============================================================

gold_cols = [
    "conversation_id",
    "candidate_intent",
    "difficulty",
    "escalation_candidate",
    "total_turns",
    "customer_turns",
    "company_turns",
    "customer_text_for_intent",
    "company_reply_text",
]

df.loc[
    df["golden_review_candidate"],
    gold_cols
].to_csv(
    OUT / "amazonhelp_review_candidates_v2.csv",
    index=False
)


# ============================================================
# 12. SUMMARY
# ============================================================

summary = {
    "total_conversations": int(len(df)),
    "candidate_intent_distribution": dist.to_dict(orient="records"),
    "clear_intent_candidates": int(df["clear_intent_candidate"].sum()),
    "ambiguous_candidates": int(df["ambiguous_candidate"].sum()),
    "strong_rag_candidates": int(df["strong_rag_candidate"].sum()),
    "golden_review_candidates": int(df["golden_review_candidate"].sum()),
    "escalation_candidates": int(df["escalation_candidate"].sum()),
    "generic_reply_count": int(df["generic_reply_heuristic"].sum()),
    "actionable_reply_count": int(df["actionable_reply_heuristic"].sum()),
    "difficulty_distribution": df["difficulty"].value_counts().to_dict(),
    "warning": (
        "All intent labels and quality flags are heuristic candidate labels. "
        "Human validation is required before using them as evaluation ground truth."
    ),
}

with open(
    OUT / "amazonhelp_analysis_v2.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(summary, f, indent=2)

print("\n==========================================")
print("AmazonHelp V2 analysis complete")
print("==========================================")
print(json.dumps(summary, indent=2))
