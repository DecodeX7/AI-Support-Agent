from pathlib import Path
import pandas as pd
import re, json

INPUT = Path("data/processed/amazonhelp_conversations_clean.csv")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

RULES = {
    "security_fraud": [r"\bfraud\b", r"\bscam\b", r"\bphishing\b", r"\bhacked\b", r"\bunauthori[sz]ed\b", r"\bsuspicious\b", r"\bstolen\b"],
    "refund_issue": [r"\brefund\b", r"\bmoney.*back\b"],
    "payment_issue": [r"\bpayment\b", r"\bcredit card\b", r"\bdebit card\b", r"\bcharged\b", r"\bbilling\b", r"\bdeclined\b"],
    "delivery_issue": [r"\bdelivery\b", r"\bdelivered\b", r"\blate\b", r"\bdelayed\b", r"\bwhere.*order\b", r"\bpackage\b", r"\bshipment\b", r"\btracking\b", r"\barriv"],
    "return_replacement": [r"\breturn\b", r"\breturned\b", r"\breplacement\b", r"\breplace\b", r"\bexchange\b"],
    "cancellation_issue": [r"\bcancel\b", r"\bcancelled\b", r"\bcanceled\b", r"\bcancellation\b"],
    "membership_issue": [r"\bprime\b", r"\bmembership\b", r"\bsubscription\b", r"\brenewal\b"],
    "account_issue": [r"\baccount\b", r"\blogin\b", r"\blog in\b", r"\bsign in\b", r"\bpassword\b", r"\blocked\b"],
    "technical_issue": [r"\bapp\b", r"\bwebsite\b", r"\bkindle\b", r"\balexa\b", r"\bcrash", r"\berror\b", r"\bnot working\b", r"\bsync"],
    "product_issue": [r"\bdamaged\b", r"\bdefective\b", r"\bbroken\b", r"\bwrong item\b", r"\bwrong product\b", r"\bcounterfeit\b", r"\bfake\b"],
}

def norm(x):
    x = "" if pd.isna(x) else str(x).lower()
    x = re.sub(r"https?://\S+", " ", x)
    x = re.sub(r"@\w+", " ", x)
    return re.sub(r"\s+", " ", x).strip()

def classify(text):
    scores = {}
    for intent, pats in RULES.items():
        score = sum(bool(re.search(p, text)) for p in pats)
        if score:
            scores[intent] = score
    if not scores:
        return "other_general", 0, ""
    best = "security_fraud" if "security_fraud" in scores else max(scores, key=scores.get)
    return best, scores[best], ";".join(f"{k}:{v}" for k,v in scores.items())

def actionable(text):
    return bool(re.search(r"\b(please|go to|click|select|check|contact|call|return|refund|replace|track|verify|reset|update|follow|try|steps?)\b", text))

def generic(text):
    return bool(re.search(r"\b(dm|direct message|send us a note|reach out|contact us|follow up|team will be in touch)\b", text))

if not INPUT.exists():
    raise FileNotFoundError(f"Missing {INPUT}. Run 01_extract_amazonhelp.py first.")

df = pd.read_csv(INPUT)
df["match_text"] = df["first_customer_message_clean"].fillna("").map(norm)

r = df["match_text"].map(classify)
df["candidate_intent"] = r.map(lambda x:x[0])
df["intent_rule_score"] = r.map(lambda x:x[1])
df["intent_matches"] = r.map(lambda x:x[2])

df["company_reply_text"] = df["company_responses_clean"].fillna("").map(norm)
df["actionable_reply_heuristic"] = df["company_reply_text"].map(actionable)
df["generic_reply_heuristic"] = df["company_reply_text"].map(generic)

df["strong_rag_candidate"] = (
    (df["company_turns"] > 0) &
    (df["customer_turns"] > 0) &
    (df["intent_rule_score"] > 0) &
    df["actionable_reply_heuristic"] &
    ~df["generic_reply_heuristic"] &
    (df["total_turns"] <= 12)
)

df["review_priority"] = "normal"
df.loc[(df["intent_rule_score"] == 0) | df["intent_matches"].str.contains(";", na=False), "review_priority"] = "high"
df.loc[df["candidate_intent"] == "security_fraud", "review_priority"] = "high"

cols = ["conversation_id","total_turns","customer_turns","company_turns",
        "first_customer_message_clean","company_responses_clean",
        "candidate_intent","intent_rule_score","intent_matches",
        "actionable_reply_heuristic","generic_reply_heuristic",
        "strong_rag_candidate","review_priority"]

df[cols].to_csv(OUT/"amazonhelp_intent_candidates.csv", index=False)

dist = df["candidate_intent"].value_counts().rename_axis("candidate_intent").reset_index(name="conversation_count")
dist["percentage"] = (dist["conversation_count"]/len(df)*100).round(2)
dist.to_csv(OUT/"amazonhelp_intent_distribution.csv", index=False)

df.loc[df["strong_rag_candidate"], cols[:8]].to_csv(OUT/"amazonhelp_rag_candidates.csv", index=False)

summary = {
    "total_conversations": len(df),
    "strong_rag_candidates": int(df["strong_rag_candidate"].sum()),
    "generic_replies": int(df["generic_reply_heuristic"].sum()),
    "actionable_replies": int(df["actionable_reply_heuristic"].sum()),
    "high_review_priority": int((df["review_priority"]=="high").sum()),
    "intent_distribution": dist.to_dict("records"),
    "warning": "Candidate labels are heuristic and must be human-validated."
}
(OUT/"amazonhelp_analysis.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

print(json.dumps(summary, indent=2))
