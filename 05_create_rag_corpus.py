"""
Step 05 — Build a historical-resolution RAG corpus for AmazonHelp.

Inputs:
  data/processed/amazonhelp_conversations_clean.csv
  data/processed/amazonhelp_golden_final.csv

Outputs:
  data/processed/amazonhelp_rag_corpus.csv
  data/processed/amazonhelp_rag_corpus.json
  data/processed/amazonhelp_rag_excluded_golden_ids.txt

Important:
  - Golden-set conversations are EXCLUDED from the RAG corpus.
  - RAG documents are historical customer-problem -> company-response examples.
  - These are retrieval documents, NOT additional evaluation labels.
"""

from pathlib import Path
import json
import re
import hashlib
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
CONV = ROOT / "data" / "processed" / "amazonhelp_conversations_clean.csv"
GOLDEN = ROOT / "data" / "processed" / "amazonhelp_golden_final.csv"

OUT_CSV = ROOT / "data" / "processed" / "amazonhelp_rag_corpus.csv"
OUT_JSON = ROOT / "data" / "processed" / "amazonhelp_rag_corpus.json"
OUT_EXCLUDED = ROOT / "data" / "processed" / "amazonhelp_rag_excluded_golden_ids.txt"

TARGET = 5000
RANDOM_STATE = 42

def clean(x):
    x = "" if pd.isna(x) else str(x)
    x = re.sub(r"https?://\S+", " ", x)
    x = re.sub(r"@\w+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def norm(x):
    return clean(x).lower()

def generic_reply(x):
    x = norm(x)
    patterns = [
        r"^\s*(please )?(dm|direct message)\s*(me|us)?\s*[.!?]*$",
        r"^\s*send (?:us )?(?:a )?dm\s*[.!?]*$",
        r"^\s*we('?re| are) sorry\s*[.!?]*$",
        r"^\s*thanks for reaching out\s*[.!?]*$",
        r"^\s*please contact (?:us|support)\s*[.!?]*$",
    ]
    return any(re.search(p, x) for p in patterns)

def useful_response(x):
    x = clean(x)
    if len(x) < 25:
        return False
    if generic_reply(x):
        return False
    # Prefer responses containing an actual action/instruction.
    action_words = [
        "check", "follow", "visit", "select", "click", "return", "refund",
        "replace", "cancel", "track", "contact", "update", "verify",
        "reset", "change", "order", "delivery", "account", "payment",
        "send", "provide", "look", "try"
    ]
    low = x.lower()
    return any(w in low for w in action_words) or len(x) >= 80

def hash_text(x):
    return hashlib.md5(norm(x).encode("utf-8")).hexdigest()

def main():
    if not CONV.exists():
        raise FileNotFoundError(CONV)
    if not GOLDEN.exists():
        raise FileNotFoundError(GOLDEN)

    conv = pd.read_csv(CONV)
    golden = pd.read_csv(GOLDEN)

    # Exclude by conversation_id when available.
    golden_ids = set()
    if "conversation_id" in golden.columns:
        golden_ids = set(golden["conversation_id"].dropna().astype(str))

    OUT_EXCLUDED.write_text("\n".join(sorted(golden_ids)), encoding="utf-8")

    if "conversation_id" in conv.columns:
        conv["_conversation_id_str"] = conv["conversation_id"].astype(str)
        before = len(conv)
        conv = conv[~conv["_conversation_id_str"].isin(golden_ids)].copy()
        excluded_count = before - len(conv)
    else:
        excluded_count = 0

    # Expected source columns from the cleaned AmazonHelp conversation file.
    customer_col = next(
        (c for c in ["customer_messages_clean", "customer_messages",
                     "customer_text_for_intent", "first_customer_message"]
         if c in conv.columns), None
    )
    response_col = next(
        (c for c in ["company_responses_clean", "company_responses",
                     "company_response", "company_text"]
         if c in conv.columns), None
    )

    if customer_col is None or response_col is None:
        raise ValueError(
            f"Could not find customer/response columns. "
            f"Customer={customer_col}, Response={response_col}. "
            f"Available={list(conv.columns)}"
        )

    conv["customer_problem"] = conv[customer_col].map(clean)
    conv["historical_response"] = conv[response_col].map(clean)

    # Quality filters: historical resolution should contain both sides,
    # should not be an extremely large mixed thread, and response should be
    # meaningfully actionable.
    if "total_turns" in conv.columns:
        conv["_turns"] = pd.to_numeric(conv["total_turns"], errors="coerce").fillna(1)
    else:
        conv["_turns"] = 1

    conv["_customer_len"] = conv["customer_problem"].str.len()
    conv["_response_len"] = conv["historical_response"].str.len()
    conv["_useful_response"] = conv["historical_response"].map(useful_response)

    cand = conv[
        (conv["_customer_len"] >= 25)
        & (conv["_response_len"] >= 25)
        & (conv["_turns"] <= 12)
        & conv["_useful_response"]
    ].copy()

    # Remove exact duplicate problem-response pairs.
    cand["_pair_hash"] = (
        cand["customer_problem"].map(norm) + " || " +
        cand["historical_response"].map(norm)
    ).map(hash_text)
    cand = cand.drop_duplicates("_pair_hash").copy()

    # A simple quality score for selecting useful historical resolutions.
    cand["_quality"] = (
        np.minimum(cand["_customer_len"], 300) / 100
        + np.minimum(cand["_response_len"], 500) / 150
        + (cand["_turns"] <= 6).astype(int)
        + (cand["_turns"] >= 2).astype(int)
    )

    # Prefer diverse examples by problem text prefix.
    cand["_problem_prefix"] = cand["customer_problem"].map(norm).str[:120]
    cand = cand.sort_values("_quality", ascending=False)
    cand = cand.drop_duplicates("_problem_prefix", keep="first").copy()

    # Stratified sampling using the same candidate taxonomy as the golden set.
    intent_patterns = {
        "delivery_shipping": [r"delivery", r"delayed", r"late", r"tracking", r"package", r"shipped", r"arrived"],
        "payment": [r"payment", r"charged", r"charge", r"card", r"paid", r"declined"],
        "refund": [r"refund", r"money back", r"reimburse"],
        "return_replacement": [r"return", r"replacement", r"replace", r"exchange", r"pickup"],
        "product_issue": [r"damaged", r"broken", r"defective", r"wrong item", r"counterfeit"],
        "account_access": [r"login", r"log in", r"password", r"account", r"sign in", r"access"],
        "prime_membership": [r"prime", r"membership", r"subscription"],
        "technical_app": [r"app", r"website", r"error", r"not working", r"technical", r"checkout"],
        "order_cancellation": [r"cancel", r"cancelled", r"canceled", r"cancellation"],
        "security_fraud": [r"fraud", r"scam", r"hacked", r"phishing", r"unauthorized", r"stolen"],
        "general_support": [r"help", r"support", r"customer service"],
    }

    def infer(text):
        low = norm(text)
        scores = {
            intent: sum(bool(re.search(p, low)) for p in pats)
            for intent, pats in intent_patterns.items()
        }
        return max(scores, key=scores.get), max(scores.values())

    inferred = cand["customer_problem"].map(infer)
    cand["rag_intent_hint"] = inferred.map(lambda x: x[0])
    cand["_intent_score"] = inferred.map(lambda x: x[1])

    # Keep a broad corpus rather than forcing exact balance.
    rng = np.random.default_rng(RANDOM_STATE)

    selected = []
    per_intent = max(1, TARGET // len(intent_patterns))

    for intent in intent_patterns:
        pool = cand[cand["rag_intent_hint"] == intent].copy()
        if pool.empty:
            continue
        pool = pool.sort_values(
            ["_intent_score", "_quality"], ascending=[False, False]
        )
        n = min(per_intent, len(pool))
        top = pool.head(min(len(pool), n * 3))
        if len(top) > n:
            idx = rng.choice(len(top), n, replace=False)
            top = top.iloc[idx]
        selected.append(top)

    rag = pd.concat(selected, ignore_index=True) if selected else cand.head(TARGET)

    # Fill to target with strongest diverse examples.
    if len(rag) < TARGET:
        used = set(rag["_pair_hash"])
        extras = cand[~cand["_pair_hash"].isin(used)].sort_values(
            "_quality", ascending=False
        )
        rag = pd.concat([rag, extras.head(TARGET-len(rag))], ignore_index=True)

    rag = rag.head(TARGET).copy()

    # Build RAG-friendly records.
    records = []
    for i, row in rag.iterrows():
        cid = str(row["conversation_id"]) if "conversation_id" in row else str(i)
        customer = row["customer_problem"]
        response = row["historical_response"]
        doc = (
            f"Historical AmazonHelp support example.\n"
            f"Customer problem: {customer}\n"
            f"Historical support response: {response}"
        )
        records.append({
            "rag_id": f"RAG{i+1:05d}",
            "conversation_id": cid,
            "intent_hint": row["rag_intent_hint"],
            "customer_problem": customer,
            "historical_response": response,
            "document_text": doc,
            "source": "TWCS AmazonHelp historical conversation",
            "golden_overlap": False,
        })

    out = pd.DataFrame(records)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    summary = {
        "source_conversations": int(len(pd.read_csv(CONV))),
        "golden_conversations_excluded": int(excluded_count),
        "quality_candidates_before_sampling": int(len(cand)),
        "rag_documents": int(len(out)),
        "golden_set_size": int(len(golden)),
        "golden_overlap_in_output": int(out["golden_overlap"].sum()),
        "intent_distribution": out["intent_hint"].value_counts().to_dict(),
        "note": (
            "RAG corpus contains historical problem-response examples and "
            "explicitly excludes the golden evaluation conversations."
        ),
    }
    (OUT_CSV.parent / "amazonhelp_rag_corpus_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("=" * 70)
    print("RAG CORPUS CREATED")
    print("=" * 70)
    print(f"Source conversations        : {len(pd.read_csv(CONV)):,}")
    print(f"Golden conversations excluded: {excluded_count:,}")
    print(f"Quality RAG candidates       : {len(cand):,}")
    print(f"Final RAG documents          : {len(out):,}")
    print(f"Golden overlap               : {int(out['golden_overlap'].sum())}")
    print()
    print("Intent hints:")
    print(out["intent_hint"].value_counts().to_string())
    print()
    print(f"CSV: {OUT_CSV}")
    print(f"JSON: {OUT_JSON}")
    print("Summary: data/processed/amazonhelp_rag_corpus_summary.json")

if __name__ == "__main__":
    main()
