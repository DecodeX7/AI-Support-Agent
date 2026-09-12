"""
Step 04 — Create a manually-reviewable golden-set candidate pool for AmazonHelp.

Input:
    data/processed/amazonhelp_conversations_clean.csv

Output:
    data/processed/amazonhelp_golden_candidates.csv
    data/processed/amazonhelp_golden_candidates_summary.json

Goal:
    Select ~300 diverse candidate conversations from the ~100k AmazonHelp
    conversations. These are CANDIDATES, not final labels.

The final golden set should be manually verified and reduced to ~200 examples.
Do not use this candidate file as ground truth until it has been reviewed.
"""

from pathlib import Path
import json
import re
import hashlib

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data" / "processed" / "amazonhelp_conversations_clean.csv"
OUT = ROOT / "data" / "processed" / "amazonhelp_golden_candidates.csv"
SUMMARY = ROOT / "data" / "processed" / "amazonhelp_golden_candidates_summary.json"

RANDOM_STATE = 42
TARGET_PER_INTENT = 27
MAX_TOTAL = 300


# Final candidate taxonomy. These are selection hints, NOT final labels.
INTENT_PATTERNS = {
    "delivery_issue": [
        r"\blate\b", r"\bdelayed\b", r"\bdelay\b", r"\bmissing package\b",
        r"\bpackage (?:is|was) missing\b", r"\bwhere is my (?:order|package)\b",
        r"\bnot arrived\b", r"\bhasn't arrived\b", r"\bhaven't received\b",
        r"\bnot received\b", r"\btracking\b", r"\bdelivery\b",
        r"\bdelivered\b", r"\bcarrier\b", r"\bshipment\b", r"\bshipping\b",
        r"\barrive\b", r"\bdelivery date\b", r"\bdelivery time\b",
    ],
    "payment_issue": [
        r"\bcharged\b", r"\bcharge\b", r"\bpayment\b", r"\bpaid\b",
        r"\bcredit card\b", r"\bdebit card\b", r"\bcard\b", r"\bpaypal\b",
        r"\bpayment method\b", r"\bpayment failed\b", r"\bdeclined\b",
        r"\bdouble charge\b", r"\bextra charge\b", r"\bunauthori[sz]ed charge\b",
    ],
    "refund_issue": [
        r"\brefund\b", r"\bmoney back\b", r"\breimburse\b", r"\breimbursed\b",
        r"\brefund pending\b", r"\brefund hasn't\b", r"\brefund not\b",
        r"\bwhere is my refund\b", r"\brefund status\b",
    ],
    "return_replacement": [
        r"\breturn\b", r"\breturning\b", r"\breturned\b", r"\breplacement\b",
        r"\breplace\b", r"\bexchange\b", r"\breturn label\b",
        r"\breturn pickup\b", r"\bpickup\b", r"\breturn window\b",
    ],
    "product_issue": [
        r"\bdamaged\b", r"\bdefective\b", r"\bbroken\b", r"\bwrong item\b",
        r"\bwrong product\b", r"\bfake\b", r"\bcounterfeit\b", r"\bmissing item\b",
        r"\bproduct issue\b", r"\bitem is\b", r"\bitem was\b",
    ],
    "account_issue": [
        r"\bcan't log in\b", r"\bcannot log in\b", r"\blogin\b", r"\blog in\b",
        r"\bsign in\b", r"\bpassword\b", r"\baccount locked\b", r"\bblocked account\b",
        r"\baccount access\b", r"\baccess my account\b", r"\bverification\b",
        r"\bverify my account\b",
    ],
    "membership_issue": [
        r"\bprime\b", r"\bmembership\b", r"\bprime membership\b",
        r"\bprime video\b", r"\bmember\b", r"\bsubscription\b",
        r"\bcancel prime\b", r"\bprime charge\b",
    ],
    "technical_issue": [
        r"\bapp\b", r"\bwebsite\b", r"\bsite\b", r"\berror\b", r"\bbug\b",
        r"\bcrash(?:es|ed)?\b", r"\bnot working\b", r"\bdoesn't work\b",
        r"\bcannot\b", r"\bcan't\b", r"\bunable\b", r"\btechnical\b",
        r"\bglitch\b", r"\bloading\b", r"\bcheckout\b",
    ],
    "cancellation_issue": [
        r"\bcancel\b", r"\bcancelled\b", r"\bcanceled\b", r"\bcancellation\b",
        r"\border was cancelled\b", r"\border got cancelled\b",
        r"\bcancel my order\b",
    ],
    "security_fraud": [
        r"\bhack(?:ed)?\b", r"\bhacked\b", r"\bfraud\b", r"\bscam\b",
        r"\bphishing\b", r"\bsecurity\b", r"\bunauthori[sz]ed\b",
        r"\bsuspicious\b", r"\bstolen\b", r"\bidentity\b", r"\bcompromised\b",
    ],
    "general_support": [
        r"\bhelp\b", r"\bsupport\b", r"\bcustomer service\b",
        r"\bcustomer care\b", r"\bcontact\b", r"\bissue\b", r"\bproblem\b",
    ],
}

# Terms that often indicate a conversational follow-up rather than a clean
# problem statement. We do not discard them completely because hard examples
# can be valuable, but they receive a penalty.
FOLLOWUP_PATTERNS = [
    r"^\s*(thanks|thank you|thx|ok|okay|yes|yeah|yep|great)\s*[.!?]*\s*$",
    r"\bthanks for (?:the|your) (?:help|response|reply)\b",
    r"\bappreciate (?:it|your help)\b",
    r"\bquick response\b",
]

GENERIC_PATTERNS = [
    r"^\s*(dm|direct message)\s*(me|us)?\s*$",
    r"^\s*please dm\s*$",
    r"^\s*send (?:us )?(?:a )?dm\s*$",
]


def norm_text(text):
    text = "" if pd.isna(text) else str(text)
    text = text.lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def score_intents(text):
    scores = {}
    for intent, patterns in INTENT_PATTERNS.items():
        score = 0
        for pattern in patterns:
            if re.search(pattern, text):
                score += 1
        scores[intent] = score
    return scores


def is_followup(text):
    return any(re.search(p, text) for p in FOLLOWUP_PATTERNS)


def is_generic(text):
    return any(re.search(p, text) for p in GENERIC_PATTERNS)


def difficulty(row):
    score = row["intent_score"]
    turns = row["total_turns"]
    if score >= 3 and turns <= 4:
        return "easy"
    if score >= 2 or turns <= 6:
        return "medium"
    return "hard"


def make_hash(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Input not found: {INPUT}")

    df = pd.read_csv(INPUT)

    # Support the known conversation-clean schema while being tolerant of
    # minor column naming differences.
    possible_customer_cols = [
        "customer_messages_clean",
        "customer_messages",
        "customer_text_for_intent",
        "first_customer_message",
    ]
    customer_col = next((c for c in possible_customer_cols if c in df.columns), None)
    if customer_col is None:
        raise ValueError(
            "Could not find a customer-message column. Expected one of: "
            + ", ".join(possible_customer_cols)
        )

    # Use the complete customer side when available. This is much better for
    # intent discovery than relying only on the first tweet.
    df["customer_text"] = df[customer_col].fillna("").astype(str).map(norm_text)

    # If customer_messages_clean is too sparse, fall back to first message.
    if "first_customer_message" in df.columns:
        fallback = df["first_customer_message"].fillna("").astype(str).map(norm_text)
        df.loc[df["customer_text"].str.len() < 8, "customer_text"] = fallback

    if "total_turns" not in df.columns:
        df["total_turns"] = 1

    df["total_turns"] = pd.to_numeric(df["total_turns"], errors="coerce").fillna(1)
    df["text_len"] = df["customer_text"].str.len()

    # Remove unusably short rows and very large threads. Large public threads
    # are more likely to contain mixed context.
    work = df[
        (df["text_len"] >= 12)
        & (df["total_turns"] <= 12)
    ].copy()

    # Deduplicate based on normalized customer-side text.
    work["text_hash"] = work["customer_text"].map(make_hash)
    work = work.drop_duplicates("text_hash").copy()

    work["followup_only"] = work["customer_text"].map(is_followup)
    work["generic_customer"] = work["customer_text"].map(is_generic)

    scores = work["customer_text"].map(score_intents)
    for intent in INTENT_PATTERNS:
        work[f"score__{intent}"] = scores.map(lambda d, k=intent: d[k])

    work["best_intent"] = scores.map(lambda d: max(d, key=d.get))
    work["intent_score"] = scores.map(lambda d: max(d.values()))

    # Penalize weak/general matches and conversational-only examples.
    work["selection_score"] = (
        work["intent_score"].astype(float) * 3
        + np.minimum(work["text_len"], 280) / 100
        + (work["total_turns"] <= 6).astype(int)
        - work["followup_only"].astype(int) * 4
        - work["generic_customer"].astype(int) * 3
    )

    work["difficulty"] = work.apply(difficulty, axis=1)

    # Candidate pool: only reasonably clear intent hints. General support is
    # allowed at a lower threshold because it is intentionally broad.
    work = work[
        (
            (work["best_intent"] != "general_support")
            & (work["intent_score"] >= 1)
        )
        | (
            (work["best_intent"] == "general_support")
            & (work["intent_score"] >= 1)
        )
    ].copy()

    rng = np.random.default_rng(RANDOM_STATE)
    selected_parts = []

    # Select approximately balanced candidates, with a mixture of difficulty.
    difficulty_order = ["easy", "medium", "hard"]

    for intent in INTENT_PATTERNS:
        pool = work[work["best_intent"] == intent].copy()

        if pool.empty:
            continue

        # Prefer clear examples, then add diversity through randomized
        # selection within difficulty buckets.
        per_diff = max(1, TARGET_PER_INTENT // 3)
        intent_selected = []

        for diff in difficulty_order:
            sub = pool[pool["difficulty"] == diff].copy()
            if sub.empty:
                continue

            n = min(per_diff, len(sub))
            # Top candidates first, then randomized tie-breaking.
            sub = sub.sort_values(
                ["selection_score", "text_len"],
                ascending=[False, False]
            )
            top_n = min(len(sub), max(n * 3, n))
            sub = sub.head(top_n)

            if len(sub) > n:
                idx = rng.choice(len(sub), size=n, replace=False)
                sub = sub.iloc[idx]

            intent_selected.append(sub)

        if intent_selected:
            picked = pd.concat(intent_selected, ignore_index=False)
            # Fill remaining slots if a difficulty bucket had too few examples.
            remaining = TARGET_PER_INTENT - len(picked)
            if remaining > 0:
                rest = pool.drop(picked.index, errors="ignore")
                rest = rest.sort_values(
                    ["selection_score", "text_len"],
                    ascending=[False, False]
                ).head(remaining)
                picked = pd.concat([picked, rest])

            selected_parts.append(picked.head(TARGET_PER_INTENT))

    if not selected_parts:
        raise RuntimeError("No candidates were selected.")

    selected = pd.concat(selected_parts, ignore_index=False)

    # Enforce the overall target.
    if len(selected) > MAX_TOTAL:
        selected = (
            selected.sort_values(
                ["best_intent", "selection_score"],
                ascending=[True, False]
            )
            .groupby("best_intent", group_keys=False)
            .head(TARGET_PER_INTENT)
        )

    selected = selected.reset_index(drop=True)

    # Escalation hints are deliberately heuristic. Final escalation labels
    # must be manually verified.
    escalation_terms = [
        r"\bhacked\b", r"\bfraud\b", r"\bscam\b", r"\bphishing\b",
        r"\bunauthori[sz]ed\b", r"\bcharge\b", r"\bstolen\b",
        r"\bmultiple times\b", r"\bcalled\b", r"\bcontacted\b",
        r"\bno response\b", r"\bstill waiting\b", r"\blegal\b",
        r"\bpolice\b", r"\bunsafe\b",
    ]
    selected["escalation_hint"] = selected["customer_text"].map(
        lambda x: any(re.search(p, x) for p in escalation_terms)
    )

    # Prepare a clean manual-review sheet.
    keep_cols = []
    for c in [
        "conversation_id",
        "total_turns",
        "customer_turns",
        "company_turns",
        "first_customer_message",
        "customer_messages",
        "company_responses",
    ]:
        if c in selected.columns:
            keep_cols.append(c)

    output = selected[keep_cols].copy()
    output.insert(0, "candidate_id", [f"G{i:04d}" for i in range(1, len(output) + 1)])
    output["candidate_intent"] = selected["best_intent"].values
    output["difficulty_hint"] = selected["difficulty"].values
    output["intent_score_hint"] = selected["intent_score"].values
    output["escalation_hint"] = selected["escalation_hint"].values
    output["selection_reason"] = (
        "Heuristic candidate for manual golden-set review; "
        "verify intent, escalation, action and resolution criteria."
    )

    # Blank columns are intentionally left for manual labeling.
    output["golden_intent"] = ""
    output["escalation_required"] = ""
    output["escalation_reason"] = ""
    output["expected_action"] = ""
    output["resolution_criteria"] = ""
    output["review_notes"] = ""

    # Put the strongest review fields first.
    front = [
        "candidate_id",
        "candidate_intent",
        "difficulty_hint",
        "escalation_hint",
        "golden_intent",
        "escalation_required",
        "escalation_reason",
        "expected_action",
        "resolution_criteria",
        "review_notes",
    ]
    rest = [c for c in output.columns if c not in front]
    output = output[front + rest]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUT, index=False, encoding="utf-8-sig")

    summary = {
        "input_file": str(INPUT),
        "output_file": str(OUT),
        "candidate_count": int(len(output)),
        "target_final_golden_set": 200,
        "candidate_intents": output["candidate_intent"].value_counts().to_dict(),
        "difficulty_hints": output["difficulty_hint"].value_counts().to_dict(),
        "escalation_hints": output["escalation_hint"].value_counts().to_dict(),
        "note": (
            "These are heuristic candidates only. Manually verify the final "
            "golden labels before using them for evaluation."
        ),
    }

    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 70)
    print("Golden candidate selection complete")
    print("=" * 70)
    print(f"Input conversations : {len(df):,}")
    print(f"Usable candidates   : {len(work):,}")
    print(f"Selected candidates : {len(output):,}")
    print()
    print("Candidate intent distribution:")
    print(output["candidate_intent"].value_counts().to_string())
    print()
    print("Difficulty hints:")
    print(output["difficulty_hint"].value_counts().to_string())
    print()
    print(f"CSV     : {OUT}")
    print(f"Summary : {SUMMARY}")
    print()
    print("IMPORTANT: candidate_intent is NOT ground truth.")
    print("Manually verify labels before using this set for evaluation.")


if __name__ == "__main__":
    main()
