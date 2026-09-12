import duckdb
import pandas as pd
import json
import re
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path("data/raw/twcs.csv")

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BRAND = "AmazonHelp"


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if text is None:
        return ""

    text = str(text)

    # Remove URLs
    text = re.sub(r"https?://\S+", " ", text)

    # Replace mentions
    text = re.sub(r"@\w+", " USER ", text)

    # Remove repeated whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# CONNECT DUCKDB
# ============================================================

con = duckdb.connect()


# ============================================================
# STEP 1
# FIND AMAZONHELP TWEETS
# ============================================================

print("Reading original dataset...")

amazon_query = f"""
SELECT
    tweet_id,
    author_id,
    inbound,
    created_at,
    text,
    response_tweet_id,
    in_response_to_tweet_id

FROM read_csv_auto(
    '{INPUT_FILE}',
    header=True,
    ignore_errors=True
)

WHERE
    author_id = '{BRAND}'
    OR text ILIKE '%@{BRAND}%'
"""

amazon_df = con.execute(amazon_query).df()

print(f"AmazonHelp-related rows: {len(amazon_df):,}")


amazon_df.to_csv(
    OUTPUT_DIR / "amazonhelp_tweets.csv",
    index=False
)


# ============================================================
# STEP 2
# BUILD TWEET LOOKUP
# ============================================================

tweet_query = f"""
SELECT
    tweet_id,
    author_id,
    inbound,
    created_at,
    text,
    response_tweet_id,
    in_response_to_tweet_id

FROM read_csv_auto(
    '{INPUT_FILE}',
    header=True,
    ignore_errors=True
)

WHERE
    tweet_id IN (
        SELECT tweet_id
        FROM read_csv_auto(
            '{INPUT_FILE}',
            header=True,
            ignore_errors=True
        )
        WHERE author_id = '{BRAND}'
    )

OR
    in_response_to_tweet_id IN (
        SELECT tweet_id
        FROM read_csv_auto(
            '{INPUT_FILE}',
            header=True,
            ignore_errors=True
        )
        WHERE author_id = '{BRAND}'
    )
"""

tweets = con.execute(tweet_query).df()

print(f"Tweets used for reconstruction: {len(tweets):,}")


# ============================================================
# STEP 3
# CREATE LOOKUP DICTIONARY
# ============================================================

tweet_map = {}

for _, row in tweets.iterrows():

    tweet_id = str(row["tweet_id"])

    tweet_map[tweet_id] = {
        "tweet_id": tweet_id,
        "author_id": str(row["author_id"]),
        "inbound": bool(row["inbound"]),
        "created_at": row["created_at"],
        "text": row["text"],
        "parent_id": (
            str(row["in_response_to_tweet_id"])
            if pd.notna(row["in_response_to_tweet_id"])
            else None
        )
    }


# ============================================================
# STEP 4
# RECONSTRUCT AMAZONHELP CONVERSATIONS
# ============================================================

print("Reconstructing conversations...")

conversations = {}

for tweet_id, tweet in tweet_map.items():

    # We are mainly interested in customer tweets
    # that interact with AmazonHelp.
    if tweet["inbound"] is not True:
        continue

    chain = []

    current = tweet

    visited = set()

    while current:

        current_id = current["tweet_id"]

        if current_id in visited:
            break

        visited.add(current_id)

        chain.append(current)

        parent_id = current["parent_id"]

        if not parent_id:
            break

        current = tweet_map.get(parent_id)

    # Reverse so conversation is chronological
    chain.reverse()

    # Only keep chains containing AmazonHelp
    has_amazon = any(
        x["author_id"] == BRAND
        for x in chain
    )

    if not has_amazon:
        continue

    # Create conversation ID
    conversation_id = tweet_id

    conversations[conversation_id] = chain


# ============================================================
# STEP 5
# CONVERT TO FLAT DATA
# ============================================================

rows = []

for conversation_id, messages in conversations.items():

    customer_turns = [
        m for m in messages
        if m["inbound"] is True
    ]

    company_turns = [
        m for m in messages
        if m["author_id"] == BRAND
    ]

    rows.append({

        "conversation_id": conversation_id,

        "total_turns": len(messages),

        "customer_turns": len(customer_turns),

        "company_turns": len(company_turns),

        "first_customer_message": (
            customer_turns[0]["text"]
            if customer_turns
            else ""
        ),

        "company_responses": " || ".join(
            str(x["text"])
            for x in company_turns
        ),

        "customer_messages": " || ".join(
            str(x["text"])
            for x in customer_turns
        )

    })


df = pd.DataFrame(rows)


# ============================================================
# STEP 6
# CLEAN TEXT
# ============================================================

df["first_customer_message_clean"] = (
    df["first_customer_message"]
    .apply(clean_text)
)

df["company_responses_clean"] = (
    df["company_responses"]
    .apply(clean_text)
)

df["customer_messages_clean"] = (
    df["customer_messages"]
    .apply(clean_text)
)


# ============================================================
# STEP 7
# QUALITY FILTERS
# ============================================================

df["has_customer"] = df["customer_turns"] > 0

df["has_company"] = df["company_turns"] > 0

df["is_multi_turn"] = df["total_turns"] >= 4

df["customer_message_length"] = (
    df["first_customer_message_clean"]
    .str.len()
)

df["is_usable"] = (
    df["has_customer"]
    & df["has_company"]
    & (df["customer_message_length"] >= 10)
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_DIR / "amazonhelp_conversations_clean.csv",
    index=False
)


# ============================================================
# STEP 8
# DATA PROFILE
# ============================================================

profile = {

    "brand": BRAND,

    "total_amazonhelp_conversations":
        int(len(df)),

    "usable_conversations":
        int(df["is_usable"].sum()),

    "multi_turn_conversations":
        int(df["is_multi_turn"].sum()),

    "median_turns":
        float(df["total_turns"].median()),

    "mean_turns":
        float(df["total_turns"].mean()),

    "max_turns":
        int(df["total_turns"].max())

}


with open(
    OUTPUT_DIR / "amazonhelp_data_profile.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        profile,
        f,
        indent=2
    )


print("\n==============================")
print("AmazonHelp extraction complete")
print("==============================")

print(
    json.dumps(
        profile,
        indent=2
    )
)