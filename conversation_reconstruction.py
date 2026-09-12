import duckdb
import pandas as pd
import os
import json
import re

# ============================================================
# CONFIGURATION
# ============================================================

# CHANGE THIS TO YOUR ACTUAL CSV PATH
CSV_PATH = r"D:\My Work\AI Support Agent\Dataset\twcs.csv"

OUTPUT_DIR = "conversation_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# We narrowed the candidate list after the first profiling pass.
BRANDS = [
    "AmazonHelp",
    "AppleSupport",
    "Uber_Support",
    "SpotifyCares",
    "Delta",
]

# Number of examples to extract for each brand
SAMPLE_CONVERSATIONS_PER_BRAND = 50


# ============================================================
# HELPERS
# ============================================================

def clean_id(value):
    """
    Convert IDs to clean strings.
    Handles NULL/NaN values.
    """
    if value is None:
        return None

    if pd.isna(value):
        return None

    value = str(value).strip()

    if value.lower() in {"", "nan", "none", "null"}:
        return None

    return value


def parse_id_list(value):
    """
    response_tweet_id can contain multiple IDs separated
    by commas.

    Example:
        '123,456,789'

    becomes:
        ['123', '456', '789']
    """

    value = clean_id(value)

    if value is None:
        return []

    # Dataset generally uses comma-separated IDs.
    parts = re.split(r"\s*,\s*", value)

    return [
        p.strip()
        for p in parts
        if p.strip()
    ]


# ============================================================
# CONNECT TO DUCKDB
# ============================================================

print("\n" + "=" * 75)
print("HIVER AI SUPPORT AGENT")
print("FULL CONVERSATION RECONSTRUCTION")
print("=" * 75)

print("\nConnecting to dataset using DuckDB...")

con = duckdb.connect()


# ============================================================
# CREATE VIEW
# ============================================================

print("Creating CSV view...")

con.execute(f"""
    CREATE OR REPLACE VIEW tweets AS
    SELECT
        tweet_id::VARCHAR AS tweet_id,
        author_id::VARCHAR AS author_id,
        inbound::BOOLEAN AS inbound,
        created_at::VARCHAR AS created_at,
        text::VARCHAR AS text,
        response_tweet_id::VARCHAR AS response_tweet_id,
        in_response_to_tweet_id::VARCHAR AS in_response_to_tweet_id
    FROM read_csv_auto(
        '{CSV_PATH}',
        header=true,
        ignore_errors=true
    )
""")


# ============================================================
# BASIC CHECK
# ============================================================

print("\nChecking dataset...")

total_rows = con.execute("""
    SELECT COUNT(*)
    FROM tweets
""").fetchone()[0]

print(f"Total rows detected: {total_rows:,}")


# ============================================================
# GET BRAND STATISTICS
# ============================================================

brand_sql = ", ".join(
    f"'{brand}'"
    for brand in BRANDS
)

print("\nCalculating brand-level conversation statistics...")


brand_stats_query = f"""
WITH brand_tweets AS (

    SELECT
        tweet_id,
        author_id AS brand,
        inbound,
        created_at,
        text,
        in_response_to_tweet_id
    FROM tweets
    WHERE author_id IN ({brand_sql})

),

direct_interactions AS (

    SELECT
        b.brand,
        b.tweet_id AS company_tweet_id,
        p.tweet_id AS customer_tweet_id,
        p.author_id AS customer_id,
        p.text AS customer_text,
        b.text AS company_text,
        b.in_response_to_tweet_id
    FROM brand_tweets b
    INNER JOIN tweets p
        ON b.in_response_to_tweet_id = p.tweet_id
    WHERE p.inbound = TRUE

)

SELECT

    brand,

    COUNT(*) AS customer_company_interactions,

    COUNT(DISTINCT customer_id)
        AS unique_customers,

    COUNT(DISTINCT customer_tweet_id)
        AS customer_messages,

    COUNT(
        DISTINCT company_tweet_id
    ) AS company_responses

FROM direct_interactions

GROUP BY brand

ORDER BY customer_company_interactions DESC
"""

brand_stats = con.execute(
    brand_stats_query
).fetchdf()

print("\n")
print(brand_stats.to_string(index=False))

brand_stats.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_brand_statistics.csv"
    ),
    index=False
)


# ============================================================
# CONVERSATION DEPTH
# ============================================================

print("\n" + "=" * 75)
print("CALCULATING CONVERSATION DEPTH")
print("=" * 75)

print("""
A conversation is approximated by following the
in_response_to_tweet_id relationship.

We focus on threads containing the selected brand.
""")


# ------------------------------------------------------------
# Create table containing only candidate-brand tweets
# ------------------------------------------------------------

candidate_tweets = con.execute(f"""
    SELECT
        tweet_id,
        author_id,
        inbound,
        created_at,
        text,
        in_response_to_tweet_id
    FROM tweets
    WHERE author_id IN ({brand_sql})
       OR tweet_id IN (

            SELECT DISTINCT in_response_to_tweet_id
            FROM tweets
            WHERE author_id IN ({brand_sql})
              AND in_response_to_tweet_id IS NOT NULL
       )
""").fetchdf()

print(
    f"\nCandidate-related tweets loaded: "
    f"{len(candidate_tweets):,}"
)


# ============================================================
# CREATE LOOKUP DICTIONARY
# ============================================================

tweet_lookup = {}

for _, row in candidate_tweets.iterrows():

    tweet_id = clean_id(row["tweet_id"])

    if tweet_id is None:
        continue

    tweet_lookup[tweet_id] = {
        "tweet_id": tweet_id,
        "author_id": clean_id(row["author_id"]),
        "inbound": bool(row["inbound"])
        if not pd.isna(row["inbound"])
        else None,
        "created_at": clean_id(row["created_at"]),
        "text": clean_id(row["text"]),
        "parent_id": clean_id(
            row["in_response_to_tweet_id"]
        ),
    }


# ============================================================
# BUILD CHILD RELATIONSHIPS
# ============================================================

children = {}

for tweet_id, tweet in tweet_lookup.items():

    parent_id = tweet["parent_id"]

    if parent_id is None:
        continue

    if parent_id not in children:
        children[parent_id] = []

    children[parent_id].append(tweet_id)


# ============================================================
# FIND ROOT OF EACH TWEET
# ============================================================

def find_root(tweet_id):

    visited = set()

    current = tweet_id

    while current in tweet_lookup:

        if current in visited:
            break

        visited.add(current)

        parent = tweet_lookup[current]["parent_id"]

        if parent is None:
            break

        if parent not in tweet_lookup:
            break

        current = parent

    return current


# ============================================================
# GROUP TWEETS INTO THREADS
# ============================================================

print("\nGrouping tweets into conversation threads...")

threads = {}

for tweet_id in tweet_lookup:

    root = find_root(tweet_id)

    if root not in threads:
        threads[root] = []

    threads[root].append(tweet_id)


# ============================================================
# ASSIGN BRANDS TO THREADS
# ============================================================

conversation_records = []

for root_id, tweet_ids in threads.items():

    thread_messages = []

    brands_in_thread = set()

    for tweet_id in tweet_ids:

        tweet = tweet_lookup[tweet_id]

        if tweet["author_id"] in BRANDS:
            brands_in_thread.add(
                tweet["author_id"]
            )

        thread_messages.append(tweet)

    # Ignore threads without our candidate brands
    if not brands_in_thread:
        continue

    # Sort chronologically when possible.
    thread_messages = sorted(
        thread_messages,
        key=lambda x: x["created_at"] or ""
    )

    for brand in brands_in_thread:

        brand_messages = [
            msg
            for msg in thread_messages
            if msg["author_id"] == brand
            or msg["inbound"] is True
        ]

        if not brand_messages:
            continue

        customer_count = sum(
            1
            for msg in brand_messages
            if msg["inbound"] is True
        )

        company_count = sum(
            1
            for msg in brand_messages
            if msg["author_id"] == brand
        )

        conversation_records.append({
            "conversation_id": root_id,
            "brand": brand,
            "total_turns": len(brand_messages),
            "customer_turns": customer_count,
            "company_turns": company_count,
            "has_multiple_turns": len(brand_messages) >= 3,
            "has_5_plus_turns": len(brand_messages) >= 5,
        })


# ============================================================
# CONVERSATION DATAFRAME
# ============================================================

conversation_df = pd.DataFrame(
    conversation_records
)

if len(conversation_df) == 0:

    print("\nWARNING:")
    print("No conversations were reconstructed.")
    print("Please check the CSV path and schema.")

else:

    conversation_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "conversation_records.csv"
        ),
        index=False
    )

    print(
        f"\nConversation records created: "
        f"{len(conversation_df):,}"
    )


# ============================================================
# DEPTH STATISTICS BY BRAND
# ============================================================

print("\n" + "=" * 75)
print("CONVERSATION DEPTH BY BRAND")
print("=" * 75)

if len(conversation_df) > 0:

    depth_stats = []

    for brand in BRANDS:

        subset = conversation_df[
            conversation_df["brand"] == brand
        ]

        if len(subset) == 0:
            continue

        total = len(subset)

        multi_turn = int(
            subset["has_multiple_turns"].sum()
        )

        five_plus = int(
            subset["has_5_plus_turns"].sum()
        )

        depth_stats.append({

            "brand": brand,

            "conversations": total,

            "avg_turns": round(
                subset["total_turns"].mean(),
                2
            ),

            "median_turns": float(
                subset["total_turns"].median()
            ),

            "2_plus_turn_conversations": multi_turn,

            "2_plus_percentage": round(
                multi_turn / total * 100,
                2
            ),

            "5_plus_turn_conversations": five_plus,

            "5_plus_percentage": round(
                five_plus / total * 100,
                2
            ),

            "max_turns": int(
                subset["total_turns"].max()
            ),
        })

    depth_df = pd.DataFrame(
        depth_stats
    )

    depth_df = depth_df.sort_values(
        "conversations",
        ascending=False
    )

    print(
        depth_df.to_string(index=False)
    )

    depth_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "conversation_depth_comparison.csv"
        ),
        index=False
    )


# ============================================================
# EXTRACT REAL FULL CONVERSATIONS
# ============================================================

print("\n" + "=" * 75)
print("EXTRACTING REPRESENTATIVE FULL CONVERSATIONS")
print("=" * 75)


def build_conversation(root_id):

    if root_id not in threads:
        return []

    messages = []

    for tweet_id in threads[root_id]:

        if tweet_id not in tweet_lookup:
            continue

        msg = tweet_lookup[tweet_id]

        messages.append(msg)

    messages = sorted(
        messages,
        key=lambda x: x["created_at"] or ""
    )

    return messages


# ------------------------------------------------------------
# Choose conversations with multiple turns
# ------------------------------------------------------------

for brand in BRANDS:

    if len(conversation_df) == 0:
        break

    subset = conversation_df[
        (conversation_df["brand"] == brand)
        &
        (conversation_df["total_turns"] >= 3)
    ].copy()

    if len(subset) == 0:

        # Fall back to any conversation
        subset = conversation_df[
            conversation_df["brand"] == brand
        ].copy()

    subset = subset.sort_values(
        "total_turns",
        ascending=False
    )

    subset = subset.head(
        SAMPLE_CONVERSATIONS_PER_BRAND
    )

    output = []

    for _, record in subset.iterrows():

        conversation_id = record[
            "conversation_id"
        ]

        messages = build_conversation(
            conversation_id
        )

        output.append({

            "conversation_id":
                conversation_id,

            "brand":
                brand,

            "total_turns":
                record["total_turns"],

            "customer_turns":
                record["customer_turns"],

            "company_turns":
                record["company_turns"],

            "messages": messages,
        })

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{brand}_full_conversations.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"{brand}: "
        f"{len(output)} conversations saved"
    )


# ============================================================
# TURN DISTRIBUTION
# ============================================================

print("\n" + "=" * 75)
print("TURN DISTRIBUTION")
print("=" * 75)

if len(conversation_df) > 0:

    turn_distribution = (
        conversation_df
        .groupby(
            ["brand", "total_turns"]
        )
        .size()
        .reset_index(
            name="conversation_count"
        )
    )

    turn_distribution.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "turn_distribution.csv"
        ),
        index=False
    )

    print(
        turn_distribution
        .head(100)
        .to_string(index=False)
    )


# ============================================================
# FINAL SUMMARY JSON
# ============================================================

summary = {

    "dataset_rows":
        int(total_rows),

    "candidate_brands":
        BRANDS,

    "conversation_records":
        int(len(conversation_df)),

    "output_directory":
        OUTPUT_DIR,
}


with open(
    os.path.join(
        OUTPUT_DIR,
        "reconstruction_summary.json"
    ),
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# FINISHED
# ============================================================

con.close()

print("\n" + "=" * 75)
print("RECONSTRUCTION COMPLETE")
print("=" * 75)

print("\nGenerated files:")

for filename in sorted(
    os.listdir(OUTPUT_DIR)
):

    print(
        f"  - {filename}"
    )

print("\nNext step:")
print(
    "Upload conversation_depth_comparison.csv, "
    "turn_distribution.csv, and the five *_full_conversations.json files."
)