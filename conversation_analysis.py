import duckdb
import os

# ============================================================
# CONFIG
# ============================================================

CSV_PATH = r"D:\My Work\AI Support Agent\Dataset\twcs.csv"

OUTPUT_DIR = "conversation_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# TOP CANDIDATE BRANDS
# ============================================================

BRANDS = [
    "AmazonHelp",
    "AppleSupport",
    "Uber_Support",
    "SpotifyCares",
    "Delta",
    "Tesco",
    "AmericanAir",
    "TMobileHelp",
    "comcastcares",
    "British_Airways",
]

# ============================================================
# DUCKDB CONNECTION
# ============================================================

con = duckdb.connect()

print("\n" + "=" * 70)
print("HIVER CONVERSATION ANALYSIS")
print("=" * 70)

print("\nReading dataset using DuckDB...")
print("This may take a few minutes.\n")


# ============================================================
# CREATE VIEW OVER CSV
# ============================================================

con.execute(f"""
    CREATE VIEW tweets AS
    SELECT *
    FROM read_csv_auto(
        '{CSV_PATH}',
        header=true,
        ignore_errors=true
    )
""")


# ============================================================
# BASIC BRAND / CONVERSATION STATISTICS
# ============================================================

brand_list = ", ".join(
    f"'{brand}'" for brand in BRANDS
)

query = f"""
WITH company_tweets AS (

    SELECT
        tweet_id,
        author_id AS brand,
        in_response_to_tweet_id,
        response_tweet_id,
        text
    FROM tweets
    WHERE author_id IN ({brand_list})

),

parent_tweets AS (

    SELECT
        c.brand,
        c.tweet_id AS company_tweet_id,
        p.tweet_id AS customer_tweet_id,
        p.author_id AS customer_id,
        p.text AS customer_text,
        c.text AS company_text,
        c.in_response_to_tweet_id
    FROM company_tweets c
    JOIN tweets p
        ON CAST(c.in_response_to_tweet_id AS VARCHAR)
        = CAST(p.tweet_id AS VARCHAR)
    WHERE p.inbound = TRUE

)

SELECT
    brand,

    COUNT(*) AS customer_company_pairs,

    COUNT(DISTINCT customer_id)
        AS unique_customers,

    COUNT(DISTINCT customer_tweet_id)
        AS customer_messages,

    AVG(
        LENGTH(customer_text)
    ) AS avg_customer_message_length,

    AVG(
        LENGTH(company_text)
    ) AS avg_company_response_length

FROM parent_tweets

GROUP BY brand

ORDER BY customer_company_pairs DESC
"""

result = con.execute(query).fetchdf()

print("\n" + "=" * 70)
print("CUSTOMER → COMPANY INTERACTION ANALYSIS")
print("=" * 70)

print(
    result.to_string(index=False)
)


# ============================================================
# SAVE RESULTS
# ============================================================

result.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "conversation_brand_comparison.csv"
    ),
    index=False
)


# ============================================================
# MULTI-TURN ANALYSIS
# ============================================================

multi_turn_query = f"""
WITH company_replies AS (

    SELECT
        tweet_id,
        author_id AS brand,
        in_response_to_tweet_id
    FROM tweets
    WHERE author_id IN ({brand_list})

),

customer_messages AS (

    SELECT
        tweet_id,
        author_id,
        in_response_to_tweet_id
    FROM tweets
    WHERE inbound = TRUE

),

interactions AS (

    SELECT
        c.brand,
        c.tweet_id AS company_tweet_id,
        p.tweet_id AS customer_tweet_id,
        p.in_response_to_tweet_id AS customer_parent
    FROM company_replies c
    JOIN customer_messages p
        ON CAST(c.in_response_to_tweet_id AS VARCHAR)
        = CAST(p.tweet_id AS VARCHAR)

)

SELECT
    brand,
    COUNT(*) AS direct_customer_company_interactions,

    COUNT(
        DISTINCT customer_tweet_id
    ) AS distinct_customer_messages

FROM interactions

GROUP BY brand

ORDER BY direct_customer_company_interactions DESC
"""

multi_result = con.execute(
    multi_turn_query
).fetchdf()

multi_result.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "interaction_statistics.csv"
    ),
    index=False
)


# ============================================================
# SAMPLE REAL CONVERSATIONS
# ============================================================

print("\nGenerating conversation samples...")


for brand in BRANDS[:5]:

    print(f"Sampling {brand}...")

    sample_query = f"""
    SELECT
        c.tweet_id AS company_tweet_id,
        p.tweet_id AS customer_tweet_id,
        p.author_id AS customer_id,
        p.created_at AS customer_time,
        p.text AS customer_text,
        c.created_at AS company_time,
        c.text AS company_text
    FROM tweets c
    JOIN tweets p
        ON CAST(c.in_response_to_tweet_id AS VARCHAR)
        = CAST(p.tweet_id AS VARCHAR)
    WHERE c.author_id = '{brand}'
      AND p.inbound = TRUE
      AND p.text IS NOT NULL
      AND c.text IS NOT NULL

    ORDER BY RANDOM()

    LIMIT 100
    """

    sample = con.execute(
        sample_query
    ).fetchdf()

    safe_brand = brand.replace("/", "_")

    sample.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{safe_brand}_samples.csv"
        ),
        index=False
    )


# ============================================================
# CLOSE
# ============================================================

con.close()

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print("\nGenerated files:")

for filename in os.listdir(OUTPUT_DIR):
    print("  -", filename)

print("\nUpload the generated CSV files here.")