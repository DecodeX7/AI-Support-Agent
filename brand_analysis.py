import pandas as pd
import os
import json
from collections import defaultdict

# ============================================================
# CONFIG
# ============================================================

CSV_PATH = r"D:\My Work\AI Support Agent\Dataset\twcs.csv"

OUTPUT_DIR = "brand_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHUNK_SIZE = 100_000

# Top brands discovered from our first profiling pass
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
    "SouthwestAir",
    "VirginTrains",
    "Ask_Spectrum",
    "XboxSupport",
    "sprintcare",
    "hulu_support",
    "sainsburys",
    "GWRHelp",
    "AskPlayStation",
    "ChipotleTweets",
]

# ============================================================
# STAT CONTAINERS
# ============================================================

stats = {}

for brand in BRANDS:
    stats[brand] = {
        "company_tweets": 0,
        "company_replies": 0,
        "customer_tweets": 0,
        "customer_tweets_with_response": 0,
        "unique_customers": set(),
    }


# ============================================================
# PROCESS CSV
# ============================================================

print("\n" + "=" * 70)
print("HIVER BRAND ANALYSIS")
print("=" * 70)

print("\nReading dataset in chunks...\n")

for chunk_no, chunk in enumerate(
    pd.read_csv(
        CSV_PATH,
        chunksize=CHUNK_SIZE,
        low_memory=False,
        usecols=[
            "tweet_id",
            "author_id",
            "inbound",
            "text",
            "response_tweet_id",
            "in_response_to_tweet_id",
        ],
    ),
    start=1,
):

    print(f"Processing chunk {chunk_no}...")

    # --------------------------------------------------------
    # Company tweets
    # --------------------------------------------------------

    company_mask = chunk["author_id"].isin(BRANDS)

    company_rows = chunk[company_mask]

    for brand, group in company_rows.groupby("author_id"):

        stats[brand]["company_tweets"] += len(group)

        # Company tweets that are replies
        reply_mask = group["in_response_to_tweet_id"].notna()

        stats[brand]["company_replies"] += int(reply_mask.sum())

    # --------------------------------------------------------
    # Customer tweets
    #
    # A customer tweet is an inbound tweet.
    # We associate it with a brand if its response_tweet_id
    # contains one of the known company tweet IDs.
    # --------------------------------------------------------

    inbound = chunk[chunk["inbound"] == True].copy()

    if len(inbound) == 0:
        continue

    response_series = inbound["response_tweet_id"].fillna("").astype(str)

    for brand in BRANDS:

        # We cannot always determine the brand from the
        # response_tweet_id alone in this chunk, so this pass
        # primarily collects general inbound information.
        #
        # Brand-specific customer attribution will be done
        # in the next conversation reconstruction stage.

        pass


# ============================================================
# CONVERT SETS TO COUNTS
# ============================================================

for brand in BRANDS:
    stats[brand]["unique_customers"] = len(
        stats[brand]["unique_customers"]
    )


# ============================================================
# CREATE DATAFRAME
# ============================================================

rows = []

for brand, data in stats.items():

    company_tweets = data["company_tweets"]
    company_replies = data["company_replies"]

    reply_rate = (
        company_replies / company_tweets
        if company_tweets > 0
        else 0
    )

    rows.append({
        "brand": brand,
        "company_tweets": company_tweets,
        "company_replies": company_replies,
        "company_reply_rate": round(reply_rate, 4),
    })


result = pd.DataFrame(rows)

result = result.sort_values(
    "company_tweets",
    ascending=False
)


# ============================================================
# SAVE RESULTS
# ============================================================

result.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "brand_comparison.csv"
    ),
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("BRAND COMPARISON")
print("=" * 70)

print(
    result.to_string(
        index=False
    )
)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print("\nGenerated:")
print("  brand_analysis/brand_comparison.csv")