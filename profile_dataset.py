import pandas as pd
import json
import os
from collections import Counter

# ============================================================
# CONFIG
# ============================================================

CSV_PATH = r"D:\My Work\AI Support Agent\Dataset\twcs.csv"

OUTPUT_DIR = "dataset_profile"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. READ BASIC INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("HIVER DATASET PROFILING")
print("=" * 70)

print("\nReading CSV in chunks...")

# We deliberately use chunks so the 500MB file
# does not need to be loaded entirely into RAM.

chunksize = 100_000

total_rows = 0
columns = None

author_counts = Counter()
inbound_counts = Counter()

min_date = None
max_date = None

sample_rows = []


# ============================================================
# 2. PROCESS DATA CHUNK BY CHUNK
# ============================================================

for chunk_number, chunk in enumerate(
    pd.read_csv(
        CSV_PATH,
        chunksize=chunksize,
        low_memory=False
    )
):

    print(f"Processing chunk {chunk_number + 1}...")

    total_rows += len(chunk)

    if columns is None:
        columns = list(chunk.columns)

    # --------------------------------------------------------
    # Author statistics
    # --------------------------------------------------------

    if "author_id" in chunk.columns:
        counts = chunk["author_id"].value_counts()

        for author, count in counts.items():
            author_counts[str(author)] += int(count)

    # --------------------------------------------------------
    # Inbound / outbound statistics
    # --------------------------------------------------------

    if "inbound" in chunk.columns:
        counts = chunk["inbound"].value_counts(dropna=False)

        for value, count in counts.items():
            inbound_counts[str(value)] += int(count)

    # --------------------------------------------------------
    # Date range
    # --------------------------------------------------------

    if "created_at" in chunk.columns:

        dates = pd.to_datetime(
            chunk["created_at"],
            errors="coerce"
        )

        chunk_min = dates.min()
        chunk_max = dates.max()

        if pd.notna(chunk_min):
            if min_date is None or chunk_min < min_date:
                min_date = chunk_min

        if pd.notna(chunk_max):
            if max_date is None or chunk_max > max_date:
                max_date = chunk_max

    # --------------------------------------------------------
    # Keep a small sample only
    # --------------------------------------------------------

    if len(sample_rows) < 5000:

        remaining = 5000 - len(sample_rows)

        sample = chunk.head(remaining)

        sample_rows.extend(
            sample.to_dict("records")
        )


# ============================================================
# 3. PRINT BASIC DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("BASIC DATASET INFORMATION")
print("=" * 70)

print(f"\nTotal rows: {total_rows}")

print("\nColumns:")
for col in columns:
    print(f"  - {col}")

print("\nInbound distribution:")
for key, value in inbound_counts.items():
    print(f"  {key}: {value}")

print(f"\nDate range:")
print(f"  Start: {min_date}")
print(f"  End:   {max_date}")


# ============================================================
# 4. TOP AUTHORS
# ============================================================

print("\n" + "=" * 70)
print("TOP AUTHORS")
print("=" * 70)

top_authors = author_counts.most_common(50)

for rank, (author, count) in enumerate(top_authors, start=1):
    print(f"{rank:2}. {author} -> {count}")


# ============================================================
# 5. SAVE AUTHOR STATISTICS
# ============================================================

author_df = pd.DataFrame(
    top_authors,
    columns=["author_id", "tweet_count"]
)

author_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "author_stats.csv"
    ),
    index=False
)


# ============================================================
# 6. SAVE SAMPLE ROWS
# ============================================================

sample_df = pd.DataFrame(sample_rows)

sample_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "sample_rows.csv"
    ),
    index=False
)


# ============================================================
# 7. SAVE GENERAL STATS
# ============================================================

stats = {
    "total_rows": total_rows,
    "columns": columns,
    "inbound_distribution": dict(inbound_counts),
    "date_start": str(min_date),
    "date_end": str(max_date),
    "top_authors": [
        {
            "author_id": author,
            "tweet_count": count
        }
        for author, count in top_authors
    ]
}

with open(
    os.path.join(
        OUTPUT_DIR,
        "overall_stats.json"
    ),
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        stats,
        f,
        indent=2
    )


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print("\nGenerated files:")

for file in os.listdir(OUTPUT_DIR):
    print(f"  - {file}")

print("\nYou can now upload the 'dataset_profile' folder/files here.")