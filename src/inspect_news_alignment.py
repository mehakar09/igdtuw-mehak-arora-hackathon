from datasets import load_dataset
from collections import defaultdict
from datetime import datetime


# ============================================================
# CONFIGURATION
# ============================================================

SELECTED_STOCKS = {
    "NVDA",
    "AAPL",
    "GE",
    "CAT",
    "PG",
    "KO",
    "WMT",
    "CVX",
    "XOM",
    "DIS",
    "LLY",
}

START_DATE = "2018-01-01"
END_DATE = "2023-12-29"

SAMPLE_LIMIT = 5


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("FNSPID NEWS ALIGNMENT INSPECTION")
print("=" * 70)

print("\nLoading FNSPID in streaming mode...")

dataset = load_dataset(
    "sabareesh88/FNSPID_nasdaq",
    split="train",
    streaming=True
)

print("Dataset loaded.")
print("\nScanning selected stocks...")
print("This may take some time because FNSPID is very large.\n")


# ============================================================
# STORAGE
# ============================================================

stats = defaultdict(lambda: {
    "count": 0,
    "first_timestamp": None,
    "last_timestamp": None,
    "dates": set(),
    "hours": defaultdict(int),
    "sample_headlines": []
})


total_rows = 0
selected_rows = 0


# ============================================================
# SCAN DATA
# ============================================================

for row in dataset:

    total_rows += 1

    ticker = row.get("Stock_symbol")

    if ticker not in SELECTED_STOCKS:
        continue

    raw_date = row.get("Date")

    if not raw_date:
        continue

    # --------------------------------------------------------
    # Parse timestamp
    # --------------------------------------------------------

    try:
        timestamp = datetime.fromisoformat(
            raw_date.replace(" UTC", "+00:00")
        )
    except Exception:

        print(
            f"Could not parse timestamp: {raw_date}"
        )

        continue

    # --------------------------------------------------------
    # Date filter
    # --------------------------------------------------------

    date_only = timestamp.date()

    if date_only.isoformat() < START_DATE:
        continue

    if date_only.isoformat() > END_DATE:
        continue

    selected_rows += 1

    # --------------------------------------------------------
    # Update statistics
    # --------------------------------------------------------

    stats[ticker]["count"] += 1

    stats[ticker]["dates"].add(
        date_only
    )

    stats[ticker]["hours"][timestamp.hour] += 1

    if (
        stats[ticker]["first_timestamp"] is None
        or timestamp < stats[ticker]["first_timestamp"]
    ):
        stats[ticker]["first_timestamp"] = timestamp

    if (
        stats[ticker]["last_timestamp"] is None
        or timestamp > stats[ticker]["last_timestamp"]
    ):
        stats[ticker]["last_timestamp"] = timestamp

    # --------------------------------------------------------
    # Save a few sample headlines
    # --------------------------------------------------------

    if len(stats[ticker]["sample_headlines"]) < SAMPLE_LIMIT:

        stats[ticker]["sample_headlines"].append(
            {
                "timestamp": raw_date,
                "headline": row.get(
                    "Article_title",
                    ""
                )
            }
        )


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("SCAN COMPLETE")
print("=" * 70)

print(f"\nTotal dataset rows scanned: {total_rows:,}")
print(f"Rows matching our universe: {selected_rows:,}")


# ============================================================
# STOCK SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("NEWS COVERAGE SUMMARY")
print("=" * 70)

print(
    f"\n{'Ticker':<8}"
    f"{'Articles':>12}"
    f"{'Days':>12}"
    f"{'First timestamp':>28}"
    f"{'Last timestamp':>28}"
)

print("-" * 90)

for ticker in sorted(SELECTED_STOCKS):

    if ticker not in stats:

        print(
            f"{ticker:<8}"
            f"{'NO DATA':>12}"
        )

        continue

    s = stats[ticker]

    print(
        f"{ticker:<8}"
        f"{s['count']:>12,}"
        f"{len(s['dates']):>12}"
        f"{str(s['first_timestamp']):>28}"
        f"{str(s['last_timestamp']):>28}"
    )


# ============================================================
# TIMESTAMP DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("TIMESTAMP / HOUR DISTRIBUTION")
print("=" * 70)

for ticker in sorted(SELECTED_STOCKS):

    if ticker not in stats:
        continue

    print(f"\n{ticker}:")

    hours = stats[ticker]["hours"]

    for hour in sorted(hours):

        print(
            f"  {hour:02d}:00 UTC"
            f" -> {hours[hour]:,} articles"
        )


# ============================================================
# SAMPLE HEADLINES
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE NEWS")
print("=" * 70)

for ticker in sorted(SELECTED_STOCKS):

    if ticker not in stats:
        continue

    print(f"\n--- {ticker} ---")

    for item in stats[ticker]["sample_headlines"]:

        print(
            f"\nTime: {item['timestamp']}"
        )

        print(
            f"Headline: {item['headline']}"
        )


# ============================================================
# DAILY AGGREGATION EXAMPLE
# ============================================================

print("\n" + "=" * 70)
print("ALIGNMENT RULE WE WILL USE")
print("=" * 70)

print(
    """
News timestamp
      |
      v
Convert to US/Eastern market date
      |
      v
Determine whether market was open
      |
      +---- Published before market close
      |          |
      |          v
      |     Same-day information
      |
      +---- Published after market close
                 |
                 v
            Next trading day
"""
)

print(
    "\nIMPORTANT:"
)

print(
    "We will NOT simply use the calendar date of the article."
)

print(
    "The publication timestamp will determine when the"
)

print(
    "information could actually have influenced a trading decision."
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)

print(
    "\nNext step:"
)

print(
    "Use the timestamp distribution and sample headlines"
)

print(
    "to design the news -> trading-day alignment pipeline."
)