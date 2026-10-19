from datasets import load_dataset
from collections import defaultdict
from datetime import datetime

print("Loading FNSPID in streaming mode...")

dataset = load_dataset(
    "sabareesh88/FNSPID_nasdaq",
    split="train",
    streaming=True
)

coverage = defaultdict(lambda: {
    "count": 0,
    "first_date": None,
    "last_date": None
})

print("Scanning dataset...")

for i, row in enumerate(dataset):

    ticker = row["Stock_symbol"]
    date = row["Date"]

    if not ticker or not date:
        continue

    date = datetime.fromisoformat(
        date.replace(" UTC", "+00:00")
    )

    coverage[ticker]["count"] += 1

    if (
        coverage[ticker]["first_date"] is None
        or date < coverage[ticker]["first_date"]
    ):
        coverage[ticker]["first_date"] = date

    if (
        coverage[ticker]["last_date"] is None
        or date > coverage[ticker]["last_date"]
    ):
        coverage[ticker]["last_date"] = date

    # Safety limit for our first inspection
    if i >= 500000:
        break

print("\nTop stocks by news coverage:\n")

results = sorted(
    coverage.items(),
    key=lambda x: x[1]["count"],
    reverse=True
)

for ticker, info in results[:30]:
    print(
        f"{ticker:8} | "
        f"{info['count']:6} articles | "
        f"{info['first_date'].date()} → "
        f"{info['last_date'].date()}"
    )