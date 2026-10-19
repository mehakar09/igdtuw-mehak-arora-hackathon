
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "scored_news.csv"
OUTPUT_FILE = ROOT / "data" / "daily_sentiment.csv"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Could not find {INPUT_FILE}")

    news = pd.read_csv(INPUT_FILE)

    required = {"ticker", "trading_date", "sentiment_score"}
    missing = required - set(news.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    news["ticker"] = news["ticker"].astype(str).str.strip().str.upper()
    news["trading_date"] = pd.to_datetime(
        news["trading_date"], errors="coerce"
    ).dt.normalize()
    news["sentiment_score"] = pd.to_numeric(
        news["sentiment_score"], errors="coerce"
    )

    news = news.dropna(
        subset=["ticker", "trading_date", "sentiment_score"]
    )
    news = news[news["ticker"].ne("")]

    if news.empty:
        raise ValueError("No valid scored headlines were found.")

    daily = (
        news.groupby(["trading_date", "ticker"], as_index=False)
        .agg(
            sentiment_score=("sentiment_score", "mean"),
            headline_count=("sentiment_score", "size"),
            sentiment_std=("sentiment_score", "std")
        )
        .sort_values(["trading_date", "ticker"])
    )

    daily["sentiment_std"] = daily["sentiment_std"].fillna(0)
    daily.to_csv(OUTPUT_FILE, index=False)

    print("\nDaily sentiment aggregation complete.")
    print("Saved to:", OUTPUT_FILE)
    print("Daily stock observations:", len(daily))
    print("Stocks covered:", daily["ticker"].nunique())
    print("First date:", daily["trading_date"].min().date())
    print("Last date:", daily["trading_date"].max().date())

    print("\nHeadlines per stock:")
    print(
        daily.groupby("ticker")["headline_count"]
        .sum()
        .sort_values(ascending=False)
        .to_string()
    )

    print("\nSample output:")
    print(daily.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
