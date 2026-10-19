
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NEWS_FILE = ROOT / "data" / "aligned_news.csv"


def main():
    if not NEWS_FILE.exists():
        print("News alignment has not created the output file yet.")
        return

    news = pd.read_csv(NEWS_FILE)

    print("Rows:", len(news))
    print("Columns:", list(news.columns))

    headline_col = next(
        (
            col for col in news.columns
            if col.lower() in {"headline", "article_title", "title"}
        ),
        None
    )

    ticker_col = next(
        (
            col for col in news.columns
            if col.lower() in {"ticker", "stock_symbol"}
        ),
        None
    )

    date_col = next(
        (
            col for col in news.columns
            if col.lower() in {"trading_date", "date"}
        ),
        None
    )

    if headline_col is None:
        print("Could not identify the headline column.")
    else:
        headlines = news[headline_col].fillna("").astype(str).str.strip()
        print("Non-empty headlines:", headlines.ne("").sum())
        print("Empty headlines:", headlines.eq("").sum())

    if ticker_col is not None:
        print("Unique stocks:", news[ticker_col].nunique())
        print("Rows per stock:")
        print(news[ticker_col].value_counts().to_string())

    if date_col is not None:
        dates = pd.to_datetime(news[date_col], errors="coerce")
        print("Invalid dates:", dates.isna().sum())
        print("Earliest date:", dates.min())
        print("Latest date:", dates.max())

    print("Exact duplicate rows:", news.duplicated().sum())


if __name__ == "__main__":
    main()
