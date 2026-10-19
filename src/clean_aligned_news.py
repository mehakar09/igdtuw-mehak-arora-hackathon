
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NEWS_FILE = ROOT / "data" / "aligned_news.csv"
CLEAN_FILE = ROOT / "data" / "aligned_news_clean.csv"


def main():
    if not NEWS_FILE.exists():
        print("No aligned_news.csv file exists yet.")
        return

    news = pd.read_csv(NEWS_FILE, low_memory=False)

    print("Original rows:", len(news))
    print("Columns:", news.columns.tolist())

    required = {"ticker", "headline", "trading_day"}
    missing = required - set(news.columns)

    if missing:
        print("Required columns missing:", sorted(missing))
        return

    news["ticker"] = news["ticker"].fillna("").astype(str).str.strip()
    news["headline"] = news["headline"].fillna("").astype(str).str.strip()
    news["trading_day"] = pd.to_datetime(
        news["trading_day"], errors="coerce"
    )

    news = news[
        news["ticker"].ne("")
        & news["headline"].ne("")
        & news["trading_day"].notna()
    ].copy()

    news["ticker"] = news["ticker"].str.upper()
    news = news[news["ticker"].isin({
        "NVDA", "AAPL", "GE", "CAT", "PG", "KO",
        "WMT", "CVX", "XOM", "DIS", "LLY"
    })]

    news = news.drop_duplicates(
        subset=["ticker", "news_timestamp", "headline", "trading_day"]
    )

    news["trading_day"] = news["trading_day"].dt.strftime("%Y-%m-%d")
    news = news.sort_values(
        ["trading_day", "ticker"]
    ).reset_index(drop=True)

    news.to_csv(CLEAN_FILE, index=False)

    print("\nClean rows:", len(news))
    print("Unique stocks:", news["ticker"].nunique())
    print("Saved to:", CLEAN_FILE)

    if not news.empty:
        print("\nArticles per stock:")
        print(news.groupby("ticker").size().to_string())
        print("\nSample headlines:")
        print(news[["ticker", "trading_day", "headline"]].head(5).to_string())
    else:
        print(
            "\nNo valid articles could be recovered from the saved CSV. "
            "We will need to change the collection approach."
        )


if __name__ == "__main__":
    main()
