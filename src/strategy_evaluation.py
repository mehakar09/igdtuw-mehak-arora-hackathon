
from pathlib import Path

import pandas as pd

from backtest import load_prices, run_backtest
from portfolio import make_portfolio_weights


ROOT = Path(__file__).resolve().parents[1]
SENTIMENT_FILE = ROOT / "data" / "daily_sentiment.csv"
OUTPUT_FILE = ROOT / "data" / "strategy_comparison.csv"


def main():
    if not SENTIMENT_FILE.exists():
        raise FileNotFoundError(
            "data/daily_sentiment.csv was not found."
        )

    prices = load_prices()
    sentiment = pd.read_csv(SENTIMENT_FILE)

    sentiment = sentiment.rename(
        columns={"trading_date": "date"}
    )
    sentiment["date"] = pd.to_datetime(
        sentiment["date"], errors="coerce"
    ).dt.normalize()

    # Generate weights from the daily sentiment observations.
    weights = make_portfolio_weights(sentiment)

    # Use the full stock-price trading calendar.
    weights = weights.reindex(
        index=prices.index,
        columns=prices.columns
    )

    # No fresh news signal means no new stock allocation.
    # Missing signal rows are held in cash rather than filled with
    # future sentiment information.
    weights = weights.fillna(0.0)

    # Apply the signal from the previous trading session.
    # Today's news cannot earn today's already-completed return.
    weights = weights.shift(1).fillna(0.0)

    # Compare both strategies on exactly the same price calendar.
    equal_weights = pd.DataFrame(
        1.0 / len(prices.columns),
        index=prices.index,
        columns=prices.columns
    )

    sentiment_results, sentiment_metrics = run_backtest(
        prices, weights
    )
    equal_results, equal_metrics = run_backtest(
        prices, equal_weights
    )

    comparison = pd.DataFrame({
        "Equal-weight baseline": equal_metrics,
        "Sentiment strategy": sentiment_metrics
    })

    comparison.to_csv(OUTPUT_FILE)

    print("\nStrategy comparison")
    print(comparison.round(4).to_string())

    print("\nSentiment strategy diagnostics")
    print(
        "Average stock allocation:",
        round(weights.sum(axis=1).mean(), 4)
    )
    print(
        "Average cash allocation:",
        round((1 - weights.sum(axis=1)).mean(), 4)
    )

    print("\nSaved comparison to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
