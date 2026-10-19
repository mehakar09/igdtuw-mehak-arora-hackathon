
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from backtest import load_prices


ROOT = Path(__file__).resolve().parents[1]
SENTIMENT_FILE = ROOT / "data" / "daily_sentiment.csv"
OUTPUT_FILE = ROOT / "data" / "reaction_fingerprint.csv"
VALIDATION_FILE = ROOT / "data" / "fingerprint_stability.csv"
SUMMARY_FILE = ROOT / "data" / "fingerprint_stability_summary.csv"

LAGS = [1, 2, 3, 5]
TRAIN_FRACTION = 0.70
MIN_OBSERVATIONS = 30


def correlation_result(signal, future_return):
    """Return correlation, p-value, and sample size."""
    valid = pd.concat(
        [
            signal.rename("signal"),
            future_return.rename("future_return")
        ],
        axis=1
    ).dropna()

    n = len(valid)

    if n < MIN_OBSERVATIONS:
        return np.nan, np.nan, n

    if (
        valid["signal"].nunique() < 2
        or valid["future_return"].nunique() < 2
    ):
        return np.nan, np.nan, n

    correlation, p_value = pearsonr(
        valid["signal"],
        valid["future_return"]
    )

    return float(correlation), float(p_value), n


def apply_fdr(results):
    """Apply the Benjamini-Hochberg false-discovery-rate correction."""
    results = results.copy()
    results["adjusted_p_value"] = np.nan

    valid = results["p_value"].notna()
    p_values = results.loc[valid, "p_value"].to_numpy()

    if len(p_values) == 0:
        return results

    order = np.argsort(p_values)
    sorted_p = p_values[order]
    count = len(sorted_p)

    adjusted = sorted_p * count / np.arange(1, count + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)

    original_positions = np.flatnonzero(valid.to_numpy())
    reordered_positions = original_positions[order]

    results.iloc[
        reordered_positions,
        results.columns.get_loc("adjusted_p_value")
    ] = adjusted

    return results


def main():
    if not SENTIMENT_FILE.exists():
        raise FileNotFoundError(
            f"Sentiment file not found: {SENTIMENT_FILE}"
        )

    prices = load_prices().sort_index()
    sentiment = pd.read_csv(SENTIMENT_FILE)

    sentiment = sentiment.rename(
        columns={"trading_date": "date"}
    )

    required = {"date", "ticker", "sentiment_score"}
    missing = required - set(sentiment.columns)

    if missing:
        raise ValueError(
            f"Missing sentiment columns: {sorted(missing)}"
        )

    sentiment["date"] = pd.to_datetime(
        sentiment["date"], errors="coerce"
    ).dt.normalize()

    sentiment["ticker"] = sentiment["ticker"].astype(str).str.strip()

    sentiment["sentiment_score"] = pd.to_numeric(
        sentiment["sentiment_score"], errors="coerce"
    )

    sentiment = sentiment.dropna(
        subset=["date", "sentiment_score"]
    )

    sentiment = sentiment[
        sentiment["ticker"].isin(prices.columns)
    ]

    sentiment = (
        sentiment.groupby(["date", "ticker"], as_index=False)
        ["sentiment_score"].mean()
    )

    common_stocks = sorted(
        set(prices.columns) & set(sentiment["ticker"])
    )

    if not common_stocks:
        raise ValueError("No stocks are shared by prices and sentiment.")

    prices = prices[common_stocks]
    daily_returns = prices.pct_change()

    split_position = int(len(prices.index) * TRAIN_FRACTION)

    if split_position < 1 or split_position >= len(prices.index):
        raise ValueError("Not enough price history for a time split.")

    split_date = prices.index[split_position]

    print("Training period ends before:", split_date.date())
    print("Test period starts on:", split_date.date())

    full_results = []
    validation_results = []

    trading_dates = pd.Series(prices.index, index=prices.index)

    for ticker in common_stocks:
        calendar = pd.DataFrame(index=prices.index)
        calendar.index.name = "date"

        observed = sentiment.loc[
            sentiment["ticker"] == ticker,
            ["date", "sentiment_score"]
        ].set_index("date")

        calendar = calendar.join(observed, how="left")

        consecutive = (
            calendar["sentiment_score"].notna()
            & calendar["sentiment_score"].shift(1).notna()
        )

        calendar["sentiment_change"] = (
            calendar["sentiment_score"].diff().where(consecutive)
        )

        for lag in LAGS:
            calendar[f"future_return_{lag}d"] = (
                daily_returns[ticker].shift(-lag)
            )

        for lag in LAGS:
            future_column = f"future_return_{lag}d"
            future_return_date = trading_dates.shift(-lag)

            valid_future_date = future_return_date.notna()

            train_mask = (
                (calendar.index < split_date)
                & (future_return_date < split_date)
                & valid_future_date
            )

            test_mask = (
                (calendar.index >= split_date)
                & (future_return_date >= split_date)
                & valid_future_date
            )

            for signal_name in [
                "sentiment_score",
                "sentiment_change"
            ]:
                signal = calendar[signal_name]
                future_return = calendar[future_column]

                corr, p_value, n = correlation_result(
                    signal, future_return
                )

                full_results.append({
                    "ticker": ticker,
                    "signal": signal_name,
                    "lag_trading_days": lag,
                    "correlation": corr,
                    "p_value": p_value,
                    "observations": n
                })

                for period, mask in [
                    ("train", train_mask),
                    ("test", test_mask)
                ]:
                    period_corr, period_p, period_n = (
                        correlation_result(
                            signal.loc[mask],
                            future_return.loc[mask]
                        )
                    )

                    validation_results.append({
                        "ticker": ticker,
                        "signal": signal_name,
                        "lag_trading_days": lag,
                        "period": period,
                        "correlation": period_corr,
                        "p_value": period_p,
                        "observations": period_n
                    })

    full = pd.DataFrame(full_results)
    validation = pd.DataFrame(validation_results)

    full = apply_fdr(full)
    full.to_csv(OUTPUT_FILE, index=False)

    test_mask = validation["period"] == "test"
    test_results = apply_fdr(validation.loc[test_mask])

    validation.loc[test_mask, "adjusted_p_value"] = (
        test_results["adjusted_p_value"].to_numpy()
    )

    validation.to_csv(VALIDATION_FILE, index=False)

    wide = validation.pivot(
        index=["ticker", "signal", "lag_trading_days"],
        columns="period",
        values="correlation"
    ).reset_index()

    if {"train", "test"}.issubset(wide.columns):
        wide["absolute_change"] = (
            wide["test"] - wide["train"]
        ).abs()

        wide.to_csv(SUMMARY_FILE, index=False)

    print("\nFull-sample fingerprint")
    print(
        full[
            [
                "ticker",
                "signal",
                "lag_trading_days",
                "correlation",
                "p_value",
                "adjusted_p_value",
                "observations"
            ]
        ].round(4).to_string(index=False)
    )

    print("\nTest-period fingerprint")
    print(
        validation[validation["period"] == "test"][
            [
                "ticker",
                "signal",
                "lag_trading_days",
                "correlation",
                "p_value",
                "adjusted_p_value",
                "observations"
            ]
        ].round(4).to_string(index=False)
    )

    print("\nSaved:")
    print(OUTPUT_FILE)
    print(VALIDATION_FILE)
    print(SUMMARY_FILE)


if __name__ == "__main__":
    main()
