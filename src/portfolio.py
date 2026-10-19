
import pandas as pd


def make_portfolio_weights(
    daily_sentiment,
    max_weight=0.20,
    smoothing_days=3,
    deadband=0.05
):
    """
    Turn daily stock sentiment into long-only portfolio weights.

    Expected columns: date, ticker, sentiment_score.
    Any unallocated weight remains in cash.
    """
    required = {"date", "ticker", "sentiment_score"}
    missing = required - set(daily_sentiment.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if not 0 < max_weight <= 1:
        raise ValueError("max_weight must be between 0 and 1.")

    if not isinstance(smoothing_days, int) or smoothing_days < 1:
        raise ValueError("smoothing_days must be a positive integer.")

    if deadband < 0:
        raise ValueError("deadband cannot be negative.")

    data = daily_sentiment.copy()
    data["date"] = pd.to_datetime(
        data["date"], errors="coerce"
    ).dt.normalize()
    data["sentiment_score"] = pd.to_numeric(
        data["sentiment_score"], errors="coerce"
    )

    data = data.dropna(
        subset=["date", "ticker", "sentiment_score"]
    )
    data["ticker"] = data["ticker"].astype(str).str.strip()
    data = data[data["ticker"] != ""]

    if data.empty:
        raise ValueError("No valid sentiment observations were provided.")

    # Combine multiple headlines for the same stock and date.
    daily = data.pivot_table(
        index="date",
        columns="ticker",
        values="sentiment_score",
        aggfunc="mean"
    ).sort_index()

    # Smooth only the observations available for each stock.
    smoothed = daily.rolling(
        window=smoothing_days,
        min_periods=1
    ).mean()

    # Small signals are treated as inconclusive.
    smoothed = smoothed.where(smoothed.abs() >= deadband, 0)

    weights = pd.DataFrame(
        0.0,
        index=smoothed.index,
        columns=smoothed.columns
    )

    for date, scores in smoothed.iterrows():
        scores = scores.fillna(0)
        positive_scores = scores.clip(lower=0)
        total = positive_scores.sum()

        # If no stock has a positive signal, remain in cash.
        if total <= 0:
            continue

        proposed = positive_scores / total
        allocated = pd.Series(0.0, index=scores.index)
        remaining = 1.0

        # Redistribute available weight without exceeding the cap.
        eligible = proposed[proposed > 0].index

        while remaining > 1e-10 and len(eligible) > 0:
            shares = proposed.loc[eligible]
            additions = remaining * shares / shares.sum()

            room = max_weight - allocated.loc[eligible]
            actual_additions = pd.concat(
                [additions, room], axis=1
            ).min(axis=1)

            allocated.loc[eligible] += actual_additions
            remaining = 1.0 - allocated.sum()

            eligible = eligible[
                allocated.loc[eligible] < max_weight - 1e-10
            ]

        weights.loc[date] = allocated

    return weights


if __name__ == "__main__":
    print("Portfolio weighting module is ready.")
