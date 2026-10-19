
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PRICE_DIR = ROOT / "data" / "prices"


def load_prices():
    """Load daily closing prices for the available stocks."""
    price_data = {}

    for file in PRICE_DIR.glob("*.csv"):
        prices = pd.read_csv(file)

        date_col = next(
            (col for col in prices.columns if col.lower() == "date"),
            None
        )
        close_col = next(
            (col for col in prices.columns if col.lower() == "close"),
            None
        )

        if date_col is None or close_col is None:
            print(f"Skipping {file.name}: date or close column missing.")
            continue

        prices = prices[[date_col, close_col]].copy()
        prices.columns = ["date", file.stem]
        prices["date"] = pd.to_datetime(
            prices["date"], errors="coerce"
        ).dt.normalize()
        prices[file.stem] = pd.to_numeric(
            prices[file.stem], errors="coerce"
        )

        prices = prices.dropna(subset=["date", file.stem])
        prices = prices.drop_duplicates("date").set_index("date")
        price_data[file.stem] = prices[file.stem]

    if not price_data:
        raise ValueError("No usable price files were found.")

    return pd.concat(price_data.values(), axis=1).sort_index()


def calculate_metrics(returns):
    """Calculate performance metrics from daily portfolio returns."""
    returns = returns.dropna()

    if returns.empty:
        raise ValueError("No returns available for evaluation.")

    growth = (1 + returns).cumprod()
    running_peak = growth.cummax()
    drawdown = growth / running_peak - 1

    daily_volatility = returns.std(ddof=1)

    if pd.notna(daily_volatility) and daily_volatility > 0:
        sharpe = returns.mean() / daily_volatility * np.sqrt(252)
    else:
        sharpe = np.nan

    return {
        "Total return": growth.iloc[-1] - 1,
        "Annualized volatility": daily_volatility * np.sqrt(252),
        "Sharpe ratio": sharpe,
        "Maximum drawdown": drawdown.min(),
    }


def run_backtest(close_prices, target_weights, transaction_cost=0.001):
    """
    Trade at each day's close and earn returns through the next close.
    Unallocated portfolio weight is held in cash.
    """
    if transaction_cost < 0:
        raise ValueError("Transaction cost cannot be negative.")

    close_prices = close_prices.sort_index()
    target_weights = target_weights.sort_index()

    common_dates = close_prices.index.intersection(target_weights.index)
    common_stocks = close_prices.columns.intersection(
        target_weights.columns
    )

    if len(common_dates) < 2 or len(common_stocks) == 0:
        raise ValueError("Prices and weights have insufficient overlap.")

    prices = close_prices.loc[common_dates, common_stocks].astype(float)
    weights = target_weights.loc[common_dates, common_stocks].fillna(0)

    if prices.isna().any().any() or (prices <= 0).any().any():
        raise ValueError("Prices contain missing or non-positive values.")

    if not np.isfinite(weights.to_numpy()).all():
        raise ValueError("Weights contain infinite or invalid values.")

    if (weights < 0).any().any():
        raise ValueError("This backtest supports long-only portfolios.")

    if (weights.sum(axis=1) > 1 + 1e-8).any():
        raise ValueError("Target stock weights cannot exceed 100%.")

    daily_returns = prices.pct_change()
    portfolio_returns = []
    turnover_values = []
    cost_values = []
    cash_values = []

    # Start fully in cash.
    previous_weights = pd.Series(0.0, index=common_stocks)

    for i, date in enumerate(common_dates):
        target = weights.loc[date]
        cash_weight = 1.0 - target.sum()

        # Stock-weight changes implicitly include the cash allocation change.
        turnover = (target - previous_weights).abs().sum()
        cost = turnover * transaction_cost

        cash_values.append(cash_weight)
        turnover_values.append(turnover)
        cost_values.append(cost)

        if i + 1 < len(common_dates):
            next_date = common_dates[i + 1]
            next_return = daily_returns.loc[next_date]

            if next_return.isna().any():
                raise ValueError(
                    f"Missing stock returns on {next_date.date()}."
                )

            gross_return = (target * next_return).sum()
            net_return = (1 - cost) * (1 + gross_return) - 1
            portfolio_returns.append(net_return)

            # Include cash in total portfolio value when calculating drift.
            stock_values = target * (1 + next_return)
            total_value = stock_values.sum() + cash_weight

            if total_value > 0:
                previous_weights = stock_values / total_value
            else:
                previous_weights = pd.Series(
                    0.0, index=common_stocks
                )
        else:
            portfolio_returns.append(np.nan)

    results = pd.DataFrame(
        {
            "Portfolio return": portfolio_returns,
            "Turnover": turnover_values,
            "Transaction cost": cost_values,
            "Cash weight": cash_values,
        },
        index=common_dates,
    )

    results = results.iloc[:-1]
    metrics = calculate_metrics(results["Portfolio return"])

    metrics["Average cash weight"] = results["Cash weight"].mean()
    metrics["Average daily turnover"] = results["Turnover"].mean()

    return results, metrics


if __name__ == "__main__":
    prices = load_prices()

    equal_weights = pd.DataFrame(
        1 / len(prices.columns),
        index=prices.index,
        columns=prices.columns,
    )

    results, metrics = run_backtest(prices, equal_weights)

    print("Loaded stock prices:", prices.shape)
    print("Date range:", prices.index.min(), "to", prices.index.max())
    print("\nEqual-weight baseline")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    results.to_csv(ROOT / "data" / "baseline_backtest.csv")
    print("\nSaved baseline results to data/baseline_backtest.csv")
