
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# --------------------------------------------------
# File paths and page setup
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
PRICE_DIR = ROOT / "data" / "prices"
BASELINE_FILE = ROOT / "data" / "baseline_backtest.csv"
SENTIMENT_FILE = ROOT / "data" / "daily_sentiment.csv"
FINGERPRINT_FILE = ROOT / "data" / "reaction_fingerprint.csv"
COMPARISON_FILE = ROOT / "data" / "strategy_comparison.csv"


st.set_page_config(
    page_title="Market Reaction Fingerprint",
    page_icon="📈",
    layout="wide"
)

st.title("Market Reaction Fingerprint")
st.caption(
    "Exploring how financial news sentiment relates to stock returns "
    "across different trading-session horizons."
)


# --------------------------------------------------
# Load stock prices
# --------------------------------------------------

@st.cache_data
def load_price_data():
    frames = []

    for file in PRICE_DIR.glob("*.csv"):
        data = pd.read_csv(file)

        date_col = next(
            (col for col in data.columns if col.lower() == "date"),
            None
        )
        close_col = next(
            (col for col in data.columns if col.lower() == "close"),
            None
        )

        if date_col is None or close_col is None:
            continue

        data = data[[date_col, close_col]].copy()
        data.columns = ["Date", file.stem]

        data["Date"] = pd.to_datetime(
            data["Date"], errors="coerce"
        ).dt.normalize()

        data[file.stem] = pd.to_numeric(
            data[file.stem], errors="coerce"
        )

        data = data.dropna(subset=["Date", file.stem])
        data = data.drop_duplicates("Date")
        data = data.set_index("Date").sort_index()

        frames.append(data)

    if not frames:
        return pd.DataFrame()

    prices = pd.concat(frames, axis=1).sort_index()
    prices = prices.loc[:, ~prices.columns.duplicated()]

    return prices


@st.cache_data
def load_daily_sentiment():
    if not SENTIMENT_FILE.exists():
        return pd.DataFrame()

    data = pd.read_csv(SENTIMENT_FILE)

    required = {"trading_date", "ticker", "sentiment_score"}

    if not required.issubset(data.columns):
        return pd.DataFrame()

    data["trading_date"] = pd.to_datetime(
        data["trading_date"], errors="coerce"
    ).dt.normalize()

    data["sentiment_score"] = pd.to_numeric(
        data["sentiment_score"], errors="coerce"
    )

    data = data.dropna(
        subset=["trading_date", "ticker", "sentiment_score"]
    )

    return data


prices = load_price_data()

if prices.empty:
    st.error("No usable stock price files found in data/prices.")
    st.stop()


# --------------------------------------------------
# Sidebar controls
# --------------------------------------------------

st.sidebar.header("Explore the data")

selected_stock = st.sidebar.selectbox(
    "Choose a stock",
    sorted(prices.columns)
)

min_date = prices.index.min().date()
max_date = prices.index.max().date()

date_range = st.sidebar.date_input(
    "Select date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
elif isinstance(date_range, (tuple, list)) and len(date_range) == 1:
    start_date = end_date = date_range[0]
else:
    start_date, end_date = min_date, max_date

if start_date > end_date:
    st.error("The start date must not be after the end date.")
    st.stop()


stock_prices = prices.loc[
    (prices.index.date >= start_date)
    & (prices.index.date <= end_date),
    selected_stock
].dropna()

if len(stock_prices) < 2:
    st.warning(
        "Select a date range containing at least two trading days."
    )
    st.stop()


# --------------------------------------------------
# Stock overview
# --------------------------------------------------

stock_return = stock_prices.iloc[-1] / stock_prices.iloc[0] - 1
daily_returns = stock_prices.pct_change().dropna()

annual_volatility = (
    daily_returns.std() * (252 ** 0.5)
    if len(daily_returns) > 1
    else float("nan")
)

growth = stock_prices / stock_prices.iloc[0]
drawdown = growth / growth.cummax() - 1

col1, col2, col3 = st.columns(3)

col1.metric(
    "Price change",
    f"{stock_return * 100:.2f}%"
)

col2.metric(
    "Annualized volatility",
    f"{annual_volatility * 100:.2f}%"
)

col3.metric(
    "Maximum drawdown",
    f"{drawdown.min() * 100:.2f}%"
)


# --------------------------------------------------
# Historical closing price
# --------------------------------------------------

st.subheader(f"{selected_stock}: closing price")

fig, ax = plt.subplots(figsize=(11, 4))

ax.plot(stock_prices.index, stock_prices.values)
ax.set_xlabel("Date")
ax.set_ylabel("Closing price")
ax.grid(alpha=0.3)

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)


# --------------------------------------------------
# Cumulative stock price return
# --------------------------------------------------

st.subheader(f"{selected_stock}: cumulative price return")

cumulative_return = (
    stock_prices / stock_prices.iloc[0] - 1
) * 100

fig, ax = plt.subplots(figsize=(11, 4))

ax.plot(cumulative_return.index, cumulative_return.values)
ax.axhline(0, linewidth=1, linestyle="--")
ax.set_xlabel("Date")
ax.set_ylabel("Return (%)")
ax.grid(alpha=0.3)

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.caption(
    "These charts show the selected stock's historical price performance. "
    "They are not returns generated by the sentiment strategy."
)


# --------------------------------------------------
# Portfolio strategy comparison
# --------------------------------------------------

st.header("Portfolio Strategy Comparison")

if COMPARISON_FILE.exists():
    comparison = pd.read_csv(COMPARISON_FILE, index_col=0)

    required_strategies = {
        "Equal-weight baseline",
        "Sentiment strategy"
    }

    if required_strategies.issubset(comparison.columns):
        metric_rows = [
            "Total return",
            "Annualized volatility",
            "Sharpe ratio",
            "Maximum drawdown",
            "Average cash weight",
            "Average daily turnover"
        ]

        available_rows = [
            row for row in metric_rows if row in comparison.index
        ]

        display_comparison = comparison.loc[available_rows].copy()

        percentage_rows = [
            "Total return",
            "Annualized volatility",
            "Maximum drawdown",
            "Average cash weight",
            "Average daily turnover"
        ]

        for row in percentage_rows:
            if row in display_comparison.index:
                display_comparison.loc[row] *= 100

        display_comparison.index = [
            f"{row} (%)" if row in percentage_rows else row
            for row in display_comparison.index
        ]

        st.dataframe(
            display_comparison.style.format("{:.2f}"),
            use_container_width=True
        )

        st.caption(
            "These are saved backtest results. Performance is preliminary "
            "until signal timing, trading-calendar alignment, and transaction "
            "cost assumptions have been fully validated."
        )
    else:
        st.warning(
            "The comparison file has unexpected columns. "
            "Run strategy_evaluation.py again."
        )
else:
    st.info(
        "Run strategy_evaluation.py to generate the portfolio comparison."
    )


# --------------------------------------------------
# Daily news sentiment
# --------------------------------------------------

st.header("Daily News Sentiment")

sentiment = load_daily_sentiment()

if not sentiment.empty:
    stock_sentiment = sentiment[
        (sentiment["ticker"] == selected_stock)
        & (sentiment["trading_date"].dt.date >= start_date)
        & (sentiment["trading_date"].dt.date <= end_date)
    ].sort_values("trading_date")

    if not stock_sentiment.empty:
        st.subheader(f"{selected_stock}: daily sentiment score")

        # Scatter plot prevents lines from implying values during
        # periods with no sentiment observations.
        fig, ax = plt.subplots(figsize=(11, 4))

        ax.scatter(
            stock_sentiment["trading_date"],
            stock_sentiment["sentiment_score"],
            s=12,
            alpha=0.65
        )

        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_xlabel("Date")
        ax.set_ylabel("Sentiment score")
        ax.grid(alpha=0.25)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Positive scores mean positive sentiment exceeds negative "
            "sentiment; negative scores mean the reverse. Missing dates "
            "indicate no recorded daily sentiment observation."
        )

        display_columns = [
            col for col in [
                "trading_date",
                "ticker",
                "sentiment_score",
                "headline_count",
                "sentiment_std"
            ]
            if col in stock_sentiment.columns
        ]

        with st.expander("View daily sentiment records"):
            st.dataframe(
                stock_sentiment[display_columns],
                use_container_width=True
            )
    else:
        st.info(
            "No sentiment observations exist for this stock and date range."
        )
else:
    st.info(
        "Daily sentiment data is unavailable or has unexpected columns. "
        "Check data/daily_sentiment.csv."
    )


# --------------------------------------------------
# Market Reaction Fingerprint
# --------------------------------------------------

st.header("Market Reaction Fingerprint")

st.write(
    "Compare observed news sentiment with subsequent stock returns "
    "across one-, two-, three-, and five-session horizons."
)

if FINGERPRINT_FILE.exists():
    fingerprint = pd.read_csv(FINGERPRINT_FILE)

    required_columns = {
        "ticker",
        "signal",
        "lag_trading_days",
        "correlation",
        "observations"
    }

    if required_columns.issubset(fingerprint.columns):
        fingerprint["correlation"] = pd.to_numeric(
            fingerprint["correlation"], errors="coerce"
        )

        fingerprint["lag_trading_days"] = pd.to_numeric(
            fingerprint["lag_trading_days"], errors="coerce"
        )

        fingerprint["observations"] = pd.to_numeric(
            fingerprint["observations"], errors="coerce"
        )

        signal_options = [
            "sentiment_score",
            "sentiment_change"
        ]

        selected_signal = st.selectbox(
            "Select sentiment measure",
            signal_options,
            format_func=lambda value: value.replace(
                "_", " "
            ).title()
        )

        selected_fingerprint = fingerprint[
            fingerprint["signal"] == selected_signal
        ].copy()

        # Selected stock's correlation by future horizon.
        stock_fingerprint = selected_fingerprint[
            selected_fingerprint["ticker"] == selected_stock
        ].sort_values("lag_trading_days")

        left, right = st.columns([2, 1])

        with left:
            st.subheader(
                f"{selected_stock}: correlation by future horizon"
            )

            if not stock_fingerprint.empty:
                chart_data = stock_fingerprint.set_index(
                    "lag_trading_days"
                )["correlation"]

                st.bar_chart(chart_data)

                display_fingerprint = stock_fingerprint[
                    [
                        "ticker",
                        "lag_trading_days",
                        "correlation",
                        "observations"
                    ]
                ].rename(columns={
                    "ticker": "Stock",
                    "lag_trading_days": "Future trading sessions",
                    "correlation": "Correlation",
                    "observations": "Observations"
                })

                st.dataframe(
                    display_fingerprint.style.format({
                        "Correlation": "{:.4f}",
                        "Observations": "{:.0f}"
                    }),
                    use_container_width=True
                )
            else:
                st.info(
                    "No fingerprint results are available for this stock."
                )

        with right:
            st.subheader("How to interpret the correlations")

            st.write(
                "A positive correlation indicates that higher values of "
                "the sentiment measure tend to coincide with higher "
                "subsequent returns in this sample."
            )

            st.write(
                "A negative correlation indicates that higher values "
                "tend to coincide with lower subsequent returns."
            )

            st.caption(
                "Correlation does not imply causation or guarantee "
                "future performance."
            )

        # All-stock fingerprint table.
        st.subheader("Fingerprint across all stocks")

        heatmap_data = selected_fingerprint.pivot(
            index="ticker",
            columns="lag_trading_days",
            values="correlation"
        ).sort_index()

        heatmap_data.columns = [
            f"{int(column)}-session"
            for column in heatmap_data.columns
        ]

        st.dataframe(
            heatmap_data.style.format("{:.3f}").background_gradient(
                cmap="coolwarm",
                vmin=-0.2,
                vmax=0.2,
                axis=None
            ),
            use_container_width=True
        )

        st.caption(
            "Blue indicates negative correlation; red indicates positive "
            "correlation. These correlations are exploratory and may not "
            "remain stable outside this sample."
        )

        with st.expander("View complete fingerprint data"):
            st.dataframe(
                selected_fingerprint.sort_values(
                    ["ticker", "lag_trading_days"]
                ),
                use_container_width=True
            )
    else:
        st.warning(
            "The fingerprint file has unexpected columns. "
            "Run reaction_fingerprint.py again."
        )
else:
    st.info(
        "Fingerprint data is not available. Run reaction_fingerprint.py "
        "to generate it."
    )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Market Reaction Fingerprint | Educational research project. "
    "Historical correlations and backtests do not guarantee future results."
)
