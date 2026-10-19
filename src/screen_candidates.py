import pandas as pd

# =========================================================
# FNSPID Candidate Stock Selection
# =========================================================
#
# Purpose:
# Select a defensible stock universe for the hackathon
# using:
#   1. News coverage
#   2. Historical coverage duration
#   3. Sector diversification
#
# IMPORTANT:
# The FNSPID dataset is NOT downloaded here.
# We are using the coverage statistics collected during
# the earlier dataset inspection.
# =========================================================


# ---------------------------------------------------------
# 1. Candidate stock information
# ---------------------------------------------------------

stocks = [
    # Ticker, Articles, First Date, Last Date, Sector

    ("AAPL", 9338, "2020-05-27", "2023-12-16", "Technology"),
    ("MSFT", 8737, "2022-04-26", "2023-12-16", "Technology"),
    ("NVDA", 11862, "2011-03-03", "2023-12-16", "Technology"),

    ("AMZN", 4782, "2023-02-09", "2023-12-16", "Consumer"),
    ("GOOGL", 175, "2020-05-02", "2020-06-11", "Technology"),

    ("BAC", 1811, "2020-06-01", "2020-06-10", "Financials"),
    ("JPM", 2928, "2011-06-14", "2020-06-11", "Financials"),

    ("JNJ", 2928, "2018-09-15", "2020-06-11", "Healthcare"),
    ("LLY", 3545, "2018-12-29", "2023-12-16", "Healthcare"),

    ("XOM", 7346, "2020-06-03", "2023-12-16", "Energy"),
    ("CVX", 8688, "2016-08-03", "2023-12-16", "Energy"),

    ("WMT", 8686, "2020-06-03", "2023-12-16", "Consumer"),
    ("KO", 10521, "2020-06-11", "2023-12-16", "Consumer"),
    ("PG", 13172, "2020-06-11", "2023-12-16", "Consumer"),

    ("CAT", 7514, "2020-05-27", "2023-12-16", "Industrials"),
    ("GE", 8680, "2012-04-20", "2023-12-16", "Industrials"),
    ("UPS", 3828, "2018-09-15", "2020-06-11", "Industrials"),

    ("DIS", 9654, "2020-06-11", "2023-12-16", "Communication"),
    ("HD", 4453, "2018-04-23", "2020-06-10", "Consumer"),

    ("META", 0, None, None, "Technology"),
]


# ---------------------------------------------------------
# 2. Create DataFrame
# ---------------------------------------------------------

df = pd.DataFrame(
    stocks,
    columns=[
        "ticker",
        "article_count",
        "first_date",
        "last_date",
        "sector"
    ]
)


# ---------------------------------------------------------
# 3. Convert dates
# ---------------------------------------------------------

df["first_date"] = pd.to_datetime(
    df["first_date"],
    errors="coerce"
)

df["last_date"] = pd.to_datetime(
    df["last_date"],
    errors="coerce"
)


# ---------------------------------------------------------
# 4. Calculate historical coverage
# ---------------------------------------------------------

df["coverage_days"] = (
    df["last_date"] - df["first_date"]
).dt.days


# ---------------------------------------------------------
# 5. Define minimum data requirements
# ---------------------------------------------------------

MIN_ARTICLES = 3000
MIN_COVERAGE_DAYS = 365


# ---------------------------------------------------------
# 6. Filter eligible stocks
# ---------------------------------------------------------

eligible = df[
    (df["article_count"] >= MIN_ARTICLES)
    &
    (df["coverage_days"] >= MIN_COVERAGE_DAYS)
].copy()


# ---------------------------------------------------------
# 7. Calculate data-quality scores
# ---------------------------------------------------------
#
# We do NOT simply rank stocks by number of articles.
#
# 50% weight -> news coverage
# 50% weight -> historical coverage
#
# This gives us a more defensible selection criterion.
# ---------------------------------------------------------

eligible["article_score"] = (
    eligible["article_count"]
    /
    eligible["article_count"].max()
)

eligible["coverage_score"] = (
    eligible["coverage_days"]
    /
    eligible["coverage_days"].max()
)

eligible["data_score"] = (
    0.5 * eligible["article_score"]
    +
    0.5 * eligible["coverage_score"]
)


# ---------------------------------------------------------
# 8. Sort eligible stocks
# ---------------------------------------------------------

eligible = eligible.sort_values(
    "data_score",
    ascending=False
)


# ---------------------------------------------------------
# 9. Display eligible stocks
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("ELIGIBLE STOCKS")
print("=" * 70)

print(
    eligible[
        [
            "ticker",
            "sector",
            "article_count",
            "coverage_days",
            "data_score"
        ]
    ].to_string(
        index=False,
        formatters={
            "data_score": "{:.3f}".format
        }
    )
)


# =========================================================
# 10. FINAL STOCK UNIVERSE
# =========================================================
#
# We explicitly define the final universe instead of
# letting an automatic ranking choose everything.
#
# Reason:
# We want sector diversification AND sufficient data.
#
# Selected sectors:
# Technology
# Industrials
# Consumer
# Energy
# Communication
# Healthcare
# =========================================================

SELECTED_STOCKS = [
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

    "LLY"
]


# ---------------------------------------------------------
# 11. Create final DataFrame
# ---------------------------------------------------------

selected_df = eligible[
    eligible["ticker"].isin(SELECTED_STOCKS)
].copy()


# ---------------------------------------------------------
# 12. Preserve our chosen order
# ---------------------------------------------------------

selection_order = {
    ticker: index
    for index, ticker
    in enumerate(SELECTED_STOCKS)
}

selected_df["selection_order"] = (
    selected_df["ticker"].map(selection_order)
)

selected_df = selected_df.sort_values(
    "selection_order"
)


# ---------------------------------------------------------
# 13. Display final stock universe
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL STOCK UNIVERSE")
print("=" * 70)

print(
    selected_df[
        [
            "ticker",
            "sector",
            "article_count",
            "coverage_days",
            "data_score"
        ]
    ].to_string(
        index=False,
        formatters={
            "data_score": "{:.3f}".format
        }
    )
)


# ---------------------------------------------------------
# 14. Display selected tickers
# ---------------------------------------------------------

print("\nSelected stocks:")
print(", ".join(SELECTED_STOCKS))


# ---------------------------------------------------------
# 15. Check that all selected stocks exist
# ---------------------------------------------------------

missing = set(SELECTED_STOCKS) - set(
    selected_df["ticker"]
)

if missing:

    print("\nWARNING!")
    print(
        "The following selected stocks were not found "
        "in the eligible dataset:"
    )

    print(", ".join(sorted(missing)))

else:

    print("\nAll selected stocks passed the data filters.")


# ---------------------------------------------------------
# 16. Show sector distribution
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("SECTOR DISTRIBUTION")
print("=" * 70)

sector_counts = (
    selected_df["sector"]
    .value_counts()
)

print(sector_counts.to_string())


# ---------------------------------------------------------
# 17. Save final stock universe
# ---------------------------------------------------------

output_path = "../data/stock_universe.csv"

selected_df[
    [
        "ticker",
        "sector",
        "article_count",
        "first_date",
        "last_date",
        "coverage_days",
        "article_score",
        "coverage_score",
        "data_score"
    ]
].to_csv(
    output_path,
    index=False
)


# ---------------------------------------------------------
# 18. Final message
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("SELECTION COMPLETE")
print("=" * 70)

print(
    f"\nSaved stock universe to:\n{output_path}"
)

print(
    "\nSelection principle:"
)

print(
    "Stocks were selected using news coverage, "
    "historical coverage and sector diversification "
    "rather than popularity alone."
)

print(
    "\nThe FNSPID dataset is not accessed again by this script."
)