import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRICE_DIR = PROJECT_ROOT / "data" / "prices"

STOCKS = [
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
    "LLY",
]


# ============================================================
# HEADER
# ============================================================

print("=" * 65)
print("PRICE DATA VALIDATION")
print("=" * 65)

print(f"\nPrice directory:")
print(PRICE_DIR)

if not PRICE_DIR.exists():
    print("\nERROR: Price directory does not exist.")
    print("Expected location:")
    print(PRICE_DIR)
    raise SystemExit(1)


# ============================================================
# CHECK FILES
# ============================================================

print("\n" + "=" * 65)
print("CHECKING STOCK FILES")
print("=" * 65)

missing_files = []

for ticker in STOCKS:

    file_path = PRICE_DIR / f"{ticker}.csv"

    if file_path.exists():
        print(f"[OK] {ticker}")
    else:
        print(f"[MISSING] {ticker}")
        missing_files.append(ticker)


if missing_files:
    print("\nMissing stocks:")
    print(", ".join(missing_files))
    raise SystemExit(1)


# ============================================================
# LOAD AND VALIDATE EACH STOCK
# ============================================================

print("\n" + "=" * 65)
print("STOCK DATA SUMMARY")
print("=" * 65)

all_data = {}
date_sets = {}

for ticker in STOCKS:

    file_path = PRICE_DIR / f"{ticker}.csv"

    try:
        df = pd.read_csv(file_path)

        # ----------------------------------------------------
        # Normalize column names
        # ----------------------------------------------------

        df.columns = [
            str(col).strip().lower()
            for col in df.columns
        ]

        # ----------------------------------------------------
        # Find date column
        # ----------------------------------------------------

        if "date" not in df.columns:

            # Sometimes yfinance data may use Datetime
            possible_date_columns = [
                col for col in df.columns
                if "date" in col or "time" in col
            ]

            if possible_date_columns:
                df.rename(
                    columns={
                        possible_date_columns[0]: "date"
                    },
                    inplace=True
                )
            else:
                print(f"\nERROR: No date column found for {ticker}")
                print("Columns:", list(df.columns))
                raise SystemExit(1)

        # ----------------------------------------------------
        # Convert date
        # ----------------------------------------------------

        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

        # Remove timezone if present
        try:
            df["date"] = df["date"].dt.tz_localize(None)
        except TypeError:
            pass

        # ----------------------------------------------------
        # Check required price columns
        # ----------------------------------------------------

        required_columns = ["close"]

        missing_columns = [
            col
            for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:
            print(
                f"\nERROR: {ticker} is missing columns:"
            )
            print(missing_columns)
            print("Available columns:", list(df.columns))
            raise SystemExit(1)

        # ----------------------------------------------------
        # Remove invalid rows
        # ----------------------------------------------------

        df = df.dropna(subset=["date"])

        df = df.sort_values("date")

        # ----------------------------------------------------
        # Check duplicate dates
        # ----------------------------------------------------

        duplicate_dates = df["date"].duplicated().sum()

        # ----------------------------------------------------
        # Check missing Close prices
        # ----------------------------------------------------

        missing_close = df["close"].isna().sum()

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        all_data[ticker] = df

        date_sets[ticker] = set(df["date"])

        print(
            f"\n{ticker}"
            f"\n  Rows:           {len(df)}"
            f"\n  Start:          {df['date'].min().date()}"
            f"\n  End:            {df['date'].max().date()}"
            f"\n  Missing Close:  {missing_close}"
            f"\n  Duplicate Date: {duplicate_dates}"
        )

    except Exception as e:

        print(f"\nERROR loading {ticker}:")
        print(e)

        raise SystemExit(1)


# ============================================================
# CHECK DATE CONSISTENCY
# ============================================================

print("\n" + "=" * 65)
print("DATE CONSISTENCY CHECK")
print("=" * 65)

reference_ticker = STOCKS[0]

reference_dates = date_sets[reference_ticker]

print(
    f"\nUsing {reference_ticker} as reference."
)

all_same_dates = True

for ticker in STOCKS[1:]:

    current_dates = date_sets[ticker]

    missing_dates = reference_dates - current_dates
    extra_dates = current_dates - reference_dates

    if not missing_dates and not extra_dates:

        print(f"[OK] {ticker}: dates match {reference_ticker}")

    else:

        all_same_dates = False

        print(f"\n[DIFFERENCE] {ticker}")

        print(
            f"  Missing relative to {reference_ticker}: "
            f"{len(missing_dates)}"
        )

        print(
            f"  Extra relative to {reference_ticker}: "
            f"{len(extra_dates)}"
        )


# ============================================================
# COMMON DATE RANGE
# ============================================================

print("\n" + "=" * 65)
print("COMMON DATE RANGE")
print("=" * 65)

common_dates = set.intersection(
    *date_sets.values()
)

if not common_dates:

    print("\nERROR: No common trading dates found.")
    raise SystemExit(1)

common_dates = sorted(common_dates)

print(
    f"\nCommon trading days: {len(common_dates)}"
)

print(
    f"Common start date:   {common_dates[0].date()}"
)

print(
    f"Common end date:     {common_dates[-1].date()}"
)


# ============================================================
# MISSING VALUES CHECK
# ============================================================

print("\n" + "=" * 65)
print("MISSING VALUE CHECK")
print("=" * 65)

numeric_columns = [
    "open",
    "high",
    "low",
    "close",
    "volume"
]

any_missing = False

for ticker, df in all_data.items():

    available_columns = [
        col
        for col in numeric_columns
        if col in df.columns
    ]

    missing_values = df[available_columns].isna().sum()

    total_missing = missing_values.sum()

    if total_missing == 0:

        print(f"[OK] {ticker}: no missing price values")

    else:

        any_missing = True

        print(f"\n[WARNING] {ticker}")

        for column, count in missing_values.items():

            if count > 0:
                print(
                    f"  {column}: {count} missing"
                )


# ============================================================
# PRICE SANITY CHECK
# ============================================================

print("\n" + "=" * 65)
print("PRICE SANITY CHECK")
print("=" * 65)

sanity_passed = True

for ticker, df in all_data.items():

    invalid_close = (
        (df["close"] <= 0)
        | (~df["close"].apply(pd.api.types.is_number))
    ).sum()

    if invalid_close == 0:

        print(
            f"[OK] {ticker}: close prices valid"
        )

    else:

        sanity_passed = False

        print(
            f"[ERROR] {ticker}: "
            f"{invalid_close} invalid close prices"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("VALIDATION SUMMARY")
print("=" * 65)

print(f"\nStocks checked:        {len(STOCKS)}")
print(f"Common trading days:  {len(common_dates)}")
print(
    f"Common period:        "
    f"{common_dates[0].date()} -> "
    f"{common_dates[-1].date()}"
)

print(
    f"\nDate consistency:     "
    f"{'PASS' if all_same_dates else 'CHECK REQUIRED'}"
)

print(
    f"Missing values:       "
    f"{'PASS' if not any_missing else 'CHECK REQUIRED'}"
)

print(
    f"Price sanity:         "
    f"{'PASS' if sanity_passed else 'CHECK REQUIRED'}"
)


# ============================================================
# FINAL DECISION
# ============================================================

if (
    all_same_dates
    and not any_missing
    and sanity_passed
):

    print("\n" + "=" * 65)
    print("PRICE DATA VALIDATION PASSED")
    print("=" * 65)

    print(
        "\nThe price dataset is ready for the backtesting pipeline."
    )

else:

    print("\n" + "=" * 65)
    print("PRICE DATA NEEDS REVIEW")
    print("=" * 65)

    print(
        "\nDo not start the backtest until the issues above "
        "have been reviewed."
    )