import os
import pandas as pd
import yfinance as yf


# ============================================================
# CONFIGURATION
# ============================================================

UNIVERSE_FILE = "../data/stock_universe.csv"
OUTPUT_DIR = "../data/prices"

# FNSPID news data covers historical periods.
# We download a wider price window so we have enough
# information for return calculations and lag analysis.
START_DATE = "2018-01-01"
END_DATE = "2024-01-01"


# ============================================================
# LOAD STOCK UNIVERSE
# ============================================================

print("=" * 60)
print("LOADING STOCK UNIVERSE")
print("=" * 60)

universe = pd.read_csv(UNIVERSE_FILE)

stocks = universe["ticker"].dropna().unique().tolist()

print("\nStocks selected:")
print(", ".join(stocks))

print(f"\nTotal stocks: {len(stocks)}")


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# DOWNLOAD HISTORICAL PRICE DATA
# ============================================================

print("\n" + "=" * 60)
print("DOWNLOADING HISTORICAL PRICE DATA")
print("=" * 60)

successful = []
failed = []

for ticker in stocks:

    print(f"\nDownloading {ticker}...")

    try:

        data = yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE,
            auto_adjust=True,
            progress=False
        )

        if data.empty:
            print(f"WARNING: No price data found for {ticker}")
            failed.append(ticker)
            continue

        # Handle MultiIndex columns returned by newer yfinance versions
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.reset_index()

        # Keep only useful columns
        required_columns = [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        available_columns = [
            col for col in required_columns
            if col in data.columns
        ]

        data = data[available_columns]

        # Make sure Date is clean
        data["Date"] = pd.to_datetime(data["Date"])

        # Remove timezone if present
        if data["Date"].dt.tz is not None:
            data["Date"] = data["Date"].dt.tz_localize(None)

        # Sort chronologically
        data = data.sort_values("Date")

        # Save individual stock file
        output_file = os.path.join(
            OUTPUT_DIR,
            f"{ticker}.csv"
        )

        data.to_csv(output_file, index=False)

        print(
            f"Saved {ticker}: "
            f"{len(data)} trading days | "
            f"{data['Date'].min().date()} -> "
            f"{data['Date'].max().date()}"
        )

        successful.append(ticker)

    except Exception as e:

        print(f"ERROR downloading {ticker}: {e}")
        failed.append(ticker)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("DOWNLOAD COMPLETE")
print("=" * 60)

print(f"\nSuccessful: {len(successful)}")
print("Stocks:", ", ".join(successful))

if failed:
    print(f"\nFailed: {len(failed)}")
    print("Stocks:", ", ".join(failed))

print("\nPrice files saved to:")
print(OUTPUT_DIR)