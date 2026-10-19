
from pathlib import Path
import json
import time

import pandas as pd
from datasets import load_dataset


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
PRICE_DIR = DATA_DIR / "prices"

OUTPUT_FILE = DATA_DIR / "aligned_news.csv"
CHECKPOINT_FILE = DATA_DIR / "alignment_checkpoint.json"

DATASET_ID = "sabareesh88/FNSPID_nasdaq"
DATASET_SPLIT = "train"

STOCKS = {
    "NVDA", "AAPL", "GE", "CAT", "PG", "KO",
    "WMT", "CVX", "XOM", "DIS", "LLY"
}

MARKET_TZ = "America/New_York"
SAVE_EVERY_ROWS = 5000
RETRY_LIMIT = 8

TICKER_FIELDS = [
    "Stock_symbol", "stock_symbol", "ticker",
    "Ticker", "symbol", "Symbol"
]

HEADLINE_FIELDS = [
    "Article_title", "Headline", "headline",
    "title", "Title", "article_title"
]

TIMESTAMP_FIELDS = [
    "Date", "date", "timestamp",
    "published_at", "publishedAt", "datetime"
]

OUTPUT_COLUMNS = [
    "ticker", "news_timestamp", "headline", "trading_day"
]


def normalize_name(name):
    return "".join(
        character.lower()
        for character in str(name)
        if character.isalnum()
    )


def get_field(row, candidates):
    available = {
        normalize_name(key): key for key in row.keys()
    }

    for candidate in candidates:
        actual_key = available.get(normalize_name(candidate))

        if actual_key is None:
            continue

        value = row[actual_key]

        if value is None:
            continue

        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass

        text = str(value).strip()

        if text and text.lower() not in {"nan", "none", "null"}:
            return value

    return None


def parse_timestamp(value):
    if value is None:
        return pd.NaT

    try:
        timestamp = pd.Timestamp(value)

        if pd.isna(timestamp):
            return pd.NaT

        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")

        return timestamp

    except (ValueError, TypeError, OverflowError):
        return pd.NaT


def load_trading_dates(ticker):
    path = PRICE_DIR / f"{ticker}.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing price file: {path}")

    prices = pd.read_csv(path)

    date_column = next(
        (
            column for column in prices.columns
            if normalize_name(column) in {
                "date", "datetime", "timestamp"
            }
        ),
        None
    )

    if date_column is None:
        raise ValueError(f"No date column found in {path}")

    # Daily price dates are exchange-session labels.
    # Preserve the date instead of converting midnight UTC to Eastern.
    dates = pd.to_datetime(
        prices[date_column], errors="coerce"
    ).dropna().dt.normalize()

    dates = pd.DatetimeIndex(dates.unique()).sort_values()

    if len(dates) == 0:
        raise ValueError(f"No valid trading dates in {path}")

    return dates


def assign_trading_day(timestamp, trading_dates):
    local_time = timestamp.tz_convert(MARKET_TZ)
    local_date = pd.Timestamp(local_time.date())

    if local_date in trading_dates and local_time.time() < pd.Timestamp(
        "16:00"
    ).time():
        return local_date

    # After close: next session.
    # Weekend/holiday: first session after that calendar date.
    side = "right" if local_date in trading_dates else "left"
    next_index = trading_dates.searchsorted(local_date, side=side)

    if next_index >= len(trading_dates):
        return pd.NaT

    return trading_dates[next_index]


def load_checkpoint():
    if not CHECKPOINT_FILE.exists():
        return 0

    try:
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as file:
            checkpoint = json.load(file)

        if checkpoint.get("dataset_id") != DATASET_ID:
            raise ValueError("Checkpoint belongs to another dataset.")

        return int(checkpoint.get("scanned_rows", 0))

    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Could not read checkpoint {CHECKPOINT_FILE}: {exc}. "
            "Do not delete it until you have checked your saved CSV."
        ) from exc


def save_checkpoint(scanned_rows):
    temporary_file = CHECKPOINT_FILE.with_suffix(".tmp")

    with open(temporary_file, "w", encoding="utf-8") as file:
        json.dump(
            {
                "dataset_id": DATASET_ID,
                "split": DATASET_SPLIT,
                "scanned_rows": scanned_rows
            },
            file
        )

    temporary_file.replace(CHECKPOINT_FILE)


def save_rows(rows, scanned_rows):
    if not rows:
        save_checkpoint(scanned_rows)
        return

    frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    # Append completed batches so they survive a connection failure.
    write_header = not OUTPUT_FILE.exists() or OUTPUT_FILE.stat().st_size == 0

    frame.to_csv(
        OUTPUT_FILE,
        mode="a",
        header=write_header,
        index=False,
        encoding="utf-8"
    )

    rows.clear()
    save_checkpoint(scanned_rows)

    print(
        f"Checkpoint saved | scanned: {scanned_rows:,} | "
        f"saved rows: {sum(1 for _ in [])}"
    )


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading trading calendars...")
    calendars = {
        ticker: load_trading_dates(ticker)
        for ticker in sorted(STOCKS)
    }

    for ticker, dates in calendars.items():
        print(f"[OK] {ticker}: {len(dates)} trading sessions")

    # Resume only if both the checkpoint and partial CSV are available.
    scanned_rows = load_checkpoint()

    if scanned_rows and not OUTPUT_FILE.exists():
        raise RuntimeError(
            "Checkpoint exists but aligned_news.csv is missing. "
            "Cannot safely resume. Inspect the checkpoint before proceeding."
        )

    print(f"\nStarting scan at dataset row {scanned_rows:,}.")
    print("A broken connection will trigger an automatic retry.")

    retries = 0

    while retries <= RETRY_LIMIT:
        rows = []
        rows_since_save = 0
        current_row = 0

        try:
            dataset = load_dataset(
                DATASET_ID,
                split=DATASET_SPLIT,
                streaming=True
            )

            for row in dataset:
                current_row += 1

                # Replay earlier rows after reconnecting.
                if current_row <= scanned_rows:
                    if current_row % 250_000 == 0:
                        print(f"Resuming: skipped {current_row:,} rows")
                    continue

                ticker_value = get_field(row, TICKER_FIELDS)

                if ticker_value is not None:
                    ticker = str(ticker_value).strip().upper()

                    if ticker in STOCKS:
                        headline_value = get_field(row, HEADLINE_FIELDS)
                        timestamp_value = get_field(row, TIMESTAMP_FIELDS)

                        if headline_value is not None:
                            timestamp = parse_timestamp(timestamp_value)

                            if not pd.isna(timestamp):
                                trading_day = assign_trading_day(
                                    timestamp, calendars[ticker]
                                )

                                if not pd.isna(trading_day):
                                    rows.append({
                                        "ticker": ticker,
                                        "news_timestamp": timestamp.isoformat(),
                                        "headline": " ".join(
                                            str(headline_value).split()
                                        ),
                                        "trading_day": (
                                            trading_day.strftime("%Y-%m-%d")
                                        )
                                    })

                                    rows_since_save += 1

                # Save periodically even if few matching rows appear.
                if rows_since_save >= SAVE_EVERY_ROWS:
                    save_rows(rows, current_row)
                    scanned_rows = current_row
                    rows_since_save = 0

                if current_row % 250_000 == 0:
                    print(
                        f"Scanned {current_row:,} rows | "
                        f"current batch: {len(rows):,}"
                    )

            # Stream reached the end.
            save_rows(rows, current_row)
            scanned_rows = current_row

            print("\nStream completed. Removing duplicate records...")

            if not OUTPUT_FILE.exists():
                raise RuntimeError("No aligned articles were saved.")

            result = pd.read_csv(OUTPUT_FILE)
            result = result.drop_duplicates(
                subset=[
                    "ticker", "news_timestamp",
                    "headline", "trading_day"
                ]
            )

            result = result.sort_values(
                ["trading_day", "ticker", "news_timestamp"]
            ).reset_index(drop=True)

            result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

            if result.empty:
                raise RuntimeError("No usable aligned articles were found.")

            if not result["headline"].fillna("").str.strip().ne("").all():
                raise RuntimeError("Empty headlines remain in the output.")

            print("\nALIGNMENT COMPLETE")
            print(f"Dataset rows scanned: {current_row:,}")
            print(f"Final aligned articles: {len(result):,}")
            print(f"Unique tickers: {result['ticker'].nunique()}")
            print("\nArticles per ticker:")
            print(result.groupby("ticker").size().to_string())
            print(f"\nSaved to: {OUTPUT_FILE}")
            return

        except KeyboardInterrupt:
            # Preserve the current batch before exiting.
            save_rows(rows, current_row)
            print("\nStopped by user. Saved progress is available to resume.")
            return

        except Exception as exc:
            # Save all collected rows and the last completed position.
            save_rows(rows, current_row)
            scanned_rows = current_row

            retries += 1
            if retries > RETRY_LIMIT:
                print(f"\nRetry limit reached: {exc}")
                raise

            wait_seconds = min(10 * retries, 60)
            print(
                f"\nConnection or stream error: {exc}\n"
                f"Retry {retries}/{RETRY_LIMIT} in {wait_seconds} seconds..."
            )
            time.sleep(wait_seconds)

    raise RuntimeError("News alignment did not complete.")


if __name__ == "__main__":
    main()
