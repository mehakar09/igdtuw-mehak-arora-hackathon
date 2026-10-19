
from pathlib import Path

import pandas as pd
import torch

from sentiment_engine import SentimentEngine


ROOT = Path(__file__).resolve().parents[1]
NEWS_FILE = ROOT / "data" / "aligned_news_clean.csv"
OUTPUT_FILE = ROOT / "data" / "scored_news.csv"

BATCH_SIZE = 32
SAVE_EVERY = 1000


def main():
    if not NEWS_FILE.exists():
        raise FileNotFoundError(f"News file not found: {NEWS_FILE}")

    news = pd.read_csv(NEWS_FILE)

    required = {"ticker", "headline", "trading_day"}
    missing = required - set(news.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    news["ticker"] = news["ticker"].astype(str).str.strip()
    news["trading_date"] = pd.to_datetime(
        news["trading_day"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    news["headline"] = news["headline"].fillna("").astype(str).str.strip()

    news = news[
        news["ticker"].ne("")
        & news["headline"].ne("")
        & news["trading_date"].notna()
    ].reset_index(drop=True)

    if news.empty:
        raise ValueError("No usable headlines found.")

    print(f"Headlines to score: {len(news):,}")
    engine = SentimentEngine()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Write a fresh output for this run.
    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    total = len(news)

    for start in range(0, total, BATCH_SIZE):
        batch = news.iloc[start:start + BATCH_SIZE]

        inputs = engine.tokenizer(
            batch["headline"].tolist(),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = engine.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)

        label_map = {
            int(index): str(label).lower()
            for index, label in engine.labels.items()
        }

        scores = []

        for row in probabilities.tolist():
            values = {
                label_map[index]: row[index]
                for index in range(len(row))
            }

            positive = values.get("positive", 0.0)
            negative = values.get("negative", 0.0)
            neutral = values.get("neutral", 0.0)

            scores.append({
                "positive": positive,
                "negative": negative,
                "neutral": neutral,
                "sentiment_score": positive - negative
            })

        output = pd.DataFrame({
            "ticker": batch["ticker"].tolist(),
            "trading_date": batch["trading_date"].tolist(),
            "headline": batch["headline"].tolist(),
            **{
                key: [score[key] for score in scores]
                for key in scores[0]
            }
        })

        output.to_csv(
            OUTPUT_FILE,
            mode="a",
            header=not OUTPUT_FILE.exists(),
            index=False
        )

        completed = min(start + len(batch), total)

        if completed % SAVE_EVERY < BATCH_SIZE or completed == total:
            print(f"Scored {completed:,}/{total:,} headlines")

    print("\nSentiment scoring complete.")
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
