
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "reaction_fingerprint.csv"
OUTPUT_DIR = ROOT / "docs"


def main():
    data = pd.read_csv(INPUT_FILE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for signal in ["sentiment_score", "sentiment_change"]:
        subset = data[data["signal"] == signal]

        table = subset.pivot(
            index="ticker",
            columns="lag_trading_days",
            values="correlation"
        )

        table = table.sort_index()

        fig, ax = plt.subplots(figsize=(8, 5))

        image = ax.imshow(
            table.values,
            aspect="auto",
            cmap="coolwarm",
            vmin=-0.2,
            vmax=0.2
        )

        ax.set_xticks(range(len(table.columns)))
        ax.set_xticklabels(
            [f"{lag} day(s)" for lag in table.columns]
        )
        ax.set_yticks(range(len(table.index)))
        ax.set_yticklabels(table.index)

        ax.set_title(
            f"Market Reaction Fingerprint: {signal.replace('_', ' ').title()}"
        )
        ax.set_xlabel("Future trading-session horizon")
        ax.set_ylabel("Stock ticker")

        for i in range(len(table.index)):
            for j in range(len(table.columns)):
                value = table.iloc[i, j]
                if pd.notna(value):
                    ax.text(
                        j, i, f"{value:.2f}",
                        ha="center", va="center", fontsize=8
                    )

        fig.colorbar(image, ax=ax, label="Correlation")
        fig.tight_layout()

        output_file = OUTPUT_DIR / f"fingerprint_{signal}.png"
        fig.savefig(output_file, dpi=200)
        plt.close(fig)

        print("Saved:", output_file)


if __name__ == "__main__":
    main()
