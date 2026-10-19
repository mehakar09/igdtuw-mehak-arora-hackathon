from datasets import load_dataset

print("Loading a small streaming sample...")

dataset = load_dataset(
    "sabareesh88/FNSPID_nasdaq",
    split="train",
    streaming=True
)

for i, row in enumerate(dataset):
    print("\n--- ROW", i + 1, "---")
    print("Date:", row["Date"])
    print("Ticker:", row["Stock_symbol"])
    print("Headline:", row["Article_title"])

    if i == 9:
        break