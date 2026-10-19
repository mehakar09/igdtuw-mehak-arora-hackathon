
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


class SentimentEngine:
    def __init__(self):
        model_name = "ProsusAI/finbert"

        print("Loading FinBERT model...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        )
        self.model.eval()

        self.labels = self.model.config.id2label

        print("FinBERT is ready.")

    def predict(self, headline):
        if not isinstance(headline, str) or not headline.strip():
            raise ValueError("Please provide a valid news headline.")

        inputs = self.tokenizer(
            headline,
            return_tensors="pt",
            truncation=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)[0]

        scores = {
            self.labels[i].lower(): probabilities[i].item()
            for i in range(len(probabilities))
        }

        positive = scores.get("positive", 0.0)
        negative = scores.get("negative", 0.0)
        neutral = scores.get("neutral", 0.0)

        return {
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
            "sentiment_score": positive - negative
        }


if __name__ == "__main__":
    engine = SentimentEngine()

    headlines = [
        "Company reports stronger quarterly profits than expected",
        "Company shares fall after disappointing earnings",
        "Company announces its quarterly earnings results"
    ]

    for headline in headlines:
        result = engine.predict(headline)

        print("\nHeadline:", headline)
        print("Positive:", round(result["positive"], 4))
        print("Negative:", round(result["negative"], 4))
        print("Neutral:", round(result["neutral"], 4))
        print("Sentiment score:", round(result["sentiment_score"], 4))
