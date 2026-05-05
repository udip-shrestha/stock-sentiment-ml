import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
PROCESSED = ROOT / "data" / "processed"

with open(MODELS / "sentiment_metrics.json", "r", encoding="utf-8") as f:
    sentiment_metrics = json.load(f)

with open(MODELS / "volatility_metrics.json", "r", encoding="utf-8") as f:
    volatility_metrics = json.load(f)

daily_sentiment = pd.read_csv(PROCESSED / "tsla_daily_sentiment.csv")

fig_dir = ROOT / "data" / "processed"
fig_dir.mkdir(parents=True, exist_ok=True)

# 1. Sentiment label distribution
label_counts = pd.Series(sentiment_metrics["label_counts"])
plt.figure(figsize=(7, 4))
sns.barplot(x=label_counts.index, y=label_counts.values, palette="Set2")
plt.title("Sentiment Label Distribution")
plt.xlabel("Sentiment Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(fig_dir / "sentiment_label_distribution.png", dpi=200)
plt.close()

# 2. Sentiment model F1 by class
sent_report = sentiment_metrics["classification_report"]
sent_f1 = pd.Series({
    "negative": sent_report["negative"]["f1-score"],
    "neutral": sent_report["neutral"]["f1-score"],
    "positive": sent_report["positive"]["f1-score"],
})
plt.figure(figsize=(7, 4))
sns.barplot(x=sent_f1.index, y=sent_f1.values, palette="coolwarm")
plt.title("Sentiment Model F1-Score by Class")
plt.xlabel("Sentiment Class")
plt.ylabel("F1-Score")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(fig_dir / "sentiment_f1_by_class.png", dpi=200)
plt.close()

# 3. Daily sentiment over time
daily_sentiment["date"] = pd.to_datetime(daily_sentiment["date"])
plt.figure(figsize=(10, 4))
sns.lineplot(data=daily_sentiment, x="date", y="avg_sentiment", marker="o")
plt.title("Average Daily Sentiment Over Time")
plt.xlabel("Date")
plt.ylabel("Average Sentiment")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(fig_dir / "daily_sentiment_over_time.png", dpi=200)
plt.close()

# 4. Volatility model metrics
vol_report = volatility_metrics["classification_report"]
vol_metrics = pd.Series({
    "accuracy": vol_report["accuracy"],
    "macro_f1": vol_report["macro avg"]["f1-score"],
    "weighted_f1": vol_report["weighted avg"]["f1-score"],
})
plt.figure(figsize=(7, 4))
sns.barplot(x=vol_metrics.index, y=vol_metrics.values, palette="magma")
plt.title("Volatility Model Performance")
plt.xlabel("Metric")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(fig_dir / "volatility_model_metrics.png", dpi=200)
plt.close()

print("Saved plots to:", fig_dir)
