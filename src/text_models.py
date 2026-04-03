from __future__ import annotations

from dataclasses import dataclass
import re

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import MODELS_DIR


LABEL_ORDER = ["negative", "neutral", "positive"]


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9$\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def weak_label_with_vader(posts: pd.DataFrame) -> pd.DataFrame:
    analyzer = SentimentIntensityAnalyzer()
    labeled = posts.copy()
    labeled["clean_text"] = labeled["text"].fillna("").map(normalize_text)
    labeled["vader_compound"] = labeled["clean_text"].map(
        lambda text: analyzer.polarity_scores(text)["compound"]
    )

    def assign_label(score: float) -> str:
        if score >= 0.05:
            return "positive"
        if score <= -0.05:
            return "negative"
        return "neutral"

    labeled["sentiment_label"] = labeled["vader_compound"].map(assign_label)
    return labeled


@dataclass
class SentimentModelArtifacts:
    model: Pipeline
    metrics: dict[str, object]


def train_sentiment_svm(posts: pd.DataFrame, min_class_count: int = 2) -> SentimentModelArtifacts:
    labeled = weak_label_with_vader(posts)
    class_counts = labeled["sentiment_label"].value_counts()
    valid_labels = class_counts[class_counts >= min_class_count].index.tolist()
    trainable = labeled[labeled["sentiment_label"].isin(valid_labels)].copy()

    if trainable.empty or trainable["sentiment_label"].nunique() < 2:
        raise ValueError(
            "Not enough label diversity to train the sentiment SVM. "
            "Collect more Reddit posts or expand the timeframe."
        )

    stratify = trainable["sentiment_label"] if trainable["sentiment_label"].nunique() > 1 else None
    test_size = 0.25 if len(trainable) >= 8 else 0.5

    X_train, X_test, y_train, y_test = train_test_split(
        trainable["clean_text"],
        trainable["sentiment_label"],
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )

    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95)),
            ("svm", LinearSVC(C=1.0)),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    report = classification_report(
        y_test,
        predictions,
        labels=[label for label in LABEL_ORDER if label in y_test.unique()],
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "num_examples": int(len(trainable)),
        "label_counts": class_counts.to_dict(),
        "classification_report": report,
    }
    return SentimentModelArtifacts(model=model, metrics=metrics)


def attach_model_predictions(posts: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
    enriched = weak_label_with_vader(posts)
    enriched["svm_sentiment"] = model.predict(enriched["clean_text"])

    label_to_score = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}
    enriched["svm_sentiment_score"] = enriched["svm_sentiment"].map(label_to_score).astype(float)
    return enriched


def save_sentiment_model(model: Pipeline, filename: str = "sentiment_svm.joblib") -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / filename)