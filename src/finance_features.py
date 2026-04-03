from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def aggregate_daily_sentiment(posts: pd.DataFrame) -> pd.DataFrame:
    if posts.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "post_count",
                "avg_sentiment",
                "sentiment_std",
                "avg_vader",
                "engagement_weighted_sentiment",
                "avg_score",
                "avg_comments",
            ]
        )

    enriched = posts.copy()
    enriched["date"] = pd.to_datetime(enriched["created_date"])
    enriched["engagement_weight"] = (
        enriched["score"].clip(lower=0) + enriched["num_comments"].clip(lower=0) + 1
    )
    enriched["weighted_sentiment"] = enriched["svm_sentiment_score"] * enriched["engagement_weight"]

    aggregated = (
        enriched.groupby("date", as_index=False)
        .agg(
            post_count=("id", "count"),
            avg_sentiment=("svm_sentiment_score", "mean"),
            sentiment_std=("svm_sentiment_score", "std"),
            avg_vader=("vader_compound", "mean"),
            weighted_sentiment_sum=("weighted_sentiment", "sum"),
            engagement_sum=("engagement_weight", "sum"),
            avg_score=("score", "mean"),
            avg_comments=("num_comments", "mean"),
        )
        .sort_values("date")
    )
    aggregated["sentiment_std"] = aggregated["sentiment_std"].fillna(0.0)
    aggregated["engagement_weighted_sentiment"] = (
        aggregated["weighted_sentiment_sum"] / aggregated["engagement_sum"]
    )
    return aggregated.drop(columns=["weighted_sentiment_sum", "engagement_sum"])


def download_market_data(ticker: str, lookback_days: int) -> pd.DataFrame:
    history = yf.download(
        tickers=ticker,
        period=f"{lookback_days}d",
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if history.empty:
        raise ValueError(f"No market data returned for ticker '{ticker}'.")

    history = history.reset_index()
    history["date"] = pd.to_datetime(history["Date"]).dt.tz_localize(None)
    history["return_1d"] = history["Close"].pct_change()
    history["realized_vol_5d"] = history["return_1d"].rolling(5).std()
    history["volume_log"] = np.log1p(history["Volume"])
    history["next_day_volatility"] = history["realized_vol_5d"].shift(-1)
    threshold = history["next_day_volatility"].median(skipna=True)
    history["high_volatility_next_day"] = (
        history["next_day_volatility"] > threshold
    ).astype(int)
    return history[
        [
            "date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "return_1d",
            "realized_vol_5d",
            "volume_log",
            "next_day_volatility",
            "high_volatility_next_day",
        ]
    ].copy()


def build_modeling_frame(sentiment_daily: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
    frame = (
        market_data.merge(sentiment_daily, on="date", how="left")
        .sort_values("date")
        .reset_index(drop=True)
    )
    fill_zero_columns = [
        "post_count",
        "avg_sentiment",
        "sentiment_std",
        "avg_vader",
        "engagement_weighted_sentiment",
        "avg_score",
        "avg_comments",
    ]
    for column in fill_zero_columns:
        if column in frame:
            frame[column] = frame[column].fillna(0.0)

    frame["lag_return_1d"] = frame["return_1d"].shift(1)
    frame["lag_realized_vol_5d"] = frame["realized_vol_5d"].shift(1)
    return frame.dropna(
        subset=["high_volatility_next_day", "lag_return_1d", "lag_realized_vol_5d"]
    ).copy()


@dataclass
class VolatilityModelArtifacts:
    model: Pipeline
    metrics: dict[str, object]


def train_volatility_classifier(modeling_frame: pd.DataFrame) -> VolatilityModelArtifacts:
    feature_columns = [
        "post_count",
        "avg_sentiment",
        "sentiment_std",
        "avg_vader",
        "engagement_weighted_sentiment",
        "avg_score",
        "avg_comments",
        "lag_return_1d",
        "lag_realized_vol_5d",
        "volume_log",
    ]
    available_columns = [column for column in feature_columns if column in modeling_frame.columns]
    X = modeling_frame[available_columns]
    y = modeling_frame["high_volatility_next_day"].astype(int)

    if y.nunique() < 2 or len(modeling_frame) < 8:
        raise ValueError(
            "Not enough market observations to train the volatility classifier. "
            "Increase lookback_days or use a ticker with longer history."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                available_columns,
            )
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessing),
            ("svm", SVC(kernel="rbf", C=1.0, gamma="scale")),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    metrics = {
        "num_examples": int(len(modeling_frame)),
        "feature_columns": available_columns,
        "classification_report": report,
    }
    return VolatilityModelArtifacts(model=model, metrics=metrics)