from __future__ import annotations

import argparse
import json

import joblib
import pandas as pd

from config import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, ensure_directories, get_reddit_credentials
from finance_features import (
    aggregate_daily_sentiment,
    build_modeling_frame,
    download_market_data,
    train_volatility_classifier,
)
from reddit_scraper import RedditQuery, scrape_stock_posts
from text_models import attach_model_predictions, save_sentiment_model, train_sentiment_svm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the stock sentiment ML pipeline.")
    parser.add_argument("--ticker", default="TSLA", help="Stock ticker to analyze.")
    parser.add_argument("--subreddit", default="wallstreetbets", help="Subreddit to scrape.")
    parser.add_argument(
        "--window",
        default="day",
        choices=["2h", "day", "week"],
        help="Reddit collection window.",
    )
    parser.add_argument(
        "--reddit-limit",
        type=int,
        default=250,
        help="Maximum number of Reddit submissions to inspect.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=90,
        help="Historical market lookback window used for downloading stock data.",
    )
    return parser.parse_args()


def _write_frame(frame: pd.DataFrame, path) -> None:
    frame.to_csv(path, index=False)


def _write_metrics(metrics: dict[str, object], filename: str) -> None:
    output_path = MODELS_DIR / filename
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def run_pipeline(
    ticker: str,
    subreddit: str,
    window: str,
    reddit_limit: int,
    lookback_days: int,
) -> dict[str, object]:
    ensure_directories()
    try:
        credentials = get_reddit_credentials()
    except Exception:
        credentials = None


    raw_posts = scrape_stock_posts(
        credentials=credentials,
        query=RedditQuery(
            ticker=ticker,
            subreddit=subreddit,
            window=window,
            limit=reddit_limit,
        ),
    )
    if raw_posts.empty:
        raise ValueError(
            "No Reddit posts matched the query and timeframe. "
            "Try a longer window, different subreddit, or higher post limit."
        )

    raw_posts_path = RAW_DATA_DIR / f"{ticker.lower()}_{window}_reddit_posts.csv"
    _write_frame(raw_posts, raw_posts_path)

    sentiment_artifacts = train_sentiment_svm(raw_posts)
    save_sentiment_model(sentiment_artifacts.model)

    predicted_posts = attach_model_predictions(raw_posts, sentiment_artifacts.model)
    predicted_posts_path = PROCESSED_DATA_DIR / f"{ticker.lower()}_{window}_reddit_scored.csv"
    _write_frame(predicted_posts, predicted_posts_path)

    daily_sentiment = aggregate_daily_sentiment(predicted_posts)
    daily_sentiment_path = PROCESSED_DATA_DIR / f"{ticker.lower()}_daily_sentiment.csv"
    _write_frame(daily_sentiment, daily_sentiment_path)

    market_data = download_market_data(ticker=ticker, lookback_days=lookback_days)
    market_path = RAW_DATA_DIR / f"{ticker.lower()}_market_data.csv"
    _write_frame(market_data, market_path)

    modeling_frame = build_modeling_frame(sentiment_daily=daily_sentiment, market_data=market_data)

    modeling_frame_path = PROCESSED_DATA_DIR / f"{ticker.lower()}_modeling_frame.csv"
    _write_frame(modeling_frame, modeling_frame_path)

    volatility_artifacts = train_volatility_classifier(modeling_frame)
    joblib.dump(volatility_artifacts.model, MODELS_DIR / "volatility_svm.joblib")

    _write_metrics(sentiment_artifacts.metrics, "sentiment_metrics.json")
    _write_metrics(volatility_artifacts.metrics, "volatility_metrics.json")

    return {
        "ticker": ticker,
        "subreddit": subreddit,
        "window": window,
        "reddit_posts": int(len(raw_posts)),
        "daily_rows": int(len(daily_sentiment)),
        "modeling_rows": int(len(modeling_frame)),
        "sentiment_metrics_path": str(MODELS_DIR / "sentiment_metrics.json"),
        "volatility_metrics_path": str(MODELS_DIR / "volatility_metrics.json"),
    }


def main() -> None:
    args = parse_args()
    summary = run_pipeline(
        ticker=args.ticker.upper(),
        subreddit=args.subreddit,
        window=args.window,
        reddit_limit=args.reddit_limit,
        lookback_days=args.lookback_days,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()