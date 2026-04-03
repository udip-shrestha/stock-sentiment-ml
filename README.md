 Stock Sentiment ML

This project analyzes whether Reddit discussions about a stock contain signals that relate to the stock's next-day volatility.

It is organized around two supervised learning tasks:

1. Sentiment classification on Reddit posts using TF-IDF features and an SVM.
2. Volatility classification using aggregated daily sentiment features plus market features.

## Folder structure

```text
stock-sentiment-ml/
├── data/
│   ├── processed/
│   └── raw/
├── notebooks/
├── src/
│   ├── config.py
│   ├── finance_features.py
│   ├── reddit_scraper.py
│   ├── run_pipeline.py
│   └── text_models.py
├── .env
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your Reddit API credentials.

## Run

```bash
python src/run_pipeline.py --ticker TSLA --subreddit wallstreetbets --window week --lookback-days 90
```

## What the pipeline does

- scrapes Reddit submissions mentioning a ticker within a chosen timeframe
- weakly labels sentiment with VADER
- trains an SVM sentiment classifier
- aggregates daily sentiment features
- downloads stock data with `yfinance`
- creates a next-day high-vs-low volatility label
- trains a second SVM classifier for volatility
- saves datasets and model artifacts for analysis

## Suggested next steps

- add a notebook in `notebooks/` for EDA and plots
- replace weak labels with a hand-labeled sentiment dataset
- compare `2h`, `1d`, and `1w` Reddit windows
- test weighting posts by upvotes and comment counts