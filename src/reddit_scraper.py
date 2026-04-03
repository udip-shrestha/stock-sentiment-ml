from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable
import re

import pandas as pd
import praw

from config import RedditCredentials


WINDOW_TO_DELTA = {
    "2h": timedelta(hours=2),
    "day": timedelta(days=1),
    "week": timedelta(days=7),
}


@dataclass(frozen=True)
class RedditQuery:
    ticker: str
    subreddit: str
    window: str = "day"
    limit: int = 250
    sort: str = "new"


def build_reddit_client(credentials: RedditCredentials) -> praw.Reddit:
    return praw.Reddit(
        client_id=credentials.client_id,
        client_secret=credentials.client_secret,
        user_agent=credentials.user_agent,
    )


def _combine_submission_text(title: str, body: str) -> str:
    return f"{title}\n{body}".strip()


def _clean_text(text: str) -> str:
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _iter_matching_posts(
    reddit: praw.Reddit, query: RedditQuery
) -> Iterable[dict[str, object]]:
    now = datetime.now(timezone.utc)
    cutoff = now - WINDOW_TO_DELTA[query.window]
    subreddit = reddit.subreddit(query.subreddit)

    for submission in subreddit.search(
        query.ticker,
        sort=query.sort,
        time_filter="week",
        limit=query.limit,
    ):
        created_at = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc)
        if created_at < cutoff:
            continue

        text = _clean_text(_combine_submission_text(submission.title, submission.selftext))
        if not text:
            continue

        yield {
            "id": submission.id,
            "ticker": query.ticker.upper(),
            "subreddit": query.subreddit,
            "created_utc": created_at,
            "created_date": created_at.date().isoformat(),
            "title": submission.title,
            "selftext": submission.selftext,
            "text": text,
            "score": int(getattr(submission, "score", 0) or 0),
            "num_comments": int(getattr(submission, "num_comments", 0) or 0),
            "url": submission.url,
        }


def scrape_stock_posts(credentials: RedditCredentials, query: RedditQuery) -> pd.DataFrame:
    if query.window not in WINDOW_TO_DELTA:
        allowed = ", ".join(sorted(WINDOW_TO_DELTA))
        raise ValueError(f"Unsupported window '{query.window}'. Use one of: {allowed}")

    reddit = build_reddit_client(credentials)
    posts = list(_iter_matching_posts(reddit, query))
    if not posts:
        return pd.DataFrame(
            columns=[
                "id",
                "ticker",
                "subreddit",
                "created_utc",
                "created_date",
                "title",
                "selftext",
                "text",
                "score",
                "num_comments",
                "url",
            ]
        )

    return pd.DataFrame(posts).sort_values("created_utc").reset_index(drop=True)