from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


@dataclass(frozen=True)
class RedditCredentials:
    client_id: str
    client_secret: str
    user_agent: str


def load_environment() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


def get_reddit_credentials() -> RedditCredentials:
    load_environment()
    client_id = os.getenv("REDDIT_CLIENT_ID", "")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
    user_agent = os.getenv("REDDIT_USER_AGENT", "")

    missing = [
        name
        for name, value in (
            ("REDDIT_CLIENT_ID", client_id),
            ("REDDIT_CLIENT_SECRET", client_secret),
            ("REDDIT_USER_AGENT", user_agent),
        )
        if not value
    ]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(
            f"Missing Reddit credentials in .env: {missing_text}. "
            "Copy .env.example to .env and fill in the values."
        )

    return RedditCredentials(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )


def ensure_directories() -> None:
    for directory in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)