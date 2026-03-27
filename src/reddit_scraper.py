import os
import praw
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")

# Initialize Reddit API
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT
)


def scrape_stock_posts(stock_ticker, subreddit_name="wallstreetbets", limit=500, hours=24):
    posts = []
    
    subreddit = reddit.subreddit(subreddit_name)
    
    time_threshold = datetime.utcnow() - timedelta(hours=hours)

    for submission in subreddit.search(stock_ticker, limit=limit):
        created_time = datetime.utcfromtimestamp(submission.created_utc)

        # Timeframe filter
        if created_time >= time_threshold:
            posts.append({
                "title": submission.title,
                "selftext": submission.selftext,
                "score": submission.score,
                "num_comments": submission.num_comments,
                "created_utc": created_time,
                "text": submission.title + " " + submission.selftext
            })

    df = pd.DataFrame(posts)
    return df


if __name__ == "__main__":
    df = scrape_stock_posts("TSLA", hours=24)
    df.to_csv("data/tsla_recent.csv", index=False)
    print(f"Saved {len(df)} posts.")