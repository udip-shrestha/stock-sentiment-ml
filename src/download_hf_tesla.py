from pathlib import Path
import pandas as pd

URL = "https://huggingface.co/datasets/emilpartow/reddit_finance_posts_sp500/resolve/main/00_combined.csv"
OUT = Path("data/raw/reddit_posts_seed.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

rows = []
limit = 300

for chunk in pd.read_csv(URL, chunksize=50000):
    tesla = chunk[chunk["company"].astype(str).str.lower() == "tesla"].copy()
    if tesla.empty:
        continue

    tesla["ticker"] = "TSLA"
    tesla["created_date"] = pd.to_datetime(tesla["created_datetime"], errors="coerce").dt.strftime("%Y-%m-%d")
    tesla["selftext"] = tesla["text"].fillna("")
    tesla["text"] = (tesla["title"].fillna("") + " " + tesla["selftext"]).str.strip()

    keep = tesla.rename(
        columns={
            "id": "id",
            "subreddit": "subreddit",
            "created_datetime": "created_utc",
            "score": "score",
            "num_comments": "num_comments",
            "url": "url",
            "title": "title",
        }
    )[
        [
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
    ]

    rows.append(keep)

    current_total = sum(len(x) for x in rows)
    print(f"Collected {current_total} Tesla rows so far...")

    if current_total >= limit:
        break

df = pd.concat(rows, ignore_index=True).head(limit)
df.to_csv(OUT, index=False)
print(f"Saved {len(df)} Tesla rows to {OUT}")
