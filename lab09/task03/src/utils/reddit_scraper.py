"""Defines the scraper for fetching posts from Reddit based on a search query."""

import os

import pandas as pd
import praw
from dotenv import load_dotenv

load_dotenv()


reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT"),
)


def scrape_reddit(
    query: str,
    limit: int = 100,
    subreddit_name: str = "all",
) -> pd.DataFrame:

    posts: list[dict[str, str]] = []

    subreddit = reddit.subreddit(subreddit_name)

    search_results = subreddit.search(
        query,
        limit=20,
        sort="relevance",
    )

    for submission in search_results:
        # submission title
        posts.append(
            {
                "source": "submission",
                "text": submission.title,
            }
        )

        if len(posts) >= limit:
            break

        submission.comments.replace_more(limit=0)

        for comment in submission.comments.list():
            body = getattr(comment, "body", None)

            if not isinstance(body, str):
                continue

            if len(body.strip()) < 20:
                continue

            posts.append(
                {
                    "source": "comment",
                    "text": body,
                }
            )

            if len(posts) >= limit:
                break

        if len(posts) >= limit:
            break

    return pd.DataFrame(posts)
