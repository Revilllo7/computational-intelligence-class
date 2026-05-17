"""Defines a fallback mechanism for fetching posts from Nitter, Reddit API, and no-API Reddit search."""

import pandas as pd

from .no_api_reddit_scraper import scrape_reddit_fallback
from .reddit_scraper import scrape_reddit
from .scraper import scrape_posts


def get_posts(query: str, limit: int = 100) -> pd.DataFrame:
    try:
        print("Trying Nitter...")

        df = scrape_posts(query, limit)

        if len(df) > 0:
            return df

    except Exception as e:
        print(f"Nitter failed: {e}")

    print("Falling back to Reddit API...")

    try:
        df = scrape_reddit(query, limit)

        if len(df) > 0:
            return df
    except Exception as e:
        print(f"Reddit API failed: {e}")

    print("Falling back to no-API Reddit search...")

    return scrape_reddit_fallback(limit)
