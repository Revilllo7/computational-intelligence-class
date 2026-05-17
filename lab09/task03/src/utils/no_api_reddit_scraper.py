"""Defines a no-API fallback scraper for Reddit search results."""

import time
from typing import Any, cast

import pandas as pd
import requests

HEADERS = {"User-Agent": ("UniversityEmotionAnalysisBot/1.0")}

FALLBACK_SEARCHES: tuple[tuple[str, str], ...] = (
    ("artificial intelligence", "technology"),
    ("anxiety", "mentalhealth"),
    ("frustration", "gaming"),
    ("happiness", "aww"),
    ("depression", "depression"),
    ("loneliness", "socialskills"),
    ("stress", "stress"),
    ("motivation", "GetMotivated"),
    ("sadness", "sad"),
    ("excitement", "excited"),
)


def scrape_reddit(
    query: str,
    subreddit: str = "mentalhealth",
    limit: int = 100,
) -> pd.DataFrame:

    posts: list[dict[str, str]] = []

    url = f"https://www.reddit.com/r/{subreddit}/search.json"

    params = {
        "q": query,
        "restrict_sr": "1",
        "sort": "relevance",
        "limit": 25,
    }

    response = requests.get(
        url,
        headers=HEADERS,
        params=params,
        timeout=10,
    )

    response.raise_for_status()

    data: dict[str, Any] = response.json()

    children = data["data"]["children"]

    for post in children:
        post_data = post["data"]

        title = post_data.get("title")

        selftext = post_data.get("selftext")

        if isinstance(title, str) and len(title) > 10:
            posts.append(
                {
                    "source": "title",
                    "text": title,
                }
            )

        if isinstance(selftext, str) and len(selftext.strip()) > 20:
            posts.append(
                {
                    "source": "body",
                    "text": selftext,
                }
            )

        permalink = post_data.get("permalink")

        if isinstance(permalink, str):
            comments_url = f"https://www.reddit.com{permalink}.json"

            try:
                comments_response = requests.get(
                    comments_url,
                    headers=HEADERS,
                    timeout=10,
                )

                comments_data = comments_response.json()

                comments = comments_data[1]["data"]["children"]

                for comment in comments:
                    comment_data = comment.get("data", {})

                    body = comment_data.get("body")

                    if isinstance(body, str) and len(body.strip()) > 20:
                        posts.append(
                            {
                                "source": "comment",
                                "text": body,
                            }
                        )

                    if len(posts) >= limit:
                        break

            except Exception:
                continue

        if len(posts) >= limit:
            break

        time.sleep(1)

    return pd.DataFrame(posts[:limit])


def scrape_reddit_fallback(limit: int = 100) -> pd.DataFrame:
    """Collect posts from a fixed set of subreddit/query pairs."""

    posts: list[dict[str, Any]] = []

    for query, subreddit in FALLBACK_SEARCHES:
        remaining = limit - len(posts)

        if remaining <= 0:
            break

        try:
            frame = scrape_reddit(
                query=query,
                subreddit=subreddit,
                limit=remaining,
            )
        except Exception as exc:
            print(f"No-API Reddit fallback failed for {subreddit}/{query}: {exc}")
            continue

        posts.extend(cast(list[dict[str, Any]], frame.to_dict(orient="records")))

    return pd.DataFrame(posts[:limit])
