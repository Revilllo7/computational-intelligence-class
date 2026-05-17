"""Defines the scraper for fetching posts from Nitter based on a search query."""

import time
from dataclasses import dataclass
from urllib.parse import quote

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag

NITTER_INSTANCE = "https://nitter.poast.org"


@dataclass
class Post:
    username: str
    date: str
    text: str


def scrape_posts(query: str, limit: int = 100) -> pd.DataFrame:
    posts: list[dict[str, str]] = []

    encoded_query = quote(query)

    url: str | None = f"{NITTER_INSTANCE}/search?f=tweets&q={encoded_query}"

    headers = {"User-Agent": "Mozilla/5.0"}

    while len(posts) < limit and url:
        session = requests.Session()
        response = session.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            print("Failed:", response.status_code)
            break

        soup = BeautifulSoup(response.text, "lxml")

        timeline: list[Tag] = soup.find_all("div", class_="timeline-item")

        for item in timeline:
            try:
                text_div = item.find("div", class_="tweet-content")

                user = item.find("a", class_="username")

                date = item.find("span", class_="tweet-date")

                if not isinstance(text_div, Tag):
                    continue

                if not isinstance(user, Tag):
                    continue

                if not isinstance(date, Tag):
                    continue

                text = text_div.get_text(strip=True)
                username = user.get_text(strip=True)
                date_text = date.get_text(strip=True)

                posts.append(
                    {
                        "username": username,
                        "date": date_text,
                        "text": text,
                    }
                )

                if len(posts) >= limit:
                    break

            except Exception as e:
                print(f"Error parsing item: {e}")
                continue

        # next page
        next_page = soup.find("div", class_="show-more")

        if isinstance(next_page, Tag):
            next_link = next_page.find("a")

            if isinstance(next_link, Tag):
                href = next_link.get("href")

                if isinstance(href, str):
                    url = NITTER_INSTANCE + href
                else:
                    break
            else:
                break
        else:
            break

        time.sleep(1)

    return pd.DataFrame(posts)
