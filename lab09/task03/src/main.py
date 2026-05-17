"""Defines the main entry point for the post scraping and emotion analysis task 03."""

from pathlib import Path
from typing import cast

import pandas as pd

from .utils.emotion_analysis import analyse_post
from .utils.fallback import get_posts

TOPIC = "artificial intelligence"
TASK_ROOT = Path(__file__).resolve().parents[1]
RAW_OUTPUT = TASK_ROOT / "data" / "raw" / "posts.csv"
PROCESSED_OUTPUT = TASK_ROOT / "data" / "processed" / "emotions.csv"


def main():
    df = get_posts(TOPIC, limit=100)

    analyses = []

    for _, row in df.iterrows():
        text = cast(str, row["text"])

        result = analyse_post(text)

        analyses.append(
            {
                "text": text,
                "compound": result["compound"],
                "positive": result["positive"],
                "neutral": result["neutral"],
                "negative": result["negative"],
                "t2e": str(result["t2e"]),
                "nrc": str(result["nrc"]),
            }
        )

    result_df = pd.DataFrame(analyses)

    RAW_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(RAW_OUTPUT, index=False)
    result_df.to_csv(PROCESSED_OUTPUT, index=False)

    print("Saved results.")


if __name__ == "__main__":
    main()
