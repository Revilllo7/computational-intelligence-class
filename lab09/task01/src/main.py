"""Main script for Task01 - Bag of Words."""

import json
from pathlib import Path

from .utils.process_document import process_document


def run_article_pipeline(
    article_path: str | Path,
    *,
    show_plots: bool = True,
    save_json: str | Path | None = None,
) -> dict[str, int]:
    """Run the processing pipeline on the article at `article_path`.

    Args:
            article_path: path to the article text file.
            show_plots: whether to generate/save plot images.
            save_json: optional path to save the full word-count JSON.

    Returns:
            counts dictionary mapping word -> count
    """
    p = Path(article_path)
    text = p.read_text(encoding="utf-8")

    counts, _tokens = process_document(text, top_k=10, show_plots=show_plots)

    # Print top-10 words
    from collections import Counter

    top10 = Counter(counts).most_common(10)
    print("\nTop 10 words:")
    for word, cnt in top10:
        print(f"{word}: {cnt}")

    # Optionally save full counts to JSON (ordered by count desc)
    if save_json:
        outpath = Path(save_json)
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        outpath.write_text(
            json.dumps(sorted_counts, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return counts


if __name__ == "__main__":
    # Default article path relative to this file
    article = Path(__file__).resolve().parents[1] / "data" / "article.txt"
    run_article_pipeline(
        article,
        show_plots=True,
        save_json=Path(__file__).resolve().parents[1] / "output" / "counts.json",
    )
