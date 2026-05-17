"""Process a document: tokenize, remove stop words, lemmatize, count and plot.

This module implements a simple pipeline and returns a frequency mapping
so callers can reuse the results programmatically. It also plots the top-N
frequent words and creates a word cloud (uses existing `create_word_cloud`).
"""

import contextlib
from collections import Counter
from pathlib import Path

OUTPUT_DIR = Path("output")


def _ensure_nltk_resources() -> None:
    """Download the NLTK corpora required by the text pipeline if missing."""
    import nltk

    resources = (
        ("punkt_tab", "tokenizers/punkt_tab/english/"),
        ("punkt", "tokenizers/punkt"),
        ("stopwords", "corpora/stopwords"),
        ("wordnet", "corpora/wordnet"),
    )
    for resource_name, resource_path in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_name, quiet=True)


def process_document(
    text: str, top_k: int = 10, show_plots: bool = True
) -> tuple[dict[str, int], list[str]]:
    """Process the document and return a word-count vector + final token list.

    Steps performed:
    1. Tokenize text using `tokenize_text_punct`.
    2. Remove English stop words using `remove_stop_words`.
    3. Lemmatize tokens using NLTK `WordNetLemmatizer` (see `src/utils/lemmatize.py`).
    4. Count token frequencies with `collections.Counter`.
    5. Optionally plot top-k bar chart and show a word cloud.

    Returns:
        (counts_dict, final_tokens)
    """

    # Local imports to avoid heavy imports at module load time
    from .lemmatize import lemmatize_text
    from .stop_words import remove_stop_words
    from .tokenize import tokenize_text_punct
    from .word_cloud import create_word_cloud

    _ensure_nltk_resources()

    # 1. Tokenize
    tokens = tokenize_text_punct(text)
    token_count = len(tokens)

    # 2. Remove stop words
    tokens_no_stop = remove_stop_words(tokens)
    no_stop_count = len(tokens_no_stop)

    # 3. Lemmatize (NLTK WordNetLemmatizer is used in lemmatize_text)
    #    We lemmatize the no-stop tokens by joining them to text input for the
    #    existing `lemmatize_text` helper which tokenizes internally.
    lemmatized_tokens = lemmatize_text(" ".join(tokens_no_stop))
    lemmatized_count = len(lemmatized_tokens)

    # 4. Count frequencies
    counts = Counter(lemmatized_tokens)

    # 5. Plot top-k bar chart and create a word cloud if requested
    if show_plots:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        try:
            # Use a non-interactive backend to avoid blocking in headless CI/servers
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            most_common = counts.most_common(top_k)
            if most_common:
                words, freqs = zip(*most_common, strict=False)
            else:
                words, freqs = (), ()

            plt.figure(figsize=(10, 5))
            plt.bar(words, freqs, color="tab:blue")
            plt.xlabel("Words")
            plt.ylabel("Count")
            plt.title(f"Top {top_k} most frequent words")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_DIR}/top_words.png")
            plt.close()
        except Exception:
            # continue silently
            pass
        with contextlib.suppress(Exception):
            create_word_cloud(lemmatized_tokens)

    # Print counts after each major step for quick inspection (non-verbose)
    print(f"Token count: {token_count}")
    print(f"After stop-word removal: {no_stop_count}")
    print(f"After lemmatization: {lemmatized_count}")

    return dict(counts), lemmatized_tokens
