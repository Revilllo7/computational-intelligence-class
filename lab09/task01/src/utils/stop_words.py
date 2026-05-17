"""Removes stop words from a list of tokens."""

from nltk.corpus import stopwords


def remove_stop_words(tokens: list[str]) -> list[str]:
    """Removes stop words from a list of tokens."""
    # Ensure the necessary NLTK resources are downloaded
    import nltk

    nltk.download("stopwords", quiet=True)

    # Get the set of English stop words
    stop_words = set(stopwords.words("english"))

    # Filter out stop words from the list of tokens
    filtered_tokens = [token for token in tokens if token not in stop_words]

    return filtered_tokens
