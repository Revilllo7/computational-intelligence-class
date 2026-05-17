"""Lemmatizes text by reducing words to their base form.

For example, "running" becomes "run", and "better" becomes "good"."""

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def lemmatize_text(text: str) -> list[str]:
    """Lemmatizes text by reducing words to their base form.

    For example, "running" becomes "run", and "better" becomes "good"."""
    # Ensure the necessary NLTK resources are downloaded
    nltk.download("punkt", quiet=True)
    nltk.download("wordnet", quiet=True)

    # Initialize the WordNet lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Tokenize the text into words
    tokens = word_tokenize(text)

    # Lemmatize each token and convert to lowercase
    lemmatized_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalpha()]

    return lemmatized_tokens
