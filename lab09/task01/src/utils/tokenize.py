"""Tokenizes text into words and sentences.

Splits text into alphabetic and non-alphabetic tokens,
drops punctuation, and converts to lowercase."""

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize


def tokenize_text_punct(text: str) -> list[str]:
    """Tokenizes text into words and sentences.

    Splits text into alphabetic and non-alphabetic tokens,
    drops punctuation, and converts to lowercase."""
    # Ensure the necessary NLTK resources are downloaded
    nltk.download("punkt", quiet=True)

    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Tokenize each sentence into words, filter out non-alphabetic tokens, and convert to lowercase
    tokens = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        filtered_words = [word.lower() for word in words if word.isalpha()]
        tokens.extend(filtered_words)

    return tokens
