"""Sentiment analysis helpers for task02."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypedDict, cast


class VaderScores(TypedDict):
    neg: float
    neu: float
    pos: float
    compound: float


class TextBlobScores(TypedDict):
    polarity: float
    subjectivity: float


class Text2EmotionScores(TypedDict):
    emotions: dict[str, float]
    dominant_emotion: str | None


class TopEmotion(TypedDict):
    emotion: str
    score: float


class NRClexScores(TypedDict):
    raw_emotion_scores: dict[str, int]
    affect_frequencies: dict[str, float]
    top_emotions: list[TopEmotion]
    dominant_emotion: str | None


class OpinionAnalysis(TypedDict):
    vader: VaderScores
    textblob: TextBlobScores
    text2emotion: Text2EmotionScores
    nrclex: NRClexScores


def _ensure_vader_lexicon() -> None:
    """Make sure the Vader lexicon is available for NLTK sentiment scoring."""

    import nltk

    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)


def _ensure_text2emotion_compatibility() -> None:
    """Patch the emoji API expected by text2emotion on newer emoji releases."""

    import emoji

    if not hasattr(emoji, "UNICODE_EMOJI"):
        setattr(emoji, "UNICODE_EMOJI", getattr(emoji, "EMOJI_DATA", {}))  # noqa: B010


def _dominant_score(scores: Mapping[str, float]) -> str | None:
    """Return the label with the highest score or None when all scores are zero."""

    if not scores:
        return None

    dominant_label, dominant_value = max(scores.items(), key=lambda item: item[1])
    if dominant_value <= 0:
        return None
    return dominant_label


def analyse_vader(text: str) -> VaderScores:
    """Score text with NLTK Vader and return the standard sentiment fields."""

    _ensure_vader_lexicon()

    from nltk.sentiment import SentimentIntensityAnalyzer

    scores = SentimentIntensityAnalyzer().polarity_scores(text)
    vader_scores: VaderScores = {
        "neg": float(scores["neg"]),
        "neu": float(scores["neu"]),
        "pos": float(scores["pos"]),
        "compound": float(scores["compound"]),
    }
    return vader_scores


def analyse_textblob(text: str) -> TextBlobScores:
    """Score text with TextBlob."""

    from textblob import TextBlob

    sentiment = cast(Any, TextBlob(text)).sentiment
    return {
        "polarity": float(sentiment.polarity),
        "subjectivity": float(sentiment.subjectivity),
    }


def analyse_text2emotion(text: str) -> Text2EmotionScores:
    """Score text with text2emotion and normalize the keys for downstream use."""

    _ensure_text2emotion_compatibility()

    from text2emotion import get_emotion

    emotion_scores = cast(dict[str, float], get_emotion(text) or {})
    emotions = {key.lower(): float(value) for key, value in emotion_scores.items()}
    return {
        "emotions": emotions,
        "dominant_emotion": _dominant_score(emotions),
    }


def analyse_nrclex(text: str) -> NRClexScores:
    """Score text with NRCLex and expose both raw and normalized emotion counts."""

    from nrclex import NRCLex

    analyzer = NRCLex(None)
    analyzer.load_raw_text(text)
    raw_emotion_scores = cast(dict[str, int], analyzer.raw_emotion_scores)
    affect_frequencies = cast(dict[str, float], analyzer.affect_frequencies)
    raw_scores = {key: int(value) for key, value in raw_emotion_scores.items()}
    normalized_scores = {key: float(value) for key, value in affect_frequencies.items()}
    top_emotions = cast(list[tuple[str, float]], analyzer.top_emotions or [])
    return {
        "raw_emotion_scores": raw_scores,
        "affect_frequencies": normalized_scores,
        "top_emotions": [
            {"emotion": emotion, "score": float(score)} for emotion, score in top_emotions
        ],
        "dominant_emotion": _dominant_score(normalized_scores),
    }


def analyse_opinion(text: str) -> OpinionAnalysis:
    """Run every analysis tool on a single opinion."""

    vader = analyse_vader(text)
    textblob = analyse_textblob(text)
    text2emotion = analyse_text2emotion(text)
    nrclex = analyse_nrclex(text)
    return {
        "vader": vader,
        "textblob": textblob,
        "text2emotion": text2emotion,
        "nrclex": nrclex,
    }


def build_summary(opinions: Mapping[str, OpinionAnalysis]) -> dict[str, Any]:
    """Build a compact comparison summary across the analyzed opinions."""

    vader_compounds = {
        label: float(result["vader"]["compound"]) for label, result in opinions.items()
    }
    textblob_polarity = {
        label: float(result["textblob"]["polarity"]) for label, result in opinions.items()
    }
    text2emotion_dominant = {
        label: result["text2emotion"]["dominant_emotion"] for label, result in opinions.items()
    }
    nrclex_dominant = {
        label: result["nrclex"]["dominant_emotion"] for label, result in opinions.items()
    }

    return {
        "vader": {
            "highest_compound": max(vader_compounds, key=lambda label: vader_compounds[label]),
            "lowest_compound": min(vader_compounds, key=lambda label: vader_compounds[label]),
            "compound_scores": vader_compounds,
        },
        "textblob": {
            "highest_polarity": max(textblob_polarity, key=lambda label: textblob_polarity[label]),
            "lowest_polarity": min(textblob_polarity, key=lambda label: textblob_polarity[label]),
            "polarity_scores": textblob_polarity,
        },
        "dominant_emotions": {
            "text2emotion": text2emotion_dominant,
            "nrclex": nrclex_dominant,
        },
    }
