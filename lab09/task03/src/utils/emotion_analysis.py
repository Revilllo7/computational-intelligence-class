"""Defines functions for emotion analysis of text using different libraries."""

import emoji

if not hasattr(emoji, "UNICODE_EMOJI"):
    setattr(emoji, "UNICODE_EMOJI", getattr(emoji, "EMOJI_DATA", {}))  # noqa: B010

from nrclex import NRCLex
from text2emotion import get_emotion
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

vader = SentimentIntensityAnalyzer()


def analyse_post(text: str):
    vader_scores = vader.polarity_scores(text)

    emotions_t2e = get_emotion(text)

    nrc = NRCLex()
    nrc.load_raw_text(text)
    nrc_emotions = nrc.raw_emotion_scores

    return {
        "compound": vader_scores["compound"],
        "positive": vader_scores["pos"],
        "neutral": vader_scores["neu"],
        "negative": vader_scores["neg"],
        "t2e": emotions_t2e,
        "nrc": nrc_emotions,
    }
