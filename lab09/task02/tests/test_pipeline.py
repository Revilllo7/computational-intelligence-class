"""Task02 sentiment comparison tests."""

from __future__ import annotations

import json
from pathlib import Path

from task02.src.main import run_sentiment_comparison
from task02.src.utils.analysis import analyse_vader


def test_vader_returns_expected_fields() -> None:
    """Vader should expose the four standard sentiment fields."""

    scores = analyse_vader("This hotel is excellent and wonderful.")

    assert set(scores) == {"neg", "neu", "pos", "compound"}
    assert all(isinstance(value, float) for value in scores.values())
    assert 0.0 <= scores["pos"] <= 1.0
    assert -1.0 <= scores["compound"] <= 1.0


def test_pipeline_writes_json_and_plot(tmp_path: Path) -> None:
    """The public pipeline should create both task artifacts."""

    report = run_sentiment_comparison(output_dir=tmp_path)

    json_path = tmp_path / "sentiment_comparison.json"
    plot_path = tmp_path / "sentiment_comparison.png"

    assert json_path.exists()
    assert plot_path.exists()
    assert report["output_paths"] == {"json": str(json_path), "plot": str(plot_path)}

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert set(payload) == {"opinions", "summary", "output_paths"}
    assert set(payload["opinions"]) == {"positive", "neutral", "negative"}
    assert set(payload["summary"]) == {"vader", "textblob", "dominant_emotions"}

    positive = payload["opinions"]["positive"]
    assert set(positive) == {"vader", "textblob", "text2emotion", "nrclex"}
    assert set(positive["vader"]) == {"neg", "neu", "pos", "compound"}
    assert set(positive["textblob"]) == {"polarity", "subjectivity"}
    assert "emotions" in positive["text2emotion"]
    assert "affect_frequencies" in positive["nrclex"]
