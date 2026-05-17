"""Input and output helpers for task02 sentiment comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

OPINION_FILENAMES = {
    "positive": "positive_opinion.txt",
    "neutral": "neutral_opinion.txt",
    "negative": "negative_opinion.txt",
}


def default_data_dir() -> Path:
    """Return the default input directory for the task02 opinions."""

    return Path(__file__).resolve().parents[2] / "data"


def default_output_dir() -> Path:
    """Return the default output directory for task02 artifacts."""

    return Path(__file__).resolve().parents[2] / "output"


def load_opinion_texts(data_dir: str | Path | None = None) -> dict[str, str]:
    """Load the three opinion files into a label-to-text mapping."""

    base_dir = Path(data_dir) if data_dir is not None else default_data_dir()
    return {
        label: (base_dir / filename).read_text(encoding="utf-8")
        for label, filename in OPINION_FILENAMES.items()
    }


def prepare_output_dir(output_dir: str | Path | None = None) -> Path:
    """Create and return the output directory for generated artifacts."""

    path = Path(output_dir) if output_dir is not None else default_output_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json_report(payload: dict[str, Any], output_path: str | Path) -> Path:
    """Serialize a report as JSON using a stable, human-readable format."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
