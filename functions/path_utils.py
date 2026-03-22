from __future__ import annotations

import re
from pathlib import Path


def ensure_directory(path: str | Path) -> Path:
    # Create directory if needed and return it as Path.
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def safe_filename(name: str) -> str:
    # Convert label-like text into a filesystem-friendly filename fragment.
    lowered = name.strip().lower()
    lowered = re.sub(r"\s+", "_", lowered)
    lowered = re.sub(r"[^a-z0-9_\-]", "", lowered)
    lowered = re.sub(r"_+", "_", lowered)
    return lowered.strip("_")
