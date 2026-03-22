from __future__ import annotations

import pandas as pd


def validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    # Raise ValueError if any required column is missing from DataFrame.
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
