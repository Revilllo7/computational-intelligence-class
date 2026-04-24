"""Reusable helpers for simple genetic algorithm workflows."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def decode_selected_items(
    chromosome: list[int] | tuple[int, ...],
    item_ids: list[str],
    item_names: list[str],
    item_values: list[float],
    item_weights: list[float],
) -> list[dict[str, Any]]:
    """Convert a 0/1 chromosome into a list of selected item rows."""
    selected: list[dict[str, Any]] = []
    for gene, item_id, name, value, weight in zip(
        chromosome, item_ids, item_names, item_values, item_weights, strict=True
    ):
        if int(gene) == 1:
            selected.append(
                {
                    "id": item_id,
                    "name": name,
                    "value": float(value),
                    "weight": float(weight),
                }
            )
    return selected


def summarize_totals(selected_items: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate totals for selected knapsack items."""
    return {
        "total_value": float(sum(row["value"] for row in selected_items)),
        "total_weight": float(sum(row["weight"] for row in selected_items)),
    }


def write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    """Write JSON payload with consistent formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write dictionaries to CSV using explicit field order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
