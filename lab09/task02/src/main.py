"""Main script for task02 sentiment comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict

try:
    from .utils.analysis import OpinionAnalysis, analyse_opinion, build_summary
    from .utils.data import load_opinion_texts, prepare_output_dir
    from .utils.plotting import save_comparison_plot
except ImportError:  # pragma: no cover - fallback for direct script execution
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.utils.analysis import OpinionAnalysis, analyse_opinion, build_summary
    from src.utils.data import load_opinion_texts, prepare_output_dir
    from src.utils.plotting import save_comparison_plot


class OutputPaths(TypedDict):
    json: str
    plot: str


class ComparisonReport(TypedDict):
    opinions: dict[str, OpinionAnalysis]
    summary: dict[str, Any]
    output_paths: OutputPaths


def run_sentiment_comparison(
    data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> ComparisonReport:
    """Run the full sentiment comparison pipeline and save its artifacts."""

    opinions = load_opinion_texts(data_dir)
    analysis = {label: analyse_opinion(text) for label, text in opinions.items()}
    summary = build_summary(analysis)

    output_path = prepare_output_dir(output_dir)
    json_path = output_path / "sentiment_comparison.json"
    plot_path = output_path / "sentiment_comparison.png"

    report: ComparisonReport = {
        "opinions": analysis,
        "summary": summary,
        "output_paths": {"json": str(json_path), "plot": str(plot_path)},
    }

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    save_comparison_plot(analysis, plot_path)

    return report


def main() -> int:
    """Execute the task02 comparison pipeline."""

    report = run_sentiment_comparison()
    print("Saved JSON:", report["output_paths"]["json"])
    print("Saved plot:", report["output_paths"]["plot"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
