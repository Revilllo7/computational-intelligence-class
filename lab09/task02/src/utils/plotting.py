"""Plot helpers for task02 sentiment comparison."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .analysis import OpinionAnalysis


def _plot_stacked_bars(
    ax,
    labels: list[str],
    opinion_results: dict[str, OpinionAnalysis],
    value_accessor,
    categories: list[str],
    colors: list[str],
    title: str,
    ylabel: str,
) -> None:
    bottom = np.zeros(len(labels))
    for index, category in enumerate(categories):
        values = np.array(
            [float(value_accessor(opinion_results[label]).get(category, 0.0)) for label in labels],
            dtype=float,
        )
        ax.bar(labels, values, bottom=bottom, color=colors[index % len(colors)], label=category)
        bottom += values

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, loc="upper right")


def save_comparison_plot(
    opinion_results: dict[str, OpinionAnalysis],
    output_path: str | Path,
) -> Path:
    """Save a combined comparison plot for all analyzed opinions."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    labels = list(opinion_results)
    vader_categories = ["neg", "neu", "pos"]
    emotion_categories = ["happy", "angry", "surprise", "sad", "fear"]
    nrc_categories = [
        "fear",
        "anger",
        "anticipation",
        "trust",
        "surprise",
        "positive",
        "negative",
        "sadness",
        "disgust",
        "joy",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    vader_axis = axes[0, 0]
    vader_bottom = np.zeros(len(labels))
    for index, category in enumerate(vader_categories):
        values = np.array(
            [float(opinion_results[label]["vader"][category]) for label in labels],
            dtype=float,
        )
        vader_axis.bar(
            labels,
            values,
            bottom=vader_bottom,
            label=category,
            color=["#d1495b", "#f7b267", "#4f772d"][index],
        )
        vader_bottom += values
    vader_axis.set_title("Vader scores")
    vader_axis.set_ylabel("score")
    vader_axis.set_ylim(bottom=0)
    vader_axis2 = vader_axis.twinx()
    vader_compounds = [float(opinion_results[label]["vader"]["compound"]) for label in labels]
    vader_axis2.plot(
        labels, vader_compounds, color="#1d3557", marker="o", linewidth=2, label="compound"
    )
    vader_axis2.set_ylim(-1, 1)
    vader_axis2.set_ylabel("compound")
    vader_handles, vader_labels = vader_axis.get_legend_handles_labels()
    compound_handles, compound_labels = vader_axis2.get_legend_handles_labels()
    vader_axis.legend(
        vader_handles + compound_handles,
        vader_labels + compound_labels,
        fontsize=8,
        loc="upper left",
    )

    textblob_axis = axes[0, 1]
    polarity = [float(opinion_results[label]["textblob"]["polarity"]) for label in labels]
    subjectivity = [float(opinion_results[label]["textblob"]["subjectivity"]) for label in labels]
    textblob_axis.bar(labels, polarity, color="#457b9d", label="polarity")
    textblob_axis.set_title("TextBlob scores")
    textblob_axis.set_ylabel("polarity")
    textblob_axis.set_ylim(-1, 1)
    textblob_axis2 = textblob_axis.twinx()
    textblob_axis2.plot(
        labels,
        subjectivity,
        color="#e76f51",
        marker="o",
        linewidth=2,
        label="subjectivity",
    )
    textblob_axis2.set_ylim(0, 1)
    textblob_axis2.set_ylabel("subjectivity")
    polarity_handles, polarity_labels = textblob_axis.get_legend_handles_labels()
    subjectivity_handles, subjectivity_labels = textblob_axis2.get_legend_handles_labels()
    textblob_axis.legend(
        polarity_handles + subjectivity_handles,
        polarity_labels + subjectivity_labels,
        fontsize=8,
        loc="upper left",
    )

    _plot_stacked_bars(
        axes[1, 0],
        labels,
        opinion_results,
        lambda analysis: analysis["text2emotion"]["emotions"],
        emotion_categories,
        ["#ffb703", "#fb8500", "#8ecae6", "#219ebc", "#023047"],
        "text2emotion emotions",
        "score",
    )

    _plot_stacked_bars(
        axes[1, 1],
        labels,
        opinion_results,
        lambda analysis: analysis["nrclex"]["affect_frequencies"],
        nrc_categories,
        [
            "#264653",
            "#2a9d8f",
            "#e9c46a",
            "#f4a261",
            "#e76f51",
            "#8ab17d",
            "#577590",
            "#c9ada7",
            "#9d8189",
            "#b56576",
        ],
        "NRCLex emotions",
        "frequency",
    )

    fig.suptitle("Task02 sentiment comparison", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
