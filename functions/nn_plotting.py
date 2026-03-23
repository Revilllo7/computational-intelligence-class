from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_learning_curves(history_df: pd.DataFrame, output_path: Path, title: str) -> None:
    # Save side-by-side training and validation curves.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="Train", linewidth=2)
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="Validation", linewidth=2)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["val_accuracy"], label="Validation", linewidth=2)
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_confusion_matrix_plot(
    confusion_df: pd.DataFrame,
    title: str,
    output_path: Path,
    class_labels: list[str] | None = None,
    x_axis_label: str = "Predicted",
    y_axis_label: str = "Actual",
) -> None:
    # Save confusion matrix heatmap from a labeled DataFrame.
    plot_df = confusion_df.copy()
    if class_labels is not None:
        if len(class_labels) != len(plot_df.index):
            raise ValueError(
                "class_labels length must match confusion matrix dimensions "
                f"({len(plot_df.index)})."
            )
        plot_df.index = class_labels
        plot_df.columns = class_labels

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(plot_df, annot=True, fmt="d", cmap="BuGn", linewidths=0.5, ax=ax)
    ax.set_title(title)
    ax.set_ylabel(y_axis_label)
    ax.set_xlabel(x_axis_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
