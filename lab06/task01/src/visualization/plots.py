from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from src.utils.io import ensure_parent


def save_feature_histograms(frame: pd.DataFrame, output_path: Path, dpi: int) -> None:
    ensure_parent(output_path)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=dpi)
    for axis, column in zip(axes.flatten(), frame.columns[:-1], strict=True):
        sns.histplot(data=frame, x=column, hue="species", kde=True, ax=axis)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_pairplot(frame: pd.DataFrame, output_path: Path, dpi: int) -> None:
    ensure_parent(output_path)
    axes = scatter_matrix(
        frame[frame.columns],
        figsize=(10, 10),
        diagonal="hist",
        c=frame["species"].astype("category").cat.codes,
    )
    for row in axes:
        for axis in row:
            axis.set_xlabel(axis.get_xlabel(), fontsize=8)
            axis.set_ylabel(axis.get_ylabel(), fontsize=8)
    plt.gcf().set_dpi(dpi)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_training_curves(history: pd.DataFrame, output_path: Path, dpi: int) -> None:
    ensure_parent(output_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=dpi)
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["validation_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history["epoch"], history["train_accuracy"], label="train")
    axes[1].plot(history["epoch"], history["validation_accuracy"], label="validation")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    class_names: list[str],
    output_path: Path,
    dpi: int,
) -> None:
    ensure_parent(output_path)
    fig, axis = plt.subplots(figsize=(6, 6), dpi=dpi)
    ConfusionMatrixDisplay.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        labels=class_names,
        ax=axis,
        colorbar=False,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
