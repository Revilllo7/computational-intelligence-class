from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from src.utils.io import ensure_parent


def save_training_curves(history: pd.DataFrame, output_path: Path, dpi: int) -> None:
    ensure_parent(output_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=dpi)

    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["validation_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["epoch"], history["train_accuracy"], label="train")
    axes[1].plot(history["epoch"], history["validation_accuracy"], label="validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
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


def save_accuracy_comparison(results: pd.DataFrame, output_path: Path, dpi: int) -> None:
    ensure_parent(output_path)
    fig, axis = plt.subplots(figsize=(12, 5), dpi=dpi)
    axis.bar(results["experiment_name"], results["test_accuracy"])
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Test accuracy")
    axis.set_title("Final accuracy by experiment")
    axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_image_grid(
    image_paths: list[Path],
    titles: list[str],
    output_path: Path,
    dpi: int,
    title: str,
) -> None:
    if not image_paths:
        raise ValueError("image_paths cannot be empty.")

    ensure_parent(output_path)
    cols = min(3, len(image_paths))
    rows = math.ceil(len(image_paths) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), dpi=dpi)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for index, image_path in enumerate(image_paths):
        axes_flat[index].imshow(plt.imread(image_path))
        axes_flat[index].set_title(titles[index], fontsize=10)
        axes_flat[index].axis("off")

    for axis in axes_flat[len(image_paths) :]:
        axis.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
