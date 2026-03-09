"""Dataset normalization comparison for Iris sepal measurements.

This script:
1. Loads iris_big.csv dataset
2. Extracts sepal length and sepal width
3. Creates three scatter plots: original, min-max normalized, z-score scaled
4. Computes and reports statistics for each version
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR.parent / "data" / "iris_big.csv"
OUTPUT_DIR = BASE_DIR / "output"

SEPAL_LENGTH_COL = "sepal length (cm)"
SEPAL_WIDTH_COL = "sepal width (cm)"
TARGET_COLUMN = "target_name"


def min_max_normalize(data: np.ndarray) -> np.ndarray:
    """Normalize data to [0, 1] range using min-max scaling."""
    min_val = data.min(axis=0)
    max_val = data.max(axis=0)
    range_val = max_val - min_val
    range_val[range_val == 0] = 1.0
    return (data - min_val) / range_val


def z_score_normalize(data: np.ndarray) -> np.ndarray:
    """Normalize data using z-score scaling (standardization)."""
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std[std == 0] = 1.0
    return (data - mean) / std


def compute_statistics(data: np.ndarray, labels: list[str]) -> dict[str, dict[str, float]]:
    """Compute min, max, mean, and std for each column."""
    stats = {}
    for idx, label in enumerate(labels):
        stats[label] = {
            "min": float(data[:, idx].min()),
            "max": float(data[:, idx].max()),
            "mean": float(data[:, idx].mean()),
            "std": float(data[:, idx].std()),
        }
    return stats


def create_scatter_plot(
    data: np.ndarray,
    species: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Create scatter plot with species-based coloring."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    classes = sorted(species.unique())
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    color_map = {name: colors[idx % len(colors)] for idx, name in enumerate(classes)}
    
    fig, ax = plt.subplots(figsize=(9, 7))
    
    for class_name in classes:
        mask = species == class_name
        ax.scatter(
            data[mask, 0],
            data[mask, 1],
            label=class_name,
            alpha=0.6,
            s=30,
            c=color_map[class_name],
        )
    
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Species", loc="best")
    
    plt.tight_layout()
    plt.savefig(output_path, format="png", dpi=300)
    plt.close(fig)
    
    legacy_path = OUTPUT_DIR / output_path.stem
    if legacy_path.exists() and legacy_path != output_path:
        legacy_path.unlink()


def print_statistics_table(stats: dict[str, dict[str, float]], version_name: str) -> None:
    """Print formatted statistics table."""
    print(f"\n{version_name}:")
    print(f"{'Metric':<20} {'Sepal Length':<15} {'Sepal Width':<15}")
    print("-" * 50)
    
    for metric in ["min", "max", "mean", "std"]:
        length_val = stats[SEPAL_LENGTH_COL][metric]
        width_val = stats[SEPAL_WIDTH_COL][metric]
        print(f"{metric.upper():<20} {length_val:<15.6f} {width_val:<15.6f}")


def main() -> None:
    df = pd.read_csv(DATA_FILE)
    
    sepal_data = df[[SEPAL_LENGTH_COL, SEPAL_WIDTH_COL]].to_numpy(dtype=float)
    species = df[TARGET_COLUMN]
    
    original_stats = compute_statistics(sepal_data, [SEPAL_LENGTH_COL, SEPAL_WIDTH_COL])
    minmax_data = min_max_normalize(sepal_data)
    minmax_stats = compute_statistics(minmax_data, [SEPAL_LENGTH_COL, SEPAL_WIDTH_COL])
    zscore_data = z_score_normalize(sepal_data)
    zscore_stats = compute_statistics(zscore_data, [SEPAL_LENGTH_COL, SEPAL_WIDTH_COL])
    
    print("=" * 60)
    print("DATASET NORMALIZATION - IRIS SEPAL MEASUREMENTS")
    print("=" * 60)
    print(f"Dataset: {DATA_FILE}")
    print(f"Rows: {len(df)}")
    print(f"Features analyzed: Sepal Length vs Sepal Width")
    
    print_statistics_table(original_stats, "ORIGINAL DATA")
    print_statistics_table(minmax_stats, "MIN-MAX NORMALIZED [0, 1]")
    print_statistics_table(zscore_stats, "Z-SCORE SCALED")
    
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)
    
    original_plot_path = OUTPUT_DIR / "original_plot.png"
    create_scatter_plot(
        sepal_data, species, "Original Data - Sepal Measurements",
        "Sepal Length (cm)", "Sepal Width (cm)", original_plot_path,
    )
    print(f"Original plot: {original_plot_path}")
    
    minmax_plot_path = OUTPUT_DIR / "min_max_normalised_plot.png"
    create_scatter_plot(
        minmax_data, species, "Min-Max Normalized - Sepal Measurements",
        "Sepal Length (normalized)", "Sepal Width (normalized)", minmax_plot_path,
    )
    print(f"Min-max normalized plot: {minmax_plot_path}")
    
    zscore_plot_path = OUTPUT_DIR / "z_score_scaled_plot.png"
    create_scatter_plot(
        zscore_data, species, "Z-Score Scaled - Sepal Measurements",
        "Sepal Length (z-score)", "Sepal Width (z-score)", zscore_plot_path,
    )
    print(f"Z-score scaled plot: {zscore_plot_path}")

if __name__ == "__main__":
    main()
