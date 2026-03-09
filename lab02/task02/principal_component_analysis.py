from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR.parent / "data" / "iris_big.csv"
OUTPUT_DIR = BASE_DIR / "output"

NUMERIC_COLUMNS = [
	"sepal length (cm)",
	"sepal width (cm)",
	"petal length (cm)",
	"petal width (cm)",
]
TARGET_COLUMN = "target_name"
RETAINED_VARIANCE_THRESHOLD = 0.95


# Validation to ensure the dataset contains the expected columns before processing.
def validate_schema(df: pd.DataFrame) -> None:
	missing = [col for col in NUMERIC_COLUMNS + [TARGET_COLUMN] if col not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns: {missing}")


# Standardizes features to have mean 0 and std 1
def standardize_features(features: np.ndarray) -> np.ndarray:
	means = features.mean(axis=0)
	stds = features.std(axis=0)
	stds[stds == 0] = 1.0
	return (features - means) / stds


def run_pca(
	features_scaled: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	covariance_matrix = np.cov(features_scaled, rowvar=False)
	eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

	sorted_indices = np.argsort(eigenvalues)[::-1]
	eigenvalues = eigenvalues[sorted_indices]
	eigenvectors = eigenvectors[:, sorted_indices]

	explained_variance_ratio = eigenvalues / eigenvalues.sum()
	cumulative_variance = np.cumsum(explained_variance_ratio)
	return eigenvalues, eigenvectors, explained_variance_ratio, cumulative_variance


def choose_min_components(cumulative_variance: np.ndarray, threshold: float) -> int:
	# Index is 0-based, component count is 1-based.
	return int(np.argmax(cumulative_variance >= threshold) + 1)

# Projects the standardized data onto the selected principal components.
def project_data(features_scaled: np.ndarray, eigenvectors: np.ndarray, components: int) -> np.ndarray:
	return features_scaled @ eigenvectors[:, :components]

# Plotting
def make_plot(projected: np.ndarray, labels: pd.Series, components: int, retained_ratio: float) -> Path:
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	classes = sorted(labels.unique())
	colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
	color_map = {name: colors[idx % len(colors)] for idx, name in enumerate(classes)}

	if components >= 3:
		output_path = OUTPUT_DIR / "plot_3D.png"
		fig = plt.figure(figsize=(9, 7))
		ax = fig.add_subplot(111, projection="3d")
		for class_name in classes:
			mask = labels == class_name
			ax.scatter(
				projected[mask, 0],
				projected[mask, 1],
				projected[mask, 2],
				label=class_name,
				alpha=0.7,
				s=25,
				c=color_map[class_name],
			)
		ax.set_xlabel("PC1")
		ax.set_ylabel("PC2")
		ax.set_zlabel("PC3")
		ax.set_title(
			f"PCA (3D) - retained variance: {retained_ratio * 100:.2f}%"
		)
	else:
		output_path = OUTPUT_DIR / "plot_2D.png"
		fig, ax = plt.subplots(figsize=(9, 7))
		for class_name in classes:
			mask = labels == class_name
			ax.scatter(
				projected[mask, 0],
				projected[mask, 1],
				label=class_name,
				alpha=0.7,
				s=25,
				c=color_map[class_name],
			)
		ax.set_xlabel("PC1")
		ax.set_ylabel("PC2")
		ax.set_title(
			f"PCA (2D) - retained variance: {retained_ratio * 100:.2f}%"
		)
		ax.grid(True, alpha=0.3)

	plt.legend(title="Species")
	plt.tight_layout()
	plt.savefig(output_path, format="png", dpi=300)
	plt.close(fig)

	# Remove alternative plot (both legacy extensionless and png naming).
	other_stem = "plot_2D" if components >= 3 else "plot_3D"
	for candidate in (OUTPUT_DIR / other_stem, OUTPUT_DIR / f"{other_stem}.png"):
		if candidate.exists():
			candidate.unlink()

	# Remove legacy extensionless file for the generated plot name.
	legacy_current = OUTPUT_DIR / output_path.stem
	if legacy_current.exists() and legacy_current != output_path:
		legacy_current.unlink()

	return output_path


def main() -> None:
	df = pd.read_csv(DATA_FILE)
	validate_schema(df)

	features = df[NUMERIC_COLUMNS].to_numpy(dtype=float)
	labels = df[TARGET_COLUMN]
	features_scaled = standardize_features(features)

	eigenvalues, eigenvectors, explained_ratio, cumulative_ratio = run_pca(features_scaled)
	min_components = choose_min_components(cumulative_ratio, RETAINED_VARIANCE_THRESHOLD)
	min_components = min(max(min_components, 2), 3)

	retained_variance = float(explained_ratio[:min_components].sum())
	removed_columns = len(NUMERIC_COLUMNS) - min_components

	print("=" * 55)
	print("PCA ANALYSIS - IRIS DATASET")
	print("=" * 55)
	print(f"Input file: {DATA_FILE}")
	print(f"Rows: {len(df)}")
	print(f"Numeric columns: {len(NUMERIC_COLUMNS)}")
	print()

	print("Explained variance ratio per principal component:")
	for index, ratio in enumerate(explained_ratio, start=1):
		print(f"PC{index}: {ratio:.6f}")
	print()

	print("Cumulative explained variance:")
	for index, ratio in enumerate(cumulative_ratio, start=1):
		print(f"PC1..PC{index}: {ratio:.6f}")
	print()

	numerator = float(explained_ratio[:min_components].sum())
	denominator = float(explained_ratio.sum())
	print("Formula justification:")
	print(
		"sum(var(column[k]) for k=n-1..n-i) / sum(var(column[k]) for k=n-1..0)"
	)
	print(f"= {numerator:.6f} / {denominator:.6f} = {numerator / denominator:.6f}")
	print()

	print(f"Minimum components for >= 95% variance: {min_components}")
	print(f"Columns that can be removed: {removed_columns}")
	print(f"Retained variance: {retained_variance * 100:.2f}%")
	print(f"Information loss: {(1.0 - retained_variance) * 100:.2f}%")
	print()

	projected = project_data(features_scaled, eigenvectors, min_components)
	plot_path = make_plot(projected, labels, min_components, retained_variance)

	print(f"Generated minimal required plot: {plot_path}")


if __name__ == "__main__":
	main()
