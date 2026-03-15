from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import pandas as pd

from PIL import Image

from sklearn.metrics import (
	accuracy_score,
	confusion_matrix,
	precision_score,
	recall_score,
)

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR.parent / "data" / "diagnosis.csv"
OUTPUT_DIR = BASE_DIR / "output"

FEATURE_COLUMNS = ["param1", "param2", "param3"]
TARGET_COLUMN = "diagnosis"
TEST_SIZE = 0.3
RANDOM_STATE = 292583

ROTATING_GIF_FILE = OUTPUT_DIR / "diagnosis_3d_rotating.gif"
SCATTER_PNG_FILE = OUTPUT_DIR / "diagnosis_3d_scatter.png"
METRICS_CHART_FILE = OUTPUT_DIR / "metrics_comparison.png"

LOG_FILE = OUTPUT_DIR / "diagnosis_log.csv"



# Validate all required columns are present

def validate_schema(df: pd.DataFrame) -> None:
	required = FEATURE_COLUMNS + [TARGET_COLUMN]
	missing = [column for column in required if column not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns: {missing}")



# Split dataset into train/test and separate features/target

def split_dataset(
	df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
	train_set, test_set = train_test_split(
		df,
		test_size=TEST_SIZE,
		random_state=RANDOM_STATE,
		shuffle=True,
	)
	X_train = train_set[FEATURE_COLUMNS].copy()
	X_test = test_set[FEATURE_COLUMNS].copy()
	y_train = train_set[TARGET_COLUMN].copy()
	y_test = test_set[TARGET_COLUMN].copy()
	return train_set, test_set, X_train, X_test, y_train, y_test



# Confusion matrix construction and plotting

def build_confusion_matrix(y_true: pd.Series, y_pred: pd.Series, labels: list[int]) -> pd.DataFrame:
	matrix = confusion_matrix(y_true, y_pred, labels=labels)
	return pd.DataFrame(matrix, index=labels, columns=labels)



# 3D scatter plot and rotating GIF generation

def save_3d_scatter_and_rotating_gif(df: pd.DataFrame) -> tuple[Path, Path]:
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	healthy = df[df[TARGET_COLUMN] == 0]
	sick = df[df[TARGET_COLUMN] == 1]

	fig = plt.figure(figsize=(8, 6))
	ax = fig.add_subplot(111, projection="3d")
	ax.scatter(
		healthy["param1"],
		healthy["param2"],
		healthy["param3"],
		color="blue",
		alpha=0.65,
		s=20,
		label="Healthy (0)",
	)
	ax.scatter(
		sick["param1"],
		sick["param2"],
		sick["param3"],
		color="red",
		alpha=0.75,
		s=25,
		label="Sick (1)",
	)
	ax.set_xlabel("param1")
	ax.set_ylabel("param2")
	ax.set_zlabel("param3")
	ax.set_title("Diagnosis dataset")
	ax.legend(loc="upper left")
	fig.tight_layout()
	fig.savefig(SCATTER_PNG_FILE, dpi=160)
	plt.close(fig)

	frames: list[Image.Image] = []
	with TemporaryDirectory() as tmp_dir:
		tmp_path = Path(tmp_dir)
		angles = list(range(0, 360, 8))
		for idx, azim in enumerate(angles):
			frame_fig = plt.figure(figsize=(8, 6))
			frame_ax = frame_fig.add_subplot(111, projection="3d")
			frame_ax.scatter(
				healthy["param1"],
				healthy["param2"],
				healthy["param3"],
				color="blue",
				alpha=0.65,
				s=20,
				label="Healthy (0)",
			)
			frame_ax.scatter(
				sick["param1"],
				sick["param2"],
				sick["param3"],
				color="red",
				alpha=0.75,
				s=25,
				label="Sick (1)",
			)
			frame_ax.set_xlabel("param1")
			frame_ax.set_ylabel("param2")
			frame_ax.set_zlabel("param3")
			frame_ax.set_title("Diagnosis dataset (rotating)")
			frame_ax.legend(loc="upper left")
			frame_ax.view_init(elev=24, azim=azim)

			frame_file = tmp_path / f"frame_{idx:03d}.png"
			frame_fig.tight_layout()
			frame_fig.savefig(frame_file, dpi=120)
			plt.close(frame_fig)
			frames.append(Image.open(frame_file))

	if not frames:
		raise RuntimeError("Could not generate GIF frames.")

	frames[0].save(
		ROTATING_GIF_FILE,
		save_all=True,
		append_images=frames[1:],
		duration=180,
		loop=0,
	)
	return SCATTER_PNG_FILE, ROTATING_GIF_FILE


def save_confusion_matrix_plot(
	confusion_df: pd.DataFrame,
	classifier_name: str,
	accuracy: float,
	filepath: Path,
) -> None:
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	fig, ax = plt.subplots(figsize=(6, 5))
	im = ax.imshow(confusion_df.values, cmap="BuGn")
	fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

	ax.set_xticks(range(len(confusion_df.columns)))
	ax.set_yticks(range(len(confusion_df.index)))
	ax.set_xticklabels(confusion_df.columns)
	ax.set_yticklabels(confusion_df.index)
	ax.set_xlabel("Predicted")
	ax.set_ylabel("Actual")
	ax.set_title(f"{classifier_name}\nAccuracy: {accuracy:.2%}")

	for row_idx in range(confusion_df.shape[0]):
		for col_idx in range(confusion_df.shape[1]):
			value = confusion_df.iloc[row_idx, col_idx]
			ax.text(col_idx, row_idx, str(value), ha="center", va="center", color="black")

	fig.tight_layout()
	fig.savefig(filepath, dpi=150)
	plt.close(fig)


def save_metrics_chart(results: list[dict], filepath: Path) -> None:
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	names = [row["name"] for row in results]
	accuracy_vals = [row["accuracy"] * 100 for row in results]
	precision_vals = [row["precision_binary"] * 100 for row in results]
	recall_vals = [row["recall_binary"] * 100 for row in results]

	family_color = {
		"KNN": "lightblue",
		"Naive Bayes": "lightcoral",
		"MLP": "lightgreen",
		"Decision Tree": "lightyellow",
	}

	bar_colors: list[str] = []
	for model_name in names:
		if model_name.startswith("KNN"):
			bar_colors.append(family_color["KNN"])
		elif model_name.startswith("Naive Bayes"):
			bar_colors.append(family_color["Naive Bayes"])
		elif model_name.startswith("MLP"):
			bar_colors.append(family_color["MLP"])
		else:
			bar_colors.append(family_color["Decision Tree"])

	fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
	metrics = [
		("Accuracy (%)", accuracy_vals),
		("Precision - binary (diag=1) (%)", precision_vals),
		("Recall/Sensitivity - binary (diag=1) (%)", recall_vals),
	]

	for ax, (title, values) in zip(axes, metrics):
		bars = ax.bar(names, values, color=bar_colors, edgecolor="black")
		ax.set_title(title)
		ax.set_ylim(0, 105)
		ax.tick_params(axis="x", rotation=35)
		for bar, value in zip(bars, values):
			ax.text(
				bar.get_x() + bar.get_width() / 2,
				bar.get_height() + 0.4,
				f"{value:.1f}",
				ha="center",
				va="bottom",
				fontsize=8,
			)

    # Legend

	legend_items = [
		Patch(facecolor=family_color["KNN"], edgecolor="black", label="KNN (k = 3, 5, 11)"),
		Patch(facecolor=family_color["Naive Bayes"], edgecolor="black", label="Naive Bayes"),
		Patch(facecolor=family_color["MLP"], edgecolor="black", label="MLP"),
		Patch(facecolor=family_color["Decision Tree"], edgecolor="black", label="Decision Tree"),
	]
	fig.legend(handles=legend_items, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
	fig.suptitle("Diagnosis classifier comparison", fontsize=14)
	fig.tight_layout()
	fig.savefig(filepath, dpi=150, bbox_inches="tight")
	plt.close(fig)



# Evaluate a single classifier and return metrics and confusion matrix

def evaluate_classifier(
	classifier,
	X_train: pd.DataFrame,
	y_train: pd.Series,
	X_test: pd.DataFrame,
	y_test: pd.Series,
	labels: list[int],
) -> dict:
	classifier.fit(X_train, y_train)
	y_pred = pd.Series(classifier.predict(X_test), index=y_test.index, name="predicted")

	accuracy = accuracy_score(y_test, y_pred)
	precision_binary = precision_score(y_test, y_pred, average="binary", pos_label=1, zero_division=0)
	recall_binary = recall_score(y_test, y_pred, average="binary", pos_label=1, zero_division=0)
	precision_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
	recall_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)

	correct = int((y_pred == y_test).sum())
	wrong = len(y_test) - correct
	confusion_df = build_confusion_matrix(y_test, y_pred, labels)

	return {
		"accuracy": accuracy,
		"precision_binary": precision_binary,
		"recall_binary": recall_binary,
		"precision_weighted": precision_weighted,
		"recall_weighted": recall_weighted,
		"correct": correct,
		"wrong": wrong,
		"confusion_df": confusion_df,
	}


def print_classifier_block(name: str, result: dict, total: int) -> None:
	print(f"  Accuracy: {result['accuracy']:.2%}")
	print(f"  Precision (binary, diagnosis=1): {result['precision_binary']:.2%}")
	print(f"  Recall/Sensitivity (binary, diagnosis=1): {result['recall_binary']:.2%}")
	print(f"  Precision (weighted): {result['precision_weighted']:.2%}")
	print(f"  Recall (weighted): {result['recall_weighted']:.2%}")
	print(f"  Good predictions: {result['correct']}/{total}")
	print(f"  Wrong predictions: {result['wrong']}/{total}")
	print(f"\n")
	print("  Confusion matrix:")
	indented = result["confusion_df"].to_string()
	for line in indented.splitlines():
		print(f"    {line}")
	print(f"\n")


def main() -> None:
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	df = pd.read_csv(DATA_FILE)
	validate_schema(df)

	scatter_png, rotating_gif = save_3d_scatter_and_rotating_gif(df)
	train_set, test_set, X_train, X_test, y_train, y_test = split_dataset(df)

	labels = sorted(df[TARGET_COLUMN].unique().tolist())
	total = len(y_test)
	class_counts = df[TARGET_COLUMN].value_counts().sort_index()

	classifiers = [
		("KNN (k= 3)", KNeighborsClassifier(n_neighbors=3)),
		("KNN (k= 5)", KNeighborsClassifier(n_neighbors=5)),
		("KNN (k= 11)", KNeighborsClassifier(n_neighbors=11)),
		("Naive Bayes", GaussianNB()),
		("MLP", MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=RANDOM_STATE)),
		("Decision Tree", DecisionTreeClassifier(random_state=RANDOM_STATE)),
	]

	print("=" * 90)
	print("DIAGNOSIS DATASET - MULTI-CLASSIFIER BENCHMARK")
	print("=" * 90)
	print(f"Input file: {DATA_FILE}")
	print(f"Full dataset shape: {df.shape}")
	print(f"Class distribution (diagnosis): {class_counts.to_dict()}")
	print(f"Split configuration: train={1 - TEST_SIZE:.0%}, test={TEST_SIZE:.0%}, random_state={RANDOM_STATE}")
	print(f"Training samples: {len(train_set)} | Test samples: {len(test_set)}")
	print(f"Labels: {labels}")
	print(f"\n")
	print(f"Static 3D scatter saved to: {scatter_png}")
	print(f"Rotating 3D GIF saved to: {rotating_gif}")
	print(f"\n")

	results: list[dict] = []
	for name, clf in classifiers:
		print("-" * 90)
		print(f"  {name}")
		print("-" * 90)
		result = evaluate_classifier(clf, X_train, y_train, X_test, y_test, labels)
		print_classifier_block(name, result, total)

		safe_name = (
			name.lower()
			.replace(" ", "_")
			.replace("(", "")
			.replace(")", "")
			.replace("=", "")
		)
		cm_path = OUTPUT_DIR / f"confusion_matrix_{safe_name}.png"
		save_confusion_matrix_plot(result["confusion_df"], name, result["accuracy"], cm_path)
		print(f"  Confusion matrix plot saved to: {cm_path}")
		print(f"\n")

		results.append(
			{
				"name": name,
				"accuracy": result["accuracy"],
				"precision_binary": result["precision_binary"],
				"recall_binary": result["recall_binary"],
				"precision_weighted": result["precision_weighted"],
				"recall_weighted": result["recall_weighted"],
				"correct": result["correct"],
				"wrong": result["wrong"],
			}
		)

	print("=" * 90)
	print("SUMMARY - METRICS COMPARISON")
	print("=" * 90)
	header = (
		f"{'Classifier':<16} {'Accuracy':>10} {'Prec(bin)':>10} {'Rec(bin)':>10} "
		f"{'Prec(w)':>10} {'Rec(w)':>10} {'Correct':>10} {'Wrong':>8}"
	)
	print(header)
	print("-" * len(header))
	for row in results:
		print(
			f"{row['name']:<16} {row['accuracy']:>10.2%} {row['precision_binary']:>10.2%} "
			f"{row['recall_binary']:>10.2%} {row['precision_weighted']:>10.2%} "
			f"{row['recall_weighted']:>10.2%} {row['correct']:>10}/{total} {row['wrong']:>7}/{total}"
		)
	print(f"\n")

	save_metrics_chart(results, METRICS_CHART_FILE)
	print(f"Metrics comparison chart saved to: {METRICS_CHART_FILE}")

	log_df = pd.DataFrame(results)
	log_df.to_csv(LOG_FILE, index=False)
	print(f"Metrics log saved to: {LOG_FILE}")


if __name__ == "__main__":
	main()
