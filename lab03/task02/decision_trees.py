from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR.parent / "data" / "iris_big.csv"
OUTPUT_DIR = BASE_DIR / "output"

FEATURE_COLUMNS = [
	"sepal length (cm)",
	"sepal width (cm)",
	"petal length (cm)",
	"petal width (cm)",
]
TARGET_COLUMN = "target_name"
TEST_SIZE = 0.7
RANDOM_STATE = 292583
TREE_PLOT_FILE = OUTPUT_DIR / "decision_tree_plot.png"


def validate_schema(df: pd.DataFrame) -> None:
	missing = [column for column in FEATURE_COLUMNS + [TARGET_COLUMN] if column not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns: {missing}")


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


def train_classifier(X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeClassifier:
	classifier = DecisionTreeClassifier(random_state=RANDOM_STATE)
	classifier.fit(X_train, y_train)
	return classifier


def save_tree_plot(classifier: DecisionTreeClassifier) -> Path:
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	fig, ax = plt.subplots(figsize=(18, 10))
	plot_tree(
		classifier,
		feature_names=FEATURE_COLUMNS,
		class_names=sorted(classifier.classes_),
		filled=True,
		rounded=True,
		ax=ax,
	)
	fig.tight_layout()
	fig.savefig(TREE_PLOT_FILE, format="png", dpi=300)
	plt.close(fig)
	return TREE_PLOT_FILE


def build_confusion_matrix(y_true: pd.Series, y_pred: pd.Series, labels: list[str]) -> pd.DataFrame:
	matrix = confusion_matrix(y_true, y_pred, labels=labels)
	return pd.DataFrame(matrix, index=labels, columns=labels)


def main() -> None:
	df = pd.read_csv(DATA_FILE)
	validate_schema(df)

	train_set, test_set, X_train, X_test, y_train, y_test = split_dataset(df)
	classifier = train_classifier(X_train, y_train)
	plot_path = save_tree_plot(classifier)

	y_pred = classifier.predict(X_test)
	y_pred_series = pd.Series(y_pred, index=y_test.index, name="predicted")
	accuracy = classifier.score(X_test, y_test)
	correct_predictions = int((y_pred_series == y_test).sum())
	wrong_predictions = len(y_test) - correct_predictions
	labels = sorted(df[TARGET_COLUMN].unique())
	confusion_df = build_confusion_matrix(y_test, y_pred_series, labels)
	text_tree = export_text(classifier, feature_names=FEATURE_COLUMNS)

	print("=" * 90)
	print("DECISION TREE CLASSIFICATION - IRIS DATASET")
	print("=" * 90)
	print(f"Input file: {DATA_FILE}")
	print(f"Full dataset shape: {df.shape}")
	print(f"Split configuration: train={1 - TEST_SIZE:.0%}, test={TEST_SIZE:.0%}, random_state={RANDOM_STATE}")
	print()

	print("Training set:")
	print(train_set)
	print()
	print("Test set:")
	print(test_set)
	print()

	print("Split into inputs and classes:")
	print(f"X_train shape: {X_train.shape}")
	print(X_train.head())
	print()
	print(f"y_train shape: {y_train.shape}")
	print(y_train.head())
	print()
	print(f"X_test shape: {X_test.shape}")
	print(X_test.head())
	print()
	print(f"y_test shape: {y_test.shape}")
	print(y_test.head())
	print()

	print("Decision tree (text form):")
	print(text_tree)
	print()
	print(f"Decision tree plot saved to: {plot_path}")
	print()

	print("Evaluation on test set:")
	print(f"Good predictions: {correct_predictions}/{len(y_test)}")
	print(f"Wrong predictions: {wrong_predictions}/{len(y_test)}")
	print(f"Accuracy (score): {accuracy:.2%}")
	print()

	print("Predictions preview:")
	predictions_preview = pd.DataFrame(
		{
			"actual": y_test.reset_index(drop=True).head(10),
			"predicted": y_pred_series.reset_index(drop=True).head(10),
		}
	)
	print(predictions_preview)
	print()

	print("Confusion matrix:")
	print(confusion_df)


if __name__ == "__main__":
	main()
