from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from algorithms import (
	evaluate_predictions,
	fit_standard_scaler,
	split_dataset,
	train_one_epoch,
	transform_standard_scaler,
	predict_from_loader,
)
from functions import (
	ensure_directory,
	loader_average_loss,
	make_dataloader,
	save_confusion_matrix_plot,
	save_learning_curves,
	validate_required_columns,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR.parent / "data" / "diagnosis.csv"
OUTPUT_DIR = BASE_DIR / "output"

FEATURE_COLUMNS = ["param1", "param2", "param3"]
TARGET_COLUMN = "diagnosis"

TEST_SIZE = 0.3
RANDOM_STATE = 292583
EPOCHS = 70
BATCH_SIZE = 32
LEARNING_RATE = 1e-3


class DiagnosisMLP(torch.nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.net = torch.nn.Sequential(
			torch.nn.Linear(3, 16),
			torch.nn.ReLU(),
			torch.nn.Linear(16, 8),
			torch.nn.ReLU(),
			torch.nn.Linear(8, 2),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)


def main() -> None:
	torch.manual_seed(RANDOM_STATE)

	output_dir = ensure_directory(OUTPUT_DIR)
	if not DATA_FILE.exists():
		raise FileNotFoundError(
			f"Missing dataset: {DATA_FILE}. Place diagnosis.csv in lab04/data."
		)

	df = pd.read_csv(DATA_FILE)
	validate_required_columns(df, FEATURE_COLUMNS + [TARGET_COLUMN])

	work_df = df.copy()
	work_df[TARGET_COLUMN] = work_df[TARGET_COLUMN].astype(int)

	train_set, val_set, x_train_df, x_val_df, y_train_sr, y_val_sr = split_dataset(
		work_df,
		feature_columns=FEATURE_COLUMNS,
		target_column=TARGET_COLUMN,
		test_size=TEST_SIZE,
		random_state=RANDOM_STATE,
		shuffle=True,
	)

	x_train_np = x_train_df.to_numpy(dtype=float)
	x_val_np = x_val_df.to_numpy(dtype=float)
	y_train_np = y_train_sr.to_numpy(dtype=int)
	y_val_np = y_val_sr.to_numpy(dtype=int)

	mean, std = fit_standard_scaler(x_train_np)
	x_train_scaled = transform_standard_scaler(x_train_np, mean, std)
	x_val_scaled = transform_standard_scaler(x_val_np, mean, std)

	train_loader = make_dataloader(x_train_scaled, y_train_np, BATCH_SIZE, shuffle=True)
	val_loader = make_dataloader(x_val_scaled, y_val_np, BATCH_SIZE, shuffle=False)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = DiagnosisMLP().to(device)
	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

	history_rows: list[dict[str, float | int]] = []

	for epoch in range(1, EPOCHS + 1):
		train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
		val_loss = loader_average_loss(model, val_loader, criterion, device)

		y_val_true, y_val_pred = predict_from_loader(model, val_loader, device)
		val_metrics = evaluate_predictions(y_val_true, y_val_pred, labels=[0, 1], positive_label=1)

		history_rows.append(
			{
				"epoch": epoch,
				"train_loss": train_loss,
				"val_loss": val_loss,
				"val_accuracy": float(val_metrics["accuracy"]),
			}
		)

		if epoch % 10 == 0 or epoch == 1 or epoch == EPOCHS:
			print(
				f"Epoch {epoch:03d}/{EPOCHS} | "
				f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
				f"val_acc={float(val_metrics['accuracy']):.2%}"
			)

	history_df = pd.DataFrame(history_rows)
	history_path = output_dir / "training_history.csv"
	history_df.to_csv(history_path, index=False)

	curves_path = output_dir / "learning_curves.png"
	save_learning_curves(history_df, curves_path, "Diagnosis MLP Learning Curves")

	y_val_true, y_val_pred = predict_from_loader(model, val_loader, device)
	final_metrics = evaluate_predictions(y_val_true, y_val_pred, labels=[0, 1], positive_label=1)
	confusion_df = final_metrics["confusion_df"]

	confusion_path = output_dir / "confusion_matrix.png"
	save_confusion_matrix_plot(
		confusion_df,
		f"Diagnosis MLP\nAccuracy: {float(final_metrics['accuracy']):.2%}",
		confusion_path,
		class_labels=["healthy (0)", "sick (1)"],
	)

	metrics_path = output_dir / "validation_metrics.csv"
	pd.DataFrame(
		[
			{
				"accuracy": float(final_metrics["accuracy"]),
				"precision_binary": float(final_metrics.get("precision_binary", 0.0)),
				"recall_binary": float(final_metrics.get("recall_binary", 0.0)),
				"precision_weighted": float(final_metrics.get("precision_weighted", 0.0)),
				"recall_weighted": float(final_metrics.get("recall_weighted", 0.0)),
			}
		]
	).to_csv(metrics_path, index=False)

	print()
	print("=" * 80)
	print("TASK03 - SUMMARY")
	print("=" * 80)
	print(f"Dataset: {DATA_FILE}")
	print(f"Train/Validation split: {len(train_set)}/{len(val_set)}")
	print("Architecture: 3 -> 16 -> 8 -> 2 (ReLU hidden, logits output)")
	print("\n")
	print(f"Final validation accuracy: {float(final_metrics['accuracy']):.2%}")
	print(f"Precision (diagnosis=1): {float(final_metrics.get('precision_binary', 0.0)):.2%}")
	print(f"Recall (diagnosis=1): {float(final_metrics.get('recall_binary', 0.0)):.2%}")
	print("Confusion matrix (validation):")
	print(confusion_df)
	print()
	print(f"Saved: {history_path}")
	print(f"Saved: {curves_path}")
	print(f"Saved: {confusion_path}")
	print(f"Saved: {metrics_path}")


if __name__ == "__main__":
	main()
