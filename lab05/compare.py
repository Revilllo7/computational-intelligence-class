from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np


OUTER_PADDING = 24
INNER_PADDING = 14
PANEL_LABEL_HEIGHT = 44
HEADER_HEIGHT = 66
DISPLAY_SCALE = 2.0
BACKGROUND_COLOR = (240, 243, 247)
HEADER_COLOR = (0, 0, 0)
PANEL_BG_COLOR = (252, 252, 252)
PANEL_BORDER_COLOR = (196, 204, 214)
TEXT_COLOR = (24, 30, 37)
HEADER_TEXT_COLOR = (245, 248, 252)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _project_root() -> Path:
	return Path(__file__).resolve().parent


def _resize_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
	height, width = image.shape[:2]
	if height == target_height:
		return image
	if height <= 0:
		return image
	new_width = max(1, int(round(width * (target_height / height))))
	interpolation = cv2.INTER_CUBIC if target_height > height else cv2.INTER_AREA
	return cv2.resize(image, (new_width, target_height), interpolation=interpolation)


def _load_required_image(path: Path) -> np.ndarray:
	image = cv2.imread(str(path), cv2.IMREAD_COLOR)
	if image is None:
		raise FileNotFoundError(f"Cannot read image: {path}")
	return image


def _read_reference_numbers(comparison_csv: Path) -> dict[str, str]:
	if not comparison_csv.exists():
		return {}

	numbers: dict[str, str] = {}
	with comparison_csv.open("r", encoding="utf-8", newline="") as handle:
		reader = csv.DictReader(handle, skipinitialspace=True)
		for row in reader:
			picture_name = (row.get("picture_name") or "").strip()
			number = (row.get("number") or "").strip()
			if picture_name:
				numbers[picture_name] = number
	return numbers


def _draw_text(img: np.ndarray, text: str, x: int, y: int, scale: float, thickness: int) -> None:
	cv2.putText(img, text, (x, y), FONT, scale, TEXT_COLOR, thickness, cv2.LINE_AA)


def _draw_header_text(img: np.ndarray, text: str, x: int, y: int, scale: float, thickness: int) -> None:
	cv2.putText(img, text, (x, y), FONT, scale, HEADER_TEXT_COLOR, thickness, cv2.LINE_AA)


def _render_panel(image: np.ndarray, label: str, target_height: int) -> np.ndarray:
	resized = _resize_to_height(image, target_height)
	panel_h, panel_w = resized.shape[:2]
	canvas = np.full(
		(panel_h + 2 * INNER_PADDING + PANEL_LABEL_HEIGHT, panel_w + 2 * INNER_PADDING, 3),
		PANEL_BG_COLOR,
		dtype=np.uint8,
	)
	cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, canvas.shape[0] - 1), PANEL_BORDER_COLOR, 1)
	canvas[PANEL_LABEL_HEIGHT + INNER_PADDING : PANEL_LABEL_HEIGHT + INNER_PADDING + panel_h, INNER_PADDING : INNER_PADDING + panel_w] = resized

	(text_w, text_h), _ = cv2.getTextSize(label, FONT, 0.75, 2)
	text_x = max(INNER_PADDING, (canvas.shape[1] - text_w) // 2)
	text_y = (PANEL_LABEL_HEIGHT + text_h) // 2
	_draw_text(canvas, label, text_x, text_y, 0.75, 2)
	return canvas


def _build_comparison_canvas(reference_number: str, left_panel: np.ndarray, middle_panel: np.ndarray, right_panel: np.ndarray) -> np.ndarray:
	panels = [left_panel, middle_panel, right_panel]
	panel_height = max(panel.shape[0] for panel in panels)

	normalized_panels = []
	for panel in panels:
		if panel.shape[0] == panel_height:
			normalized_panels.append(panel)
			continue
		padded = np.full((panel_height, panel.shape[1], 3), BACKGROUND_COLOR, dtype=np.uint8)
		y_offset = (panel_height - panel.shape[0]) // 2
		padded[y_offset : y_offset + panel.shape[0], :] = panel
		normalized_panels.append(padded)

	body_width = sum(panel.shape[1] for panel in normalized_panels) + (len(normalized_panels) - 1) * INNER_PADDING
	body_height = panel_height

	canvas_h = HEADER_HEIGHT + OUTER_PADDING + body_height + OUTER_PADDING
	canvas_w = OUTER_PADDING + body_width + OUTER_PADDING
	canvas = np.full((canvas_h, canvas_w, 3), BACKGROUND_COLOR, dtype=np.uint8)
	cv2.rectangle(canvas, (0, 0), (canvas_w - 1, HEADER_HEIGHT), HEADER_COLOR, -1)

	number_display = reference_number if reference_number else "?"
	header_text = f"Comparison #{number_display}"
	(header_w, header_h), _ = cv2.getTextSize(header_text, FONT, 0.95, 2)
	header_x = max(OUTER_PADDING, (canvas_w - header_w) // 2)
	header_y = max(header_h + 8, (HEADER_HEIGHT + header_h) // 2)
	_draw_header_text(canvas, header_text, header_x, header_y, 0.95, 2)

	x = OUTER_PADDING
	y = HEADER_HEIGHT + OUTER_PADDING
	for idx, panel in enumerate(normalized_panels):
		canvas[y : y + panel.shape[0], x : x + panel.shape[1]] = panel
		x += panel.shape[1]
		if idx < len(normalized_panels) - 1:
			x += INNER_PADDING

	return canvas


def _find_variant(annotated_dir: Path, stem: str, suffix: str) -> Path:
	candidate = annotated_dir / f"{stem}_{suffix}.jpg"
	if candidate.exists():
		return candidate
	raise FileNotFoundError(f"Missing annotated image: {candidate}")


def main() -> None:
	root = _project_root()
	input_dir = root / "common" / "input"
	comparison_csv = root / "common" / "comparison.csv"
	task02_annotated_dir = root / "task02" / "output" / "annotated"
	task03_annotated_dir = root / "task03" / "output" / "annotated"
	comparisons_dir = root / "comparisons"
	comparisons_dir.mkdir(parents=True, exist_ok=True)
	reference_numbers = _read_reference_numbers(comparison_csv)

	input_images = sorted(path for path in input_dir.glob("*.jpg") if path.is_file())
	if not input_images:
		raise FileNotFoundError(f"No JPG images found in {input_dir}")

	processed = 0
	for input_path in input_images:
		stem = input_path.stem
		reference_number = reference_numbers.get(stem, "")
		task02_path = _find_variant(task02_annotated_dir, stem, "blobs")
		task03_path = _find_variant(task03_annotated_dir, stem, "yolo")

		left = _load_required_image(task02_path)
		middle = _load_required_image(input_path)
		right = _load_required_image(task03_path)

		target_height = max(1, int(round(middle.shape[0] * DISPLAY_SCALE)))
		left_panel = _render_panel(middle, "Original", target_height)
		middle_panel = _render_panel(left, "OpenCV", target_height)
		right_panel = _render_panel(right, "YOLO", target_height)
		comparison = _build_comparison_canvas(reference_number, left_panel, middle_panel, right_panel)

		output_path = comparisons_dir / f"{stem}_comparison.jpg"
		cv2.imwrite(str(output_path), comparison)
		processed += 1

	print(f"Generated comparisons: {processed}")
	print(f"Saved to: {comparisons_dir}")


if __name__ == "__main__":
	main()
