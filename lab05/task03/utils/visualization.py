from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _color_from_class(class_id: int) -> tuple[int, int, int]:
	# Deterministic pseudo-random color based on class id.
	b = (37 * class_id + 47) % 255
	g = (17 * class_id + 97) % 255
	r = (29 * class_id + 151) % 255
	return int(b), int(g), int(r)


def draw_detections(image: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
	annotated = image.copy()
	height, width = annotated.shape[:2]

	for detection in detections:
		bbox = detection["bbox"]
		class_id = int(detection["class_id"])
		class_name = detection["class_name"]
		confidence = float(detection["confidence"])

		x_center = float(bbox["x_center"]) * width
		y_center = float(bbox["y_center"]) * height
		box_w = float(bbox["width"]) * width
		box_h = float(bbox["height"]) * height

		x1 = max(0, int(x_center - box_w / 2))
		y1 = max(0, int(y_center - box_h / 2))
		x2 = min(width - 1, int(x_center + box_w / 2))
		y2 = min(height - 1, int(y_center + box_h / 2))

		color = _color_from_class(class_id)
		cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

		label = f"{class_name} {confidence:.2f}"
		(text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
		text_y = max(0, y1 - text_h - 6)
		cv2.rectangle(annotated, (x1, text_y), (x1 + text_w + 6, text_y + text_h + 6), color, -1)
		cv2.putText(
			annotated,
			label,
			(x1 + 3, text_y + text_h + 2),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.5,
			(255, 255, 255),
			1,
			cv2.LINE_AA,
		)

	return annotated


def save_annotated(image: np.ndarray, detections: list[dict[str, Any]], output_path: Path) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	annotated = draw_detections(image, detections)
	cv2.imwrite(str(output_path), annotated)
