from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class YoloPreprocessConfig:
	upscale_factor: float = 1.0
	clahe: bool = False
	clahe_clip_limit: float = 2.0
	clahe_tile_grid: tuple[int, int] = (8, 8)
	sharpen: bool = False


def preprocess_for_yolo(image_bgr: np.ndarray, config: YoloPreprocessConfig) -> np.ndarray:
	processed = image_bgr.copy()

	if config.upscale_factor > 1.0:
		h, w = processed.shape[:2]
		new_w = max(1, int(round(w * config.upscale_factor)))
		new_h = max(1, int(round(h * config.upscale_factor)))
		processed = cv2.resize(processed, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

	if config.clahe:
		lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
		l, a, b = cv2.split(lab)
		clahe = cv2.createCLAHE(
			clipLimit=max(config.clahe_clip_limit, 0.1),
			tileGridSize=config.clahe_tile_grid,
		)
		l = clahe.apply(l)
		processed = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

	if config.sharpen:
		kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
		processed = cv2.filter2D(processed, -1, kernel)

	return processed


def extract_detections(result: Any) -> list[dict[str, Any]]:
	detections: list[dict[str, Any]] = []
	names = result.names

	if result.boxes is None or len(result.boxes) == 0:
		return detections

	classes = result.boxes.cls.tolist()
	confidences = result.boxes.conf.tolist()
	boxes_xywhn = result.boxes.xywhn.tolist()

	for cls_id, conf, xywhn in zip(classes, confidences, boxes_xywhn):
		class_id = int(cls_id)
		detections.append(
			{
				"class_id": class_id,
				"class_name": str(names[class_id]),
				"confidence": round(float(conf), 4),
				"bbox": {
					"x_center": round(float(xywhn[0]), 6),
					"y_center": round(float(xywhn[1]), 6),
					"width": round(float(xywhn[2]), 6),
					"height": round(float(xywhn[3]), 6),
				},
			}
		)

	return detections


def filter_detections_by_class(
	detections: list[dict[str, Any]],
	allowed_classes: set[str],
) -> list[dict[str, Any]]:
	if not allowed_classes:
		return detections
	allowed = {name.strip().lower() for name in allowed_classes if name.strip()}
	if not allowed:
		return detections
	return [det for det in detections if str(det["class_name"]).lower() in allowed]
