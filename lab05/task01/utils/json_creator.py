from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def confidence_tag(confidence_threshold: float) -> str:
	# Format threshold as two-digit tag, e.g. 0.1 -> 01.
	return f"{int(round(confidence_threshold * 10)):02d}"


def extract_detections(result: Any) -> list[dict[str, Any]]:
	# Convert a single Ultralytics result into strict JSON detection entries.
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


def build_image_payload(
	image_name: str, confidence_threshold: float, detections: list[dict[str, Any]]
) -> dict[str, Any]:
	return {
		"image": image_name,
		"confidence_threshold": confidence_threshold,
		"detections": detections,
	}


def build_frame_payload(
	frame_id: int, timestamp: float, detections: list[dict[str, Any]]
) -> dict[str, Any]:
	return {
		"frame_id": frame_id,
		"timestamp": round(timestamp, 3),
		"detections": detections,
	}


def build_video_payload(
	video_name: str,
	confidence_threshold: float,
	fps: float,
	frames: list[dict[str, Any]],
) -> dict[str, Any]:
	return {
		"video": video_name,
		"confidence_threshold": confidence_threshold,
		"fps": round(fps, 3),
		"frames": frames,
	}


def write_json(file_path: Path, payload: dict[str, Any]) -> None:
	file_path.parent.mkdir(parents=True, exist_ok=True)
	with file_path.open("w", encoding="utf-8") as file:
		json.dump(payload, file, indent=2)
