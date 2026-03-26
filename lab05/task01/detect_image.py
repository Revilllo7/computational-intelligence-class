from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

from utils.bb_visualisation import save_annotated
from utils.json_creator import (
	build_image_payload,
	confidence_tag,
	extract_detections,
	write_json,
)

THRESHOLDS = [0.1, 0.3, 0.5, 0.7]


def _project_root() -> Path:
	return Path(__file__).resolve().parent


def main() -> None:
	root = _project_root()
	input_image = root / "input" / "office_yolo.png"
	output_dir = root / "output" / "image"
	annotated_dir = output_dir / "annotated"

	image = cv2.imread(str(input_image))
	if image is None:
		raise FileNotFoundError(f"Image not found or unreadable: {input_image}")

	model = YOLO("yolov8n.pt")

	for threshold in THRESHOLDS:
		result = model.predict(source=image, conf=threshold, verbose=False)[0]
		detections = extract_detections(result)

		payload = build_image_payload(input_image.name, threshold, detections)
		json_path = output_dir / f"detection_conf_{confidence_tag(threshold)}.json"
		write_json(json_path, payload)

		image_out_name = f"{input_image.stem}_conf_{confidence_tag(threshold)}."
		save_annotated(image, detections, annotated_dir / image_out_name)

		class_counts = Counter(det["class_name"] for det in detections)
		print(f"[image] conf={threshold:.1f} detections={len(detections)}")
		if class_counts:
			formatted = ", ".join(f"{name}:{count}" for name, count in sorted(class_counts.items()))
			print(f"[image] classes: {formatted}")
		else:
			print("[image] classes: none")


if __name__ == "__main__":
	main()
