from __future__ import annotations

from pathlib import Path

import cv2
from ultralytics import YOLO

from utils.bb_visualisation import save_annotated
from utils.json_creator import (
	build_frame_payload,
	build_video_payload,
	extract_detections,
	write_json,
	confidence_tag
)
from utils.stats import build_video_stats

THRESHOLDS = [0.1, 0.3, 0.5, 0.7]
VIDEOS = [("office_yolo.mp4", "video1"), ("street_yolo.mp4", "video2")]


def _project_root() -> Path:
	return Path(__file__).resolve().parent


def _process_video(model: YOLO, input_video: Path, output_dir: Path, threshold: float) -> None:
	capture = cv2.VideoCapture(str(input_video))
	if not capture.isOpened():
		raise FileNotFoundError(f"Video not found or unreadable: {input_video}")

	output_dir.mkdir(parents=True, exist_ok=True)
	frames_dir = output_dir / "frames"
	frames_dir.mkdir(parents=True, exist_ok=True)

	fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
	if fps <= 0:
		fps = 30.0

	frames_payload: list[dict] = []
	frame_id = 0

	while True:
		success, frame = capture.read()
		if not success:
			break

		frame_id += 1
		result = model.predict(source=frame, conf=threshold, verbose=False)[0]
		detections = extract_detections(result)
		timestamp = frame_id / fps

		frames_payload.append(build_frame_payload(frame_id, timestamp, detections))

		frame_out = frames_dir / f"frame_{frame_id:06d}.jpg"
		save_annotated(frame, detections, frame_out)

	capture.release()

	detections_payload = build_video_payload(
		video_name=input_video.name,
		confidence_threshold=threshold,
		fps=fps,
		frames=frames_payload,
	)
	write_json(output_dir / "detections.json", detections_payload)

	stats_payload = build_video_stats(
		video_name=input_video.name,
		confidence_threshold=threshold,
		frames=frames_payload,
	)
	write_json(output_dir / "stats.json", stats_payload)

	print(
		f"[video] {input_video.name} conf={threshold:.1f} "
		f"frames={stats_payload['total_frames']} detections={stats_payload['total_detections']}"
	)


def main() -> None:
	root = _project_root()
	input_dir = root / "input"
	output_root = root / "output"

	model = YOLO("yolov8n.pt")

	for input_name, output_name in VIDEOS:
		input_video = input_dir / input_name
		for threshold in THRESHOLDS:
			threshold_dir = output_root / output_name / f"conf_{confidence_tag(threshold)}"
			_process_video(model, input_video, threshold_dir, threshold)


if __name__ == "__main__":
	main()
