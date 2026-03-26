from __future__ import annotations

from collections import defaultdict
from typing import Any


def build_video_stats(
	video_name: str,
	confidence_threshold: float,
	frames: list[dict[str, Any]],
) -> dict[str, Any]:
	class_counts: dict[str, int] = defaultdict(int)
	total_detections = 0

	for frame in frames:
		detections = frame["detections"]
		total_detections += len(detections)
		for detection in detections:
			class_counts[str(detection["class_name"])] += 1

	return {
		"video": video_name,
		"confidence_threshold": confidence_threshold,
		"total_frames": len(frames),
		"total_detections": total_detections,
		"class_counts": dict(sorted(class_counts.items(), key=lambda x: x[0])),
	}
