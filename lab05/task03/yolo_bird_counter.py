from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
from ultralytics import YOLO

LAB05_ROOT = Path(__file__).resolve().parents[1]
if str(LAB05_ROOT) not in sys.path:
	sys.path.insert(0, str(LAB05_ROOT))

from common.comparison import build_result_row, read_comparison_rows, score_rows, write_output_csv
from utils.visualization import save_annotated
from utils.yolo_utils import (
	YoloPreprocessConfig,
	extract_detections,
	filter_detections_by_class,
	preprocess_for_yolo,
)

FLYING_CLASSES = "bird,airplane,kite"


@dataclass(frozen=True)
class DetectionProfile:
	name: str
	confidence: float
	imgsz: int
	classes: str
	upscale_factor: float
	clahe: bool
	sharpen: bool


def _project_root() -> Path:
	return Path(__file__).resolve().parent


def _iter_image_paths(input_dir: Path) -> list[Path]:
	return sorted(path for path in input_dir.glob("*.jpg") if path.is_file())


def _parse_allowed_classes(classes_raw: str) -> set[str]:
	return {value.strip().lower() for value in classes_raw.split(",") if value.strip()}


def _build_arg_parser() -> argparse.ArgumentParser:
	root = _project_root()
	parser = argparse.ArgumentParser(description="Count birds with YOLO on JPG images.")
	parser.add_argument("--input-dir", type=Path, default=root.parent / "common" / "input")
	parser.add_argument("--comparison-csv", type=Path, default=root.parent / "common" / "comparison.csv")
	parser.add_argument("--output-csv", type=Path, default=root / "output" / "yolo_count.csv")
	parser.add_argument("--output-dir", type=Path, default=root / "output")
	parser.add_argument("--model", type=str, default="yolov8n.pt")
	parser.add_argument("--confidence", type=float, default=0.18)
	parser.add_argument("--confidence-grid", type=str, default="")
	parser.add_argument("--imgsz", type=int, default=1280)
	parser.add_argument("--classes", type=str, default=FLYING_CLASSES)
	parser.add_argument("--upscale-factor", type=float, default=1.0)
	parser.add_argument("--clahe", action="store_true")
	parser.add_argument("--sharpen", action="store_true")
	parser.add_argument("--save-preprocessed", action="store_true")
	parser.add_argument("--auto-profile", dest="auto_profile", action="store_true", default=True)
	parser.add_argument("--no-auto-profile", dest="auto_profile", action="store_false")
	parser.add_argument("--sweep-output-csv", type=Path, default=root / "output" / "confidence_sweep.csv")
	parser.add_argument("--device", type=str, default=None)
	return parser


def _profile_from_args(args: argparse.Namespace) -> DetectionProfile:
	return DetectionProfile(
		name="user_config",
		confidence=max(0.0, min(1.0, args.confidence)),
		imgsz=max(64, args.imgsz),
		classes=args.classes,
		upscale_factor=max(args.upscale_factor, 1.0),
		clahe=args.clahe,
		sharpen=args.sharpen,
	)


def _default_recovery_profiles(user_profile: DetectionProfile) -> list[DetectionProfile]:
	profiles = [user_profile]

	# Tiny-object profile while still keeping flying class filter.
	profiles.append(
		DetectionProfile(
			name="tiny_flying",
			confidence=min(user_profile.confidence, 0.05),
			imgsz=max(user_profile.imgsz, 1920),
			classes=FLYING_CLASSES,
			upscale_factor=max(user_profile.upscale_factor, 2.0),
			clahe=True,
			sharpen=True,
		)
	)

	# Relax class filtering as a last resort if flying classes miss tiny blobs.
	profiles.append(
		DetectionProfile(
			name="tiny_all_classes",
			confidence=min(user_profile.confidence, 0.03),
			imgsz=max(user_profile.imgsz, 1920),
			classes="",
			upscale_factor=max(user_profile.upscale_factor, 2.0),
			clahe=True,
			sharpen=True,
		)
	)

	return profiles


def _parse_confidence_grid(raw: str) -> list[float]:
	if not raw.strip():
		return []
	values: list[float] = []
	for part in raw.split(","):
		part = part.strip()
		if not part:
			continue
		values.append(max(0.0, min(1.0, float(part))))
	return sorted(set(values))


def _write_sweep_csv(path: Path, rows: list[dict[str, str]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(
			handle,
			fieldnames=["confidence", "numeric_correct", "ambiguity_correct", "images"],
		)
		writer.writeheader()
		for row in rows:
			writer.writerow(row)


def _run_single_confidence(
	image_paths: list[Path],
	comparison_rows,
	model,
	allowed_classes: set[str],
	preprocess_config: YoloPreprocessConfig,
	imgsz: int,
	confidence: float,
	device: str | None,
	annotated_dir: Path,
	save_preprocessed: bool,
	preprocessed_dir: Path,
	model_name: str,
) -> list:
	results = []
	for image_path in image_paths:
		image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
		if image_bgr is None:
			results.append(
				build_result_row(
					image_name=image_path.name,
					algorithm_count=None,
					comparison=comparison_rows.get(image_path.stem),
					extra_note="unreadable_image",
				)
			)
			continue

		processed = preprocess_for_yolo(image_bgr, preprocess_config)
		predict_kwargs = {
			"source": processed,
			"conf": confidence,
			"imgsz": imgsz,
			"verbose": False,
		}
		if device:
			predict_kwargs["device"] = device

		prediction = model.predict(**predict_kwargs)[0]
		detections = extract_detections(prediction)
		detections = filter_detections_by_class(detections, allowed_classes)

		algorithm_count = len(detections)
		note = (
			f"yolo_model:{model_name};confidence:{confidence:.3f};imgsz:{imgsz};"
			f"classes:{','.join(sorted(allowed_classes)) if allowed_classes else 'all'};"
			f"upscale_factor:{preprocess_config.upscale_factor:.2f};clahe:{preprocess_config.clahe};"
			f"sharpen:{preprocess_config.sharpen};detections:{algorithm_count}"
		)

		annotated_path = annotated_dir / f"{image_path.stem}_yolo.jpg"
		save_annotated(processed, detections, annotated_path)

		if save_preprocessed:
			preprocessed_dir.mkdir(parents=True, exist_ok=True)
			cv2.imwrite(str(preprocessed_dir / f"{image_path.stem}_prep.jpg"), processed)

		results.append(
			build_result_row(
				image_name=image_path.name,
				algorithm_count=algorithm_count,
				comparison=comparison_rows.get(image_path.stem),
				extra_note=note,
			)
		)

	return results


def _run_detection_profile(
	image_paths: list[Path],
	comparison_rows,
	model,
	profile: DetectionProfile,
	device: str | None,
	annotated_dir: Path,
	save_preprocessed: bool,
	preprocessed_dir: Path,
	model_name: str,
) -> tuple[list, int, int, int]:
	allowed_classes = _parse_allowed_classes(profile.classes)
	preprocess_config = YoloPreprocessConfig(
		upscale_factor=profile.upscale_factor,
		clahe=profile.clahe,
		sharpen=profile.sharpen,
	)
	results = _run_single_confidence(
		image_paths=image_paths,
		comparison_rows=comparison_rows,
		model=model,
		allowed_classes=allowed_classes,
		preprocess_config=preprocess_config,
		imgsz=profile.imgsz,
		confidence=profile.confidence,
		device=device,
		annotated_dir=annotated_dir,
		save_preprocessed=save_preprocessed,
		preprocessed_dir=preprocessed_dir,
		model_name=model_name,
	)
	result_by_name = {row.picture_name: row for row in results}
	ordered_results = [result_by_name[path.stem] for path in image_paths if path.stem in result_by_name]
	numeric_correct, ambiguity_correct = score_rows(ordered_results)
	total_detections = sum(int(row.algorithm_count) for row in ordered_results if row.algorithm_count.isdigit())
	return ordered_results, numeric_correct, ambiguity_correct, total_detections


def main() -> None:
	parser = _build_arg_parser()
	args = parser.parse_args()

	image_paths = _iter_image_paths(args.input_dir)
	if not image_paths:
		raise FileNotFoundError(f"No JPG files found in {args.input_dir}")

	comparison_rows = read_comparison_rows(args.comparison_csv)
	base_profile = _profile_from_args(args)

	model = YOLO(args.model)

	annotated_dir = args.output_dir / "annotated"
	preprocessed_dir = args.output_dir / "preprocessed"
	profiles = [base_profile]
	if args.auto_profile:
		profiles = _default_recovery_profiles(base_profile)

	candidates: list[tuple[DetectionProfile, list, int, int, int]] = []
	for idx, profile in enumerate(profiles):
		ordered_results, numeric_score, ambiguity_score, total_detections = _run_detection_profile(
			image_paths=image_paths,
			comparison_rows=comparison_rows,
			model=model,
			profile=profile,
			device=args.device,
			annotated_dir=annotated_dir,
			save_preprocessed=args.save_preprocessed and idx == 0,
			preprocessed_dir=preprocessed_dir,
			model_name=args.model,
		)
		candidates.append((profile, ordered_results, numeric_score, ambiguity_score, total_detections))

	best_profile, ordered_results, numeric_correct, ambiguity_correct, total_detections = max(
		candidates,
		key=lambda item: (item[3], item[2], item[4]),
	)

	write_output_csv(args.output_csv, ordered_results)

	print(f"Processed images: {len(ordered_results)}")
	print(f"Correctly matched counts (numeric rows): {numeric_correct}")
	print(f"Correctly matched counts (with ambiguity rules): {ambiguity_correct}")
	print(f"Total detections in selected run: {total_detections}")
	print(
		"Selected profile: "
		f"{best_profile.name} "
		f"(conf={best_profile.confidence:.3f}, imgsz={best_profile.imgsz}, "
		f"classes={best_profile.classes or 'all'}, upscale={best_profile.upscale_factor:.2f}, "
		f"clahe={best_profile.clahe}, sharpen={best_profile.sharpen})"
	)
	print(f"CSV written to: {args.output_csv}")
	print(f"Annotated images in: {annotated_dir}")

	confidence_grid = _parse_confidence_grid(args.confidence_grid)
	if confidence_grid:
		sweep_rows: list[dict[str, str]] = []
		sweep_allowed_classes = _parse_allowed_classes(best_profile.classes)
		sweep_preprocess = YoloPreprocessConfig(
			upscale_factor=best_profile.upscale_factor,
			clahe=best_profile.clahe,
			sharpen=best_profile.sharpen,
		)
		for conf in confidence_grid:
			conf_results = _run_single_confidence(
				image_paths=image_paths,
				comparison_rows=comparison_rows,
				model=model,
				allowed_classes=sweep_allowed_classes,
				preprocess_config=sweep_preprocess,
				imgsz=best_profile.imgsz,
				confidence=conf,
				device=args.device,
				annotated_dir=annotated_dir,
				save_preprocessed=False,
				preprocessed_dir=preprocessed_dir,
				model_name=args.model,
			)
			result_by_name = {row.picture_name: row for row in conf_results}
			ordered_conf_results = [result_by_name[path.stem] for path in image_paths if path.stem in result_by_name]
			numeric_score, ambiguity_score = score_rows(ordered_conf_results)
			sweep_rows.append(
				{
					"confidence": f"{conf:.3f}",
					"numeric_correct": str(numeric_score),
					"ambiguity_correct": str(ambiguity_score),
					"images": str(len(ordered_conf_results)),
				}
			)

		_write_sweep_csv(args.sweep_output_csv, sweep_rows)
		best = max(
			sweep_rows,
			key=lambda row: (int(row["ambiguity_correct"]), int(row["numeric_correct"])),
		)
		print(f"Confidence sweep written to: {args.sweep_output_csv}")
		print(
			"Best confidence from sweep: "
			f"{best['confidence']} (ambiguity={best['ambiguity_correct']}, numeric={best['numeric_correct']})"
		)


if __name__ == "__main__":
	main()
