from __future__ import annotations

import argparse
import csv
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2

from utils.blob_counter import BlobFilterConfig, count_blobs
from utils.preprocessing import PreprocessConfig, preprocess_for_blobs
from utils.review_export import save_review_artifacts


@dataclass(frozen=True)
class RuntimeConfig:
	input_dir: Path
	comparison_csv: Path
	output_csv: Path
	output_dir: Path
	preprocess: PreprocessConfig
	blob_filter: BlobFilterConfig
	hybrid: HybridConfig
	max_workers: int
	save_intermediate: bool


@dataclass(frozen=True)
class HybridConfig:
	switch_threshold: int
	secondary_preprocess: PreprocessConfig
	secondary_blob_filter: BlobFilterConfig


@dataclass(frozen=True)
class ComparisonRow:
	number: str
	picture_name: str
	official_count_raw: str
	notes: str


@dataclass(frozen=True)
class ImageResult:
	number: str
	picture_name: str
	official_count: str
	algorithm_count: str
	found: str
	missed: str
	notes: str


def _project_root() -> Path:
	return Path(__file__).resolve().parent


def _iter_image_paths(input_dir: Path) -> list[Path]:
	# Return paths only so image content is loaded lazily inside worker threads.
	return sorted(path for path in input_dir.glob("*.jpg") if path.is_file())


def _read_comparison_rows(comparison_csv: Path) -> dict[str, ComparisonRow]:
	rows: dict[str, ComparisonRow] = {}
	with comparison_csv.open("r", encoding="utf-8", newline="") as handle:
		reader = csv.DictReader(handle, skipinitialspace=True)
		for row in reader:
			picture_name = (row.get("picture_name") or "").strip()
			if not picture_name:
				continue

			rows[picture_name] = ComparisonRow(
				number=(row.get("number") or "").strip(),
				picture_name=picture_name,
				official_count_raw=(row.get("official_count") or "").strip(),
				notes=(row.get("notes") or "").strip(),
			)
	return rows


def _parse_official_count(raw_value: str) -> tuple[int | None, str | None]:
	value = raw_value.strip()
	if not value:
		return None, None

	if value.endswith("?") and value[:-1].isdigit():
		return int(value[:-1]), f"official_count_uncertain:{value}"

	if value.isdigit():
		return int(value), None

	match = re.search(r"\d+", value)
	if match:
		return int(match.group(0)), f"official_count_nonstandard:{value}"

	return None, f"official_count_unparsed:{value}"


def _merge_notes(*notes: str | None) -> str:
	merged = [note.strip() for note in notes if note and note.strip()]
	if not merged:
		return ""
	# Preserve order while deduplicating.
	unique = list(dict.fromkeys(merged))
	return " | ".join(unique)


def _build_result_row(
	image_name: str,
	algorithm_count: int | None,
	comparison: ComparisonRow | None,
	extra_note: str | None,
) -> ImageResult:
	if comparison is None:
		return ImageResult(
			number="",
			picture_name=Path(image_name).stem,
			official_count="",
			algorithm_count="" if algorithm_count is None else str(algorithm_count),
			found="",
			missed="",
			notes=_merge_notes(extra_note, "missing_in_comparison_csv"),
		)

	official_numeric, parse_note = _parse_official_count(comparison.official_count_raw)
	found = ""
	missed = ""
	if official_numeric is not None and algorithm_count is not None:
		found = str(min(official_numeric, algorithm_count))
		missed = str(max(official_numeric - algorithm_count, 0))

	return ImageResult(
		number=comparison.number,
		picture_name=comparison.picture_name,
		official_count=comparison.official_count_raw,
		algorithm_count="" if algorithm_count is None else str(algorithm_count),
		found=found,
		missed=missed,
		notes=_merge_notes(comparison.notes, parse_note, extra_note),
	)


def _count_single_image(
	image_path: Path,
	comparison_rows: dict[str, ComparisonRow],
	config: RuntimeConfig,
) -> ImageResult:
	image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
	if image_bgr is None:
		return _build_result_row(
			image_name=image_path.name,
			algorithm_count=None,
			comparison=comparison_rows.get(image_path.stem),
			extra_note="unreadable_image",
		)

	primary_preprocessed = preprocess_for_blobs(image_bgr, config.preprocess)
	primary_count, primary_components, _ = count_blobs(
		primary_preprocessed.separated,
		config.blob_filter,
	)

	algorithm_count = primary_count
	components = primary_components
	preprocessed = primary_preprocessed
	chosen_profile = "primary"
	secondary_count: int | None = None

	if primary_count >= config.hybrid.switch_threshold:
		secondary_preprocessed = preprocess_for_blobs(image_bgr, config.hybrid.secondary_preprocess)
		secondary_count, secondary_components, _ = count_blobs(
			secondary_preprocessed.separated,
			config.hybrid.secondary_blob_filter,
		)
		if secondary_count > primary_count:
			algorithm_count = secondary_count
			components = secondary_components
			preprocessed = secondary_preprocessed
			chosen_profile = "secondary"

	mask_path, overlay_path = save_review_artifacts(
		output_dir=config.output_dir,
		image_name=image_path.name,
		original_bgr=image_bgr,
		preprocess=preprocessed,
		components=components,
		save_intermediate=config.save_intermediate,
	)

	accepted = sum(1 for component in components if component.accepted)
	rejected = len(components) - accepted
	secondary_note = "none" if secondary_count is None else str(secondary_count)
	detail_note = (
		f"accepted_components:{accepted};rejected_components:{rejected};"
		f"hybrid:true;chosen_profile:{chosen_profile};"
		f"primary_count:{primary_count};secondary_count:{secondary_note};"
		f"switch_threshold:{config.hybrid.switch_threshold};"
		f"mask:{mask_path.name};overlay:{overlay_path.name}"
	)

	return _build_result_row(
		image_name=image_path.name,
		algorithm_count=algorithm_count,
		comparison=comparison_rows.get(image_path.stem),
		extra_note=detail_note,
	)


def _write_output_csv(output_csv: Path, rows: list[ImageResult]) -> None:
	output_csv.parent.mkdir(parents=True, exist_ok=True)
	with output_csv.open("w", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(
			handle,
			fieldnames=[
				"number",
				"picture_name",
				"official_count",
				"algorithm_count",
				"found",
				"missed",
				"notes",
			],
		)
		writer.writeheader()
		for row in rows:
			writer.writerow(
				{
					"number": row.number,
					"picture_name": row.picture_name,
					"official_count": row.official_count,
					"algorithm_count": row.algorithm_count,
					"found": row.found,
					"missed": row.missed,
					"notes": row.notes,
				}
			)


def _ambiguity_accepted(number: str, algorithm_count: str, official_count: str) -> bool:
	if not algorithm_count.isdigit():
		return False

	algo = int(algorithm_count)
	if number == "6":
		return algo in {5, 6}
	if number == "14":
		return algo in {15, 16, 17}
	return official_count.isdigit() and algo == int(official_count)


def _build_arg_parser() -> argparse.ArgumentParser:
	root = _project_root()
	parser = argparse.ArgumentParser(description="Count birds in JPG images using OpenCV blobs.")
	parser.add_argument("--input-dir", type=Path, default=root / "input")
	parser.add_argument("--comparison-csv", type=Path, default=root / "comparison.csv")
	parser.add_argument("--output-csv", type=Path, default=root / "output" / "opencv_count.csv")
	parser.add_argument("--output-dir", type=Path, default=root / "output")
	parser.add_argument("--hybrid-switch-threshold", type=int, default=10)
	parser.add_argument("--max-workers", type=int, default=max(1, min(32, (os.cpu_count() or 1) * 2)))
	parser.add_argument("--save-intermediate", action="store_true")
	return parser


def main() -> None:
	parser = _build_arg_parser()
	args = parser.parse_args()

	config = RuntimeConfig(
		input_dir=args.input_dir,
		comparison_csv=args.comparison_csv,
		output_csv=args.output_csv,
		output_dir=args.output_dir,
		preprocess=PreprocessConfig(
			clip_limit=1.77,
			tile_grid_size=(8, 8),
			blur_kernel_size=(5, 5),
			adaptive_block_size=9,
			adaptive_c=2.94,
			erode_iterations=0,
			erode_kernel_size=(3, 3),
			morph_kernel_size=(3, 3),
			watershed_enabled=True,
			watershed_fg_ratio=0.48,
			watershed_bg_dilate_iterations=1,
		),
		blob_filter=BlobFilterConfig(
			min_area=2,
			max_area=780,
		),
		hybrid=HybridConfig(
			switch_threshold=args.hybrid_switch_threshold,
			secondary_preprocess=PreprocessConfig(
				clip_limit=1.95,
				tile_grid_size=(8, 8),
				blur_kernel_size=(7, 7),
				adaptive_block_size=9,
				adaptive_c=1.08,
				erode_iterations=0,
				erode_kernel_size=(3, 3),
				morph_kernel_size=(3, 3),
				watershed_enabled=False,
				watershed_fg_ratio=0.41,
				watershed_bg_dilate_iterations=3,
			),
			secondary_blob_filter=BlobFilterConfig(
				min_area=6,
				max_area=950,
			),
		),
		max_workers=max(1, args.max_workers),
		save_intermediate=args.save_intermediate,
	)

	image_paths = _iter_image_paths(config.input_dir)
	if not image_paths:
		raise FileNotFoundError(f"No JPG files found in {config.input_dir}")

	comparison_rows = _read_comparison_rows(config.comparison_csv)
	results: list[ImageResult] = []

	with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
		future_to_path = {
			executor.submit(_count_single_image, image_path, comparison_rows, config): image_path
			for image_path in image_paths
		}
		for future in as_completed(future_to_path):
			image_path = future_to_path[future]
			try:
				result = future.result()
			except Exception as exc:  # pragma: no cover
				result = _build_result_row(
					image_name=image_path.name,
					algorithm_count=None,
					comparison=comparison_rows.get(image_path.stem),
					extra_note=f"processing_error:{exc}",
				)
			results.append(result)

	result_by_name = {row.picture_name: row for row in results}
	ordered_results = [result_by_name[path.stem] for path in image_paths if path.stem in result_by_name]
	_write_output_csv(config.output_csv, ordered_results)

	numeric_correct = 0
	ambiguity_correct = 0
	for result in ordered_results:
		official_raw = result.official_count.strip().rstrip("?")
		if official_raw.isdigit() and result.algorithm_count.isdigit() and int(official_raw) == int(result.algorithm_count):
			numeric_correct += 1

		if _ambiguity_accepted(result.number.strip(), result.algorithm_count.strip(), result.official_count.strip()):
			ambiguity_correct += 1

	print(f"Processed images: {len(ordered_results)}")
	print(f"Correctly matched counts (numeric rows): {numeric_correct}")
	print(f"Correctly matched counts (with ambiguity rules): {ambiguity_correct}")
	print(f"CSV written to: {config.output_csv}")
	print(f"Review artifacts in: {config.output_dir}")


if __name__ == "__main__":
	main()
