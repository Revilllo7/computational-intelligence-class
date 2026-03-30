from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2

LAB05_ROOT = Path(__file__).resolve().parents[1]
if str(LAB05_ROOT) not in sys.path:
	sys.path.insert(0, str(LAB05_ROOT))

from common.comparison import (
	ComparisonRow,
	CountResultRow,
	ambiguity_accepted,
	build_result_row,
	read_comparison_rows,
	score_rows,
	write_output_csv,
	write_output_json,
)

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


def _project_root() -> Path:
	return Path(__file__).resolve().parent


def _iter_image_paths(input_dir: Path) -> list[Path]:
	# Return paths only so image content is loaded lazily inside worker threads.
	return sorted(path for path in input_dir.glob("*.jpg") if path.is_file())


def _count_single_image(
	image_path: Path,
	comparison_rows: dict[str, ComparisonRow],
	config: RuntimeConfig,
) -> CountResultRow:
	image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
	if image_bgr is None:
		return build_result_row(
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

	return build_result_row(
		image_name=image_path.name,
		algorithm_count=algorithm_count,
		comparison=comparison_rows.get(image_path.stem),
		extra_note=detail_note,
	)


def _build_arg_parser() -> argparse.ArgumentParser:
	root = _project_root()
	parser = argparse.ArgumentParser(description="Count birds in JPG images using OpenCV blobs.")
	parser.add_argument("--input-dir", type=Path, default=root.parent / "common" / "input")
	parser.add_argument("--comparison-csv", type=Path, default=root.parent / "common" / "comparison.csv")
	parser.add_argument("--output-csv", type=Path, default=root / "output" / "opencv_count.csv")
	parser.add_argument("--output-json", type=Path, default=root / "output" / "opencv_count.json")
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

	comparison_rows = read_comparison_rows(config.comparison_csv)
	results: list[CountResultRow] = []

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
				result = build_result_row(
					image_name=image_path.name,
					algorithm_count=None,
					comparison=comparison_rows.get(image_path.stem),
					extra_note=f"processing_error:{exc}",
				)
			results.append(result)

	result_by_name = {row.picture_name: row for row in results}
	ordered_results = [result_by_name[path.stem] for path in image_paths if path.stem in result_by_name]
	write_output_csv(config.output_csv, ordered_results)
	write_output_json(args.output_json, ordered_results)

	for row in ordered_results:
		is_match = ambiguity_accepted(
			row.number.strip(),
			row.algorithm_count.strip(),
			row.official_count.strip(),
		)
		bird_counter = row.algorithm_count if row.algorithm_count else "N/A"
		print(
			f"Image #{row.number or '?'} ({row.picture_name}): "
			f"bird_counter={bird_counter}, matches_with_ambiguity={is_match}"
		)

	numeric_correct, ambiguity_correct = score_rows(ordered_results)

	print(f"Processed images: {len(ordered_results)}")
	print(f"Correctly matched counts (numeric rows): {numeric_correct}")
	print(f"Correctly matched counts (with ambiguity rules): {ambiguity_correct}")
	print(f"CSV written to: {config.output_csv}")
	print(f"JSON written to: {args.output_json}")
	print(f"Review artifacts in: {config.output_dir}")


if __name__ == "__main__":
	main()
