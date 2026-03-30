from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ComparisonRow:
	number: str
	picture_name: str
	official_count_raw: str
	notes: str


@dataclass(frozen=True)
class CountResultRow:
	number: str
	picture_name: str
	official_count: str
	algorithm_count: str
	found: str
	missed: str
	notes: str


def read_comparison_rows(comparison_csv: Path) -> dict[str, ComparisonRow]:
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


def parse_official_count(raw_value: str) -> tuple[int | None, str | None]:
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


def merge_notes(*notes: str | None) -> str:
	merged = [note.strip() for note in notes if note and note.strip()]
	if not merged:
		return ""
	unique = list(dict.fromkeys(merged))
	return " | ".join(unique)


def build_result_row(
	image_name: str,
	algorithm_count: int | None,
	comparison: ComparisonRow | None,
	extra_note: str | None,
) -> CountResultRow:
	if comparison is None:
		return CountResultRow(
			number="",
			picture_name=Path(image_name).stem,
			official_count="",
			algorithm_count="" if algorithm_count is None else str(algorithm_count),
			found="",
			missed="",
			notes=merge_notes(extra_note, "missing_in_comparison_csv"),
		)

	official_numeric, parse_note = parse_official_count(comparison.official_count_raw)
	found = ""
	missed = ""
	if official_numeric is not None and algorithm_count is not None:
		found = str(min(official_numeric, algorithm_count))
		missed = str(max(official_numeric - algorithm_count, 0))

	return CountResultRow(
		number=comparison.number,
		picture_name=comparison.picture_name,
		official_count=comparison.official_count_raw,
		algorithm_count="" if algorithm_count is None else str(algorithm_count),
		found=found,
		missed=missed,
		notes=merge_notes(comparison.notes, parse_note, extra_note),
	)


def write_output_csv(output_csv: Path, rows: list[CountResultRow]) -> None:
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


def write_output_json(output_json: Path, rows: list[CountResultRow]) -> None:
	output_json.parent.mkdir(parents=True, exist_ok=True)
	payload: list[dict[str, object]] = []
	for row in rows:
		image_number: int | str = int(row.number) if row.number.strip().isdigit() else row.number
		bird_counter: int | None = int(row.algorithm_count) if row.algorithm_count.strip().isdigit() else None
		payload.append(
			{
				"image_number": image_number,
				"image_name": row.picture_name,
				"bird_counter": bird_counter,
				"matches_with_ambiguity": ambiguity_accepted(
					row.number.strip(),
					row.algorithm_count.strip(),
					row.official_count.strip(),
				),
			}
		)

	with output_json.open("w", encoding="utf-8") as handle:
		json.dump(payload, handle, ensure_ascii=True, indent=2)


def ambiguity_accepted(number: str, algorithm_count: str, official_count: str) -> bool:
	if not algorithm_count.isdigit():
		return False

	algo = int(algorithm_count)
	if number == "6":
		return algo in {5, 6}
	if number == "14":
		return algo in {15, 16, 17}
	return official_count.isdigit() and algo == int(official_count)


def score_rows(rows: list[CountResultRow]) -> tuple[int, int]:
	numeric_correct = 0
	ambiguity_correct = 0
	for result in rows:
		official_raw = result.official_count.strip().rstrip("?")
		if official_raw.isdigit() and result.algorithm_count.isdigit() and int(official_raw) == int(result.algorithm_count):
			numeric_correct += 1

		if ambiguity_accepted(result.number.strip(), result.algorithm_count.strip(), result.official_count.strip()):
			ambiguity_correct += 1

	return numeric_correct, ambiguity_correct
