from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.io import ensure_parent

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class SplitManifests:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def unpack_zip_dataset(raw_zip: Path, extracted_dir: Path) -> None:
    if not raw_zip.exists():
        raise FileNotFoundError(f"Configured zip file does not exist: {raw_zip}")

    if extracted_dir.exists() and discover_image_paths(extracted_dir):
        return

    extracted_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(raw_zip, mode="r") as archive:
        archive.extractall(extracted_dir)


def discover_image_paths(extracted_dir: Path) -> list[Path]:
    if not extracted_dir.exists():
        return []
    image_paths = [
        path
        for path in extracted_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    ]
    return sorted(image_paths)


def parse_label_from_filename(path: Path, class_names: list[str]) -> str:
    stem = path.stem.lower()
    for class_name in class_names:
        prefix = class_name.lower()
        if stem == prefix:
            return class_name
        if (
            stem.startswith(f"{prefix}.")
            or stem.startswith(f"{prefix}_")
            or stem.startswith(f"{prefix}-")
        ):
            return class_name
    raise ValueError(f"Could not infer label from file name: {path.name}")


def build_manifest(image_paths: list[Path], class_names: list[str]) -> pd.DataFrame:
    records = [
        {"image_path": str(path), "label": parse_label_from_filename(path, class_names)}
        for path in image_paths
    ]
    return pd.DataFrame(records)


def split_manifest(
    manifest: pd.DataFrame,
    test_size: float,
    validation_size: float,
    random_seed: int,
    max_samples: int | None = None,
) -> SplitManifests:
    if manifest.empty:
        raise ValueError("Input manifest is empty.")

    selected = manifest
    if max_samples is not None and max_samples < len(manifest):
        sample_split = train_test_split(
            manifest,
            train_size=max_samples,
            stratify=manifest["label"],
            random_state=random_seed,
        )
        selected = cast(pd.DataFrame, sample_split[0])

    split_one = train_test_split(
        selected,
        test_size=test_size,
        stratify=selected["label"],
        random_state=random_seed,
    )
    train_validation = cast(pd.DataFrame, split_one[0])
    test = cast(pd.DataFrame, split_one[1])

    validation_fraction = validation_size / (1.0 - test_size)
    split_two = train_test_split(
        train_validation,
        test_size=validation_fraction,
        stratify=train_validation["label"],
        random_state=random_seed,
    )
    train = cast(pd.DataFrame, split_two[0])
    validation = cast(pd.DataFrame, split_two[1])

    return SplitManifests(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )


def write_manifest(path: Path, manifest: pd.DataFrame) -> None:
    ensure_parent(path)
    manifest.to_csv(path, index=False)


def read_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    expected = {"image_path", "label"}
    if set(frame.columns) != expected:
        raise ValueError(f"Manifest {path} must contain columns: {sorted(expected)}")
    return frame


def build_dataset_metadata(
    manifest: pd.DataFrame,
    raw_zip: Path,
    extracted_dir: Path,
    class_names: list[str],
    max_samples: int | None,
) -> dict[str, object]:
    class_distribution = {
        class_name: int((manifest["label"] == class_name).sum()) for class_name in class_names
    }
    return {
        "source_zip": str(raw_zip),
        "extracted_dir": str(extracted_dir),
        "rows": len(manifest),
        "class_names": class_names,
        "class_distribution": class_distribution,
        "max_samples": max_samples,
    }
