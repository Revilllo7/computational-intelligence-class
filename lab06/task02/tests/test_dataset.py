from pathlib import Path

import pandas as pd
import pytest

from src.data.dataset import parse_label_from_filename, split_manifest


def test_parse_label_from_filename_works_for_cat_and_dog() -> None:
    class_names = ["cat", "dog"]
    assert parse_label_from_filename(Path("cat.12.jpg"), class_names) == "cat"
    assert parse_label_from_filename(Path("dog_7.png"), class_names) == "dog"


def test_parse_label_from_filename_raises_on_unknown_prefix() -> None:
    with pytest.raises(ValueError):
        parse_label_from_filename(Path("horse.1.jpg"), ["cat", "dog"])


def test_split_manifest_is_deterministic_and_preserves_rows() -> None:
    rows = []
    for index in range(20):
        rows.append({"image_path": f"/tmp/cat.{index}.jpg", "label": "cat"})
        rows.append({"image_path": f"/tmp/dog.{index}.jpg", "label": "dog"})
    manifest = pd.DataFrame(rows)

    split_a = split_manifest(
        manifest=manifest,
        test_size=0.1,
        validation_size=0.1,
        random_seed=42,
        max_samples=30,
    )
    split_b = split_manifest(
        manifest=manifest,
        test_size=0.1,
        validation_size=0.1,
        random_seed=42,
        max_samples=30,
    )

    assert len(split_a.train) + len(split_a.validation) + len(split_a.test) == 30
    assert split_a.train.equals(split_b.train)
    assert split_a.validation.equals(split_b.validation)
    assert split_a.test.equals(split_b.test)
