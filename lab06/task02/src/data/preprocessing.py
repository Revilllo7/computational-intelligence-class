from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from src.utils.torch_runtime import prepare_torch_import

prepare_torch_import()

import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from torchvision import transforms  # noqa: E402

from src.data.dataset import read_manifest  # noqa: E402
from src.utils.config import PreprocessingConfig  # noqa: E402
from src.utils.io import write_json  # noqa: E402


class CatsDogsImageDataset(Dataset[tuple[torch.Tensor, int, str]]):
    def __init__(
        self,
        manifest: pd.DataFrame,
        class_names: list[str],
        transform: Callable[[Image.Image], torch.Tensor],
    ) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.transform = transform
        self.class_to_index = {name: index for index, name in enumerate(class_names)}

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        row = self.manifest.iloc[index]
        image_path = Path(str(row["image_path"]))
        label = str(row["label"])
        if label not in self.class_to_index:
            raise ValueError(f"Unknown label '{label}' found in manifest.")

        with Image.open(image_path) as image:
            image_rgb = image.convert("RGB")
            tensor = self.transform(image_rgb)

        return tensor, self.class_to_index[label], str(image_path)


def build_train_transform(config: PreprocessingConfig) -> transforms.Compose:
    steps: list[Any] = [
        transforms.Resize((config.image_size, config.image_size)),
    ]

    if config.augment_train:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(p=config.random_horizontal_flip_p),
                transforms.RandomRotation(degrees=config.random_rotation_degrees),
                transforms.ColorJitter(
                    brightness=config.color_jitter_brightness,
                    contrast=config.color_jitter_contrast,
                ),
            ]
        )

    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=config.normalize_mean, std=config.normalize_std),
        ]
    )
    return transforms.Compose(steps)


def build_eval_transform(config: PreprocessingConfig) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.normalize_mean, std=config.normalize_std),
        ]
    )


def build_dataloader(
    manifest_path: Path,
    class_names: list[str],
    preprocessing_config: PreprocessingConfig,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    is_train: bool,
) -> DataLoader:
    manifest = read_manifest(manifest_path)
    transform = (
        build_train_transform(preprocessing_config)
        if is_train
        else build_eval_transform(preprocessing_config)
    )
    dataset = CatsDogsImageDataset(manifest, class_names, transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def write_preprocessing_artifact(
    output_path: Path,
    class_names: list[str],
    preprocessing_config: PreprocessingConfig,
    train_size: int,
    validation_size: int,
    test_size: int,
) -> None:
    write_json(
        output_path,
        {
            "class_names": class_names,
            "image_size": preprocessing_config.image_size,
            "normalize_mean": preprocessing_config.normalize_mean,
            "normalize_std": preprocessing_config.normalize_std,
            "augment_train": preprocessing_config.augment_train,
            "random_horizontal_flip_p": preprocessing_config.random_horizontal_flip_p,
            "random_rotation_degrees": preprocessing_config.random_rotation_degrees,
            "color_jitter_brightness": preprocessing_config.color_jitter_brightness,
            "color_jitter_contrast": preprocessing_config.color_jitter_contrast,
            "split_sizes": {
                "train": train_size,
                "validation": validation_size,
                "test": test_size,
            },
        },
    )
