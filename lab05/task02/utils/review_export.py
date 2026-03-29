from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .blob_counter import BlobComponent
from .preprocessing import PreprocessResult

# Draw accepted and rejected components for quick manual review.
def draw_blob_overlay(image_bgr: np.ndarray, components: list[BlobComponent]) -> np.ndarray:
   
    annotated = image_bgr.copy()
    for component in components:
        color = (0, 220, 0) if component.accepted else (0, 0, 220)
        x1 = component.left
        y1 = component.top
        x2 = component.left + component.width
        y2 = component.top + component.height

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
        cv2.circle(
            annotated,
            (int(component.centroid_x), int(component.centroid_y)),
            radius=1,
            color=color,
            thickness=-1,
        )
        cv2.putText(
            annotated,
            "",
            (x1, max(0, y1 - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )
    return annotated


def save_review_artifacts(
    output_dir: Path,
    image_name: str,
    original_bgr: np.ndarray,
    preprocess: PreprocessResult,
    components: list[BlobComponent],
    save_intermediate: bool,
) -> tuple[Path, Path]:
    # Persist mask and overlay images for manual validation.
    mask_dir = output_dir / "blob_masks"
    overlay_dir = output_dir / "annotated"
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    image_stem = Path(image_name).stem
    mask_path = mask_dir / f"{image_stem}_mask.png"
    overlay_path = overlay_dir / f"{image_stem}_blobs.jpg"

    cv2.imwrite(str(mask_path), preprocess.separated)
    overlay = draw_blob_overlay(original_bgr, components)
    cv2.imwrite(str(overlay_path), overlay)

    return mask_path, overlay_path
