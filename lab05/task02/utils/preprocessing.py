from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocessConfig:
    clip_limit: float = 2.0
    tile_grid_size: tuple[int, int] = (8, 8)
    blur_kernel_size: tuple[int, int] = (5, 5)
    adaptive_block_size: int = 11
    adaptive_c: float = 2.0
    erode_iterations: int = 0
    erode_kernel_size: tuple[int, int] = (3, 3)
    morph_kernel_size: tuple[int, int] = (3, 3)
    watershed_enabled: bool = False
    watershed_fg_ratio: float = 0.4
    watershed_bg_dilate_iterations: int = 2


@dataclass(frozen=True)
class PreprocessResult:
    gray: np.ndarray
    clahe_gray: np.ndarray
    blurred: np.ndarray
    thresholded: np.ndarray
    eroded: np.ndarray
    opened: np.ndarray
    closed: np.ndarray
    separated: np.ndarray


def _ensure_odd(value: int, name: str) -> int:
    if value % 2 == 0:
        raise ValueError(f"{name} must be odd. Got {value}.")
    return value


def _ensure_non_negative(value: int, name: str) -> int:
    if value < 0:
        raise ValueError(f"{name} must be non-negative. Got {value}.")
    return value


def _watershed_split(closed_mask: np.ndarray, image_bgr: np.ndarray, fg_ratio: float, bg_dilate_iterations: int) -> np.ndarray:
    if closed_mask.max() == 0:
        return closed_mask.copy()

    bg_dilate_iterations = _ensure_non_negative(bg_dilate_iterations, "watershed_bg_dilate_iterations")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_bg = cv2.dilate(closed_mask, kernel, iterations=bg_dilate_iterations)

    distance = cv2.distanceTransform(closed_mask, cv2.DIST_L2, 5)
    if distance.max() <= 0:
        return closed_mask.copy()

    threshold_value = float(fg_ratio) * float(distance.max())
    _, sure_fg = cv2.threshold(distance, threshold_value, 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    watershed_markers = cv2.watershed(image_bgr.copy(), markers)
    separated = np.zeros_like(closed_mask, dtype=np.uint8)
    separated[watershed_markers > 1] = 255
    return separated

# Run the requested OpenCV preprocessing pipeline for blob counting.
def preprocess_for_blobs(image_bgr: np.ndarray, config: PreprocessConfig) -> PreprocessResult:
    
    if image_bgr.ndim == 2:
        gray = image_bgr
    else:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=config.clip_limit, tileGridSize=config.tile_grid_size)
    clahe_gray = clahe.apply(gray)

    blur_w, blur_h = config.blur_kernel_size
    _ensure_odd(blur_w, "blur_kernel_size width")
    _ensure_odd(blur_h, "blur_kernel_size height")
    blurred = cv2.GaussianBlur(clahe_gray, config.blur_kernel_size, 0)

    block_size = _ensure_odd(config.adaptive_block_size, "adaptive_block_size")
    thresholded = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        config.adaptive_c,
    )

    _ensure_non_negative(config.erode_iterations, "erode_iterations")
    eroded = thresholded
    if config.erode_iterations > 0:
        erode_kernel = np.ones(config.erode_kernel_size, np.uint8)
        eroded = cv2.erode(thresholded, erode_kernel, iterations=config.erode_iterations)

    kernel = np.ones(config.morph_kernel_size, np.uint8)
    opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    separated = closed

    if config.watershed_enabled:
        separated = _watershed_split(
            closed_mask=closed,
            image_bgr=image_bgr,
            fg_ratio=config.watershed_fg_ratio,
            bg_dilate_iterations=config.watershed_bg_dilate_iterations,
        )

    return PreprocessResult(
        gray=gray,
        clahe_gray=clahe_gray,
        blurred=blurred,
        thresholded=thresholded,
        eroded=eroded,
        opened=opened,
        closed=closed,
        separated=separated,
    )
