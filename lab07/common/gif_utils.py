"""Reusable helpers for GIF creation from matplotlib-rendered frames."""

from __future__ import annotations

from collections.abc import Sequence
from io import BytesIO
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from PIL import Image


def figure_to_rgb_array(figure: Figure) -> np.ndarray:
    """Render a matplotlib figure into an RGB image array."""
    buffer = BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)

    with Image.open(buffer) as image:
        rgb_image = image.convert("RGB")
        return np.asarray(rgb_image, dtype=np.uint8)


def save_gif_from_arrays(
    frames: Sequence[np.ndarray],
    output_path: Path,
    *,
    duration_ms: int = 600,
    loop: int = 0,
) -> None:
    """Persist image frames as an animated GIF."""
    if not frames:
        raise ValueError("Expected at least one frame to build a GIF")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames: list[Image.Image] = []

    for frame in frames:
        if frame.ndim != 3 or frame.shape[2] not in (3, 4):
            raise ValueError("Each frame must have shape (height, width, 3|4)")

        normalized = frame.astype(np.uint8, copy=False)
        if normalized.shape[2] == 4:
            image = Image.fromarray(normalized, mode="RGBA").convert("RGB")
        else:
            image = Image.fromarray(normalized, mode="RGB")
        pil_frames.append(image.convert("P", palette=Image.Palette.ADAPTIVE))

    first_frame, *rest_frames = pil_frames
    first_frame.save(
        output_path,
        save_all=True,
        append_images=rest_frames,
        duration=duration_ms,
        loop=loop,
        optimize=False,
    )


def save_gif_from_figures(
    figures: Sequence[Figure],
    output_path: Path,
    *,
    duration_ms: int = 300,
    loop: int = 0,
) -> None:
    """Convert matplotlib figures into frames and save them as a GIF."""
    frame_arrays = [figure_to_rgb_array(figure) for figure in figures]
    save_gif_from_arrays(frame_arrays, output_path, duration_ms=duration_ms, loop=loop)
