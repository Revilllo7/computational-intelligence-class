from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class BlobFilterConfig:
    min_area: int = 3
    max_area: int = 300


@dataclass(frozen=True)
class BlobComponent:
    label: int
    area: int
    left: int
    top: int
    width: int
    height: int
    centroid_x: float
    centroid_y: float
    accepted: bool

# Count connected components within an area range and return full component metadata.
def count_blobs(
    binary_image: np.ndarray,
    filter_config: BlobFilterConfig,
) -> tuple[int, list[BlobComponent], int]:
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(binary_image)

    components: list[BlobComponent] = []
    count = 0
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        left = int(stats[label, cv2.CC_STAT_LEFT])
        top = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        centroid_x = float(centroids[label][0])
        centroid_y = float(centroids[label][1])

        accepted = filter_config.min_area < area < filter_config.max_area

        if accepted:
            count += 1

        components.append(
            BlobComponent(
                label=label,
                area=area,
                left=left,
                top=top,
                width=width,
                height=height,
                centroid_x=centroid_x,
                centroid_y=centroid_y,
                accepted=accepted,
            )
        )

    return count, components, num_labels
