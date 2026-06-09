from __future__ import annotations

import cv2
import numpy as np

CONNECTIVITY = 8


def connected_component_filter(
    mask: np.ndarray,
    min_area_ratio: float = 0.0005,
    keep_largest: bool = False,
) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=CONNECTIVITY)
    if count <= 1:
        return binary.astype(bool)

    areas = stats[:, cv2.CC_STAT_AREA]
    keep = np.zeros(count, dtype=bool)

    if keep_largest:
        keep[1 + int(np.argmax(areas[1:]))] = True
    else:
        minimum_area = min_area_ratio * binary.size
        for label in range(1, count):
            if areas[label] >= minimum_area:
                keep[label] = True

    return keep[labels]
