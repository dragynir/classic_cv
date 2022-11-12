import cv2
import numpy as np


def get_relative_size(
    image: np.ndarray,
    contours: np.ndarray,
    area_thresholds=(0, 50, 80, 100),
    names=("small", "medium", "large"),
):

    assert len(names) == len(area_thresholds) - 1

    image_area = image.shape[0] * image.shape[1]

    cnt_areas = tuple(cv2.contourArea(cnt) for cnt in contours)

    relative_area = tuple(100 * area / image_area for area in cnt_areas)

    ranges = list(zip(area_thresholds, area_thresholds[1:]))

    relative_sizes = []
    for rel_area in relative_area:
        for ind, r in enumerate(ranges):
            if r[0] <= rel_area <= r[1]:
                relative_sizes.append(names[ind])
                break

    return {
        "contours": contours,
        "relative_sizes": relative_sizes,
        "relative_area": relative_area,
    }
