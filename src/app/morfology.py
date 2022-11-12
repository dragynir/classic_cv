import cv2
import numpy as np

from image_utils import random_color


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


def choose_shape(approx_cnt):

    dots_count = len(approx_cnt)
    # 3 dots
    if dots_count == 3:
        return 'triangle'
    elif dots_count == 4:
        x, y, w, h = cv2.boundingRect(approx_cnt)

        if abs(1 - w / h) < 0.15:
            return 'square'
        return 'rectangle'
    elif dots_count > 4:
        return 'circle'
    # TODO add ellipse
    return 'unknown'


def get_contours_types(image: np.ndarray, contours: np.ndarray):

    contours_image = image.copy()

    shapes_types = []

    for cnt in contours:
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx_cnt = cv2.approxPolyDP(cnt, epsilon, True)
        shapes_types.append(choose_shape(approx_cnt))

        cv2.drawContours(
            contours_image,
            [approx_cnt],
            contourIdx=-1,
            color=random_color(),
            thickness=5,
        )

    return {'contours_image': contours_image, 'shapes_types': shapes_types}
