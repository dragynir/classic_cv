import cv2
import numpy as np

from image_utils import random_color, put_text_to_image


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
    # SimpleBlobDetector_Params
    dots_count = len(approx_cnt)
    perimeter = cv2.arcLength(approx_cnt, True)

    if perimeter == 0:
        return 'unknown'

    area = cv2.contourArea(approx_cnt)
    circularity = 4 * np.pi * area / perimeter ** 2.
p    # 3 dots
    if dots_count == 3:
        return 'triangle'
    elif dots_count == 4:
        if abs(1 - w / h) < 0.15:
            return 'square'
        return 'rectangle'
    elif dots_count > 4 and circularity > 0.9:
        return 'circle'
    # TODO add ellipse
    return 'unknown'


def get_contours_types(image: np.ndarray, contours: np.ndarray):

    contours_image = image.copy()

    shapes_types = []
    approx_cnts = []

    for cnt in contours:
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx_cnt = cv2.approxPolyDP(cnt, epsilon, True)
        shape_name = choose_shape(approx_cnt)
        shapes_types.append(shape_name)
        approx_cnts.append(approx_cnt)

        contours_image = put_text_to_image(contours_image, shape_name, approx_cnt[0][0])

        cv2.drawContours(
            contours_image,
            [approx_cnt],
            contourIdx=-1,
            color=random_color(),
            thickness=5,
        )

    return {'contours_image': contours_image, 'shapes_types': shapes_types, 'approx_cnts': approx_cnts}
