from image_utils import IMAGES, plot_two_images, random_color
from edge import create_contours_image
from morfology import get_relative_size, get_contours_types

import cv2
import numpy as np


def unite_contours(image, cnt):
    contour_image = image.copy()
    cnt = np.concatenate(cnt, axis=0)

    hull = cv2.convexHull(cnt, False)

    cv2.drawContours(
        contour_image,
        [hull],
        contourIdx=-1,
        color=random_color(),
        thickness=5,
    )

    return {"out_image": contour_image, 'approx_cnt': hull}


if __name__ == "__main__":
    # TODO надо определять один объект
    image = IMAGES["hockey"]

    edge_result = create_contours_image(image, edge_thresholds=(50, 160))
    shape_result = get_contours_types(image, edge_result['contours'])
    print(shape_result['shapes_types'])

    approx_results = unite_contours(image, shape_result['approx_cnts'])
    plot_two_images(image, edge_result["edges"])
    plot_two_images(edge_result["contours_image"], shape_result["contours_image"])
    plot_two_images(edge_result["contours_image"], approx_results["out_image"])
