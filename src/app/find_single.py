from image_utils import IMAGES, plot_two_images, random_color
from edge import create_contours_image
from morfology import get_relative_size, get_contours_types, choose_shape

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
    # image = IMAGES["hockey"]
    image = IMAGES["mario_template"]
    edge_result = create_contours_image(image, edge_thresholds=(50, 160))
    shapes_result = get_contours_types(image, edge_result['contours'])
    approx_results = unite_contours(image, shapes_result['approx_cnts'])

    size = get_relative_size(image, np.array([approx_results['approx_cnt']]))
    shape_result = choose_shape(approx_results['approx_cnt'])
    print(size['relative_sizes'], shape_result)

    plot_two_images(image, edge_result["edges"])
    plot_two_images(edge_result["contours_image"], shapes_result["contours_image"])
    plot_two_images(edge_result["contours_image"], approx_results["out_image"])
