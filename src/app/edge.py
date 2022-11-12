from typing import Tuple

import numpy as np
import cv2
from utils import IMAGES, plot_two_images


def find_edges(image: np.ndarray, blur_kernal=(5, 5), thresholds=(30, 120)):

    assert len(blur_kernal) == 2
    assert len(thresholds) == 2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, blur_kernal, 0)

    # L2gradient more accurate but slower
    edges = cv2.Canny(
        blurred, threshold1=thresholds[0], threshold2=thresholds[1], L2gradient=True
    )

    return edges


def create_contours_image(image: np.ndarray, edge_thresholds: Tuple[int, int]):

    edges = find_edges(image, thresholds=edge_thresholds)
    contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    # hierarchy [Next, Previous, First_Child, Parent]

    contours_image = np.zeros_like(image)
    image_with_contours = image.copy()
    cv2.drawContours(
        contours_image,
        contours,
        contourIdx=-1,
        color=(0, 255, 0),
        thickness=-1,
    )
    cv2.drawContours(
        image_with_contours,
        contours,
        contourIdx=-1,
        color=(0, 255, 0),
        thickness=-1,
    )

    return {
        "image": image,
        "contours": contours,
        "édges": edges,
        "out_image": image_with_contours,
        "contours_image": contours_image,
    }


if __name__ == "__main__":
    image = IMAGES["simple_edges"]
    result = create_contours_image(image, edge_thresholds=(30, 120))

    plot_two_images(image, result["out_image"])
