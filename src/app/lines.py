from typing import Dict

import numpy as np
import cv2

from image_utils import IMAGES, plot_two_images
from edge import create_contours_image


def find_lines(image: np.ndarray, edges: np.ndarray):

    # rho - точность в пикселях для rho, theta - точность в радианах для угла
    # threshold - порог для голосования за линию

    image_out = image.copy()

    lines = cv2.HoughLines(edges, rho=1, theta=np.pi / 180, threshold=150)

    for line in lines:
        r, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)

        x0 = a * r
        y0 = b * r

        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(image_out, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return {"lines": lines, "out_image": image_out}


def find_lines_p(image: np.ndarray, edges: np.ndarray, hough_args: Dict):

    image_out = image.copy()

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        **hough_args,
    )

    if lines is None:
        return {}

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image_out, (x1, y1), (x2, y2), (255, 0, 0), 3)

    return {"lines": lines, "out_image": image_out}


if __name__ == "__main__":
    image = IMAGES["lines"]
    edge_result = create_contours_image(image, edge_thresholds=(50, 120))

    hough_args = {
        "threshold": 70,
        "minLineLength": int(image.shape[1] * 0.2),
        "maxLineGap": 20,
    }
    lines_result = find_lines_p(image, edge_result["edges"], hough_args)
    # lines_result = find_lines(image, edge_result["edges"])

    plot_two_images(image, edge_result["edges"])
    plot_two_images(image, lines_result["out_image"])
