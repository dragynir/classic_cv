from typing import Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
import dataclasses as dc


@dc.dataclass
class Circle:
    radius: Tuple
    center: Tuple
    angle: int


def draw_mask(image, circles, polygons=None):
    color = (255, 0, 0)
    thickness = 5

    for circle in circles:
        cv2.ellipse(image, circle.center, circle.radius, circle.angle, 0, 360, color, thickness)

    for poly in polygons:
        x_coords, y_coords = poly
        res_poly = []
        for x, y in zip(x_coords, y_coords):
            res_poly.append([x, y])
        res_poly = np.array(res_poly, np.int32)
        cv2.polylines(image, [res_poly], True, color, thickness)
    return image


if __name__ == '__main__':

    circles = [Circle(radius=(80, 58), center=(610, 830), angle=0),
               Circle(radius=(80, 60), center=(950, 730), angle=120)]
    polygons = [
        [[749,816,899,852], [965,896,934,1017]],
        [[825,977,1037,1072,1137,1164,1184],[384,87,51,78,176,372,577]],
        [[183,163,230,283,511,230], [292,377,586,653,450,279]]
    ]

    image = cv2.imread(r'C:\Users\dkoro\PythonProjects\classic_cv\images\cat.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = draw_mask(image, circles, polygons)

    cv2.imwrite(r'C:\Users\dkoro\PythonProjects\classic_cv\images\cat_mask.jpg', image)
    plt.imshow(image)
    plt.show()




