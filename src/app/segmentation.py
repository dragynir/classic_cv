from image_utils import plot_two_images, IMAGES

import cv2
import numpy as np


def water_shed(image, markers_threshold=0.2):

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = 255 - thresh

    kernel = np.ones((3, 3), np.uint8)
    closing_fg = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel, iterations=5)
    dist = cv2.distanceTransform(closing_fg, cv2.DIST_L2, cv2.DIST_MASK_3)

    ret, sure_fg = cv2.threshold(dist, markers_threshold * dist.max(), 255, 0)

    sure_fg = sure_fg.astype('uint8')
    ret, markers = cv2.connectedComponents(sure_fg)

    out_image = image.copy()
    markers = cv2.watershed(image, markers)
    out_image[markers == -1] = [255, 0, 0]

    return {"morpology": closing_fg, "dist": sure_fg, "markers": markers, "out_image": out_image}


if __name__ == "__main__":
    # image = cv2.cvtColor(IMAGES["rocks"], cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(IMAGES["coins"], cv2.COLOR_BGR2RGB)

    results = water_shed(image, markers_threshold=0.5)
    plot_two_images(image, results['morpology'])
    plot_two_images(image, results['dist'])
    plot_two_images(image, results['markers'])
    plot_two_images(image, results['out_image'])
