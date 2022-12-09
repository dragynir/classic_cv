from __future__ import print_function
from imutils.object_detection import non_max_suppression
import numpy as np
import imutils
import cv2

from image_utils import IMAGES, plot_two_images


def draw_boxes(image, boxes):
    for (xA, yA, xB, yB) in boxes:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
    return image


def detect(hog, image: np.nditer):

    image = imutils.resize(image, width=min(400, image.shape[1]))
    print(image.shape)
    orig = image.copy()
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
                                            padding=(8, 8), scale=1.05)
    for (x, y, w, h) in rects:
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    print('Before non max suppression:', len(rects))

    before_image = draw_boxes(image.copy(), rects)

    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    print('After non max suppression:', len(pick))

    after_image = draw_boxes(image.copy(), pick)

    return {'out_image': after_image, 'before_image': before_image}


if __name__ == '__main__':
    image = IMAGES['pedastrian2']
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    results = detect(hog, image)
    plot_two_images(image, results['out_image'])
    plot_two_images(results['before_image'], results['out_image'])
