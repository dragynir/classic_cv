import cv2
import numpy as np

from image_utils import IMAGES, plot_two_images


def find_features(image: np.ndarray, threshold=0.01):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    features = cv2.cornerHarris(gray, blockSize=2, ksize=5, k=0.07)
    features[features < threshold * features.max()] = 0

    features = cv2.dilate(features, None)
    image_out = image.copy()
    image_out[features > 0] = [0, 255, 0]

    return {
        'features': features,
        'image_out': image_out,
    }


if __name__ == '__main__':
    # like repeatability
    # metrics to estimate: https://sbme-tutorials.github.io/2018/cv/notes/9_week9.html
    # надо построить графиг для повторяемости для разных углов и т д
    image = IMAGES["lines"]

    result = find_features(image, threshold=0.1)

    plot_two_images(image, result["image_out"])
