import cv2
import numpy as np
from image_utils import IMAGES, plot_two_images
import albumentations as A
from tqdm import tqdm
import matplotlib.pyplot as plt


def find_harris_features(image: np.ndarray, threshold=0.01):

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


def find_tomas_features(image: np.ndarray, top_n=20):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    features = cv2.goodFeaturesToTrack(gray, top_n, 0.01, 10)
    features = np.int0(features)

    image_out = image.copy()

    # Iterate over the corners and draw a circle at that location
    for i in features:
        x, y = i.ravel()
        cv2.circle(image_out, (x, y), 5, (0, 0, 255), -1)

    return {
        'image_out': image_out,
        'features': features,
    }

# def test_rotation(image, target_features, angles=(0, 360)):
#
#     rot = imutils.rotate(image, angle=45)


if __name__ == '__main__':
    # like repeatability
    # metrics to estimate: https://sbme-tutorials.github.io/2018/cv/notes/9_week9.html
    # надо построить графиг для повторяемости для разных углов и т д
    image = IMAGES["blox"]

    agg_count = 20  # TODO add perspective transform
    transform = A.Compose([
        A.Blur(blur_limit=3),
        A.RandomBrightnessContrast(p=0.2),
        A.OpticalDistortion(),
        A.HueSaturationValue(),
    ])

    features_heat_map = np.zeros(image.shape[:2])
    for step in tqdm(range(agg_count)):
        aug_image = transform(image=image)['image']
        # result = find_tomas_features(image)
        result = find_harris_features(aug_image, threshold=0.2)
        features_heat_map += result['features'] > 0

    plot_two_images(image, result['image_out'])
    plot_two_images(image, features_heat_map, cmap='gray')

    plt.figure()
    plt.hist(features_heat_map[features_heat_map > 0].ravel())
    plt.show()
