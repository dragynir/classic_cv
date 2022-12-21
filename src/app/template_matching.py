from image_utils import plot_two_images, IMAGES

import cv2
import numpy as np
from PIL import Image

def detect_object(image, template, detection_threshold=0.8, method=cv2.TM_CCOEFF_NORMED):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    out_image = image.copy()
    h, w = template.shape[:2]
    res = cv2.matchTemplate(image_gray, template, method)
    loc = np.where(res >= detection_threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(out_image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    return {'out_image': out_image}


if __name__ == "__main__":
    template = cv2.cvtColor(IMAGES["mario_template"], cv2.COLOR_BGR2RGB)

    noise = np.zeros(IMAGES["mario"].shape, dtype=np.uint8)
    grid = {
        'rgb': (0.8, cv2.cvtColor(IMAGES["mario"], cv2.COLOR_BGR2RGB)),
        'bgr': (0.8, IMAGES["mario"]),
        'rotated_90': (0.2, cv2.rotate(IMAGES["mario"], cv2.ROTATE_90_CLOCKWISE)),
        'random_rotate': (0.8, cv2.rotate(IMAGES["mario"], cv2.ROTATE_90_CLOCKWISE)),
        'filtered': (0.8, cv2.blur(IMAGES["mario"], ksize=(7, 7))),
        'noise': (0.8, IMAGES["mario"] + cv2.randn(noise, 40, 20)),
        'rotate': (0.8, np.array(Image.fromarray(IMAGES["mario"]).rotate(3))),
    }

    for name, config in grid.items():
        print(name)
        threshold, image = config
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = detect_object(image, template, detection_threshold=threshold)
        # plot_two_images(image, template)
        plot_two_images(template, results['out_image'], title=name)
