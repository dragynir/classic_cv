import cv2
import numpy as np

from image_utils import IMAGES, plot_two_images


def apply_filter(image, kernel=None, callable=None, args=None):
    assert (filter is not None or callable is not None)

    if kernel is not None:
        result = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    elif callable is not None:
        result = callable(image, **args)
    else:
        assert False, 'Bad arguments'

    return {'image': image, 'out_image': result}


if __name__ == '__main__':

    operations = {
        'blur': (cv2.blur, {'ksize': (5, 5)}),
        'median_blur': (cv2.medianBlur, {'ksize': 5}),
        'sobel': (cv2.Sobel, {'ddepth': -1, 'dx': 1, 'dy': 1, 'ksize': 3}),
        'laplacian': (cv2.Laplacian, {'ddepth': -1, 'ksize': 3}),
    }

    kernels = {
        'blur': np.ones((3, 3)),
        'sobel_gx': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
        'soble_gy': np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]),
    }

    image = IMAGES["lines"]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for name, config in operations.items():
        op, args = config
        result = apply_filter(image, callable=op, args=args)
        plot_two_images(image, result["out_image"], title=name)

    for name, kernel in kernels.items():
        result = apply_filter(gray_image, kernel=kernel)
        plot_two_images(gray_image, result["out_image"], title=name)
