from image_utils import plot_two_images, IMAGES

import cv2
import numpy as np
import matplotlib.pyplot as plt


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    return (((image / 255) ** invGamma) * 255).astype("uint8")


def adjast_brightnes_contrast(image, alpha=1.0, beta=0.0):
    return np.clip(image * alpha + image * beta, 0, 255).astype("uint8")


def equalize_hist(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    before = img_yuv[:, :, 0].copy()
    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    return {'img_out': img_output, 'before': before, 'after': img_yuv[:, :, 0]}


if __name__ == "__main__":
    image = cv2.cvtColor(IMAGES["board"], cv2.COLOR_BGR2RGB)

    gamma_image = adjust_gamma(image, gamma=0.6)
    plot_two_images(image, gamma_image)

    bc_image = adjast_brightnes_contrast(image=image, alpha=1.3, beta=0.0)
    plot_two_images(image, bc_image)

    result = equalize_hist(image)
    plot_two_images(image, result['img_out'])

    fig, axs = plt.subplots(1, 2)
    axs[0].hist(result['before'].ravel(), bins=255)
    axs[1].hist(result['after'].ravel(), bins=255)
    plt.show()
