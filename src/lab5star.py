import numpy as np
import cv2
import matplotlib.pyplot as plt


def add_gaussian_noise(images):
    noise_imgs = []
    row, col = images[0].shape

    for image in images:
        noise = np.random.randint(low=-100, high=100, size=(row, col), dtype=int)
        image_noisy = np.clip(image + noise, 0, 255).astype(np.uint8)
        noise_imgs.append(image_noisy)

    return noise_imgs


def remove_noise(images):
    images_st = np.stack(images, axis=0)
    return np.sum(images_st, axis=0) / len(images)


def load_gray_image(path):
    return cv2.resize(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY), (200, 200))


image = load_gray_image(r'C:\Users\dkoro\PythonProjects\classic_cv\images\lena.png')

images_count_for_restore = 40
original_images = [image] * images_count_for_restore
noisy_images = add_gaussian_noise(original_images)

plt.figure()
plt.imshow(image, cmap='gray')
plt.show()

plt.figure()
plt.imshow(noisy_images[0], cmap='gray')
plt.show()

restored_image = remove_noise(noisy_images)
plt.figure()
plt.imshow(restored_image, cmap='gray')
plt.show()
