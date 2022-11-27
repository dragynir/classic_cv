import skimage
from image_utils import IMAGES, plot_two_images
import numpy as np
import matplotlib.pyplot as plt


def insert_pattern(background_img, pattern, location):
    img = background_img.copy()
    r0, c0 = location
    r1, c1 = r0 + pattern.shape[0], c0 + pattern.shape[1]
    if r1 < background_img.shape[0] and c1 < background_img.shape[1]:
        img[r0:r1, c0:c1, :] = skimage.img_as_float(pattern)
    return img


def tile_horizontally(background_img, pattern, start_location, repetitions, shift):
    img = background_img.copy()
    for i in range(repetitions):
        r, c = start_location
        c += i * shift
        img = insert_pattern(img, pattern, location=(r, c))
    return img


def make_pattern(shape=(16, 16), levels=64):
    return np.random.randint(0, levels - 1, shape) / levels


def create_circular_depthmap(shape=(600, 800), center=None, radius=100):
    depthmap = np.zeros(shape, dtype=np.float)
    r = np.arange(depthmap.shape[0])
    c = np.arange(depthmap.shape[1])
    R, C = np.meshgrid(r, c, indexing='ij')
    if center is None:
        center = np.array([r.max() / 2, c.max() / 2])
    d = np.sqrt((R - center[0])**2 + (C - center[1])**2)
    depthmap += (d < radius)
    return depthmap


def normalize(depthmap):
    if depthmap.max() > depthmap.min():
        return (depthmap - depthmap.min()) / (depthmap.max() - depthmap.min())
    else:
        return depthmap


def make_autostereogram(depthmap, pattern, shift_amplitude=0.1, invert=False):
    depthmap = normalize(depthmap)
    if invert:
        depthmap = 1 - depthmap
    autostereogram = np.zeros_like(depthmap, dtype=pattern.dtype)
    for r in np.arange(autostereogram.shape[0]):
        for c in np.arange(autostereogram.shape[1]):
            if c < pattern.shape[1]:
                autostereogram[r, c] = pattern[r % pattern.shape[0], c]
            else:
                shift = int(depthmap[r, c] * shift_amplitude * pattern.shape[1])
                autostereogram[r, c] = autostereogram[r, c - pattern.shape[1] + shift]
    return autostereogram


if __name__ == '__main__':
    "https://flothesof.github.io/making-stereograms-Python.html"
    coin = IMAGES['mario_template']
    img = np.ones((800, 800, 3))
    img = tile_horizontally(img, coin, (10, 20), 3, 128)
    img = tile_horizontally(img, coin, (10 + 150, 10), 5, shift=150)
    img = tile_horizontally(img, coin, (10 + 2 * 150, 10), 5, shift=140)

    pattern = make_pattern(shape=(128, 64))
    depthmap = create_circular_depthmap(radius=150)
    autostereogram = make_autostereogram(depthmap, pattern)
    plot_two_images(pattern, depthmap)

    plt.figure()
    plt.imshow(autostereogram)
    plt.savefig('autostereogram.png')
    plt.show()
