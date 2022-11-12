import cv2
import matplotlib.pyplot as plt
import os

images_path = r'C:\Users\dkoro\PythonProjects\classic_cv\images'

IMAGES = {
    "simple_edges": cv2.imread(
        os.path.join(images_path, "simple_edges.JPG"),
    ),
    "shapes": cv2.imread(os.path.join(images_path, "shapes.png")),
}


def plot_two_images(image1, image2, figsize=(13, 13)):
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    axs[0].imshow(image1)
    axs[1].imshow(image2)
    plt.show()


def random_color():
    return (255, 0, 0)
