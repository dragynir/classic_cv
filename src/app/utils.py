import cv2
import matplotlib.pyplot as plt

IMAGES = {
    'simple_edges': cv2.imread(r'C:\Users\dkoro\PythonProjects\classic_cv\images\simple_edges.JPG'),
}


def plot_two_images(image1, image2, figsize=(13, 13)):
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    axs[0].imshow(image1)
    axs[1].imshow(image2)
    plt.show()
