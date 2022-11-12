import cv2
import matplotlib.pyplot as plt


def load_gray_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def show_gray_threshold(image, channel=0, threshold=155):
    original_image = image.copy()
    image = image[:, :, channel]
    per = (image > 0).sum() / (image.shape[0] * image.shape[1])

    fig, axs = plt.subplots(2, 1, figsize=(20, 20))
    axs[0].imshow(image > threshold, cmap='gray')
    axs[1].imshow(original_image, cmap='gray')
    axs[0].set_title(f'Above threshold: {round(per * 100, 3)} %')
    plt.show()

image = load_gray_image(r'C:\Users\dkoro\PythonProjects\classic_cv\images\image2.jpg')
show_gray_threshold(image, threshold=200, channel=2)


