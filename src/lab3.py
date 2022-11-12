import cv2
import matplotlib.pyplot as plt


def load_gray_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)


def local_thresholding(image):

    # blurred_image = cv2.GaussianBlur(image, ksize=(3, 3), sigmaX=0, sigmaY=0)
    blurred_image = image
    plt.figure()
    plt.imshow(blurred_image, cmap='gray')
    plt.show()

    image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 17, 7)
    image = 255 - image
    return image


image = load_gray_image(r'/images/lena_son.JPG')

plt.figure()
plt.imshow(image, cmap='gray')
plt.show()

image_thr = local_thresholding(image)

plt.figure()
plt.imshow(image_thr, cmap='gray')
plt.show()
