from image_utils import IMAGES, plot_two_images


import cv2
import numpy as np
import matplotlib.pyplot as plt


def match_images(image_left, image_right):
    surf = cv2.xfeatures2d.SURF_create(400)
    # find the keypoints and descriptors with ORB
    kp1, des1 = surf.detectAndCompute(image_left, None)
    kp2, des2 = surf.detectAndCompute(image_right, None)

    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw first 10 matches.
    img_out = cv2.drawMatches(
        image_left,
        kp1,
        image_right,
        kp2,
        matches[:10],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    plt.imshow(img_out)
    plt.show()



# Решение в колабе https://colab.research.google.com/drive/1lFcDbWeIlDhcsmgofbOOyZesyrhZS28c?authuser=1#scrollTo=Wdf8YEQ4Ldqt
if __name__ == "__main__":
    # like repeatability
    # metrics to estimate: https://sbme-tutorials.github.io/2018/cv/notes/9_week9.html
    # надо построить графиг для повторяемости для разных углов и т д
    image_left = IMAGES["left"]
    image_right = IMAGES["right"]

    match_images(image_left, image_right)


# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
#
# image_left = cv2.cvtColor(cv2.imread('/content/left_book.jpg'), cv2.COLOR_BGR2GRAY)
# image_right = cv2.cvtColor(cv2.imread('/content/right_book.jpg'), cv2.COLOR_BGR2GRAY)
#
# orb = cv2.ORB_create()
# key_points_left, desc_left = orb.detectAndCompute(image_left, None)
# key_points_right, desc_right = orb.detectAndCompute(image_right, None)
#
# index_params = dict(algorithm=6,
#                     table_number=12,
#                     key_size=12,
#                     multi_probe_level=2)
#
# search_params = dict(checks=100)
#
# flann = cv2.FlannBasedMatcher(index_params, search_params)
# matches = flann.knnMatch(desc_left, desc_right, k=2)
#
# good_keypoints, left_p, right_p = [], [], []
# threshold = 0.7
# for m, n in matches:
#     if m.distance < threshold * n.distance:
#         good_keypoints.append(m)
#         left_p.append(key_points_left[m.queryIdx].pt)
#         right_p.append(key_points_right[m.trainIdx].pt)
#
# left_p = np.int32(left_p)
# right_p = np.int32(right_p)
#
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# matches = bf.match(desc_left, desc_right)
# matches = sorted(matches, key=lambda x: x.distance)
#
# iml = image_left
# imr = image_right
#
# count = 20
# matches_image = cv2.drawMatches(iml, key_points_left, imr, key_points_right, matches[:count], None, flags=2)
#
# plt.figure(figsize=(12, 12))
# plt.title("Key points matches")
# plt.imshow(matches_image)
# plt.show()
