from __future__ import print_function
from imutils.object_detection import non_max_suppression
import numpy as np
import imutils
import cv2


def draw_boxes(image, boxes):
    for (xA, yA, xB, yB) in boxes:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
    return image


def detect(hog, image: np.nditer):

    image = imutils.resize(image, width=min(400, image.shape[1]))
    print(image.shape)
    orig = image.copy()
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
                                            padding=(8, 8), scale=1.05)
    for (x, y, w, h) in rects:
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    print('Before non max suppression:', len(rects))

    before_image = draw_boxes(image.copy(), rects)

    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    print('After non max suppression:', len(pick))

    after_image = draw_boxes(image.copy(), pick)

    return {'out_image': after_image, 'before_image': before_image}


def show_camera_view(hog, video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        results = detect(hog, frame)

        cv2.imshow('frame', results['out_image'])

        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    show_camera_view(hog, r"C:\Users\dkoro\PythonProjects\classic_cv\images\pedastrian.mp4")
