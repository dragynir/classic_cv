from image_utils import plot_two_images, IMAGES

import cv2
import numpy as np


def detect_object(image, template, detection_threshold=0.8, method=cv2.TM_CCOEFF_NORMED):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    out_image = image.copy()
    h, w = template.shape[:2]
    res = cv2.matchTemplate(image_gray, template, method)
    loc = np.where(res >= detection_threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(out_image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    return {'out_image': out_image}


if __name__ == "__main__":
    """
    Алгоритм:
        1. Ресайз исходной картинки до 1024, 1024 с сохранением aspect ratio
        2. RGB -> Gray
        3. Размазывание шаблона вдоль y (чтобы цифры стали похожи)
        4. Template matching с рамкой времени
        5. Вырезание рамки времени по шаблону
        6. Матчинг цифр по шаблонам
        7. Формирование ответа
    """
    template = cv2.cvtColor(IMAGES["dota"], cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(IMAGES["dota"], cv2.COLOR_BGR2RGB)
    results = detect_object(image, template, detection_threshold=0.8)
    # plot_two_images(image, template)
    plot_two_images(template, results['out_image'])
