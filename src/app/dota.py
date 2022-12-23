from typing import List

from image_utils import plot_two_images, IMAGES, TEMPLATES, DOTA

import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_object(image, template, detection_threshold=0.8, method=cv2.TM_CCOEFF_NORMED):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    h, w = template.shape[:2]
    res = cv2.matchTemplate(image_gray, template, method)

    loc = np.where(res >= detection_threshold)

    cropped = None
    max_threshold = -1
    for pt in zip(*loc[::-1]):
        threshold = res[pt[1], pt[0]]
        if threshold > max_threshold:
            max_threshold = threshold
            out_image = image.copy()
            cv2.rectangle(out_image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
            cropped = image[pt[1]:pt[1]+h, pt[0]: pt[0]+w]
    if cropped is None:
        return {'cropped': None}
    return {'out_image': out_image, 'points': pt, 'cropped': cropped, 'threshold': max_threshold}


def match_digits(image: np.array, templates: List[np.array]):

    w = image.shape[1]
    image = image[:, w//2:]

    matched = []
    for i in range(4):
        max_threshold = -1
        max_det = None
        for i, t in enumerate(templates):
            det = detect_object(image, t, detection_threshold=0.70)
            if det['cropped'] is None:
                continue
            det['number'] = i

            detected = det['points'] in [m['points'] for m in matched]

            if det['threshold'] > max_threshold and not detected:
                max_det = det
                max_threshold = det['threshold']

        matched.append(max_det)
    return matched


def parse_digits(matched):
    sorted_index = np.argsort(np.array(tuple(m['points'][0] for m in matched)))
    time = ''
    for iter, i in enumerate(sorted_index):
        n = matched[i]['number']
        if iter == 2:
            time += ':'
        time += str(n)
    return time



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
    template = cv2.cvtColor(IMAGES["dota_template"], cv2.COLOR_BGR2RGB)
    dota_image = DOTA[2]
    image = cv2.cvtColor(dota_image, cv2.COLOR_BGR2RGB)
    results = detect_object(image, template, detection_threshold=0.8)
    matched = match_digits(results['cropped'], TEMPLATES)
    found_time = parse_digits(matched)
    print('Time is:', found_time)

    plt.figure(figsize=(30, 30))
    plt.imshow(dota_image)

    plot_two_images(template, results['cropped'])

    for m in matched:
        plot_two_images(m['out_image'], m['cropped'], title=m['number'])
