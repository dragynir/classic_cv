from image_utils import plot_two_images, IMAGES, TEMPLATES, DOTA

import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_object(image, template, detection_threshold=0.8, method=cv2.TM_CCOEFF_NORMED, return_all=False):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    h, w = template.shape[:2]
    matched = cv2.matchTemplate(image_gray, template, method)

    loc = np.where(matched >= detection_threshold)
    artifacts_list = []
    locations = list(zip(*loc[::-1]))
    for pt in locations:
        threshold = matched[pt[1], pt[0]]
        out_image = image.copy()
        cv2.rectangle(out_image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
        cropped = image[pt[1]:pt[1]+h, pt[0]: pt[0]+w]

        artifacts = {
            'out_image': out_image,
            'point': pt,
            'crop': cropped,
            'threshold': threshold,
            'width': w,
        }
        artifacts_list.append(artifacts)
        if not return_all:
            break
    return artifacts_list


def crop_time(image):
    w = image.shape[1]
    return image[:, w // 2 + int(w * 0.1):]


def detect_digits(image, templates, thresholds):
    results = []
    for search_number, template in enumerate(templates):
        search_threshold = thresholds[search_number]
        search_det = detect_object(image, template, detection_threshold=search_threshold, method=cv2.TM_CCOEFF_NORMED, return_all=True)

        if len(search_det) == 0:
            continue

        for s in search_det:
            s['number'] = search_number

        results.extend(search_det)

    return results


def compare_location(a, b, threshold=0.7):
    a_x = a['point'][0]
    b_x = b['point'][0]

    a_xe = a_x + a['width']
    b_xe = b_x + b['width']
    mean_w = np.mean((a['width'], b['width']))

    if b_x <= a_x <= b_xe:
        return ((b_xe - a_x) / mean_w) > threshold

    if b_x <= a_xe <= b_xe:
        return ((a_xe - b_x) / mean_w) > threshold

    return False


def digits_max_supression(digits, cmp_threshold=0.7):
    marked = [False] * len(digits)

    for d_i, d in enumerate(digits):
        if marked[d_i]:
            continue

        for c_i, c in enumerate(digits):
            if c_i == d_i:
                continue
            if compare_location(d, c, cmp_threshold):
                if marked[c_i]:
                    continue

                if d['threshold'] > c['threshold']:
                    marked[c_i] = True
                else:
                    marked[d_i] = True
    return [digits[i] for i, m in enumerate(marked) if not m]


def parse_digits(digits):
    sorted_index = np.argsort(np.array(tuple(d['point'][0] for d in digits)))
    time = ''
    for iter, i in enumerate(sorted_index):
        n = digits[i]['number']
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

    thresholds = {
        0: 0.8,
        1: 0.8,
        2: 0.8,
        3: 0.8,
        4: 0.8,
        5: 0.8,
        6: 0.8,
        7: 0.8,
        8: 0.8,
        9: 0.8,
    }

    template = cv2.cvtColor(IMAGES["dota_template"], cv2.COLOR_BGR2RGB)
    dota_image = DOTA[2]
    image = cv2.cvtColor(dota_image, cv2.COLOR_BGR2RGB)
    results = detect_object(image, template, detection_threshold=0.8)
    win_crop = results[0]['crop']
    time_crop = crop_time(win_crop)
    digits = detect_digits(time_crop, TEMPLATES, thresholds)
    digits = digits_max_supression(digits, cmp_threshold=0.7)
    str_time = parse_digits(digits)
    print('Time is:', str_time)

    plt.figure(figsize=(30, 30))
    plt.imshow(dota_image)
    plot_two_images(template, win_crop)
    plot_two_images(win_crop, time_crop)

    for m in digits:
        plot_two_images(m['out_image'], m['crop'], title=m['number'])
