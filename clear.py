import os

import cv2
import numpy as np

def fun(image, data):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(image)
    rgba = [b, g, r, alpha]
    mer = cv2.merge(rgba, 4)
    # 创建一个和透明图一样的全0数组作为掩膜背景幕布
    masks = np.zeros((mer.shape[0],mer.shape[1],4), np.uint8)

    for content in data["content"]:
        contours = np.array(content['outline'])
        contours_img = cv2.drawContours(masks, [contours], -1, (255, 255, 255,255), -1)
        masks=contours_img

    img = np.dstack([image, np.ones((image.shape[0], image.shape[1]), dtype="uint8") * 255])
    # 截取需要的区域
    result= cv2.bitwise_and(img,masks)
    return result

def clear_img(img_file, save_dir, data):
    target_img = cv2.imread(img_file)
    file_name = os.path.basename(img_file)
    target = fun(target_img,data)
    if os.path.exists(save_dir):
        cv2.imwrite(os.path.join(save_dir, file_name), target)
