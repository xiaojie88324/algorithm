
import os

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import cv2
import numpy as np
from matplotlib import pyplot as plt

def find_contours(img):
    image=hsv(img)
    mor=close_img(image)
    contours, hierarchy = cv2.findContours(mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 检测物体的轮廓
    c = sorted(contours, key=cv2.contourArea, reverse=True)[0]  # 计算最大轮廓的旋转包围盒
    rect = cv2.minAreaRect(c)  # 获取包围盒（中心点，宽高，旋转角度）
    ins = np.int0(cv2.boxPoints(rect))
    return ins

def close_img(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_img, (9, 9), 0)  # 高斯模糊去噪（设定卷积核大小影响效果）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 定义矩形结构元素
    mean, stddv = cv2.meanStdDev(blurred)  # 计算灰度图均值与方差
    _, RedThresh = cv2.threshold(blurred, 0, 255, cv.THRESH_OTSU)  # 设定阈值（阈值影响开闭运算效果）
    canny = cv2.Canny(RedThresh, int(mean - stddv), int(mean + stddv))  # 边缘检测
    mor = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mor

def hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # HSV空间

    lower_blue = np.array([110, 80,80])  # blue
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    lower_green = np.array([80, 15, 15])  # green
    upper_green = np.array([110, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    lower_yellow = np.array([10, 5, 5])  # 黄色
    upper_yellow = np.array([25, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    res = green_mask + blue_mask + yellow_mask
    open_img = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel=np.ones((3, 3), np.uint8))

    picture = cv2.bitwise_and(img, img, mask=open_img)

    return picture




def Perspective_transform(original_img, points):

    rect = order_points(points)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0])**2) + ((br[1] - bl[1])**2))
    widthB = np.sqrt(((tr[0] - tl[0])**2) + ((tr[1] - tl[1])**2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0])**2) + ((tr[1] - br[1])**2))
    heightB = np.sqrt(((tl[0] - bl[0])**2) + ((tl[1] - bl[1])**2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array(
        [[0, 0], [maxWidth-1 , 0], [maxWidth-1 , maxHeight-1 ], [0, maxHeight-1 ]],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(rect, dst)
    change_img = cv2.warpPerspective(original_img, M, (maxWidth, maxHeight))

    return change_img


def order_points(pts):

    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)

    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect



def correct_img(img_file, save_dir):
    target_img = cv2.imread(img_file)
    file_name = os.path.basename(img_file)
    points = find_contours(target_img)
    target = Perspective_transform(target_img, points)
    if os.path.exists(save_dir):
        cv2.imwrite(os.path.join(save_dir, file_name), target)




# file_dir = r'D:\\program\\code\\cv\\Datas\\data3s\\'
# file_path = r'D:\program\code\cv\Datas\\result4s\\'
# for img_name in os.listdir(file_dir):
#     # print(img_name)
#     img_path = file_dir + img_name
#     img = cv2.imread(img_path)
#     # 目标处理
# #     mean_img = cv2.pyrMeanShiftFiltering(img, 10, 3)  # 色彩平滑，后面两个参数可以根据自己的效果进行调整
#     points = find_contours(img)
#     # print("points=",points)
#     target = Perspective_transform(img, points)
#     cv2.imwrite(file_path+img_name[0:-4]+'.jpg',target)