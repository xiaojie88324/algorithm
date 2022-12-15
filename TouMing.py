import os

import cv2
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np


def Img_Outline(original_img):
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_img, (9, 9), 0)  # 高斯模糊去噪（设定卷积核大小影响效果）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 定义矩形结构元素
    mean, stddv = cv2.meanStdDev(blurred)  # 计算灰度图均值与方差
    _, RedThresh = cv2.threshold(blurred, int(mean), 255, cv2.THRESH_BINARY)  # 设定阈值（阈值影响开闭运算效果）
    canny = cv2.Canny(RedThresh, int(mean - stddv), int(mean + stddv))  # 边缘检测
    mor=cv2.morphologyEx(canny, cv2.MORPH_CLOSE,kernel, iterations=1)
    return mor



def findContours_img(closed,img):

    contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 检测物体的轮廓
    # c=contours

    # print(c)
    contours_img = cv2.drawContours(img.copy(), contours, -1, (0, 0, 255), 3)
    # cv.imshow("c",contours_img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # 创建一个和原图一样的全0数组作为掩膜背景幕布
    # print(contours)
    masks=np.zeros(img.shape,np.uint8)
    # cv.imshow("r",contours_img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # 描绘边缘
    im = cv2.polylines(masks, contours, True, (0,0,255),1)
    # 封闭区域内填充白色(把所有点连接起来，形成封闭区域)
    mask = cv2.fillPoly(im, contours,(255,255,255))
    # 将连接起来的区域对应的数组和原图对应位置按位相与，这样就截取到了需要的区域了
    image = cv2.bitwise_and(img, mask)

    tmp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    # print(alpha)
    b, g, r = cv2.split(image)
    rgba = [b, g, r, alpha]
    dst = cv2.merge(rgba, 4)

    return dst

    # cv.imwrite(r"D:\program\code\cv\Strong\code\7.png", dst)


target_img= cv2.imread(r"D:\personal\U\cv\14.png")
closed = Img_Outline(target_img)

result=findContours_img(closed,target_img)
cv.imwrite(r"D:\personal\U\cv\15.png",result)

# file_dir = r'D:\\program\\code\\cv\\data\\input\\'
# file_path = r'D:\\program\\code\cv\\data\\result\\'
# for img_name in os.listdir(file_dir):
#     img_path = file_dir + img_name
#     img = cv2.imread(img_path)
#     # 目标处理
# #     mean_img = cv2.pyrMeanShiftFiltering(img, 10, 3)  # 色彩平滑，后面两个参数可以根据自己的效果进行调整
#     closed = Img_Outline(img)
#     contours = findContours_img(closed,img)
#     # print("points=",points)
#     cv2.imwrite(file_path+img_name[0:-4]+'.jpg',contours)


