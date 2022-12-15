# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# def Img_Outline(original_img):
#     gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray_img, (9, 9), 0)  # 高斯模糊去噪（设定卷积核大小影响效果）
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 定义矩形结构元素
#     mean, stddv = cv2.meanStdDev(blurred)  # 计算灰度图均值与方差
#     _, RedThresh = cv2.threshold(blurred, int(mean), 255, cv2.THRESH_BINARY)  # 设定阈值（阈值影响开闭运算效果）
#     canny = cv2.Canny(RedThresh, int(mean - stddv), int(mean + stddv))  # 边缘检测
#     return cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel, iterations=1) # 返回其形态学的开,闭运算，这里是闭运算
#
#
# def findContours_img(closed):
#     contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 检测物体的轮廓
#     c = sorted(contours, key=cv2.contourArea, reverse=True)[0]  # 计算最大轮廓的旋转包围盒
#     rect = cv2.minAreaRect(c)  # 获取包围盒（中心点，宽高，旋转角度）
#     # print("rect=",rect)
#     return np.int0(cv2.boxPoints(rect)) #使用cv2.boxPoints()可获取该矩形的四个顶点坐标
#
#
# def Perspective_transform(original_img, points):
#     # obtain a consistent order of the points and unpack them
#     # individually
#     rect = order_points(points)
#     (tl, tr, br, bl) = rect
#
#
#     # compute the width of the new image, which will be the
#     # maximum distance between bottom-right and bottom-left
#     # x-coordiates or the top-right and top-left x-coordinates
#     widthA = np.sqrt(((br[0] - bl[0])**2) + ((br[1] - bl[1])**2))
#     widthB = np.sqrt(((tr[0] - tl[0])**2) + ((tr[1] - tl[1])**2))
#     maxWidth = max(int(widthA), int(widthB))
#
#
#     # compute the height of the new image, which will be the
#     # maximum distance between the top-right and bottom-right
#     # y-coordinates or the top-left and bottom-left y-coordinates
#     heightA = np.sqrt(((tr[0] - br[0])**2) + ((tr[1] - br[1])**2))
#     heightB = np.sqrt(((tl[0] - bl[0])**2) + ((tl[1] - bl[1])**2))
#     maxHeight = max(int(heightA), int(heightB))
#
#
#     # now that we have the dimensions of the new image, construct
#     # the set of destination points to obtain a "birds eye view",
#     # (i.e. top-down view) of the image, again specifying points
#     # in the top-left, top-right, bottom-right, and bottom-left
#     # order
#     dst = np.array(
#         [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
#         dtype="float32",
#     )
#
#     # compute the perspective transform matrix and then apply it
#     M = cv2.getPerspectiveTransform(rect, dst)
#
#
#     return cv2.warpPerspective(original_img, M, (maxWidth, maxHeight))
#
#
# def order_points(pts):
#     # initialzie a list of coordinates that will be ordered
#     # such that the first entry in the list is the top-left,
#     # the second entry is the top-right, the third is the
#     # bottom-right, and the fourth is the bottom-left
#     rect = np.zeros((4, 2), dtype="float32")
#
#
#     # the top-left point will have the smallest sum, whereas
#     # the bottom-right point will have the largest sum
#     # print("pts=",pts)
#     s = pts.sum(axis=1)
#     rect[0] = pts[np.argmin(s)]
#     rect[2] = pts[np.argmax(s)]
#
#     # now, compute the difference between the points, the
#     # top-right point will have the smallest difference,
#     # whereas the bottom-left will have the largest difference
#     diff = np.diff(pts, axis=1)
#     rect[1] = pts[np.argmin(diff)]
#     rect[3] = pts[np.argmax(diff)]
#
#     # return the ordered coordinates
#     return rect
#
# # 读取图片
#
# target_img= cv2.imread('D:\\program\\code\\cv\\picture\\input\\15-1.jpg')
#
#  # 目标处理
# closed = Img_Outline(target_img)
# points = findContours_img(closed)
# # print("points=",points)
# target = Perspective_transform(target_img, points)
#
# cv2.imwrite("D:\program\code\cv\8-31.jpg",target)
# # 解决中文显示问题
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# plt.subplot(1,2,1)
# plt.title("原始图片")
# plt.imshow(target_img)
# plt.subplot(1,2,2)
# plt.title("校正后的图片")
# plt.imshow(target)
# plt.show()
#
#
#

# import numpy as np
# import cv2
# import os
# import tifffile as tif
# import matplotlib.pyplot as plt
# import math
#
# def read_img(path):
#
#     if os.path.exists(path):
#         #获取文件夹下所有文件名
#         files = os.listdir(path)
#         #存储所有文件的绝对路径
#         path_detail=[]
#         #存储所有文件内容
#         data=[]
#         for f in files:
#              path_detail.append(os.path.join(path,f ))
#              img=cv2.imread(os.path.join(path,f))
#              img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#              data.append(img)
#     return data
#
#
#
#
# def laplace(img, filter):
#
#     img=img.astype(np.int16)
#
#     row, col = img.shape
#
#     img_temp = np.zeros((row + 2, col + 2))
#     img_temp=img_temp.astype(np.int16)
#     img_temp[1:row + 1, 1:col + 1] = img
#
#     img_empty = np.zeros(img.shape)
#     img_empty=img_empty.astype(np.int16)
#
#     for i in range(row):
#         for j in range(col):
#             img_empty[i,j] = np.sum(filter * img_temp[i:i+3,j:j+3])
#
#     for i in range(1,row-1):
#         for j in range(1,col-1):
#             img[i,j]=img[i,j]+img_empty[i, j]
#             if (img[i,j])<0:
#                 img[i,j]=0
#
#
#     return [img,img_empty]
#
# def Laplc(img,filter):
#
#     img=img.astype(np.int16)
#
#     row,col=img.shape
#
#     img_temp=np.zeros(img.shape)
#     img_temp = img_temp.astype(np.int16)
#
#     for i in range(1,row-1):
#         for j in range(1,col-1):
#             img_temp[i,j]=np.sum(filter*img[i-1:i+2,j-1:j+2])
#
#     for i in range(1,row-1):
#         for j in range(1,col-1):
#             img[i,j]=img[i,j]+img_temp[i, j]
#             if (img[i,j])<0:
#                 img[i,j]=0
#
#
#     return [img,img_temp]
#
#
#
#
# if __name__=="__main__":
#     #拉普拉斯算子
#     fil1_1 = np.asarray([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
#     fil1_2 = -1 * fil1_1
#     fil2_1 = np.asarray([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
#     fil2_2 = -1 * fil2_1
#
#
#     ###读取图像
#     # img = tif.imread("D:\program\code\cv\9-1.jpg")
#     img=cv2.imread("D:\program\code\cv\9-1.jpg")
#
#
#     plt.imshow(img,'gray')
#     plt.show()
#    # cv2.imshow("img",img)
#
#     #平滑滤波
#
#     #img = cv2.GaussianBlur (img, (3, 3), 0)
#
#     [res,lp]=Laplc(img,fil2_1)
#     plt.imshow(res,'gray')
#     plt.show()
#
#     """
#     显示标定的拉普拉斯图像只要保持其负数区域，不对其置0即可
#     一般拉普拉斯图像显示要将负数区域置0即可
#     原始图像加上拉普拉斯图像后，对于小于0的部分也是置0处理
#     """

#导入库
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import random
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# #导入图片
# img = mpimg.imread("D:\program\code\cv\9-1.jpg")
# #转换灰度
# gimg=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# #拉普拉斯算子锐化
# kernel=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],np.float32)#定义拉普拉斯算子
# dst=cv2.filter2D(img,-1,kernel=kernel)#调用opencv图像锐化函数
# # sobel算子锐化
# # 对x方向梯度进行sobel边缘提取
# # x=cv2.Sobel(gimg,cv2.CV_64F,1,0)
# # 对y方向梯度进行sobel边缘提取
# y=cv2.Sobel(gimg,cv2.CV_64F,0,1)
# #对x方向转回uint8
# absX=cv2.convertScaleAbs(x)
# #对y方向转会uint8
# absY=cv2.convertScaleAbs(y)
# #x，y方向合成边缘检测结果
# dst1=cv2.addWeighted(absX,0.5,absY,0.5,0)
# #与原图像堆叠
# res=dst1+gimg
# #测试
# #print("dstshape:",dst1)
# #print("resshape:",res)
# #按要求左右显示原图与拉普拉斯处理结果
# # imges1=np.hstack([img,dst])
# imges1=np.hstack([dst])
# plt.subplot(1,2,1)
# plt.title("原始图像")
# plt.imshow(img)
#
# plt.subplot(1,2,2)
# plt.title("拉普拉斯变换")
# plt.imshow(dst1)
# plt.show()
# cv2.imshow('lapres',imges1)
#按要求左右显示原图与sobel处理结果
# image=np.hstack([gimg,res])
# cv2.imshow('sobelres',image)
#去缓存
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# img = mpimg.imread("D:\program\code\cv\9-1.jpg")
# # #转换灰度
# # gimg=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# dst=cv.Laplacian(img,cv.CV_32F)
# lpls=cv.convertScaleAbs(dst)
# plt.subplot(1,2,1)
# plt.title("拉普拉斯变换")
# plt.imshow(dst)
# plt.show()
import matplotlib.image as mpimg

# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os.path
import copy


# 椒盐噪声
def SaltAndPepper(src, percetage):
    SP_NoiseImg = src.copy()
    SP_NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(SP_NoiseNum):
        randR = np.random.randint(0, src.shape[0] - 1)
        randG = np.random.randint(0, src.shape[1] - 1)
        randB = np.random.randint(0, 3)
        if np.random.randint(0, 1) == 0:
            SP_NoiseImg[randR, randG, randB] = 0
        else:
            SP_NoiseImg[randR, randG, randB] = 255
    return SP_NoiseImg


# 高斯噪声
def addGaussianNoise(image, percetage):
    G_Noiseimg = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    G_NoiseNum = int(percetage * image.shape[0] * image.shape[1])
    for i in range(G_NoiseNum):
        temp_x = np.random.randint(0, h)
        temp_y = np.random.randint(0, w)
        G_Noiseimg[temp_x][temp_y][np.random.randint(3)] = np.random.randn(1)[0]
    return G_Noiseimg


# 昏暗
def darker(image, percetage=0.9):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    # get darker
    for xi in range(0, w):
        for xj in range(0, h):
            image_copy[xj, xi, 0] = int(image[xj, xi, 0] * percetage)
            image_copy[xj, xi, 1] = int(image[xj, xi, 1] * percetage)
            image_copy[xj, xi, 2] = int(image[xj, xi, 2] * percetage)
    return image_copy


# 亮度
def brighter(image, percetage=1.5):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    # get brighter
    for xi in range(0, w):
        for xj in range(0, h):
            image_copy[xj, xi, 0] = np.clip(int(image[xj, xi, 0] * percetage), a_max=255, a_min=0)
            image_copy[xj, xi, 1] = np.clip(int(image[xj, xi, 1] * percetage), a_max=255, a_min=0)
            image_copy[xj, xi, 2] = np.clip(int(image[xj, xi, 2] * percetage), a_max=255, a_min=0)
    return image_copy


# 旋转
def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    # If no rotation center is specified, the center of the image is set as the rotation center
    if center is None:
        center = (w / 2, h / 2)
    m = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, m, (w, h))
    return rotated


# 翻转
def flip(image):
    flipped_image = np.fliplr(image)
    return flipped_image
#
# img=cv.imread("D:\\program\\code\\cv\\20-3.jpg")
# b,g,r=cv.split(img)
# img=cv.merge([r,g,b])
# imgs=brighter(img)
# # imgs=darker(img)
# plt.subplot(1,2,1)
# plt.title("original_image")
# plt.imshow(img)
# plt.subplot(1,2,2)
# plt.title("变换后图像")
# plt.imshow(imgs)
# plt.show()


# 图片文件夹路径
file_dir = r'D:\program\code\cv\data\i\\'
file_result=r"D:\program\code\cv\data\r\\"
for img_name in os.listdir(file_dir):
    img_path = file_dir + img_name
    img = cv2.imread(img_path)
    # cv2.imshow("1",img)
    # cv2.waitKey(5000)
    # 旋转
    rotated_90 = rotate(img, 90)
    cv2.imwrite(file_dir + img_name[0:-4] + '_r90.jpg', rotated_90)
    rotated_180 = rotate(img, 180)
    cv2.imwrite(file_dir + img_name[0:-4] + '_r180.jpg', rotated_180)

for img_name in os.listdir(file_dir):
    img_path = file_dir + img_name
    img = cv2.imread(img_path)
    # 镜像
    flipped_img = flip(img)
    cv2.imwrite(file_dir + img_name[0:-4] + '_fli.jpg', flipped_img)

    # 增加噪声
    # img_salt = SaltAndPepper(img, 0.3)
    # cv2.imwrite(file_dir + img_name[0:7] + '_salt.jpg', img_salt)
    # img_gauss = addGaussianNoise(img, 0.3)
    # cv2.imwrite(file_dir + img_name[0:-4] + '_noise.jpg', img_gauss)
    #
    # # 变亮、变暗
    # img_darker = darker(img)
    # cv2.imwrite(file_dir + img_name[0:-4] + '_darker.jpg', img_darker)
    # img_brighter = brighter(img)
    # cv2.imwrite(file_dir + img_name[0:-4] + '_brighter.jpg', img_brighter)
    #
    # blur = cv2.GaussianBlur(img, (7, 7), 1.5)
    # #      cv2.GaussianBlur(图像，卷积核，标准差）
    # cv2.imwrite(file_dir + img_name[0:-4] + '_blur.jpg', blur)


