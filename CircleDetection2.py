import os
import math
import cv2
import time
import numpy as np

from Image import *
from scipy import misc, ndimage


# 1. 图像预处理
def dataPreprocess(Image):
    """
    :param Image: 灰度图
    :return: 二值图、二值01图
    """
    binImage = bin(4, Image)
    # 二值图01化
    bin01Image = binImage.copy()
    bin01Image[binImage == 255] = 0
    bin01Image[binImage == 0] = 1
    return binImage, bin01Image


# 1.1 输入图像进行裁剪，在指定区域进行连通域查找，默认设置为[0~0.15]、[0.85~1]：
def getSpecificAreaConnects(originImage, thresholdH=[0.15, 0.85]):
    """
    :param originImage: 原始图像
    :param threshold: 裁剪阈值，默认设置为[0~0.15]、[0.85~1]
    :return:裁剪后的顶部图像、底部图像，以及底部图像对应顶部边界的位置
    """
    H, W = originImage.shape
    topEndPos = int(H * thresholdH[0])
    midStartPos = int(H * 0.4)
    midEndPos = int(H * 0.6)
    buttonStartPos = int(H * thresholdH[1])
    topImageArea = originImage[0:topEndPos, :]
    midImageArea = originImage[midStartPos:midEndPos, :]
    buttonImageArea = originImage[buttonStartPos:H, :]

    return topImageArea, midImageArea, buttonImageArea, midStartPos


# 1.2 灰度图二值化
def bin(num, img):
    if num == 1:
        # 1.全局阈值法
        ret, out = cv2.threshold(src=img,  # 要二值化的图片
                                 thresh=90,  # 全局阈值
                                 maxval=255,  # 大于全局阈值后设定的值
                                 type=cv2.THRESH_BINARY)  # 设定的二值化类型，THRESH_BINARY：表示小于阈值置0，大于阈值置填充色
    elif num == 2:
        # 2.自适应阈值法
        out = cv2.adaptiveThreshold(src=img,  # 要进行处理的图片
                                    maxValue=255,  # 大于阈值后设定的值
                                    adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                    # 自适应方法，ADAPTIVE_THRESH_MEAN_C：表区域内均值；ADAPTIVE_THRESH_GAUSSIAN_C：表区域内像素点加权求和
                                    thresholdType=cv2.THRESH_BINARY,  # 同全局阈值法中的参数一样
                                    blockSize=11,  # 方阵（区域）大小，
                                    C=1)  # 常数项，每个区域计算出的阈值的基础上在减去这个常数作为这个区域的最终阈值，可以为负数
    elif num == 3:
        # 3.OTSU二值化
        ret2, out = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    else:
        ret2, out = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return out

    # 获取图像中所有的连通域


# 2. 提取连通域
def getConnectDomin(binImage, value):
    """
    :param binImage: 二值图
    :param value: 连通域边界值为value
    :return: 连通域坐标，并非最外面一层的连通域，也包含着连通域内部的value值的坐标
    """
    s = solution(binImage)
    result = s.bsfsolver(value)
    return result


# 3. 获取当前连通域上下左右边界
def getConnectDominBorders(connect):
    """
    :param connect: 连通域集合
    :return: 每个连通域的上下左右边界集合
    """
    borders = []
    for i in connect:
        rowMax = np.array(i)[:, 0].max()  # 行最大
        rowMin = np.array(i)[:, 0].min()  # 行最小
        colMax = np.array(i)[:, 1].max()  # 列最大
        colMin = np.array(i)[:, 1].min()  # 列最小
        borders.append([rowMax, rowMin, colMax, colMin])
    return borders


# 4. 判断一个连通域是不是圆形
def isCirCle(border, connect, binaryImage):
    """
    :param border: 当前连通域的上下左右边界。具体格式:【行大，行小，列大，列小】
    :param connect: 连通域的边界坐标
    :param binaryImage: 原始二值图
    :return:
    """
    H = border[0] - border[1]
    W = border[2] - border[3]
    if H * W < 100 or H > W * 3 or H * 3 < W:
        return False
    else:
        # 从原始二值图中裁取出来目标区域
        out = binaryImage[border[1]:border[0] + 1, border[3]:border[2] + 1]
        height, width = np.array(out).shape
        total = height * width
        whitePoint = 0
        for y in range(height):
            for x in range(width):
                if out[y, x] == 255:
                    whitePoint += 1
        blackPoint = total - whitePoint
        if abs(1 - blackPoint / whitePoint) < 0.3:
            area = conArea(connect)
            perimeter = 2 * math.pi * ((height + width) / 4)
            metric = 4 * math.pi * area / (perimeter ** 2)
            if metric > 0.90:
                return True
        return False


# 4.1 计算连通域面积
def conArea(connect):
    """
    :param connect: 连通域坐标
    :return: 连通域内的像素点个数=面积
    """
    temp_list = np.array(connect)
    col = list((set(temp_list[:, 1])))
    area = 0
    for j in col:
        temp_row = []
        for k in temp_list:
            if k[1] == j:
                temp_row.append(k[0])
        area += np.array(temp_row).max() - np.array(temp_row).min()
    return area


# 5.1 按照目标圆最外层轮廓的外接圆的圆心为圆心
def getCenterThroughMinCircumcircleCenter(connect):
    """
    :param connect: 连通域最外围像素
    :return: 圆心、半径
    """
    return cv2.minEnclosingCircle(np.array(connect))


# 5.1.1 获取连通域的最外围轮廓
def outLayerConnect(border, connect):
    """
    :param border: 连通域的上下左右边界
    :param connect: 连通域
    :return: 连通域的最外层轮廓
    """
    res = []
    connect = np.array(connect)
    connect = connect[np.lexsort(connect[:, ::-1].T)]
    symbol = True
    for i in range(border[1], border[0] + 1):
        row = np.where(connect[:, 0] == i)[0]
        if i == border[0] or i == border[1]:
            for j in row:
                res.append(connect[j])
        else:
            res.append(connect[row[0]])
            res.append(connect[row[len(row) - 1]])
    return res


# 5.2 根据直方图统计的结果找出圆心
def getCenterThroughHistogram(border, histogramCol, histogramRow, binaryImage):
    """
    :param border: 目标圆所处矩形框的上下左右边界
    :param histogramCol: 目标圆所含黑色像素的列的直方图
    :param histogramRow: 目标圆所含黑色像素的行的直方图
    :param binaryImage: 二值图像
    :return: 圆心
    """
    col = 0
    colMaxNum = histogramCol.count(max(histogramCol))
    row = 0
    rowMaxNum = histogramRow.count(max(histogramRow))
    if colMaxNum == 1:
        col = border[3] + histogramCol.index(max(histogramCol))
    else:
        for i in range(colMaxNum):
            col += histogramCol.index(max(histogramCol))
            histogramCol[histogramCol.index(max(histogramCol))] = min(histogramCol)
        col = border[3] + int(col / colMaxNum)
    if rowMaxNum == 1:
        row = border[1] + histogramRow.index(max(histogramRow))
    else:
        for i in range(rowMaxNum):
            row += histogramRow.index(max(histogramRow))
            histogramRow[histogramRow.index(max(histogramRow))] = min(histogramRow)
        row = border[1] + int(row / rowMaxNum)

    return (row, col)


# 5.2.1 获取圆的直方图
def circleHistogram(border, connect, binaryImage):
    """
    :param border:连通域的边界
    :param connect:连通域
    :param binaryImage:连通域所在图像的二值图
    :return:列的直方图、行的直方图
    """
    minCol = border[3]
    minRow = border[1]
    blackCol = [0 for _ in range(border[3], border[2] + 1)]
    blackRow = [0 for _ in range(border[1], border[0] + 1)]
    for i in connect:
        blackCol[i[1] - minCol] += 1 if binaryImage[i[0]][i[1]] == 0 else 0
        blackRow[i[0] - minRow] += 1 if binaryImage[i[0]][i[1]] == 0 else 0
    return blackCol, blackRow


# 5.3 按照目标圆所处矩形框的对角线交点为圆心
def getCenterThroughMatrixCenter(border):
    """
    :param border: 目标圆所在矩形框的上下左右边界
    :return: 圆心
    """
    return (border[0] - int((border[0] - border[1]) / 2),
            border[3] + int((border[2] - border[3]) / 2))


# 5.4 获取圆心 按照目标圆所处的矩形区域的最大的两个白色区域的白像素点的边界的最短距离确定圆心
def getCenter(connects):
    """
    :param connects: 目标圆的连通域所在的矩形区域
    :return: 圆心坐标
    """
    center = []  # 数据格式：[firstConnect中的点坐标f1，secondConnect中与f1的距离等于最小距离的所有点坐标（一般为两个）]
    minDistance = 100000000
    conLen = [len(i) for i in connects]
    firstConnect = connects.pop(conLen.index(max(conLen)))
    conLen.remove(max(conLen))
    secondConnect = connects[conLen.index(max(conLen))]
    for i in range(len(firstConnect)):
        distance = [abs(coordinate[0] - firstConnect[i][0]) + abs(coordinate[1] - firstConnect[i][1]) for coordinate in
                    secondConnect]
        minDistanceTemp = min(distance)
        minDisList = []  # 第一个坐标点为firstConnect数组的，后面的坐标点都是secondConnect中与第一个坐标点距离最小的坐标
        if minDistanceTemp < minDistance:
            # 更新最小距离
            minDistance = minDistanceTemp
            center.clear()
            minDisList.append(firstConnect[i])
            for index, value in enumerate(distance):
                if value == minDistanceTemp:
                    minDisList.append(secondConnect[index])
            center.append(minDisList)
        elif minDistanceTemp == minDistance:

            minDisList.append(firstConnect[i])
            for index, value in enumerate(distance):
                if value == minDistanceTemp:
                    minDisList.append(secondConnect[index])
            center.append(minDisList)
    print(center)
    a = [int(np.array([i[0] for i in center])[:, 0].mean()), int(np.array([i[0] for i in center])[:, 1].mean())]
    b = np.array([[int(np.array(i[1:])[:, 0].mean()), int(np.array(i[1:])[:, 1].mean())] for i in center])
    c = [int(b[:, 0].mean()), int(b[:, 1].mean())]
    res = [int((a[0] + c[0]) / 2), int((a[1] + c[1]) / 2)]
    print(res)
    return res


# 6. 圆心优化
def centerPosOptimize(topLeftVertex, centerMatrix, centerCount):
    """
    :param topLeftVertex: 矩阵的顶点坐标
    :param centerMatrix: 圆心矩阵
    :param centerCount: 当前圆是一幅图中的第几个圆
    :return:
    """
    cx, cy = topLeftVertex
    centerMatrix = np.array(centerMatrix)
    n = len(centerMatrix)

    blackNum = 0

    if centerCount == 0:
        # 计算最后一行0的个数
        for i in range(n - 1, -1, -1):
            if centerMatrix[n - 1][i] == 0:
                blackNum += 1
            else:
                break
        if blackNum == 0:
            return (cx + 5, cy + 5)
        # 从centerMatrix中按最后一行0的个数进行切分
        centerMatrixTemp = centerMatrix[:, n - blackNum:n]
        # 当某一行0的个数小于最后一行0的个数的一半时，退出。
        for i in range(n - 1, -1, -1):
            if centerMatrixTemp[i].sum() / 255 < int((blackNum + 1) / 2):
                continue
            else:
                return (cx + (i + 1 if i != n - 1 else i), cy + n - blackNum)
    elif centerCount == 1:
        if centerMatrix[n - 1][0] == 0:
            # 计算最后一行0的个数
            for i in range(n):
                if centerMatrix[n - 1][i] == 0:
                    blackNum += 1
                else:
                    break
            # 从centerMatrix中按最后一行0的个数进行切分
            centerMatrixTemp = centerMatrix[:, 0:blackNum]
            # 从最后一行往上数
            for i in range(n - 1, -1, -1):
                if centerMatrixTemp[i].sum() / 255 < int((blackNum + 1) / 2):
                    continue
                else:
                    return (cx + (i + 1 if i != n - 1 else i), cy + blackNum - 1)
        elif centerMatrix[n - 1][0] == 255:
            # 计算最后一行255的个数
            for i in range(n):
                if centerMatrix[n - 1][i] == 255:
                    blackNum += 1
                else:
                    break
            # 从centerMatrix中按最后一行0的个数进行切分
            centerMatrixTemp = centerMatrix[:, 0:blackNum]
            for i in range(n - 1, -1, -1):
                if blackNum - (centerMatrixTemp[i].sum() / 255) < int((blackNum + 1) / 2):
                    continue
                else:
                    return (cx + (i + 1 if i != n - 1 else i), cy + blackNum - 1)
    elif centerCount == 2:
        if centerMatrix[0][n - 1] == 0:
            # 计算第一行0的个数
            for i in range(n - 1, -1, -1):
                if centerMatrix[0][i] == 0:
                    blackNum += 1
                else:
                    break
            # 从centerMatrix中按最后一行0的个数进行切分
            centerMatrixTemp = centerMatrix[:, n - blackNum:n]
            for i in range(n):
                if centerMatrixTemp[i].sum() / 255 < int((blackNum + 1) / 2):
                    continue
                else:
                    return (cx + (i - 1 if i != 0 else i), cy + n - blackNum)
        elif centerMatrix[0][n - 1] == 255:
            # 计算最后一行0的个数
            for i in range(n - 1, -1, -1):
                if centerMatrix[0][i] == 255:
                    blackNum += 1
                else:
                    break
            # 从centerMatrix中按最后一行0的个数进行切分
            centerMatrixTemp = centerMatrix[:, n - blackNum:n]
            for i in range(n):
                if blackNum - (centerMatrixTemp[i].sum() / 255) < int((blackNum + 1) / 2):
                    continue
                else:
                    return (cx + (i - 1 if i != 0 else i), cy + n - blackNum)
    elif centerCount == 3:
        if centerMatrix[0][0] == 0:
            # 计算最后一行0的个数
            for i in range(n):
                if centerMatrix[0][i] == 0:
                    blackNum += 1
                else:
                    break
            # 从centerMatrix中按最后一行0的个数进行切分
            centerMatrixTemp = centerMatrix[:, 0:blackNum]
            for i in range(n):
                if centerMatrixTemp[i].sum() / 255 < int((blackNum + 1) / 2):
                    continue
                else:
                    return (cx + (i - 1 if i != 0 else i), cy + blackNum - 1)
        elif centerMatrix[0][0] == 255:
            # 计算最后一行0的个数
            for i in range(n):
                if centerMatrix[0][i] == 255:
                    blackNum += 1
                else:
                    break
            # 从centerMatrix中按最后一行0的个数进行切分
            centerMatrixTemp = centerMatrix[:, 0:blackNum]
            for i in range(n):
                if blackNum - (centerMatrixTemp[i].sum() / 255) < int((blackNum + 1) / 2):
                    continue
                else:
                    return (cx + (i - 1 if i != 0 else i), cy + blackNum - 1)
    elif centerCount == 4:
        if centerMatrix[0][n - 1] == 0:
            # 计算第一行0的个数
            for i in range(n - 1, -1, -1):
                if centerMatrix[0][i] == 0:
                    blackNum += 1
                else:
                    break
            # 从centerMatrix中按最后一行0的个数进行切分
            centerMatrixTemp = centerMatrix[:, n - blackNum:n]
            for i in range(n):
                if centerMatrixTemp[i].sum() / 255 < int((blackNum + 1) / 2):
                    continue
                else:
                    return (cx + (i - 1 if i != 0 else i), cy + n - blackNum)
        elif centerMatrix[0][n - 1] == 255:
            # 计算最后一行0的个数
            for i in range(n - 1, -1, -1):
                if centerMatrix[0][i] == 255:
                    blackNum += 1
                else:
                    break
            # 从centerMatrix中按最后一行0的个数进行切分
            centerMatrixTemp = centerMatrix[:, n - blackNum:n]
            for i in range(n):
                if blackNum - (centerMatrixTemp[i].sum() / 255) < int((blackNum + 1) / 2):
                    continue
                else:
                    return (cx + (i - 1 if i != 0 else i), cy + n - blackNum)
    else:
        # 计算最后一行0的个数
        for i in range(n):
            if centerMatrix[0][i] == 0:
                blackNum += 1
            else:
                break
        # 从centerMatrix中按最后一行0的个数进行切分
        centerMatrixTemp = centerMatrix[:, 0:blackNum]
        for i in range(n):
            if centerMatrixTemp[i].sum() / 255 < int((blackNum + 1) / 2):
                continue
            else:
                return (cx + (i - 1 if i != 0 else i), cy + blackNum - 1)
    return (cx + 5, cy + 5)


# 6.1 圆心处二次截取
def centerAreaSecondCut(center, originImage, binaryImage):
    """
    :param center: 圆心坐标
    :param originImage: 原始图
    :param binaryImage: 二值图
    :return: 矩阵的左上角顶点像素，包含初步圆心的一个11x11的矩阵
    """
    cx, cy = center
    topLeftx, topLefty = cx - 5, cy - 5
    # centerAreaInBinaryImage = binaryImage[cx - 5:cx + 6, cy - 5:cy + 6]
    centerAreaInOriginImage = originImage[cx - 5:cx + 6, cy - 5:cy + 6]  # 左闭右开 [cx - 5:cx + 5)
    centerAreaBinaryImage = bin(3, centerAreaInOriginImage)

    # originImageSavePath = f'./result/5/' + f"{center[0]}x{center[1]}_" + filename
    # binImageSavePath = f'./result/6/' + f"{center[0]}x{center[1]}_" + filename
    #
    # cv2.imwrite(originImageSavePath, centerAreaInOriginImage)
    # cv2.imwrite(binImageSavePath, centerAreaBinaryImage)
    return (topLeftx, topLefty), centerAreaBinaryImage


# 6.2 判断当前圆心是哪个位置的目标圆
def getTargetCirclePos(originImage, center):
    H, W = originImage.shape
    if center[0] <= H / 2 and center[1] <= W / 2:
        return 0
    elif center[0] < H / 2 and center[1] > W / 2:
        return 1
    elif center[0] > H / 2 and center[1] < W / 2:
        return 2
    else:
        return 3


# 6.2 判断当前圆心是哪个位置的目标圆
def getTargetCirclePos2(W, center):
    if center[1] <= W / 2:
        return 0
    else:
        return 1


# 7. 在二值图与原始图上裁取目标圆所在矩形框并画出圆心所在位置并保存
def cropAndDraw(border, originImage, binaryImage, center, filename, circleNum):
    """
        :param border: 圆形区域的上下边界
        :param originImage: 原始RGB图
        :param filename: 图像名
        :param circleNum: 圆的个数
        """
    center = (center[0] - border[1], center[1] - border[3])
    circleAreaInBinaryImage = binaryImage[border[1]:border[0] + 1, border[3]:border[2] + 1]
    circleAreaInOriginImage = originImage[border[1]:border[0] + 1, border[3]:border[2] + 1]
    cv2.circle(circleAreaInBinaryImage, (center[1], center[0]), 5, (122, 255, 0), 2)
    cv2.circle(circleAreaInOriginImage, (center[1], center[0]), 5, (121, 255, 0), 2)
    originImageSavePath = f'./result/5/' + f"{circleNum}_" + filename
    binImageSavePath = f'./result/6/' + f"{circleNum}_" + filename

    cv2.imwrite(originImageSavePath, circleAreaInOriginImage)
    cv2.imwrite(binImageSavePath, circleAreaInBinaryImage)


# 7.1 从原始图上裁取目标圆所在区域
def getTargetCircleMatrixArea(border, originImage):
    return originImage[border[1]:border[0] + 1, border[3]:border[2] + 1]


def houghTrans(img, filename):
    # 先通过hough transform检测图片中的图片，计算直线的倾斜角度并实现对图片的旋转

    # filepath = 'E:/peking_rw/hough transform/tilt image correction/test image'
    # for filename in os.listdir(filepath):
    #     img = cv2.imread('E:/peking_rw/hough transform/tilt image correction/test image/%s' % filename)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img, 50, 150, apertureSize=3)

    # 霍夫变换
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 0)
    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
    if x1 == x2 or y1 == y2:
        return img
    t = float(y2 - y1) / (x2 - x1)
    rotate_angle = math.degrees(math.atan(t))
    if rotate_angle > 45:
        rotate_angle = -90 + rotate_angle
    elif rotate_angle < -45:
        rotate_angle = 90 + rotate_angle
    rotate_img = ndimage.rotate(img, rotate_angle)
    cv2.imwrite(f'./result/hough/' + "_111x471_" + filename, rotate_img)
    return rotate_img
    # misc.imsave('E:/peking_rw/hough transform/tilt image correction/test result/%s' % filename, rotate_img)


# 8. 透视变换
# def Perspective_transform(original_img, points):
#     # obtain a consistent order of the points and unpack them
#     # individually
#     rect = order_points(points)
#     (tl, tr, br, bl) = rect
#
#     # compute the width of the new image, which will be the
#     # maximum distance between bottom-right and bottom-left
#     # x-coordiates or the top-right and top-left x-coordinates
#     widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
#     widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
#     maxWidth = max(int(widthA), int(widthB))
#
#     # compute the height of the new image, which will be the
#     # maximum distance between the top-right and bottom-right
#     # y-coordinates or the top-left and bottom-left y-coordinates
#     heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
#     heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
#     maxHeight = max(int(heightA), int(heightB))
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
#     return cv2.warpPerspective(original_img, M, (maxWidth, maxHeight))
#
#
# # 8.1 圆心排序
# def order_points(pts):
#     # initialzie a list of coordinates that will be ordered
#     # such that the first entry in the list is the top-left,
#     # the second entry is the top-right, the third is the
#     # bottom-right, and the fourth is the bottom-left
#     rect = np.zeros((4, 2), dtype="float32")
#
#     # the top-left point will have the smallest sum, whereas
#     # the bottom-right point will have the largest sum
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


# 2.1 提取连通域
class solution:
    def __init__(self, im):
        self.im = im
        m, n = im.shape
        self.mask = [[0 for _ in range(n)] for _ in range(m)]

    def isValid(self, i, j, mask, im, borderValue):
        m, n = im.shape
        return i >= 0 and i < m and j >= 0 and j < n and mask[i][j] == 0 and im[i][j] == borderValue

    def add(self, i, j, mask, im, q, borderValue):
        if self.isValid(i, j, mask, im, borderValue):
            q.append([i, j])
            self.mask[i][j] = 1

    def bsfsolver(self, borderValue):
        m, n = self.im.shape
        res = []
        for i in range(m):
            for j in range(n):
                if self.mask[i][j] == 1 or self.im[i][j] != borderValue:
                    continue
                P, Q = list(), list()
                P.append([i, j])
                self.mask[i][j] = 1
                while P:
                    temp = P.pop(0)
                    Q.append(temp)
                    self.add(temp[0] - 1, temp[1], self.mask, self.im, P, borderValue)
                    self.add(temp[0] + 1, temp[1], self.mask, self.im, P, borderValue)
                    self.add(temp[0], temp[1] - 1, self.mask, self.im, P, borderValue)
                    self.add(temp[0], temp[1] + 1, self.mask, self.im, P, borderValue)
                res.append(Q)

        return res


def adsa(originImage, filename, cirNum):
    centers = []
    binaryImage, bin01Image = dataPreprocess(originImage)
    binarySavePath = f'./binary_thresh200/{cirNum}_' + filename
    cv2.imwrite(binarySavePath, binaryImage)

    # 2. 提取连通域
    connects = getConnectDomin(bin01Image, 1)
    # 3. 获取每个连通域的上下左右边界
    borders = getConnectDominBorders(connects)

    for idx in range(len(borders)):
        border = borders[idx]
        connect = connects[idx]
        if isCirCle(border, connects[idx], binaryImage):
            outConnect = outLayerConnect(border, connect)

            # 5.1
            minCircumcircleCenter, radius = getCenterThroughMinCircumcircleCenter(outConnect)
            print(f"最小外接圆圆心：{minCircumcircleCenter}")

            # 5.2 此处为获取圆域直方图并保存
            histogramCol, histogramRow = circleHistogram(border, connect, binaryImage)
            histogramStatisticalCenter = getCenterThroughHistogram(border, histogramCol, histogramRow, binaryImage)
            print(f"直方图统计圆心：{histogramStatisticalCenter}")

            # 5.3 矩形对角线 圆心
            matrixDiagonalCenter = getCenterThroughMatrixCenter(border)
            print(f"矩形对角线圆心：{matrixDiagonalCenter}")

            # 6.1 裁取包含初步圆心目标矩阵
            topLeftVertex, centerAreaBinaryImage = centerAreaSecondCut(matrixDiagonalCenter, originImage,
                                                                       binaryImage)
            circleNum = cirNum + getTargetCirclePos2(originImage.shape[1], matrixDiagonalCenter)
            # 6.2 圆心优化
            centerOpt = centerPosOptimize(topLeftVertex, centerAreaBinaryImage, circleNum)
            print(f"优化后的圆心：{centerOpt}")

            print(borders[idx])

            # 二值图像=根据目标圆所在原始图位置裁取出来重新二值化
            targetCircleAreaBinaryImage = bin(3, getTargetCircleMatrixArea(border, originImage))
            # 添加到圆心列表中去
            centers.append(np.array([centerOpt[1], centerOpt[0]]))
            cv2.circle(targetCircleAreaBinaryImage, (centerOpt[0] - border[1], centerOpt[1] - border[3]), 30,
                       (0, 255, 0), 2)
            # 7. 裁取并保存
            cropAndDraw(border, originImage, binaryImage, centerOpt, filename, circleNum)
    return centers


if __name__ == "__main__":
    time_start = time.time()

    directory_name = "./dataset"  # 要读取的文件夹名

    imgList = os.listdir(directory_name)
    for count in range(0, len(imgList)):
        filename = imgList[count]
        path = directory_name + "/" + filename
        originImage = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # originImage = houghTrans(originImage, filename)
        # centers = []  # 圆心
        H, W = originImage.shape

        # 1. 数据预处理
        topImageArea, midImageArea, buttonImageArea, midStartPos = getSpecificAreaConnects(originImage)
        centerTop = adsa(topImageArea, filename, 0)
        centerMid = adsa(midImageArea, filename, 2)
        centerButton = adsa(buttonImageArea, filename, 4)
        """
        binaryImage, bin01Image = dataPreprocess(originImage)
        binarySavePath = './binary_thresh200/' + filename
        cv2.imwrite(binarySavePath, binaryImage)
        # 2. 提取连通域
        connects = getConnectDomin(bin01Image, 1)
        # 3. 获取每个连通域的上下左右边界
        borders = getConnectDominBorders(connects)

        for idx in range(len(borders)):
            border = borders[idx]
            connect = connects[idx]
            if isCirCle(border, connects[idx], binaryImage):
                outConnect = outLayerConnect(border, connect)

                # 5.1
                minCircumcircleCenter, radius = getCenterThroughMinCircumcircleCenter(outConnect)
                print(f"最小外接圆圆心：{minCircumcircleCenter}")

                # 5.2 此处为获取圆域直方图并保存
                histogramCol, histogramRow = circleHistogram(border, connect, binaryImage)
                histogramStatisticalCenter = getCenterThroughHistogram(border, histogramCol, histogramRow, binaryImage)
                print(f"直方图统计圆心：{histogramStatisticalCenter}")

                # 5.3 矩形对角线 圆心
                matrixDiagonalCenter = getCenterThroughMatrixCenter(border)
                print(f"矩形对角线圆心：{matrixDiagonalCenter}")

                # 6.1 裁取包含初步圆心目标矩阵
                topLeftVertex, centerAreaBinaryImage = centerAreaSecondCut(matrixDiagonalCenter, originImage,
                                                                           binaryImage)
                # 6.2 圆心优化

                circleNum = getTargetCirclePos(originImage, matrixDiagonalCenter)
                centerOpt = centerPosOptimize(topLeftVertex, centerAreaBinaryImage, circleNum)
                print(f"优化后的圆心：{centerOpt}")

                print(borders[idx])

                # 二值图像=根据目标圆所在原始图位置裁取出来重新二值化
                targetCircleAreaBinaryImage = bin(3, getTargetCircleMatrixArea(border, originImage))
                # 添加到圆心列表中去
                centers.append(np.array([centerOpt[1], centerOpt[0]]))
                cv2.circle(targetCircleAreaBinaryImage, (centerOpt[0] - border[1], centerOpt[1] - border[3]), 30,
                           (0, 255, 0), 2)
                # 7. 裁取并保存
                cropAndDraw(border, originImage, binaryImage, centerOpt, filename, circleNum)
        """
        # 圆心重新变回来
        for i in range(len(centerButton)):
            centerMid[i][1] = midStartPos + centerMid[i][1]
            centerButton[i][1] = originImage.shape[0] - (buttonImageArea.shape[0] - centerButton[i][1])
        print(f"顶部圆心：{centerTop}  中部圆心{centerMid}  底部圆心：{centerButton}")
        # 此处应该再加上圆心数量判断
        # centers = [centerTop[0], centerTop[1], centerButton[0], centerButton[1]]
        centersTop = [centerTop[0], centerTop[1], centerMid[0], centerMid[1]]
        centersButton = [centerMid[0], centerMid[1], centerButton[0], centerButton[1]]

        # 8. 透视变换+resize
        targetImgTop = Perspective_transform(originImage, np.array(centersTop))
        targetImgTop = cv2.resize(targetImgTop, (135, 230))
        targetImgButton = Perspective_transform(originImage, np.array(centersButton))
        targetImgButton = cv2.resize(targetImgButton, (135, 231))
        targetImg = np.vstack((targetImgTop, targetImgButton))
        # 9. 结果图保存
        connectSavePath = f'./result/1/' + "_111x471_" + filename
        targetImg = cv2.cvtColor(targetImg, cv2.COLOR_RGBA2BGR)  # 灰度转彩色
        cv2.imwrite(connectSavePath, targetImg)

        time_end = time.time()
        print(f"-----------------------程序运行时间：{time_end - time_start}--------------------------------")
