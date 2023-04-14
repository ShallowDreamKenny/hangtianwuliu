import cv2 as cv


# img1 img2 分别是;两张地图 p1,p2是对应点 于两张地图的像素
import numpy as np


def concat_img(img1,img2,p1,p2):
    img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    print(w1,h1)
    print(w2, h2)
    img1 = cv.resize(img1, (w1//2,h1//2))
    img2 = cv.resize(img2, (w2//2,h2//2))
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    print(w1,h1)
    print(w2, h2)
    lw = max(p1[0],p2[0])
    rw = max(w1-p1[0],w2-p2[0])

    th = max(p1[1],p2[1])
    bh = max(h1 - p1[1], h2 - p2[1])

    w = lw + rw
    h = th + bh

    # 创建一个满足地图尺寸的底片
    # 行列
    creat1 = np.full((h, w), 205,dtype=np.uint8)
    creat2 = np.full((h, w), 205,dtype=np.uint8)
    #creat1 = np.zeros((h,w),dtype=np.uint8)
    #creat2 = np.zeros((h,w),dtype=np.uint8)
    cv.imshow('img1',img1)
    cv.imshow('img2',img2)
    cv.waitKey(0)
    # 将两个地图 分别放入底片对应的位置
    # 找到对应点在新地图的位置
    newp = (lw,th)
    img1posleft = (newp[0]-p1[0],newp[1]-p1[1])
    creat1[img1posleft[1]:img1posleft[1]+h1,img1posleft[0]:img1posleft[0]+w1] = img1

    img2posleft = (newp[0]-p2[0],newp[1]-p2[1])
    creat2[img2posleft[1]:img2posleft[1]+h2,img2posleft[0]:img2posleft[0]+w2] = img2

    # print(creat)
    # print(img1posleft)
    cv.imshow("create1", creat1)
    cv.imshow("create2", creat2)
    cv.waitKey(0)
    # 取大 及白色和灰色区域  白 > 灰 > 黑
    res1 = cv.max(creat1, creat2)
    # 这里是 res1  无黑 仅灰白

    #cv.imshow('img1',img1)
    #cv.imshow('img2',img2)
    cv.imshow('noblack',res1)
    cv.waitKey(0)

    # 只保留 黑色
    _, img1black = cv.threshold(creat1, 200, 255, cv.THRESH_BINARY)
    _, img2black = cv.threshold(creat2, 200, 255, cv.THRESH_BINARY)

    res2 = cv.bitwise_and(img1black,img2black)
    cv.imshow("black",res2)
    cv.waitKey(0)

    # 合成结果
    res3 = cv.bitwise_and(res1,res2)
    cv.imshow("res3",res3)
    cv.waitKey(0)

    # 优化结果
    retval = cv.getStructuringElement(2, (5,5))
    retval1 = cv.getStructuringElement(1, (3, 3))
    # 腐蚀
    res4 = cv.morphologyEx(res3, 0, retval, 3)
    cv.imshow("lastres4", res4)
    cv.waitKey(0)
    # 膨胀
    res4 = cv.morphologyEx(res4, 1, retval1, 2)
    cv.imshow("lastres5", res4)
    cv.waitKey(0)
    # 中值滤波
    res4 = cv.medianBlur(res4,3)
    cv.imshow("lastres6",res4)
    cv.waitKey(0)

    cv.imwrite("res.png",res4)
img1 = cv.imread("map1.pgm")
img2 = cv.imread("map2.pgm")


cv.imwrite("map1.png",img1)
cv.imwrite("map2.png",img2)
#cv.imshow("1",img1)
#cv.imshow("2",img2)
concat_img(img1,img2,(512//2,670//2),(500//2,570//2))
# print(np.max(img1))
cv.waitKey(0)