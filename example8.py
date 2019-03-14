# -*- coding: UTF-8 -*-
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.ndimage import label, generate_binary_structure


def sobel(gray,size):
    #ksize是指核的大小,只能取奇数，影响边缘的粗细
    x = cv2.Sobel(gray,cv2.CV_16S,1,0,ksize=size)
    y = cv2.Sobel(gray,cv2.CV_16S,0,1,ksize=size)
    
    # 转回uint8
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)

    dst = cv2.addWeighted(absX,0.5,absY,0.5,0)
    return dst

def canny(img):
    #拆分色彩通道
    b, g, r = cv2.split(img)
    #锐化
    b = cv2.equalizeHist(b)
    g = cv2.equalizeHist(g)
    r = cv2.equalizeHist(r)
    target_b = cv2.Canny(b, 200, 1200)
    target_g = cv2.Canny(g, 200, 1200)
    target_r = cv2.Canny(r, 200, 1200)
    target=cv2.add(target_b,target_g)
    target=cv2.add(target,target_r)
    return target



os.chdir(os.path.dirname(__file__))

#因为真实的图片色彩难以区分低梯度图像的差异，因此用plt显示图片赋予对比明显的颜色
#生成画布
fig=plt.figure()



img = cv2.imdecode(np.fromfile('8.png', dtype=np.uint8), 1)
#生成子图，将画布分割成1行5列，图像画在从左到右从上到下的第1块
ax1=fig.add_subplot(231)
ax1.imshow(img)
ax1.set_title("origin")

#寻找文本边缘
edge=canny(img)

#确定未知区域
h_structure = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
d_edge = cv2.dilate(edge,h_structure,1)


#图片先转成灰度的
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#分水岭算法需要对背景、前景（目标）、未知区域进行标记。
#标记的数据类型必须是int32，否则后面会报错
#确定的背景标记为1
markers=np.ones_like(gray, dtype=np.int32)

#未知区域标记为0（黑色）
markers[np.where(d_edge==255)]=0
ax2=fig.add_subplot(232)
ax2.imshow(markers)
ax2.set_title("background")

#转换二值图
ret,binary=cv2.threshold(gray,240,255,cv2.THRESH_BINARY)
#反色
binary=cv2.bitwise_not(binary)

#去除小的区域
binary,contours,hierarch=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
for i in range(len(contours)):
    area = cv2.contourArea(contours[i])
    #设定阈值
    if area < 20:
        cv2.drawContours(binary,[contours[i]],0,0,-1)

#确定前景种子区域
foreground=cv2.subtract(binary,d_edge)
#label是标签，该函数把图像背景标成0，其他封闭目标用从1开始的整数标记
#stats 是bounding box的信息，N*5的矩阵，行对应每个label，五列分别为[x0, y0, 宽度, 高度, 面积]
#centroids 是每个域的质心坐标
_, labels, stats, centroids = cv2.connectedComponentsWithStats(foreground)
#使背景标记为1
labels=labels+1

ax4=fig.add_subplot(233)
ax4.imshow(labels)
ax4.set_title("foreground")


#确定的前景目标标记为2、3、4......(不同目标标记为不同序号，方面后面进行粘连前景的分割)
markers[np.where(labels!=1)]=labels[np.where(labels!=1)]

ax5=fig.add_subplot(234)
ax5.imshow(markers)
ax5.set_title("markers")


#只接受三通道的图像
#分水岭变换的结果会保存在markers中
cv2.watershed(img,markers)



ax6=fig.add_subplot(235)
ax6.imshow(markers)
ax6.set_title("obj")

plt.show()
