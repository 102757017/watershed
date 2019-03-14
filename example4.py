# -*- coding: UTF-8 -*-
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.ndimage import label, generate_binary_structure


os.chdir(os.path.dirname(__file__))

#因为真实的图片色彩难以区分低梯度图像的差异，因此用plt显示图片赋予对比明显的颜色
# 生成画布
fig=plt.figure()
img = cv2.imdecode(np.fromfile('4.png', dtype=np.uint8), 1)
#图片先转成灰度的
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


#生成子图，将画布分割成1行5列，图像画在从左到右从上到下的第1块
ax1=fig.add_subplot(161)
ax1.imshow(gray)
ax1.set_title("origin")

#分水岭算法需要对背景、前景（目标）、未知区域进行标记。
#标记的数据类型必须是int32，否则后面会报错
#未知区域标记为0（黑色）
markers=np.zeros_like(gray, dtype=np.int32)


#确定的背景标记为1
markers[np.where(gray>253)]=1

ax2=fig.add_subplot(162)
ax2.imshow(markers)
ax2.set_title("background")


#分离前景区域
ret,foreground=cv2.threshold(gray,220,255,cv2.THRESH_BINARY)
#反色
foreground=cv2.bitwise_not(foreground)

#进行距离变换，计算每个像素离最近最近0值像素(黑色区域)的距离
foreground=cv2.distanceTransform(foreground, cv2.DIST_L1,3)

ax3=fig.add_subplot(163)
ax3.imshow(foreground)
ax3.set_title("distanceTransform")


#分离前景区域
ret,foreground=cv2.threshold(foreground,25,255,cv2.THRESH_BINARY)
#转换数据类型
foreground=foreground.astype(np.uint8)

#label是标签，该函数把图像背景标成0，其他封闭目标用从1开始的整数标记
#stats 是bounding box的信息，N*5的矩阵，行对应每个label，五列分别为[x0, y0, 宽度, 高度, 面积]
#centroids 是每个域的质心坐标
_, labels, stats, centroids = cv2.connectedComponentsWithStats(foreground)
#使背景标记为1
labels=labels+1

ax4=fig.add_subplot(164)
ax4.imshow(labels)
ax4.set_title("foreground")


#确定的前景目标标记为2、3、4......(不同目标标记为不同序号，方面后面进行粘连前景的分割)
markers[np.where(labels!=1)]=labels[np.where(labels!=1)]
markers2=markers.copy()

ax5=fig.add_subplot(165)
ax5.imshow(markers)
ax5.set_title("markers")


#只接受三通道的图像
#分水岭变换的结果会保存在markers2中
cv2.watershed(img,markers2)



ax6=fig.add_subplot(166)
ax6.imshow(markers2)
ax6.set_title("obj")

plt.show()
