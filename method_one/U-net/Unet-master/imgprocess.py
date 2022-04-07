import cv2
import numpy as np
import cv2
from PIL import Image
import numpy as np
import scipy.misc

yuantu = cv2.imread("/home/bell-chuangxingang/桌面/code/U-net/Unet-master/raw1/test/008860_014430.jpg")
mask = cv2.imread("/home/bell-chuangxingang/桌面/code/U-net/Unet-master/results/008860_014430_dia.jpg")


# 在二值图上寻找轮廓
contours, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
for cont in contours:
	# 外接矩形
    x, y, w, h = cv2.boundingRect(cont)
    # 在原图上画出预测的矩形
    cv2.rectangle(mask, (x, y), (x+w, y+h), (0, 0, 255), 10)

# masked = cv2.bitwise_and(yuantu,mask)

# cv2.imwrite("/home/bell-chuangxingang/桌面/code/U-net/Unet-master/results/008860_014430_rectan.jpg",masked) #抠出前景






