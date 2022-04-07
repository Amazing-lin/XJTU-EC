import cv2
import PIL.Image as Image

imgpath = "/home/bell-chuangxingang/桌面/code/U-net/Unet-master/results/52A0_16.jpg"
src = cv2.imread(imgpath)
dst = cv2.threshold(src,0.1,255,cv2.THRESH_BINARY)
cv2.imwrite("/home/bell-chuangxingang/桌面/code/U-net/Unet-master/results/temp.jpg",dst) #抠出前景

