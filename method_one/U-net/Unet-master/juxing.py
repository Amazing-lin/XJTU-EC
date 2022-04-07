import cv2
import math
imgpath = "/home/bell-chuangxingang/桌面/code/U-net/Unet-master/results/008860_014430_dia.jpg"
oripath = "/home/bell-chuangxingang/桌面/code/U-net/Unet-master/results/008860_014430_qian.jpg"
# 读取图片
mask = cv2.imread(imgpath)
image = cv2.imread(oripath)
# 转换为灰度图
gray = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
# 将图片二值化
_, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
# 在二值图上寻找轮廓
_, contours, _ = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
i=0
for cont in contours:
	# 外接矩形
    x, y, w, h = cv2.boundingRect(cont)
    print(x,y,w,h)
    crop = image[y:y+h,x:x+w]
    top = math.ceil((512-h)/2)
    left = math.ceil((512-w)/2)
    img_temp = cv2.copyMakeBorder(crop,top,512-top,left,512-left,cv2.BORDER_CONSTANT,value=[0,0,0])
    # # 在原图上画出预测的矩形
    # res = cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 1)
    # cv2.imwrite("/home/bell-chuangxingang/桌面/code/U-net/Unet-master/results/008860_014430_rectan.jpg",res) #抠出前景
    cv2.imwrite("/home/bell-chuangxingang/桌面/code/U-net/Unet-master/results/008860_014430_crop"+str(i)+".jpg",img_temp) #抠出前景
    i+=1
