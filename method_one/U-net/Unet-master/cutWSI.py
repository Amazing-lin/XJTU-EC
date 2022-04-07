import openslide
from openslide.deepzoom import DeepZoomGenerator
import numpy as np
# import skimage.io as io
import os
import cv2


img_path = '/home/bell-chuangxingang/桌面/微组织/7C-5.svs'
result_path = '/home/bell-chuangxingang/桌面/微组织/patch/7C-5/1024*1024/'
slide = openslide.open_slide(img_path)
# slide_suo = slide.get_thumbnail((100,100))
# slide_suo.save(result_path + "suo.jpg")

data_gen = DeepZoomGenerator(slide,tile_size=1024,overlap=1,limit_bounds=False)
[w,h] = slide.level_dimensions[0]
num_w = int(np.floor(w/1024))+1
num_h = int(np.floor(h/1024))+1

# 将图片转为灰度图
# img = cv2.imread('/home/bell-chuangxingang/桌面/svs/tosvs1_31.jpg')
# img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# count = 0
# for i in range(512):
#     for j in range(512):
#         if(img_gray[i][j] > 230):
#             count = count + 1
#
# print(count/(512*512 -count))
# # cv2.imshow("img_gray", img_gray)

print(w,h)
print(data_gen.tile_count)
print(data_gen.level_count)

for i in range(num_w):
    for j in range(num_h):
        img = np.array(data_gen.get_tile(data_gen.level_count-1,(i,j)))
        # if img.var() > 500 :
        # img = cv2.resize(img,(512,512))
        # io.imsave(result_path+'05A'+str(i)+'_'+ str(j) + ".jpg",img)
        cv2.imwrite(result_path+'7C-5'+str(i)+'_'+ str(j) + ".jpg",img)


