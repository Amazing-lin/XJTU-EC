from unet import *
from data import *
import matplotlib.pyplot as plt
import numpy as np
import os


def xingtai(maskaddress):
    # 读图
    img = cv2.imread(maskaddress, 0)
    # 设置核
    kernel = np.ones((5, 5), np.uint8)
    # 膨胀
    dialation = cv2.dilate(img, kernel, iterations=3)
    erosion = cv2.erode(dialation, kernel, iterations=3)
    return erosion
    # 闭运算
    # closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)
    # cv2.imwrite(imgaddress, erosion)

def rectangle(maskaddr,imageaddr):
    # mask = cv2.imread(maskaddr)
    mask = xingtai(maskaddr)
    image = cv2.imread(imageaddr)
    # 转换为灰度图
    # gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # 将图片二值化
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    # 在二值图上寻找轮廓
    _, contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = []
    for c in range(len(contours)):
        areas.append(cv2.contourArea(contours[c]))
    max_id = areas.index(max(areas))
    x, y, w, h = cv2.boundingRect(contours[max_id])
    print(x, y, w, h)
    res = cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 1)
    cv2.imwrite('/home/bell-chuangxingang/Data/res/'+maskaddr.split('/')[-1],res)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.jpg', '.png'])

# 预测
def make_twozhi_image():
    mydata = dataProcess(512,512)
    imgs_test = mydata.load_test_data()

    myunet = myUnet()
    model = myunet.get_unet()
    model.load_weights('my_unet.hdf5')
    imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
    print("save imgs_mask_test.npy")
    np.save('./results/imgs_mask_test.npy', imgs_mask_test)
    print("array to image")
    imgs = np.load('./results/imgs_mask_test.npy')
    imgs_name = sorted(glob.glob("./raw1/test" + "/*." + "jpg"))
    for i in range(imgs.shape[0]):
        img = imgs[i]
        imgname = imgs_name[i]
        midname = imgname[imgname.rindex("/") + 1:]
        img_order = midname[:-4]
        img = array_to_img(img)
        img.save("./results/%s.jpg" % (img_order))
if __name__ == "__main__":
    make_twozhi_image()

# #########################################################
# dir1 = "/home/bell-chuangxingang/Data/mask"
# dir2 = "/home/bell-chuangxingang/Data/image"
# image_filenames = [os.path.join(dir1,x) for x in os.listdir(dir1) if is_image_file(x)]
# for image_filename in image_filenames:
#     rectangle(image_filename,dir2+'/'+image_filename.split('/')[-1])
