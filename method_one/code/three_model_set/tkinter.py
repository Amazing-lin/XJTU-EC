from keras.models import load_model
import os
import cv2
from PIL import Image
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# 与保证cpu显存够用
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def Dataset_loader(DIR, RESIZE, sigmaX=10):
    IMG = []
    read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))
    for IMAGE_NAME in tqdm(os.listdir(DIR)):
        PATH = os.path.join(DIR, IMAGE_NAME)
        _, ftype = os.path.splitext(PATH)
        if ftype == ".jpg":
            img = read(PATH)

            img = cv2.resize(img, (RESIZE, RESIZE))

            IMG.append(np.array(img))
    return IMG


def random_cut():
    imgsize = 256
    benign_test = np.array(Dataset_loader('/home/bell-chuangxingang/桌面/person_exp/阴/张久香77/512', imgsize))

    BATCH_SIZE = 8

    train_generator = ImageDataGenerator(
        zoom_range=2,  # 设置范围为随机缩放
        rotation_range=90,
        horizontal_flip=True,  # 随机翻转图片
        vertical_flip=True,  # 随机翻转图片
    )
    filepath = "/home/bell-chuangxingang/桌面/person_exp/save/Densenet_person_weights.hdf5"
    model = load_model(filepath)
    # 测试一个person的阳性或者阴性判断
    Y_pred = model.predict(benign_test)
    ying_count0 = 0
    yang_count1 = 0
    for i in range(86):
        if (np.argmax(Y_pred[i]) == 0):
            ying_count0 = ying_count0 +1
        if (np.argmax(Y_pred[i]) == 1):
            yang_count1 = yang_count1 +1
    print(ying_count0,yang_count1)
