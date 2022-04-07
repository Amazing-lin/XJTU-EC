#coding=utf-8
import keras
import keras.backend.tensorflow_backend as K
from keras.models import load_model
from skimage import io, transform
import tensorflow as tf
from sklearn import preprocessing
import random
import numpy as np
import os
import shutil

config = tf.ConfigProto()
config.gpu_options.allow_growth = True      # TensorFlow按需分配显存
config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 指定显存分配比例
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))





def readlist():
    dir = "/home/bell-chuangxingang/Data/HIE/数据集/"
    doc = open("all_files.txt", 'w')
    for root, dirs, files in os.walk(dir):
        for file in files:
            # print(file)
            if file.split('.')[-1] in ['jpg', 'JPG']:
                print(os.path.join(root, file), file=doc)
    doc.close()


def read_image(imagePath, width=600, height=600, normalization = True):
    img = io.imread(imagePath.split('\n')[0])
    if normalization == True:
        imageData = transform.resize(img,(width, height, 3))
        imageData = np.transpose(imageData,(2,0,1))
        imageData[0] = preprocessing.scale(imageData[0])
        imageData[1] = preprocessing.scale(imageData[1])
        imageData[2] = preprocessing.scale(imageData[2])
        imageData = np.transpose(imageData,(1,2,0))
        # imageData = transform.resize(img,(width, height,3))
    else:
        imageData = transform.resize(img,(width, height,3))
    return imageData

def fdss(path):
    lists = []
    with open(path) as f:
        line = f.readline()
        while line:
            # print(line)
            lists.append(line)
            line = f.readline()
    f.close()
    return np.array(lists)

def batch_fill(all_data, batch_size, shuffle):
    arr=range(len(all_data))
    if shuffle:
        indices = np.arange(len(all_data))
        random.shuffle(indices)

    for start_idx in range(0, len(all_data) - batch_size + 1, batch_size):
        data = []
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = arr[slice(start_idx, start_idx + batch_size)]

        for di in excerpt:
            # tmp_data = all_data[di]

            data.append(all_data[di])

        yield np.array(data)


# 预测
def pre_unknow(test_data_path,batch_size,filepath):
    test_data = []
    for path in test_data_path:
        test_data.append(read_image(path, 224, 224, True))
    model = load_model(filepath)
    all_y_pred = []
    for test_data_batch in batch_fill(np.array(test_data),32,False):
        y_pred = model.predict(test_data_batch, batch_size)
        for y_p in y_pred:
            all_y_pred.append(np.where(y_p == max(y_p))[0][0])
    return all_y_pred

#拷贝图片到两个文件夹
def copy_img(list,all_y_pred):
    for i in range(len(all_y_pred)):
        if all_y_pred[i] == 1 or all_y_pred[i] == 2:
            shutil.copy(list[i], Targetfile_path)
        else:
            shutil.copy(list[i], Targetfile_path)






def main():
    list = fdss("/home/bell-chuangxingang/Data/HIE/DL4ETI-realcode/CreateRandomList/all_files.txt")
    all_y_pred = pre_unknow(list,32,"/home/bell-chuangxingang/Data/HIE/DL4ETI-realcode/ContrastExperimentCNN/tmpvgg/0-weights.03-0.8428-0.3193-0.8750.h5")

main()