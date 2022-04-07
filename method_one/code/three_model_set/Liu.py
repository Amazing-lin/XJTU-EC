import tensorflow as tf
import pickle
import os
import numpy as np
import json
import math
from keras.models import load_model
import os
import cv2

# import seaborn as sbn
import pylab as plt
# import theano
from keras.models import Sequential
from keras.layers import Dense, Activation

from keras.models import Model
from PIL import Image
import numpy as np
from keras import layers
from keras.applications import DenseNet201
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
import scipy
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import roc_curve
import tensorflow as tf
from keras import backend as K
import gc

from keras.applications.resnet50 import ResNet50
from functools import partial
from sklearn import metrics
from collections import Counter
import json
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report
import itertools
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# 与保证cpu显存够用
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# # 模型路径
# model_path = '/home/bell-chuangxingang/桌面/project/classification/save/vgg16_model_weights.h5'
# # 输入的图片路径
# input_img_path = '/home/bell-chuangxingang/桌面/LIu'
# # 输出向量保存路径
# output_folder = '/home/bell-chuangxingang/桌面/LIu/out_table'
#
# # 加一个判断保证文件夹存在
# if not os.path.exists(output_folder):
#     os.mkdir(output_folder)
#
#
# # 制作数据和标签
# def parse_img(input_img_path):
#     img_datas = []
#     img_labels = []
#     # 获得分类名和下标值 比如“0 liang,1 er等”
#     for i, j in enumerate(os.listdir(input_img_path)):
#     	# 获取每一张图片的名称
#         for evry in os.listdir(os.path.join(input_img_path,j)):
#         	# 获取每一张图片的全路径
#             evry_img_path = os.path.join(input_img_path,j,evry)
#             # 将每一张图片读取为像素矩阵
#             img_data = tf.gfile.FastGFile(evry_img_path,'rb').read()
#             # 将数据逐个添加到保存数据的列表里
#             img_datas.append(img_data)
#             # 同时保存对应的标签
#             img_labels += [i]
#     return np.array(img_datas),np.array(img_labels)
#
#
# # 调用数据处理函数 获得处理好的数据
# img_datas,img_labels = parse_img(input_img_path)
# # 加载模型
# model = load_model(model_path)
#
# #取某一层的输出为输出新建为model，采用函数模型
# dense1_layer_model = Model(inputs=model.input, outputs=model.get_layer('dense_1').output)
#
# batch_size = 8
# num_batches = int(len(img_datas) / batch_size)
#
# # 开启会话
# # config = tf.ConfigProto()
# # config.gpu_options.allow_growth = True
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     # # 通过名称获取张量
#     # tensor = sess.graph.get_tensor_by_name('pool_3/_reshape:0')
#     # 每8张图片为一个批次循环处理
#     for i in range(num_batches):
#         batch_img_data = img_datas[i*batch_size:(i+1)*batch_size]
#         batch_img_labels = img_labels[i*batch_size:(i+1)*batch_size]
#         # 制作一个存放特征向量的空列表
#         feature_v = []
#         # 将每一张图片转为的像素矩阵作为数据传入到tensor中做前向计算，得到2048的特征向量
#         for j in batch_img_data:
#             j_vector = dense1_output = dense1_layer_model.predict(batch_img_data)
#             # 逐个添加
#             feature_v.append(j_vector)
#         feature_v = np.vstack(feature_v)
#         # 保存特征向量的全路径
#         save_path = os.path.join(output_folder,'data_%d.pickle'%i)
#         # 打开这个全路径文件
#         with tf.gfile.FastGFile(save_path,'w') as f:
#         	# 写入这个批次的向量，便于后续提取
#             pickle.dump((feature_v,batch_img_labels),f)
#         print(save_path,'is_ok!')
# 设置环境
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# 模型路径
model_path = '/home/bell-chuangxingang/桌面/未命名文件夹/tensorflow_inception_graph.pb'
# 输入的图片路径
input_img_path = '/home/bell-chuangxingang/桌面/LIu'
# 输出向量保存路径
output_folder = '/home/bell-chuangxingang/桌面/LIu/out_table'

# 加一个判断保证文件夹存在
if not os.path.exists(output_folder):
    os.mkdir(output_folder)


# 制作数据和标签
def parse_img(input_img_path):
    img_datas = []
    img_labels = []
    # 获得分类名和下标值 比如“0 bandeng,1 zhuozi等”
    for i, j in enumerate(os.listdir(input_img_path)):
    	# 获取每一张图片的名称
        for evry in os.listdir(os.path.join(input_img_path,j)):
        	# 获取每一张图片的全路径
            evry_img_path = os.path.join(input_img_path,j,evry)
            # 将每一张图片读取为像素矩阵
            img_data = tf.gfile.FastGFile(evry_img_path,'rb').read()
            # 将数据逐个添加到保存数据的列表里
            img_datas.append(img_data)
            # 同时保存对应的标签
            img_labels += [i]
    return np.array(img_datas),np.array(img_labels)


# 导入计算图
def load_inception_v3(model_path):
    # 以二进制形式打开训练好的模型图
    with tf.gfile.FastGFile(model_path,'rb') as f:
        # 创建一张新图
        graph_def = tf.GraphDef()
        # 将打开的模型图写入到新图中
        graph_def.ParseFromString(f.read())
        # 将这张新图设为默认图
        _ = tf.import_graph_def(graph_def=graph_def,name='')


# 调用数据处理函数 获得处理好的数据
img_datas,img_labels = parse_img(input_img_path)
# 加载inception_v3模型
load_inception_v3(model_path)

batch_size = 200
num_batches = int(len(img_datas) / batch_size)

# 开启会话
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 通过名称获取张量
    tensor = sess.graph.get_tensor_by_name('pool_3/_reshape:0')
    # tensorname = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
    # for tensor_name in tensorname:
    #     print(tensor_name,'\n')
    # 每500张图片为一个批次循环处理
    for i in range(num_batches):
        batch_img_data = img_datas[i*batch_size:(i+1)*batch_size]
        batch_img_labels = img_labels[i*batch_size:(i+1)*batch_size]
        # 制作一个存放特征向量的空列表
        feature_v = []
        # 将每一张图片转为的像素矩阵作为数据传入到tensor中做前向计算，得到2048的特征向量
        for j in batch_img_data:
            j_vector = sess.run(tensor,feed_dict={'DecodeJpeg/contents:0':j})
            # 逐个添加
            feature_v.append(j_vector)
        feature_v = np.vstack(feature_v)
        # 保存特征向量的全路径
        save_path = os.path.join(output_folder,'data_%d.pickle'%i)
        # 打开这个全路径文件
        with tf.gfile.FastGFile(save_path,'w') as f:
        	# 写入这个批次的向量，便于后续提取
            pickle.dump((feature_v,batch_img_labels),f)
        print(save_path,'is_ok!')
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     # 通过名称获取张量
#     tensor = sess.graph.get_tensor_by_name('pool_3:0')
#     # tensorname = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
#     # for tensor_name in tensorname:
#     #     print(tensor_name,'\n')
#     # 每500张图片为一个批次循环处理
#     feature_v = []
#     for j in img_datas:
#         j_vector = sess.run(tensor,feed_dict={'DecodeJpeg/contents:0':j})
#             # 逐个添加
#         feature_v.append(j_vector)
#     feature_v = np.vstack(feature_v)
#         # 保存特征向量的全路径
#     data = pd.DataFrame(feature_v)
#     data.to_csv('/home/bell-chuangxingang/桌面/LIu/out_table/data1.csv')
#     print('is_ok!')






