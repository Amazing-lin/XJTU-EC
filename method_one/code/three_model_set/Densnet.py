import json
import math
from keras.models import load_model
import os
import cv2
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

imgsize = 256
benign_train = np.array(Dataset_loader('/home/bell-chuangxingang/桌面/project/classification/en_data/train/ying', imgsize))#良
malign_train = np.array(Dataset_loader('/home/bell-chuangxingang/桌面/project/classification/en_data/train/yang', imgsize))#恶
benign_test = np.array(Dataset_loader('/home/bell-chuangxingang/桌面/project/classification/en_data/test/ying', imgsize))
malign_test = np.array(Dataset_loader('/home/bell-chuangxingang/桌面/project/classification/en_data/test/yang', imgsize))

###################################################################


benign_train_label = np.zeros(len(benign_train))
malign_train_label = np.ones(len(malign_train))
benign_test_label = np.zeros(len(benign_test))
malign_test_label = np.ones(len(malign_test))

X_train = np.concatenate((benign_train, malign_train), axis=0)
Y_train = np.concatenate((benign_train_label, malign_train_label), axis=0)
X_test = np.concatenate((benign_test, malign_test), axis=0)
Y_test = np.concatenate((benign_test_label, malign_test_label), axis=0)

s = np.arange(X_train.shape[0])
np.random.shuffle(s)
X_train = X_train[s]
Y_train = Y_train[s]

s = np.arange(X_test.shape[0])
np.random.shuffle(s)
X_test = X_test[s]
Y_test = Y_test[s]

Y_train = to_categorical(Y_train, num_classes=2)
Y_test = to_categorical(Y_test, num_classes=2)

x_train, x_val, y_train, y_val = train_test_split(
    X_train, Y_train,
    test_size=0.2,
    random_state=11
)

BATCH_SIZE = 8

train_generator = ImageDataGenerator(
    zoom_range=2,  # 设置范围为随机缩放
    rotation_range=90,
    horizontal_flip=True,  # 随机翻转图片
    vertical_flip=True,  # 随机翻转图片
)
# ############################################################
# DenseNet
def build_model(backbone, lr=1e-4):
    model = Sequential()
    model.add(backbone)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(2, activation='softmax'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=lr),
        metrics=['accuracy']
    )
    return model


resnet = DenseNet201(
    weights='imagenet',
    include_top=False,
    input_shape=(imgsize, imgsize, 3)
)

model = build_model(resnet, lr=1e-4)
model.summary()
# ##############################################################################################
##VGG16
# def build_model(backbone, lr=1e-4):
#     model = Sequential()
#     model.add(backbone)
#     model.add(layers.GlobalAveragePooling2D())
#     model.add(layers.Dropout(0.5))
#     model.add(layers.BatchNormalization())
#     model.add(layers.Dense(2, activation='softmax'))
#
#     model.compile(
#         loss='binary_crossentropy',
#         optimizer=Adam(lr=lr),
#         metrics=['accuracy']
#     )
#     return model
#
# vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=(imgsize, imgsize, 3))
# model = build_model(vgg16, lr=1e-4)
# model.summary()

###############################################################################################
##ResNet50
# def build_model(backbone, lr=1e-4):
#     model = Sequential()
#     model.add(backbone)
#     model.add(layers.GlobalAveragePooling2D())
#     model.add(layers.Dropout(0.5))
#     model.add(layers.BatchNormalization())
#     model.add(layers.Dense(2, activation='softmax'))
#
#     model.compile(
#         loss='binary_crossentropy',
#         optimizer=Adam(lr=lr),
#         metrics=['accuracy']
#     )
#     return model
#
# ResNet50 = ResNet50(include_top=False, weights='imagenet', input_shape=(imgsize, imgsize, 3))
# model = build_model(ResNet50, lr=1e-4)
# model.summary()

#####################################################################################################
# def build_model(backbone, lr=1e-4):
#     for layer in backbone.layers:
#             layer.trainable = False
#     # model = Sequential()
#     # model.add(backbone)
#     # model.add(layers.GlobalAveragePooling2D())
#     # model.add(layers.Dropout(0.5))
#     # model.add(layers.BatchNormalization())
#     # model.add(layers.Dense(256, activation='relu'))
#     # # model.add(layers.GlobalAveragePooling2D())
#     # model.add(layers.Dropout(0.5))
#     # model.add(layers.BatchNormalization())
#     # model.add(layers.Dense(256, activation='relu'))
#     # # model.add(layers.GlobalAveragePooling2D())
#     # model.add(layers.Dropout(0.5))
#     # model.add(layers.BatchNormalization())
#     # model.add(layers.Dense(2, activation='softmax'))
#
#     model = Sequential()
#     model.add(backbone)
#     model.add(layers.GlobalAveragePooling2D())
#     model.add(layers.Dropout(0.5))
#     model.add(layers.BatchNormalization())
#     model.add(layers.Dense(2, activation='softmax'))
#     model.compile(
#         loss='binary_crossentropy',
#         optimizer=Adam(lr=lr),
#         metrics=['accuracy']
#     )
#     return model
# inceptionV3 = InceptionV3(include_top=False, weights='imagenet', input_shape=(imgsize, imgsize, 3))
# model = build_model(inceptionV3, lr=1e-4)
# model.summary()
# # ################################################################################################################
filepath = "/home/bell-chuangxingang/桌面/person_exp/save/Densenet_person_weights.hdf5"
# learn_control = ReduceLROnPlateau(monitor='val_acc', patience=5,
#                                   verbose=1, factor=0.2, min_lr=1e-7)
#
#
# checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#
# history = model.fit_generator(
#     train_generator.flow(x_train, y_train, batch_size=BATCH_SIZE),
#     steps_per_epoch=x_train.shape[0] / BATCH_SIZE,
#     epochs=100,
#     validation_data=(x_val, y_val),
#     callbacks=[learn_control, checkpoint]
# )
# model.save(filepath, overwrite=True)



model = load_model(filepath)

###############################################################################
# with open('history.json', 'w') as f:
#     json.dump(str(history.history), f)
# history_df = pd.DataFrame(history.history)
# history_df[['loss', 'val_loss']].plot()
# plt.legend()
# plt.show()
# history_df = pd.DataFrame(history.history)
# history_df[['accuracy', 'val_accuracy']].plot()
# plt.legend()
# plt.show()
# model.load_weights(filepath)
# Y_val_pred = model.predict(x_val)
# accuracy_score(np.argmax(y_val, axis=1), np.argmax(Y_val_pred, axis=1))
# print('accuracy_score :' , accuracy_score)

###############################################################################
Y_pred = model.predict(X_test)
tta_steps = 10
predictions = []

for i in tqdm(range(tta_steps)):
    preds = model.predict_generator(train_generator.flow(X_test, batch_size=BATCH_SIZE, shuffle=False),
                                    steps=len(X_test) / BATCH_SIZE)

    predictions.append(preds)
    gc.collect()

Y_pred_tta = np.mean(predictions, axis=0)
############################################################################################

# classification_report(np.argmax(Y_test, axis=1), np.argmax(Y_pred_tta, axis=1))

# def plot_confusion_matrix(cm, classes,
#                           normalize=True,
#                           title='Confusion matrix',
#                           cmap=plt.cm.RdPu):
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#
#     print(cm)
#
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes)    ##, rotation=55
#     plt.yticks(tick_marks, classes)
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.tight_layout()
#     plt.show()
#
# cm = confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(Y_pred, axis=1))
#
#
# cm_plot_label = ['benign', 'malignant']
# plot_confusion_matrix(cm, cm_plot_label, title='Confusion Metrix for Endometrial Cancer')
###############################################################################################
# classification_report(np.argmax(Y_test, axis=1), np.argmax(Y_pred_tta, axis=1))
#
# # roc and auc
#
#
# roc_log = roc_auc_score(np.argmax(Y_test, axis=1), np.argmax(Y_pred_tta, axis=1))
# false_positive_rate, true_positive_rate, threshold = roc_curve(np.argmax(Y_test, axis=1), np.argmax(Y_pred_tta, axis=1))
# area_under_curve = auc(false_positive_rate, true_positive_rate)
#
# plt.plot([0, 1], [0, 1], 'r--')
# plt.plot(false_positive_rate, true_positive_rate, label='AUC = {:.3f}'.format(area_under_curve))
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.title('ROC curve')
# plt.legend(loc='best')
# plt.show()
# plt.savefig(ROC_PLOT_FILE, bbox_inches='tight')
# plt.close()
#################################################################################################
# #保存假阴和假阳
# i = 0
# mis_ying = []
# mis_yang = []
#
# for i in range(len(Y_test)):
#     if ((np.argmax(Y_test[i]) == 0) and (np.argmax(Y_pred[i]) == 1)):
#         plt.imsave('/home/bell-chuangxingang/桌面/inception/false_yang/'+ str(i) + '.jpg', X_test[i])
#
# i = 0
# for i in range(len(Y_test)):
#     if ((np.argmax(Y_test[i]) == 1) and (np.argmax(Y_pred[i]) == 0)):
#         plt.imsave('/home/bell-chuangxingang/桌面/inception/false_ying/'+ str(i) + '.jpg', X_test[i])
#########################################################################################################
# 测试一个person的阳性或者阴性判断
# Y_pred = model.predict(benign_test)
# ying_count0 = 0
# yang_count1 = 0
# for i in range(86):
#     if (np.argmax(Y_pred[i]) == 0):
#         ying_count0 = ying_count0 +1
#     if (np.argmax(Y_pred[i]) == 1):
#         yang_count1 = yang_count1 +1
# print(ying_count0,yang_count1)

################################################################################################
#随便找出8张图片看结果
# i = 0
# prop_class = []
# mis_class = []
#
# for i in range(len(Y_test)):
#     if (np.argmax(Y_test[i]) == np.argmax(Y_pred_tta[i])):
#         prop_class.append(i)
#     if (len(prop_class) == 8):
#         break
#
# i = 0
# for i in range(len(Y_test)):
#     if (not np.argmax(Y_test[i]) == np.argmax(Y_pred_tta[i])):
#         mis_class.append(i)
#     if (len(mis_class) == 8):
#         break
#
# # # Display first 8 images of benign
# w = 60
# h = 40
# fig = plt.figure(figsize=(18, 10))
# columns = 4
# rows = 2
#
#
# def Transfername(namecode):
#     if namecode == 0:
#         return "Benign"
#     else:
#         return "Malignant"
#
#
# for i in range(len(prop_class)):
#     ax = fig.add_subplot(rows, columns, i + 1)
#     ax.set_title("Predicted result:" + Transfername(np.argmax(Y_pred_tta[prop_class[i]]))
#                  + "\n" + "Actual result: " + Transfername(np.argmax(Y_test[prop_class[i]])))
#     plt.imshow(X_test[prop_class[i]], interpolation='nearest')
# plt.show()