import os
import numpy as np
import tensorflow as tf
import random
# import seaborn as sns
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop, Adam, SGD
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.model_selection import train_test_split
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# 与保证cpu显存够用
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def read_and_process_image(data_dir, width=32, height=32, channels=3, preprocess=False):
    train_classes = [data_dir + i for i in os.listdir(data_dir)]
    train_images = []
    for train_class in train_classes:
        train_images = train_images + [train_class + "/" + i for i in os.listdir(train_class)]

    random.shuffle(train_images)

    def read_image(file_path, preprocess):
        img = image.load_img(file_path, target_size=(height, width))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        if preprocess:
            x = preprocess_input(x)
        return x

    def prep_data(images, preprocess):
        count = len(images)
        data = np.ndarray((count, height, width, channels), dtype=np.float32)

        for i, image_file in enumerate(images):
            image = read_image(image_file, preprocess)
            data[i] = image

        return data


    def read_labels(file_path):
            global label
            labels = []
            for i in file_path:
                if 'ying' in i:
                    label = 0
                elif 'yang' in i:
                    label = 1
                labels.append(label)

            return labels

    X = prep_data(train_images, preprocess)
    labels = read_labels(train_images)

    assert X.shape[0] == len(labels)

    print("Train shape: {}".format(X.shape))

    return X, labels



# 读取训练集图片
WIDTH = 128
HEIGHT = 128
CHANNELS = 3
X, y = read_and_process_image('/home/bell-156/桌面/project/classification/unen_data/train/',width=WIDTH, height=HEIGHT, channels=CHANNELS)
test_X, test_y = read_and_process_image('/home/bell-156/桌面/project/classification/unen_data/test/',width=WIDTH, height=HEIGHT, channels=CHANNELS)


train_y = np_utils.to_categorical(y)
test_y = np_utils.to_categorical(test_y)


#显示图片
# def show_picture(X, idx):
#     plt.figure(figsize=(10, 5), frameon=True)
#     img = X[idx, :, :, ::-1]
#     img = img / 255
#     plt.imshow(img)
#     plt.show()
#
#
# for idx in range(0, 3):
#     show_picture(X, idx)


def vgg16_model(input_shape=(HEIGHT, WIDTH, CHANNELS)):
    vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)

    for layer in vgg16.layers:
        layer.trainable = False
    last = vgg16.output
    # 后面加入自己的模型
    x = Flatten()(last)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(2, activation='softmax')(x)

    model = Model(inputs=vgg16.input, outputs=x)

    return model

model_vgg16 = vgg16_model()
model_vgg16.summary()
model_vgg16.compile(loss='categorical_crossentropy',optimizer = Adam(0.0001), metrics = ['accuracy'])

history = model_vgg16.fit(X,train_y, validation_split=0.2,epochs=150,batch_size=100,verbose=True)
y_pred = model_vgg16.predict(test_X, batch_size=1)

max_index = np.argmax(y_pred,axis=1)
labels = np.argmax(test_y,axis=1)
ying_correct = 0.0
ying_error = 0.0
yang_correct = 0.0
yang_error = 0.0
for i in range(len(labels)):
    if labels[i] == 0:
        if max_index[i] ==0 :
            ying_correct += 1
        else:
            ying_error += 1
    else :
        if max_index[i] == 0:
            yang_error += 1
        else:
            yang_correct += 1

print('正确率： ',(ying_correct+yang_correct)/(ying_correct+ying_error+yang_error+yang_correct),
      '假阴率： ',yang_error/(ying_correct+ying_error+yang_error+yang_correct),
      '假阳率： ',ying_error/(ying_correct+ying_error+yang_error+yang_correct))
print('ying_correct: ',ying_correct,'  yang_correct: ',yang_correct,'  ying_error: ',ying_error,'  yang_error: ',yang_error)


# print(y_pred)
# loss , score = model_vgg16.evaluate(test_X, test_y, verbose=0)
# print('loss: ',loss,'score: ',score)
# model_vgg16.save('/home/bell-156/桌面/project/classification/save/vgg16_model_weights.h5', overwrite=True)
# print("Large CNN Error: %.2f%%" %(100-score[1]*100))
print(history.history.keys())

plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])
plt.show()