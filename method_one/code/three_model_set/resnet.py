# -*- coding: utf-8 -*-
import os
from keras.utils import plot_model
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.models import Model, load_model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# 与保证cpu显存够用
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

class PowerTransferMode:
    # 数据准备
    def DataGen(self, dir_path, img_row, img_col, batch_size, is_train):
        if is_train:
            datagen = ImageDataGenerator(rescale=1. / 255,
                                         zoom_range=0.25, rotation_range=15.,
                                         channel_shift_range=25., width_shift_range=0.02, height_shift_range=0.02,
                                         horizontal_flip=True, fill_mode='constant')
        else:
            datagen = ImageDataGenerator(rescale=1. / 255)

        generator = datagen.flow_from_directory(
            dir_path, target_size=(img_row, img_col),
            batch_size=batch_size,
            # class_mode='binary',
            class_mode='categorical',
            shuffle=is_train)

        return generator

    # ResNet模型
    def ResNet50_model(self, lr=0.005, decay=1e-6, momentum=0.9, nb_classes=2, img_rows=197, img_cols=197, RGB=True,
                       is_plot_model=False):
        color = 3 if RGB else 1
        base_model = ResNet50(weights='imagenet', include_top=False, pooling=None,
                              input_shape=(img_rows, img_cols, color),
                              classes=nb_classes)

        # 冻结base_model所有层，这样就可以正确获得bottleneck特征
        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        # 添加自己的全链接分类层
        x = Flatten()(x)
        # x = GlobalAveragePooling2D()(x)
        # x = Dense(1024, activation='relu')(x)
        predictions = Dense(nb_classes, activation='softmax')(x)

        # 训练模型
        model = Model(inputs=base_model.input, outputs=predictions)
        sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        # 绘制模型
        if is_plot_model:
            plot_model(model, to_file='resnet50_model.png', show_shapes=True)

        return model

    # VGG模型
    def VGG19_model(self, lr=0.005, decay=1e-6, momentum=0.9, nb_classes=18, img_rows=197, img_cols=197, RGB=True,
                    is_plot_model=False):
        color = 3 if RGB else 1
        base_model = VGG19(weights='imagenet', include_top=False, pooling=None, input_shape=(img_rows, img_cols, color),
                           classes=nb_classes)

        # 冻结base_model所有层，这样就可以正确获得bottleneck特征
        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        # 添加自己的全链接分类层
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        predictions = Dense(nb_classes, activation='softmax')(x)

        # 训练模型
        model = Model(inputs=base_model.input, outputs=predictions)
        sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        # 绘图
        if is_plot_model:
            plot_model(model, to_file='vgg19_model.png', show_shapes=True)

        return model

    # InceptionV3模型
    def InceptionV3_model(self, lr=0.005, decay=1e-6, momentum=0.9, nb_classes=18, img_rows=197, img_cols=197, RGB=True,
                          is_plot_model=False):
        color = 3 if RGB else 1
        base_model = InceptionV3(weights='imagenet', include_top=False, pooling=None,
                                 input_shape=(img_rows, img_cols, color),
                                 classes=nb_classes)

        # 冻结base_model所有层，这样就可以正确获得bottleneck特征
        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        # 添加自己的全链接分类层
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)

        # x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(nb_classes, activation='softmax')(x)

        # 训练模型
        model = Model(inputs=base_model.input, outputs=predictions)
        sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        # 绘图
        if is_plot_model:
            plot_model(model, to_file='inception_v3_model.png', show_shapes=True)

        return model

    # 训练模型
    def train_model(self, model, epochs, train_generator, steps_per_epoch, validation_generator, validation_steps,
                    model_url, is_load_model=False):
        # 载入模型
        if is_load_model and os.path.exists(model_url):
            model = load_model(model_url)

        history_ft = model.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_steps)
        # 模型保存
        model.save(model_url, overwrite=True)
        return history_ft

    # 画图
    def plot_training(self, history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(acc))
        plt.plot(epochs, acc, 'b-')
        plt.plot(epochs, val_acc, 'r')
        plt.title('Training and validation accuracy')
        plt.figure()
        plt.plot(epochs, loss, 'b-')
        plt.plot(epochs, val_loss, 'r-')
        plt.title('Training and validation loss')
        plt.show()


if __name__ == '__main__':

    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    image_size = 128
    batch_size = 100
    epo = 100
    transfer = PowerTransferMode()
    num_train = 3037
    num_test = 536
    # 得到数据
    train_generator = transfer.DataGen('/home/bell-156/桌面/project/classification/unen_data/train/', image_size, image_size,
                                       batch_size, True)
    validation_generator = transfer.DataGen('/home/bell-156/桌面/project/classification/unen_data/test/', image_size, image_size,
                                            batch_size, False)

    # VGG19
    # model = transfer.VGG19_model(nb_classes=2, img_rows=image_size, img_cols=image_size, is_plot_model=False)
    # history_ft1 = transfer.train_model(model, epo, train_generator, 300, validation_generator,
    #                                    100, 'vgg19_model_weights.h5', is_load_model=False)

    # ResNet50
    model = transfer.ResNet50_model(nb_classes=2, img_rows=image_size, img_cols=image_size, is_plot_model=False)
    history_ft = transfer.train_model(model, 100, train_generator, 60, validation_generator, 60, 'resnet50_model_weights.h5', is_load_model=False)

    # # InceptionV3
    # model = transfer.InceptionV3_model(nb_classes=2, img_rows=image_size, img_cols=image_size, is_plot_model=False)
    # # 分多次进行训练，每训练100次，保存一下模型
    # # for _ in range(10):
    # history_ft2 = transfer.train_model(model, 100, train_generator, num_train // batch_size, validation_generator,
    #                                        num_test // batch_size, 'inception_v3_model_weights.h5',
    #                                        is_load_model=False)
    # # # 训练的acc_loss图
    transfer.plot_training(history_ft2)
    # transfer.plot_training(history_ft2)

