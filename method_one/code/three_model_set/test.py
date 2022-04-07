import matplotlib.pyplot as plt
import model
from input_data import get_files
import numpy as np
import tensorflow as tf
import os
from PIL import Image
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# 与保证cpu显存够用
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


yang = []
label_yang = []
ying = []
label_ying = []


# 获取一张图片
def get_one_image(train):
    # 输入参数：train,训练图片的路径
    # 返回参数：image，从训练图片中随机抽取一张图片
    n = len(train)
    ind = np.random.randint(0, n)
    img_dir = train[ind]  # 随机选择测试的图片

    img = Image.open(img_dir)
    plt.imshow(img)
    plt.show()
    image = np.array(img)
    return image


# 测试图片
def evaluate_one_image(image_array):
    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 2
        sumyang = 0
        sumying = 0

        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 64, 64, 3])

        logit = model.inference(image, BATCH_SIZE, N_CLASSES)

        # logit = tf.nn.softmax(logit)

        x = tf.placeholder(tf.float32, shape=[64, 64, 3])

        # you need to change the directories to yours.
        logs_train_dir = '/home/bell-156/桌面/save'

        saver = tf.train.Saver()

        with tf.Session() as sess:

            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')

            prediction = sess.run(logit, feed_dict={x: image_array})
            # print(prediction)
            max_index = np.argmax(prediction)
            if max_index == 0:
                sumyang = sumyang + 1
                result = ('这是yang的可能性为： %.6f' % prediction[:, 0])
            elif max_index == 1:
                sumying = sumying + 1

                result = ('这是ying的可能性为： %.6f' % prediction[:, 1])
            return result ,sumyang


def get_filess(file_dir):
        for file in os.listdir(file_dir + '/yang'):
            yang.append(file_dir + '/yang' + '/' + file)
            # label_yang.append(0)
        for file in os.listdir(file_dir + '/ying'):
            ying.append(file_dir + '/ying' + '/' + file)
            # label_ying.append(1)

            # step2：对生成的图片路径和标签List做打乱处理
        # image_list = np.hstack((roses, tulips, dandelion, sunflowers))
        return ying
# ------------------------------------------------------------------------

if __name__ == '__main__':
    list =  get_filess('/home/bell-156/桌面/inceptionV3/data_prepare/pic/validation')
    sum = 0
    for i in range(len(list)):
        print(list[i])
        img = Image.open(list[i])
        # plt.imshow(img)
        # plt.show()
        imag = img.resize([64, 64])
        image = np.array(imag)
        result ,sumyang = evaluate_one_image(image)
        sum = sum + sumyang
        print(result)
        print('阳性总数为:  %d '  % i,'分对总数为： %d ' % sum )


