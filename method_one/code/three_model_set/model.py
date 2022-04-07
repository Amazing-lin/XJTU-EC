# import tensorflow as tf
#
# #定义函数infence，定义CNN网络结构
# #卷积层1
# def inference(images, batch_size, n_classes):
#     ########################layer1##################################
#     with tf.variable_scope('conv1_1') as scope:
#         weights = tf.Variable(tf.truncated_normal(shape=[3,3,3,64],stddev=1.0,dtype=tf.float32),
#                              name ='weights',dtype=tf.float32)
#         biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[64]),
#                              name='biases', dtype=tf.float32)
#         conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')
#         pre_activation = tf.nn.bias_add(conv, biases)
#         conv1_1 = tf.nn.relu(pre_activation, name=scope.name)
#
#     with tf.variable_scope('conv1_2') as scope:
#         weights = tf.Variable(tf.truncated_normal(shape=[3,3,64,64],stddev=1.0,dtype=tf.float32),
#                              name = 'weights',dtype=tf.float32)
#         biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[64]),
#                              name='biases', dtype=tf.float32)
#         conv = tf.nn.conv2d(conv1_1, weights, strides=[1, 1, 1, 1], padding='SAME')
#         pre_activation = tf.nn.bias_add(conv, biases)
#         conv1_2 = tf.nn.relu(pre_activation, name=scope.name)
#
#     with tf.variable_scope('pooling1') as scope:
#         pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pooling1')
#         pooling1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
#
#     ########################layer2##################################
#     with tf.variable_scope('conv2_1') as scope:
#         weights = tf.Variable(tf.truncated_normal(shape=[3,3,64,128],stddev=1.0,dtype=tf.float32),
#                              name ='weights',dtype=tf.float32)
#         biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[128]),
#                              name='biases', dtype=tf.float32)
#         conv = tf.nn.conv2d(pooling1, weights, strides=[1, 1, 1, 1], padding='SAME')
#         pre_activation = tf.nn.bias_add(conv, biases)
#         conv2_1 = tf.nn.relu(pre_activation, name=scope.name)
#
#     with tf.variable_scope('conv2_2') as scope:
#         weights = tf.Variable(tf.truncated_normal(shape=[3,3,128,128],stddev=1.0,dtype=tf.float32),
#                              name = 'weights',dtype=tf.float32)
#         biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[128]),
#                              name='biases', dtype=tf.float32)
#         conv = tf.nn.conv2d(conv2_1, weights, strides=[1, 1, 1, 1], padding='SAME')
#         pre_activation = tf.nn.bias_add(conv, biases)
#         conv2_2 = tf.nn.relu(pre_activation, name=scope.name)
#
#     with tf.variable_scope('pooling2') as scope:
#         pool1 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pooling1')
#         pooling2 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
#
#     ########################layer3##################################
#     with tf.variable_scope('conv3_1') as scope:
#         weights = tf.Variable(tf.truncated_normal(shape=[3,3,128,256],stddev=1.0,dtype=tf.float32),
#                              name ='weights',dtype=tf.float32)
#         biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[256]),
#                              name='biases', dtype=tf.float32)
#         conv = tf.nn.conv2d(pooling2, weights, strides=[1, 1, 1, 1], padding='SAME')
#         pre_activation = tf.nn.bias_add(conv, biases)
#         conv3_1 = tf.nn.relu(pre_activation, name=scope.name)
#
#     with tf.variable_scope('conv3_2') as scope:
#         weights = tf.Variable(tf.truncated_normal(shape=[3,3,256,256],stddev=1.0,dtype=tf.float32),
#                              name = 'weights',dtype=tf.float32)
#         biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[256]),
#                              name='biases', dtype=tf.float32)
#         conv = tf.nn.conv2d(conv3_1, weights, strides=[1, 1, 1, 1], padding='SAME')
#         pre_activation = tf.nn.bias_add(conv, biases)
#         conv3_2 = tf.nn.relu(pre_activation, name=scope.name)
#
#     with tf.variable_scope('conv3_3') as scope:
#         weights = tf.Variable(tf.truncated_normal(shape=[3,3,256,256],stddev=1.0,dtype=tf.float32),
#                              name = 'weights',dtype=tf.float32)
#         biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[256]),
#                              name='biases', dtype=tf.float32)
#         conv = tf.nn.conv2d(conv3_2, weights, strides=[1, 1, 1, 1], padding='SAME')
#         pre_activation = tf.nn.bias_add(conv, biases)
#         conv3_3 = tf.nn.relu(pre_activation, name=scope.name)
#
#     with tf.variable_scope('pooling3') as scope:
#         pool1 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pooling1')
#         pooling3 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
#
#     ########################layer4##################################
#     with tf.variable_scope('conv4_1') as scope:
#         weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 512], stddev=1.0, dtype=tf.float32),
#                               name='weights', dtype=tf.float32)
#         biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[512]),
#                              name='biases', dtype=tf.float32)
#         conv = tf.nn.conv2d(pooling3, weights, strides=[1, 1, 1, 1], padding='SAME')
#         pre_activation = tf.nn.bias_add(conv, biases)
#         conv4_1 = tf.nn.relu(pre_activation, name=scope.name)
#
#     with tf.variable_scope('conv4_2') as scope:
#         weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512], stddev=1.0, dtype=tf.float32),
#                               name='weights', dtype=tf.float32)
#         biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[512]),
#                              name='biases', dtype=tf.float32)
#         conv = tf.nn.conv2d(conv4_1, weights, strides=[1, 1, 1, 1], padding='SAME')
#         pre_activation = tf.nn.bias_add(conv, biases)
#         conv4_2 = tf.nn.relu(pre_activation, name=scope.name)
#
#     with tf.variable_scope('conv4_3') as scope:
#         weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512], stddev=1.0, dtype=tf.float32),
#                               name='weights', dtype=tf.float32)
#         biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[512]),
#                              name='biases', dtype=tf.float32)
#         conv = tf.nn.conv2d(conv4_2, weights, strides=[1, 1, 1, 1], padding='SAME')
#         pre_activation = tf.nn.bias_add(conv, biases)
#         conv4_3 = tf.nn.relu(pre_activation, name=scope.name)
#
#     with tf.variable_scope('pooling4') as scope:
#         pool1 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pooling1')
#         pooling4 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
#
#     ########################layer5##################################
#     with tf.variable_scope('conv5_1') as scope:
#         weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512], stddev=1.0, dtype=tf.float32),
#                               name='weights', dtype=tf.float32)
#         biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[512]),
#                              name='biases', dtype=tf.float32)
#         conv = tf.nn.conv2d(pooling4, weights, strides=[1, 1, 1, 1], padding='SAME')
#         pre_activation = tf.nn.bias_add(conv, biases)
#         conv5_1 = tf.nn.relu(pre_activation, name=scope.name)
#
#     with tf.variable_scope('conv5_2') as scope:
#         weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512], stddev=1.0, dtype=tf.float32),
#                               name='weights', dtype=tf.float32)
#         biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[512]),
#                              name='biases', dtype=tf.float32)
#         conv = tf.nn.conv2d(conv5_1, weights, strides=[1, 1, 1, 1], padding='SAME')
#         pre_activation = tf.nn.bias_add(conv, biases)
#         conv5_2 = tf.nn.relu(pre_activation, name=scope.name)
#
#     with tf.variable_scope('conv5_3') as scope:
#         weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512], stddev=1.0, dtype=tf.float32),
#                               name='weights', dtype=tf.float32)
#         biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[512]),
#                              name='biases', dtype=tf.float32)
#         conv = tf.nn.conv2d(conv5_2, weights, strides=[1, 1, 1, 1], padding='SAME')
#         pre_activation = tf.nn.bias_add(conv, biases)
#         conv5_3 = tf.nn.relu(pre_activation, name=scope.name)
#
#     with tf.variable_scope('pooling5') as scope:
#         pool1 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pooling1')
#         pooling5 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
#
#
#
#     # 全连接层3
#     # 128个神经元，将之前pool层的输出reshape成一行，激活函数relu()
#     with tf.variable_scope('local3') as scope:
#         reshape = tf.reshape(pooling5, shape=[batch_size, -1])
#         dim = reshape.get_shape()[1].value
#         weights = tf.Variable(tf.truncated_normal(shape=[dim, 4096], stddev=0.005, dtype=tf.float32),
#                               name='weights', dtype=tf.float32)
#
#         biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[4096]),
#                              name='biases', dtype=tf.float32)
#
#         local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
#
#     with tf.variable_scope('dropout') as scope:
#         drop_out1 = tf.nn.dropout(local3, 0.6)
#
#     # 全连接层4
#     # 128个神经元，激活函数relu()
#     with tf.variable_scope('local4') as scope:
#         weights = tf.Variable(tf.truncated_normal(shape=[4096, 4096], stddev=0.005, dtype=tf.float32),
#                               name='weights', dtype=tf.float32)
#
#         biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[4096]),
#                              name='biases', dtype=tf.float32)
#
#         local4 = tf.nn.relu(tf.matmul(drop_out1, weights) + biases, name='local4')
#
#     # dropout层
#     with tf.variable_scope('dropout2') as scope:
#         drop_out2 = tf.nn.dropout(local4, 0.6)
#
#     # Softmax回归层
#     # 将前面的FC层输出，做一个线性回归，计算出每一类的得分
#     with tf.variable_scope('softmax_linear') as scope:
#         weights = tf.Variable(tf.truncated_normal(shape=[4096, n_classes], stddev=0.005, dtype=tf.float32),
#                               name='softmax_linear', dtype=tf.float32)
#
#         biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[n_classes]),
#                              name='biases', dtype=tf.float32)
#
#         softmax_linear = tf.nn.softmax(tf.add(tf.matmul(drop_out2, weights), biases, name='softmax_linear'))
#
#     return softmax_linear


# -----------------------------------------------------------------------------

# 载入几个系统库和tensorflow
from datetime import datetime
import math
import time
import tensorflow as tf

# VGG包含很多卷积，函数conv_op创建卷积层并把本层参数存入参数列表
# input_op是输入的tensor
# name是本层名称
# kh卷积核高
# kw卷积核宽
# n_out是卷积核的数量，即输出通道数
# dh步长的高
# dw步长的宽
# p是参数列表
def conv_op(input_op, name, kh, kw, n_out, dh, dw, p):
    # get_shape获得输入tensor的通道数
    # 224*224*3的图片就是最后的3
    n_in = input_op.get_shape()[-1].value
    # name_scope将scope内生成的variable自动命名为name/xxx
    # 用于区分不同卷积层的组件
    with tf.name_scope(name) as scope:
        # 卷积核参数由get_variable函数创建
        # shape即卷积核的高，宽，输入通道数，输出通道数
        kernel = tf.get_variable(scope+"w", shape=[kh,kw,n_in,n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())

        # conv2d对输入的tensor进行卷积处理
        # tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
        # input是输入的图像tensor shape=[batch, in_height, in_width, in_channels]
        # [训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]
        # filter是卷积核tensor shape=[filter_height, filter_width, in_channels, out_channels]
        # [卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]
        # strides卷积时在每一维上的步长，strides[0]=strides[3]=1
        # padding drop or zeropad
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
        # bias使用tf.constant赋值为0
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        # tf.variable再将其转换成可训练的参数
        biases = tf.Variable(bias_init_val, trainable=True, name='b')
        # tf.nn.bias_add将卷积结果和bias相加
        z = tf.nn.bias_add(conv,biases)
        # 使用relu对卷积结果进行非线性处理
        activation = tf.nn.relu(z, name=scope)
        # 把卷积层使用到的参数添加到参数列表p
        p += [kernel, biases]
        return activation

# 定义 创建全连接层 函数fc_op
def fc_op(input_op, name, n_out, p):
    # 同样获得输入图片tensor的通道数
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        # 同样使用get_variable创建全连接层的参数，只不过参数纬度只有两个:输入和输出通道数
        # 参数初始化方法使用xavier_initializer
        kernel = tf.get_variable(scope+"w", shape=[n_in, n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        # bias利用constant函数初始化为较小的值0.1,而不是0
        biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), name='b')
        # 这里使用tf.nn.relu_layer，对输入变量input_op和kernel做矩阵乘法并加上biases
        # 再做relu非线性变换得到activation
        activation = tf.nn.relu_layer(input_op, kernel, biases, name= scope)
        # 把全连接层使用到的参数添加到参数列表p
        p += [kernel, biases]
        return activation

# 定义 创建最大池化层 函数mpool_op
def mpool_op(input_op, name, kh, kw, dh, dw):
    # 直接使用tf.nn.max_pool,输入为图片tensor，池化尺寸为kh*kw,步长为dh*dw，padding为same
    return tf.nn.max_pool(input_op,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding='SAME',
                          name=name)

# 完成了 卷积层，全连接层，pooling层 的创建函数
# 下面开始创建VGG16的网络结构


# inference_op是创建网络结构的函数
# input_op是输入的图像tensor shape=[batch, in_height, in_width, in_channels]
# keep_prob是控制dropout比率的一个placeholder
def inference_op(input_op,keep_prob):
    # 初始化参数p列表
    p = []
    # VGG16包含6个部分，前面5段卷积，最后一段全连接
    # 每段卷积包含多个卷积层和pooling层

    # 下面是第一段卷积，包含2个卷积层和一个pooling层
    # 利用前面定义好的函数conv_op,mpool_op 创建这些层
    # 第一段卷积的第一个卷积层 卷积核3*3，共64个卷积核（输出通道数），步长1*1
    # input_op：224*224*3 输出尺寸224*224*64
    conv1_1 = conv_op(input_op, name="conv1_1", kh=3, kw=3, n_out=64, dh=1,
                      dw=1, p=p)
    # 第一段卷积的第2个卷积层 卷积核3*3，共64个卷积核（输出通道数），步长1*1
    # input_op：224*224*64 输出尺寸224*224*64
    conv1_2 = conv_op(conv1_1, name="conv1_2", kh=3, kw=3, n_out=64, dh=1,
                      dw=1, p=p)
    # 第一段卷积的pooling层，核2*2，步长2*2
    # input_op：224*224*64 输出尺寸112*112*64
    pool1 = mpool_op(conv1_2, name="pool1", kh=2, kw=2, dh=2, dw=2)

    # 下面是第2段卷积，包含2个卷积层和一个pooling层
    # 第2段卷积的第一个卷积层 卷积核3*3，共128个卷积核（输出通道数），步长1*1
    # input_op：112*112*64 输出尺寸112*112*128
    conv2_1 = conv_op(pool1, name="conv2_1", kh=3, kw=3, n_out=128, dh=1,
                      dw=1, p=p)
    # input_op：112*112*128 输出尺寸112*112*128
    conv2_2 = conv_op(conv2_1, name="conv2_2", kh=3, kw=3, n_out=128, dh=1,
                      dw=1, p=p)
    # input_op：112*112*128 输出尺寸56*56*128
    pool2 = mpool_op(conv2_2, name="pool2", kh=2, kw=2, dh=2, dw=2)

    # 下面是第3段卷积，包含3个卷积层和一个pooling层
    # 第3段卷积的第一个卷积层 卷积核3*3，共256个卷积核（输出通道数），步长1*1
    # input_op：56*56*128 输出尺寸56*56*256
    conv3_1 = conv_op(pool2, name="conv3_1", kh=3, kw=3, n_out=256, dh=1,
                      dw=1, p=p)
    # input_op：56*56*256 输出尺寸56*56*256
    conv3_2 = conv_op(conv3_1, name="conv3_2", kh=3, kw=3, n_out=256, dh=1,
                      dw=1, p=p)
    # input_op：56*56*256 输出尺寸56*56*256
    conv3_3 = conv_op(conv3_2, name="conv3_3", kh=3, kw=3, n_out=256, dh=1,
                      dw=1, p=p)
    # input_op：56*56*256 输出尺寸28*28*256
    pool3 = mpool_op(conv3_3, name="pool3", kh=2, kw=2, dh=2, dw=2)

    # 下面是第4段卷积，包含3个卷积层和一个pooling层
    # 第3段卷积的第一个卷积层 卷积核3*3，共512个卷积核（输出通道数），步长1*1
    # input_op：28*28*256 输出尺寸28*28*512
    conv4_1 = conv_op(pool3, name="conv4_1", kh=3, kw=3, n_out=512, dh=1,
                      dw=1, p=p)
    # input_op：28*28*512 输出尺寸28*28*512
    conv4_2 = conv_op(conv4_1, name="conv4_2", kh=3, kw=3, n_out=512, dh=1,
                      dw=1, p=p)
    # input_op：28*28*512 输出尺寸28*28*512
    conv4_3 = conv_op(conv4_2, name="conv4_3", kh=3, kw=3, n_out=512, dh=1,
                      dw=1, p=p)
    # input_op：28*28*512 输出尺寸14*14*512
    pool4 = mpool_op(conv4_3, name="pool4", kh=2, kw=2, dh=2, dw=2)

    # 前面4段卷积发现，VGG16每段卷积都是把图像面积变为1/4，但是通道数翻倍
    # 因此图像tensor的总尺寸缩小一半

    # 下面是第5段卷积，包含3个卷积层和一个pooling层
    # 第3段卷积的第一个卷积层 卷积核3*3，共512个卷积核（输出通道数），步长1*1
    # input_op：14*14*512 输出尺寸14*14*512
    conv5_1 = conv_op(pool4, name="conv5_1", kh=3, kw=3, n_out=512, dh=1,
                      dw=1, p=p)
    # input_op：14*14*512 输出尺寸14*14*512
    conv5_2 = conv_op(conv5_1, name="conv5_2", kh=3, kw=3, n_out=512, dh=1,
                      dw=1, p=p)
    # input_op：14*14*512 输出尺寸14*14*512
    conv5_3 = conv_op(conv5_2, name="conv5_3", kh=3, kw=3, n_out=512, dh=1,
                      dw=1, p=p)
    # input_op：28*28*512 输出尺寸7*7*512
    pool5 = mpool_op(conv5_3, name="pool5", kh=2, kw=2, dh=2, dw=2)

    # 将第五段卷积网络的结果扁平化
    # reshape将每张图片变为7*7*512=25088的一维向量
    shp = pool5.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    # tf.reshape(tensor, shape, name=None) 将tensor变换为参数shape的形式。
    resh1 = tf.reshape(pool5, [-1, flattened_shape], name="resh1")

    # 第一个全连接层，是一个隐藏节点数为4096的全连接层
    # 后面接一个dropout层，训练时保留率为0.5，预测时为1.0
    fc6 = fc_op(resh1, name="fc6", n_out=4096, p=p)
    fc6_drop = tf.nn.dropout(fc6, keep_prob, name="fc6_drop")

    # 第2个全连接层，是一个隐藏节点数为4096的全连接层
    # 后面接一个dropout层，训练时保留率为0.5，预测时为1.0
    fc7 = fc_op(fc6_drop, name="fc7", n_out=4096, p=p)
    fc7_drop = tf.nn.dropout(fc7, keep_prob, name="fc7_drop")

    # 最后是一个1000个输出节点的全连接层
    # 利用softmax输出分类概率
    # argmax输出概率最大的类别
    fc8 = fc_op(fc7_drop, name="fc8", n_out=2, p=p)
    softmax = tf.nn.softmax(fc8)
    predictions = tf.argmax(softmax, 1)
    return fc8


# -----------------------------------------------------------------------------
# loss计算
# 传入参数：logits，网络计算输出值。labels，真实值，在这里是0或者1
# 返回参数：loss，损失值
def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                                       name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name + '/loss', loss)
    return loss


# --------------------------------------------------------------------------
# loss损失值优化
# 输入参数：loss。learning_rate，学习速率。
# 返回参数：train_op，训练op，这个参数要输入sess.run中让模型去训练。
def trainning(loss, learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


# -----------------------------------------------------------------------
# 评价/准确率计算
# 输入参数：logits，网络计算值。labels，标签，也就是真实值，在这里是0或者1。
# 返回参数：accuracy，当前step的平均准确率，也就是在这些batch中多少张图片被正确分类了。
def evaluation(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy

