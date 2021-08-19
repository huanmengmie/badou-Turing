# -*-coding:utf-8 -*-
from tensorflow.keras.datasets import mnist
from tensorflow.keras import models, layers
from matplotlib import pyplot as plt
from tensorflow_core.python.keras.utils import to_categorical
import numpy as np


def test():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print("x_train  shape: {}".format(x_train.shape))
    print("x_test  shape: {}, data: {}".format(x_test.shape, x_test))
    print("y_train  shape: {}".format(y_train.shape))
    print("y_test  shape: {}, data: {}".format(y_test.shape, y_test))

    test_data, test_target = x_test[0], y_test[0]
    # cmap: color map. 图片可用色彩集 If None, default to rc image.cmap value.
    # cmap is ignored if X is 3-D, directly specifying RGB(A) values.
    plt.imshow(test_data)
    # plt.imshow(test_data, cmap=plt.cm.binary)
    plt.show()

    # 训练数据处理
    x_train_h = x_train.reshape((x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
    x_train_h = x_train_h.astype('float32') / 255
    x_test_h = x_test.reshape((x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
    x_test_h = x_test_h.astype('float32') / 255

    # 标签数据处理  将标签使用oneHot编码的方式展现
    y_train_h = to_categorical(y_train)
    y_test_h = to_categorical(y_test)

    # 使用tensorflow.Keras搭建一个有效识别图案的神经网络，
    # 1.layers:表示神经网络中的一个数据处理层。(dense:全连接层)
    # 2.models.Sequential():表示把每一个数据处理层串联起来.
    # 3.layers.Dense(…):构造一个数据处理层。
    # 4.input_shape(28*28,):表示当前处理层接收的数据格式必须是长和宽都是28的二维数组，
    # 后面的“,“表示数组里面的每一个元素到底包含多少个数字都没有关系.
    network = models.Sequential()
    # 隐藏层 512 个
    network.add(layer=layers.Dense(512, activation="relu", input_shape=(28 * 28,)))
    # 输出层 10 个
    network.add(layer=layers.Dense(10, activation="softmax"))
    # compile  Configures the model for training
    # optimizer: 选用的优化器    loss：使用分类交叉熵作为损失函数   metrics：精确度作为性能指标
    network.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=["accuracy"])

    # 进行5代，每批次为200的训练
    network.fit(x_train_h, y_train_h, batch_size=200, epochs=5, verbose=1)

    test_loss, test_acc = network.evaluate(x_test_h, y_test_h, verbose=1)
    print("测试损失：{}， 精确度:{}".format(test_loss, test_acc))

    # 单个数据预测
    res = network.predict(x_test_h[0].reshape(1, 784))
    print("实际值：{}, 预测值：{}， 预测输出:{}".format(test_target, np.argmax(res[0]), res))


if __name__ == '__main__':
    test()
