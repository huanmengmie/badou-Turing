# -*- coding:utf-8 -*-
"""
用tf训练一个原模型为 y = w * x + b
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def test():
    # 生成样本数据 (200, 1)
    x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
    noise = np.random.normal(0, 0.02, x_data.shape)
    y_data = np.square(x_data) + noise

    # 定义两个placeholder存放数据
    x = tf.placeholder(tf.float32, [None, 1])
    y = tf.placeholder(tf.float32, [None, 1])

    # 中间隐藏层
    weight_l1 = tf.Variable(tf.random_normal([1, 10]))
    bias_l1 = tf.Variable(tf.zeros([1, 10]))
    wx_plus_b_l1 = tf.matmul(x, weight_l1) + bias_l1
    out_l1 = tf.nn.tanh(wx_plus_b_l1)

    # 输出层
    weight_l2 = tf.Variable(tf.random_normal([10, 1]))
    bias_l2 = tf.Variable(tf.zeros([1, 1]))
    wx_plus_b_l2 = tf.matmul(out_l1, weight_l2) + bias_l2
    prediction = tf.nn.tanh(wx_plus_b_l2)

    # 计算损失(均方差作为损失函数)
    loss = tf.reduce_mean(tf.square(y - prediction))
    # 反向传播训练
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    with tf.Session() as session:
        # 变量初始化
        session.run(tf.global_variables_initializer())
        for i in range(2000):
            session.run(train_step, feed_dict={x: x_data, y: y_data})
        y_prediction = session.run(prediction, feed_dict={x: x_data})
    # 画图
    plt.figure()
    plt.scatter(x_data, y_data)  # 散点是真实值
    plt.plot(x_data, y_prediction, 'r-', lw=5)  # 曲线是预测值
    plt.show()

    # 记录图信息
    writer = tf.summary.FileWriter("logs", tf.get_default_graph())
    writer.close()


if __name__ == '__main__':
    test()