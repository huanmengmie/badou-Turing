# -*- coding:utf-8 -*-
"""
手写数字识别，神经网络实现测试和训练
"""
import numpy as np
import scipy.special


class NeuralNetWork:
    def __init__(self, input_node_count, output_node_count, hidden_node_count, learning_rate):
        """

        :param input_node_count: 输入神经元（输入点）个数   28 * 28 = 784
        :param output_node_count: 输出神经元（输出点）个数   10
        :param hidden_node_count: 隐藏神经元个数   100
        :param learning_rate: 学习率
        """
        self.in_nodes = input_node_count
        self.ou_nodes = output_node_count
        self.h_nodes = hidden_node_count
        self.learning_rate = learning_rate
        # 设置激活函数为sigmoid函数
        self.active_function = lambda x: scipy.special.expit(x)

        # 随机初始化权值
        # wih 输入层到隐藏层的权值 w = (100 * 784)  x = (784 * 1)   w*x = (100, 1)
        self.wih = np.random.rand(self.h_nodes, self.in_nodes) - 0.5
        # who 隐藏层到输出层的权值 w = (10 * 100)  x = (100 * 1)   w*x = (10, 1)
        self.who = np.random.rand(self.ou_nodes, self.h_nodes) - 0.5

    def train(self, input_list, target_list):
        """
        模型训练
        :param input_list:  特征值数组
        :param target_list:  标签数组 （类似于OneHot编码那种）
        :return:
        """
        # (784, 1)   (10, 1)
        input_data = np.array(input_list, ndmin=2).T
        target_data = np.array(target_list, ndmin=2).T
        # 输入层的 加权输入和
        sum_hidden_inputs = np.dot(self.wih, input_data)
        # 中间层神经元对输入的信号做激活函数后得到输出信号
        hidden_outputs = self.active_function(sum_hidden_inputs)

        # 隐藏层的 加权输入和
        sum_final_inputs = np.dot(self.who, hidden_outputs)
        # 隐藏层神经元对输入的信号做激活函数后得到输出信号
        final_outputs = self.active_function(sum_final_inputs)

        # 输出的误差
        output_error = target_data - final_outputs
        # 隐藏层误差 todo: 接着看看
        hidden_error = np.dot(self.who.T, output_error * final_outputs * (1 - final_outputs))

        self.who += self.learning_rate * np.dot(output_error * final_outputs * (1 - final_outputs), hidden_outputs.T)
        self.wih += self.learning_rate * np.dot(hidden_error * hidden_outputs * (1 - hidden_outputs), input_data.T)