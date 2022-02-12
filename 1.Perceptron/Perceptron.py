#coding=utf-8
#Author:Dodo
#Date:2018-11-15
#Refactor:2022-2-12
#Email:lvtengchao@pku.edu.cn

'''
数据集：Mnist
训练集数量：1000
测试集数量：100
------------------------------
运行结果：
正确率：
运行时长：
'''

import time
import numpy as np

def load_data(path):
    '''
    加载Mnist数据集
    :param fileName:要加载的数据集路径
    :return: list形式的数据集及标记
    '''
    print('start reading data from {}'.format(path))
    # 存放数据及对应标签的list
    data, label = [], []
    # 打开文件, 使用utf-8编码读取
    fr = open(path, 'r', encoding='utf-8')
    # 将文件按行读取
    for line in fr.readlines():
        # 对每一行数据按切割符','进行切割
        cur_line = line.strip().split(',')
        # Mnsit有0-9是个标记，由于该任务是二分类任务，所以将>=5的作为1，<5为-1，在这一任务中模型负责判断样本是>=5还是<5
        if int(cur_line[0]) >= 5:
            label.append(1)
        else:
            label.append(-1)
        #存放标记
        #[int(num)/255 for num in curLine[1:]] -> 将所有数据除255归一化(非必须步骤，可以不归一化)
        data.append([int(num)/255 for num in cur_line[1:]])
    #返回data和label
    return np.array(data), np.array(label)

class Perceptron():
    def __init__(self, training_data, training_label, test_data, test_label, h=0.0001, iter=50):
        self.training_data  = training_data
        self.training_label = training_label
        self.test_data      = test_data
        self.test_label     = test_label

        # 创建初始权重w，初始值全为0。
        # np.shape(dataMat)的返回值为m，n -> np.shape(dataMat)[1])的值即为n，与
        # 样本长度保持一致
        self.w = np.zeros((1, self.training_data.shape[1]))
        # 初始化偏置b为0
        self.b = 0
        # 初始化步长，也就是梯度下降过程中的n，控制梯度下降速率
        self.h = h
        # 控制迭代的轮数
        self.iter = iter

    def train(self):
        # 进行iter次迭代计算
        for iter in range(self.iter):
            # 对于每一个样本进行梯度下降
            # 李航书中在2.3.1开头部分使用的梯度下降，是全部样本都算一遍以后，统一
            # 进行一次梯度下降
            # 在2.3.1的后半部分可以看到（例如公式2.6 2.7），求和符号没有了，此时用
            # 的是随机梯度下降，即计算一个样本就针对该样本进行一次梯度下降。
            # 两者的差异各有千秋，但较为常用的是随机梯度下降。
            for i in range(self.training_data.shape[0]):
                # 获取当前样本的向量
                xi = self.training_data[i]
                # 获取当前样本所对应的标签
                yi = self.training_label[i]
                # 判断是否是误分类样本
                # 误分类样本特诊为： -yi(w*xi+b)>=0，详细可参考书中2.2.2小节
                # 在书的公式中写的是>0，实际上如果=0，说明改点在超平面上，也是不正确的
                if -1 * yi * (np.dot(self.w, xi) + self.b) >= 0:
                    # 对于误分类样本，进行梯度下降，更新w和b
                    self.w = self.w + self.h * yi * xi
                    self.b = self.b + self.h * yi
            # 打印训练进度
            print('Iter %d:%d' % (iter, self.iter))

if __name__ == '__main__':
    #获取当前时间
    #在文末同样获取当前时间，两时间差即为程序运行时间
    start = time.time()

    #获取训练集及标签
    training_data, training_label = load_data('../0.datasets/Mnist/Mnist_train.txt')
    #获取测试集及标签
    test_data, test_label = load_data('../0.datasets/Mnist/Mnist_test.txt')

    perceptron = Perceptron(training_data, training_label, test_data, test_label, h=0.0001, iter=50)
    # 训练获得权重
    perceptron.train()
    print('w')
    #
    # w, b = perceptron(trainData, trainLabel, iter = 30)
    # #进行测试，获得正确率
    # accruRate = model_test(testData, testLabel, w, b)
    #
    # #获取当前时间，作为结束时间
    # end = time.time()
    # #显示正确率
    # print('accuracy rate is:', accruRate)
    # #显示用时时长
    # print('time span:', end - start)