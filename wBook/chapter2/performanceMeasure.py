import numpy as np
import random
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import operator
from sklearn.linear_model import LogisticRegression
from holdOut import holdOut
from matplotlib.pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']


# function:计算模型的错误率
# parameter:predict:  模型对样本的分类
#           y:  样本实际的分类
# return: error:错误率
def calError(predict_y, y):
    m = len(predict_y)
    cnt = 0
    for p_y_i, y_i in zip(predict_y, y):
        if p_y_i == y_i:
            cnt += 1
    error = cnt / (m * 1.) 
    return error

# function:计算模型的精度
# parameter:predict_y:模型对样本的分类
#           y:样本实际的分类
# return:acc:准确率
def calAccuracy(predict_y, y):
    m = len(predict_y)
    cnt = 0
    for p_y_i, y_i in zip(predict_y, y):
        if p_y_i != y_i:
            cnt += 1
    acc = cnt / (m * 1.) 
    return acc


# fucntion:计算二分类模型的查准率与查全率
# parameter:predict_y:模型对样本的分类
#           y:样本实际的分类   0:负例  1:正例
# return:precision:查准率与查全率
def calPrecisionAndRecall(predict_y, y):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for p_y_i, y_i in zip(predict_y, y):
        if p_y_i == 1 and y_i == 1:
            tp += 1
        elif p_y_i == 1 and y_i == 0:
            fp += 1
        elif p_y_i == 0 and y_i == 1:
            fn += 1
        elif p_y_i == 0 and y_i == 0:
            tn += 1
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return p, r


def drawP_RCurve(predict, y):
    zipPredict = list(zip(predict, y))
    zipPredict.sort(key = operator.itemgetter(0))

    x_y = dict()
    
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for y_i in  y:
        if y_i == 1:
            fn += 1
        elif y_i == 0:
            tn += 1

    for item in zipPredict:
        if item[1] == 1:
            fn -= 1
            tp += 1
        elif item[1] == 0:
            tn -= 1
            fp += 1
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        x_y.setdefault(r, p)
    
    plotP_R(x_y)

def plotP_R(x_y):

    plt.xlabel("Recall")
    plt.ylabel("Precison")
    plt.plot(x_y.keys(), x_y.values())
    t = np.arange(0, 1, 0.01)
    plt.plot(t, t)
    plt.show()


def drawROC(predict, y):
    zipPredict = list(zip(predict, y))
    zipPredict.sort(key=operator.itemgetter(0))

    mP = 0
    mN = 0
    for y_i in y:
        if y_i == 0:
            mN += 1
        elif y_i == 1:
            mP += 1
    x_y = []
    x = []
    y = []
    tpr = 0
    fpr = 0
    for item in zipPredict:
        if item[1] == 1:
            tpr += 1 / mP
            x.append(fpr)
            y.append(tpr)
        elif item[1] == 0:
            fpr += 1 / mN
            x.append(fpr)
            y.append(tpr)
    plotROC(x, y)
    costCurve(x, y)    

def plotROC(x, y):
    plt.xlabel("fpr")
    plt.ylabel("tpr")
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)
    plt.plot(x, y,'-o')
    plt.show()

def costCurve(Fpr, Tpr):
    plt.xlabel("正例概率代价")
    plt.ylabel("归一化代价")
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)
    for fpr, tpr in zip(Fpr, Tpr):
        x = [0, 1]
        y = [fpr, 1-tpr]
        plt.plot(x, y,)
    plt.show()

cancer = load_breast_cancer()

dataSet = [cancer.data[:,:].tolist(), cancer.target.tolist()]
clf = LogisticRegression()
x_train, y_train, x_pre, y_val = holdOut(dataSet, 0.75)
clf.fit(x_train, y_train)
y_pre_prob = clf.predict_proba(x_pre)
y_pre = clf.predict(x_pre)
# 绘制P-R曲线和求BEP
drawP_RCurve(y_pre_prob[:,:1].tolist(), y_val)

# 计算F1与Fβ

# 绘制ROC曲线
drawROC(y_pre_prob[:,:1].tolist(), y_val)


# clf.fit(x_train, y_train)

# drawP_RCurve()


# # 看数据集的分布
# X, Y = dataSet
# positive = [[], []]
# negitive = [[], []]
# for x, y in zip(X, Y):
#     if y == 1:
#         positive[0].append(x[0])
#         positive[1].append(x[1])
#     else:
#         negitive[0].append(x[0])
#         negitive[1].append(x[1])
# plt.scatter(positive[0], positive[1], marker='o', c='r')
# plt.scatter(negitive[0], negitive[1], marker='x', c='b')
# plt.show()