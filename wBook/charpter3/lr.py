import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import pandas as pd


# function:
#       根据传入的数据集，训练一个多元线性回归模型
# parameter:
#       X
#       Y
#       learn_rate:学习率
#       iters:迭代次数
# return:
#       loss_list, W, B
def linearRegression(X, Y, learn_rate, iters):
    num_feature = X.shape[1]
    W = np.zeros((num_feature, 1))
    B = 0
    loss_list = []

    for i in range(iters):
        loss, dw, db = calSquareLoss(W, B, X, Y)
        loss_list.append(loss)
        W = W - learn_rate * dw
        B = B - learn_rate * db
        
        learn_rate *= 0.9
        if i % 500 == 0:
            print("iters = %d, loss = %f"%(i, loss))
    return loss_list, W, B

# function:
#       计算均方误差
# parameter:
#       W:权重w
#       B:权值b
#       X:数据集的属性 
#       Y:数据集的实际值
# return:
#       square_loss:
def calSquareLoss(W, B, X, Y):
    num_train = X.shape[0]
    H = X.dot(W) + B
    loss = np.sum(np.square(H - Y)) / num_train
    dw = X.T.dot((H - Y)) / num_train
    db = np.sum((H - Y)) / num_train
    return loss, dw, db


# function:
#       模型预测
# parameter:
#       X:测试集
#       W:
#       B
# return:
#       Y_pred:模型对测试集的预测
def predict(X, W, B):
    Y_pred = X.dot(W) + B
    return Y_pred

# funciton:
# parameter:
# return:
def plotData(Y, Y_pred):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(Y, label='y_test')
    ax.plot(Y_pred, label='y_predict')
    plt.legend()
    plt.show()


path = 'd:/MachineLearning/ML/wBook/charpter3/学生身高和体重.txt'
data = pd.read_csv(path)
X = np.array(data.iloc[:6,0:2], dtype='float')
Y = np.array(data.iloc[:6,2:], dtype='float')

loss, W, B = linearRegression(X, Y, 0.0005, 100)
print(W,B, loss)

X_test = np.array(data.iloc[6:,0:2], dtype='float')
Y_test = np.array(data.iloc[6:,2:], dtype='float')

Y_pred = predict(X_test, W, B)

plotData(Y_test, Y_pred)