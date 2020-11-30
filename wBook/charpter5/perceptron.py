import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


def sign(x):
    return np.heaviside(x, 1)



def Propagate_one(theta, X, y, learning_rate):
    X = X[None]
    y_pre = predict(theta, X)
    n = X.shape[0]
    ones = np.ones([n, 1])
    a = np.c_[X, ones]
    # 计算delta_theta
    delta_theta = (y - y_pre) @ a
    # 返回theta的更新步长

    return learning_rate * delta_theta

    


def Propagate_all(X, Y, learning_rate):
    theta = np.ones([X.shape[1] + 1, 1])
    tlearning_rate = learning_rate
    while not (predict(theta, X)[:,0] == Y).all():   
        #遍历数据X，Y进行更新
        for x, y in zip(X, Y):
            #求得要更新得theta得步长
            temp = Propagate_one(theta,x, y, tlearning_rate)
            #更新theta
            theta += temp.T
        
    #返回theta
    return theta

def predict(theta, X):
    n = X.shape[0]
    ones = np.ones([n,1])
    a = np.c_[X, ones]
    y_pre = sign(a @ theta)
    return y_pre

def plotAns(X, Y, theta):
    x1, x2 = np.mgrid[-1:2:500j, -1:2:500j]  # np.mgrid的用法自行百度
    x_pre = np.stack((x1.flat, x2.flat), axis=1)  # 合成坐标 , '#A0A0FF'
    
    y_pre = predict(theta, x_pre)
    y_pre = y_pre.reshape(x1.shape)
    cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080'])
    plt.pcolormesh(x1, x2, y_pre, cmap=cm_light)  # 网格
    
    negitive = [[],[]]
    positive = [[],[]]
    for x, y in zip(X, Y):
        if y == 1:
            positive[0].append(x[0])
            positive[1].append(x[1])
        else:
            negitive[0].append(x[0])
            negitive[1].append(x[1])
    plt.scatter(positive[0], positive[1], marker='o', c='r')
    plt.scatter(negitive[0], negitive[1], marker='x', c='b')
    
    plt.show()


            
learning_rate = 0.8


X = np.array([[0,0],[1,0],[0,1],[1,1]])
Y = np.array([[0,0,0,1], [0, 1, 1, 1]])

for tY in Y:
    theta = Propagate_all(X, tY, learning_rate)
    plotAns(X, tY, theta)
    
X = np.array([[0,0], [1,0]])
Y = np.array([1, 0])
theta = Propagate_all(X, Y, learning_rate)
plotAns(X, Y, theta)
