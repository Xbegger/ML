import numpy as np
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import pandas as pd

def sigmod(X):
    return 1 / (1 + np.exp(-X))

def cost(param, X, Y):
    belta = param.T
    
    res = 0
    for x, y in zip(X, Y):
        x.reshape(x.shape[0], 1)
        y.reshape(y.shape[0], 1)
        res += np.sum(np.multiply(-y, (x @ belta)) + np.log((1 + np.exp(x @ belta))))
    p1 = sigmod(X @ belta)
    d1 = np.sum(X @ (Y - p1))
    d2 = np.sum(X @ X @ p1 @ X @ (1 - p1))
    return res, d1, d2

def NewtonMethod(X, y, iters):
    params = np.ones((1,X.shape[1]))
    loss_list = []
    for i in range(iters):
        loss, d1, d2 = cost(params, X, Y)
        params -= np.linalg.inv(d2) * d1
        loss_list.append(loss)
        if i % 500 == 0:
            print("iters = %d, loss = %f"%(i, loss))
    return loss_list, params    

def predict(params, X):
    belta = params.T
    p = sigmod(X @ belta)
    return np.array([[1 if x>= 0.5 else 0 for x in p]]).T
    

def plotData(X, Y):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set(title="data")
    positive = [[], []]
    negitive = [[], []]
    for x, y in zip(X, Y):
        if y == 1:
            positive[0].append(x[0])
            positive[1].append(x[1])
        elif y == 0:
            negitive[0].append(x[0])
            negitive[1].append(x[1])
    ax.scatter(positive[0], positive[1], marker='o', c='r')
    ax.scatter(negitive[0], negitive[1], marker='x', c='b')
    plt.show() 

# cancer = load_breast_cancer()

# t = 3
# dataSet = [cancer.data[:,t:t+2].tolist(), cancer.target.tolist()]

# X, Y = dataSet
# plotData(X, Y)

path = 'd:/MachineLearning/ML/wBook/charpter3/西瓜数据集alpha.txt'  # 导入数据
data = pd.read_csv(path)

data['Good melon'][data['Good melon'].isin(['是'])] = 1
data['Good melon'][data['Good melon'].isin(['否'])] = 0

X = np.array(data.iloc[:, :2], dtype='float')
Y = np.array(data.iloc[:, 2], dtype='float').reshape(data.shape[0], 1)

X = np.insert(X, axis=1, values=np.ones(X.shape[0]), obj=2)

loss, params = NewtonMethod(X, Y, 100)