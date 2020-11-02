import numpy as np
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
def sigmod(X):
    return 1 / (1 + np.exp(-X))

def cost(param, X, Y):
    belta = param
    
    res = 0
    d1 = 0
    d2 = 0
    for x, y in zip(X, Y):
        x = x.reshape(x.shape[0], 1)
        y = y.reshape(y.shape[0], 1)
        res += np.multiply(-y, (belta.T @ x)) + np.log((1 + np.exp(belta.T @ x)))
        p1 = sigmod(belta.T @ x)
        d1 += -x * (y - p1)
        d2 += x @ x.T * p1 * (1 - p1)
    return res, d1, d2

def NewtonMethod(X, y, iters):
    params = np.ones((1,X.shape[1]))
    loss_list = []
    belta = params.T
    for i in range(iters):
        loss, d1, d2 = cost(belta, X, Y)
        belta -= np.linalg.inv(d2) @ d1
        loss_list.append(loss)
        
        print("iters = %d, loss = %f"%(i, loss))
    params = belta
    return loss_list, params    

def predict(params, X):
    belta = params
    Z = X @ belta
    return np.array([[1 if sigmod(z)>= 0.5 else 0 for z in Z]]).T
    

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


def plotData(Y, Y_pred):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(Y, label='y_test')
    ax.plot(Y_pred, label='y_predict')
    plt.legend()
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
yes = data[data['Good melon'].isin([1])]
no = data[data['Good melon'].isin([0])]


X = np.array(data.iloc[:12, :2], dtype='float')
Y = np.array(data.iloc[:12, 2], dtype='float')
Y = Y.reshape(Y.shape[0], 1)

X = np.insert(X, axis=1, values=np.ones(X.shape[0]), obj=2)

loss, params = NewtonMethod(X, Y, 100)

X_test = np.array(data.iloc[:, :2], dtype='float')
Y_test = np.array(data.iloc[:, 2], dtype='float')
X_test = np.insert(X_test, axis=1, values=np.ones(X_test.shape[0]), obj=2)
Y = Y.reshape(Y.shape[0], 1)
y_pre = predict(params, X_test)

x1 = X_test[:,0:1]
x2 = X_test[:,1:2]


x1_max = X_test[:, 0].max()
x1_min = X_test[:, 0].min()
x2_max = X_test[:, 1].max()
x2_min = X_test[:, 1].min()
x1, x2 = np.mgrid[x1_min:x1_max:500j, x2_min:x2_max:500j]  # np.mgrid的用法自行百度
x_pre = np.stack((x1.flat, x2.flat), axis=1)  # 合成坐标
x_pre = np.insert(x_pre, values=np.ones(x_pre.shape[0]), obj=2, axis=1)
y_pre = predict(params, x_pre)
y_pre = y_pre.reshape(x1.shape)
cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
fig, ax = plt.subplots(figsize=(12, 8))
plt.pcolormesh(x1, x2, y_pre, cmap=cm_light)  # 网格
ax.scatter(yes['Density'], yes['Sugar content'], marker='o', c='b', label='Yes')
ax.scatter(no['Density'], no['Sugar content'], marker='x', c='r', label='No')
ax.legend()
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
ax.set_xlabel('Density')
ax.set_ylabel('Sugar content')
plt.show()