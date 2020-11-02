import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def LDA(X_0, X_1):
    mu_0 = X_0.mean(axis=0).reshape(1, X_0.shape[1]).T
    mu_1 = X_1.mean(axis=0).reshape(1, X_1.shape[1]).T
    sigma_0 = (X_0 - mu_0.T).T @ (X_0 - mu_0.T)
    sigma_1 = (X_1 - mu_1.T).T @ (X_1 - mu_1.T)
    s_w = sigma_0 + sigma_1
    w = np.linalg.inv(s_w) @ (mu_0 - mu_1)
    return w

def mapping(w, X):
    k = w[1] / w[0]
    new_X = ((k * X[:, 1] + X[:, 0]) / (k * k + 1)).reshape(1, X.shape[0])
    new_X = np.insert(new_X, values=k*new_X, obj=1, axis=0)
    return new_X.T


def plot(yes, no):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(yes[:, 0], yes[:, 1], marker='^', c='b', label='new_Yes')
    ax.scatter(no[:, 0], no[:, 1], marker='*', c='r', label='new_No')
    ax.set_xlabel('Density')
    ax.set_ylabel('Sugar content')
    plt.axis([0, 1., 0, 1.])
    plt.show()

path = 'd:/MachineLearning/ML/wBook/charpter3/西瓜数据集alpha.txt'  # 导入数据
data = pd.read_csv(path)

data['Good melon'][data['Good melon'].isin(['是'])] = 1
data['Good melon'][data['Good melon'].isin(['否'])] = 0


yes = data[data['Good melon'].isin([1])]
no = data[data['Good melon'].isin([0])]
plot(np.array(yes), np.array(no))

X_0 = np.array(data[data['Good melon'].isin([0])].iloc[:, 0:2], dtype='float')
X_1 = np.array(data[data['Good melon'].isin([1])].iloc[:, 0:2], dtype='float')

w = LDA(X_0, X_1)

np.insert(w, values=0, axis=1, obj=0).T

new_yes = mapping(w, X_1)
new_no = mapping(w, X_0)
plot(new_yes, new_no)



