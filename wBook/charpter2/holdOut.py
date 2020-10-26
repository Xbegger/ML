import numpy as np
import random
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
iris = load_iris()
dataSet = [iris.data.tolist(), iris.target.tolist()]

# 分层抽样的留出法
def holdOutStatify(dataSet, randomSeed, ratio):
    X, Y = dataSet
    x_train = []
    x_validation = []
    y_train = []
    y_validation = []
    # 用来表示当前元素放入哪个数据集
    t = 0
    for x, y in zip(X, Y):
        t = random.random()
        if t < ratio:
            x_train.append(x)
            y_train.append(y)
        else:
            x_validation.append(x)
            y_validation.append(y)
    return x_train, y_train, x_validation, y_validation


def holdOut(dataSet, ratio):
    X, Y = dataSet
    length = len(X)

    SizeOfTrain = int(length * ratio)
    SizeOfValidation = length - SizeOfTrain
    t = random.randint(0, length - SizeOfValidation)

    startOfValidation = t

    x_validation = X[startOfValidation:startOfValidation + SizeOfValidation]
    y_validation = Y[startOfValidation:startOfValidation + SizeOfValidation]

    x_train = X[:startOfValidation] + X[startOfValidation + SizeOfValidation:]
    y_train = Y[:startOfValidation] + Y[startOfValidation + SizeOfValidation:]
    
    return x_train, y_train, x_validation, y_validation

x_train, y_train, x_validation, y_validation = holdOutStatify(dataSet, 42, 0.75)
