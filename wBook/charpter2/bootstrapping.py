import numpy as np
import random
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
iris = load_iris()
dataSet = [iris.data.tolist(), iris.target.tolist()]


def bootstrapping(dataSet):
    X, Y = dataSet
    sizeOfdata = len(X)

    x_train = []
    y_train = []

    setOfTrain = set()

    for i in range(sizeOfdata):
        t = random.randint(0, sizeOfdata)
        x_train.append(X[t])
        y_train.append(X[t])
        setOfTrain.add(t)
    
    setOfValidation = {i for i in range(sizeOfdata)} - setOfTrain
    x_Validation = []
    y_Validation = []
    for i in setOfValidation:
        x_Validation.append(X[i])
        y_Validation.append(Y[i])
    return x_train, y_train, x_Validation, y_Validation
