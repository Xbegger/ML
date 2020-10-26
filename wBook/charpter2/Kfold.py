import numpy as np
import random
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
iris = load_iris()
dataSet = [iris.data.tolist(), iris.target.tolist()]


def kFold(dataSet, k):
    X, Y = dataSet
    x_K = [[] for i in range(k)]
    y_K = [[] for i in range(k)]
    t = 0
    for x, y in zip(X, Y):
        t = random.randint(0, k)
        x_K[t].append(x)
        y_K[t].append(y)
    return x_K, y_K