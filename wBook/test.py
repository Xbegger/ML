import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
iris = load_iris()

X = iris.data[:,:4]
Y = iris.target

print(Y)