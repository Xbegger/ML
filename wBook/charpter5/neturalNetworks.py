import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import preprocessing
from scipy.optimize import minimize  # 优化函数
from scipy.io import loadmat

def sigmod(x):
    return 1 / (1 + np.exp(x))

def sigmod_gradient(z):
    return np.multiply(sigmod(z), (1 - sigmod(z)))

def cost(params, X, y, hidden_size, input_size, num_labels):
    a_0, z_1, a_1, z_2, y_pre = forward_propagate(params, X, hidden_size, input_size, num_labels)
    return np.sum((y_pre - y) ** 2) / 2

def forward_propagate(params, X, hidden_size, input_size, num_labels):
    theta = separate_params(params, hidden_size, input_size, num_labels)
    n = X.shape[0]
    ones = np.ones([n, 1])
    a_0 = np.c_[X, ones]
    z_1 = a_0 @ theta[0].T
    a_1 = np.c_[sigmod(z_1), ones]
    z_2 = a_1 @ theta[1].T
    y_pre = sigmod(z_2)
    return a_0, z_1, a_1, z_2, y_pre

def backprop_one(params, X, y, hidden_size, input_size, num_labels, learning_rate):
    theta = separate_params(params, hidden_size, input_size, num_labels)
    a_0, z_1, a_1, z_2, y_pre = forward_propagate(params, X,hidden_size, input_size, num_labels)
    z_1 = np.c_[z_1,1]
    gradient = [None, None]
    delta_2 = (y_pre - y) * y_pre * (1 - y_pre)
    gradient[1] = delta_2.T @ a_1
    delta_1 = delta_2 @ theta[1] * sigmod_gradient(z_1)
    gradient[0] = delta_1[:, :-1].T @ a_0
    return np.concatenate((np.ravel(gradient[0]), np.ravel(gradient[1]))) * learning_rate


def backprop_all(params, X, y, hidden_size, input_size, num_labels, learning_rate):
    n = X.shape[0]
    gradient = create_params(hidden_size, input_size, num_labels)
    gradient -= gradient

    for X_i, y_i in zip(X, y):
        temp = backprop_one(params, X_i[None], y_i[None], hidden_size, input_size, num_labels, learning_rate)
        gradient += temp

    c = cost(params, X, y, hidden_size, input_size, num_labels)
    return c, gradient


def create_params(hidden_size, input_size, num_labels):
    return (np.random.random(size=hidden_size*(input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.25


def separate_params(params, hidden_size, input_size, num_lables):
    theta1 = np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1)))
    theta2 = np.reshape(params[hidden_size * (input_size + 1):], (num_lables, (hidden_size + 1)))
    return [theta1, theta2]


data = loadmat("d:/MachineLearning/ML/wBook/charpter5/ex4data1.mat")

enc = preprocessing.OneHotEncoder()
X = np.array(data['X'])
y_prev = data['y']
y = enc.fit_transform(data['y']).toarray()

hidden_size = 30
num_labels = 10
input_size = X.shape[1]
params = create_params(hidden_size, input_size, num_labels)
learning_rate = 0.1

fmin = minimize(fun=backprop_all, x0=params, args=(X, y, hidden_size, input_size, num_labels, learning_rate), 
                method='TNC', jac=True, options={'maxiter': 250}, )
print(fmin)


a_0, z_1, a_1, z_2, y_pre = forward_propagate(fmin.x, X, hidden_size, input_size, num_labels)
y_pred = np.array(np.argmax(y_pre, axis=1) + 1)
correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y_prev)]
accuracy = (sum(map(int, correct)) / float(len(correct)))
print ('accuracy = {0}%'.format(accuracy * 100))
    
# path = "d:/MachineLearning/ML/wBook/charpter5/西瓜数据集3.0.txt"
# data = pd.read_csv(path)

# train_index = [1,2,3,6,7,10,14,15,16,17]
# test_index = [4,5,8,9,11,12,13]

# train = data
# test = data.loc[test_index]



# enc = preprocessing.OneHotEncoder()
# a = enc.fit_transform(train.iloc[:, :6]).toarray()
# b = np.array(train.iloc[:, 6:8])

# X = np.c_[a, b]
# y = enc.fit_transform(train.iloc[:, 8:]).toarray()

# hidden_size = 5
# input_size = X.shape[1]
# num_labels = y.shape[1]


# params = create_params(hidden_size, input_size, num_labels)
# learning_rate = 0.1


# fmin = minimize(fun=backprop_all, x0=params, args=(X, y, hidden_size, input_size, num_labels, learning_rate), 
#                 method='TNC', jac=True, options={'maxiter': 250}, )
# print(fmin)
# # c,t = backprop_all(params, X, y, hidden_size, input_size, num_labels, learning_rate)


# a = enc.fit_transform(test.iloc[:, :6]).toarray()
# b = np.array(test.iloc[:, 6:8])

# X_test = np.c_[a, b]
# y_test = enc.fit_transform(test.iloc[:, 8:]).toarray()

# a_0, z_1, a_1, z_2, y_pre = forward_propagate(params, X_test, hidden_size, input_size, num_labels)


