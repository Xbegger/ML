from sklearn import tree
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

import numpy as np
import pickle


def AdaBoost(X_train, y_train, T):
    Dt = np.ones(X_train.shape[0])/  X_train.shape[0]
    
    clf = tree.DecisionTreeClassifier()
    
    H = []
    Alpha = []
    for i in range(T):
        clf = tree.DecisionTreeClassifier(max_depth = 1)
        clf = clf.fit(X_train, y_train, Dt)
        accuracy = clf.score(X_train, y_train)
        error = 1 - accuracy
        if error > 0.5:
            break
        alpha = 0.5 * np.log(accuracy / error)
        y_pre = clf.predict(X_train)

        ht = pickle.dumps(clf)
        
        expon = np.multiply(-1 * alpha, y_pre * y_train)
        
        D = np.multiply(Dt, np.exp(expon))
        
        Dt = D / np.sum(D)
        
        H.append(ht)
        Alpha.append(alpha)
        if Dt.all() == 0:
            break
        
    return H, Alpha
        
def calAccuracy(H, Alpha, X_test, y_test):
    num = X_test.shape[0]
    y_pre = np.zeros(num)
    for h, alpha in zip(H, Alpha):
        hi = pickle.loads(h)
        y_pre += alpha * hi.predict(X_test)
    y_pre = np.sign(y_pre)
    accuracy = np.sum(y_pre == y_test) / num
    return accuracy


def Bagging(X, y, T):
    H = []
    for i in range(T):
        index =np.random.choice(range(X.shape[0]),size=X.shape[0],replace=None)
        X_train = X[index]
        y_train= y[index]
        clf = tree.DecisionTreeClassifier(max_depth = 1,splitter="random")
        clf = clf.fit(X_train, y_train)
        hi = pickle.dumps(clf)
        H.append(hi)
    return H


def calAccuracyBagging(H,X_test, y_test):
    num = X_test.shape[0]
    y_pre = np.zeros(num)
    for h in H:
        hi = pickle.loads(h)
        y_pre += hi.predict(X_test)
        
    y_pre =np.sign(y_pre)
    accuracy = np.sum(y_pre == y_test) / num
    return accuracy


dataSet = load_breast_cancer()
X = dataSet.data
Y = dataSet.target
Y[Y==0]=-1

X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.25, random_state=0)


clf = tree.DecisionTreeClassifier(max_depth = 1)
clf = clf.fit(X_train,y_train)
print("orignal:",clf.score(X_test, y_test))

H, Alpha = AdaBoost(X_train, y_train, 11)

accuracy = calAccuracy(H, Alpha, X_test, y_test)
print("ensemble AdaBoost:", accuracy)

H = Bagging(X_train, y_train, 11)
accuracy = calAccuracyBagging(H, X_test, y_test)
print("ensemble Bagging:", accuracy)