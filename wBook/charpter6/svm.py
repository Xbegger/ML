import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm

data = pd.read_csv("d:/MachineLearning/ML/wBook/charpter6/西瓜数据集3.0.txt")



def plotData(data):
    yes = data[data['Good melon'].isin(['是'])]
    no = data[data['Good melon'].isin(['否'])]
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(yes['Density'], yes['Sugar content'], marker='o', c='b', label='Yes')
    ax.scatter(no['Density'], no['Sugar content'], marker='x', c='r', label='No')
    ax.legend()
    ax.set_xlabel('Density')
    ax.set_ylabel('Sugar content')
    plt.show()


linear_svm = svm.SVC(kernel='linear')
rbf_svm = svm.SVC(kernel='rbf')

train_index = [0, 1,2,3,6,7,10,14,15,16]
test_index = [4,5,8,9,11,12,13]

train_data = data.loc[train_index]
test_data = data.loc[test_index]

temp = {'是': 1, '否': -1}
X = np.array(train_data.iloc[:,:2])
y = np.array(train_data.iloc[:,2].replace(temp))[None].T

linear_svm.fit(X,y)
rbf_svm.fit(X,y)


temp = {'是': 1, '否': -1}
test_X = np.array(test_data.iloc[:,:2])
test_y = np.array(test_data.iloc[:,2].replace(temp))[None].T

print("linear scores:", linear_svm.score(test_X, test_y))
print("rbf scores:", rbf_svm.score(test_X, test_y))


from sklearn import datasets
iris = datasets.load_iris()


X = iris['data']
y = iris['target'][None].T


linear_svm = svm.SVC(kernel='linear')
rbf_svm = svm.SVC(kernel='rbf')


linear_svm.fit(X, y)
rbf_svm.fit(X, y)

print("linear SVM:", linear_svm.score(X, y))
print("rbf SVM:", rbf_svm.score(X, y))

from sklearn import tree
clf = tree.DecisionTreeClassifier()

clf.fit(X, y)
print("decisionTree:" , clf.score(X, y))

