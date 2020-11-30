import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st


# 贝叶斯分类器

class Bayes:
    def train(self, X, y, con_col):
        self.con_col = con_col
        self.X = X
        self.y = y
        #离散属性项
        self.dis_X = np.delete(X, con_col, axis=1)
        #连续属性项
        self.con_X = X[:,con_col]
        #y的可能取值
        self.unique_y = np.unique(y)
        self.dis_P_all = self.get_dis_P(self.dis_X, y)
        self.P_c = ((self.unique_y == y).sum(axis=0) + 1) / (data.shape[0] + self.unique_y.shape[0])

    def get_dis_P(self, X, y):
        P_all = []
        for i, c in enumerate(self.unique_y):
            temp = X[(y==c).ravel()]
            an_temp_list = []
            #枚举离散数据项
            for x in range(temp.shape[1]):
                temp_list = {}
                temp_uni = np.unique(X[:,x])
                #枚举每项数据的可能取值
                for x_i in temp_uni:
                    temp_list[x_i] = self.P(temp, x, x_i, temp_uni.shape[0])
                an_temp_list.insert(x, temp_list)
            P_all.insert(i, an_temp_list)
        return P_all

    def get_con_P(self, X, x, x_i):
        return st.norm.pdf(x_i, loc=X[:,x].mean(), scale=X[:,x].std())

    def P(self, X, x, x_i, N):
        #拉普拉斯修正
        return ((X[:,x] == x_i).sum() + 1) / (X.shape[0] + N)
        #return (X[:,x] == x_i).sum() / X.shape[0]

    def predict(self, X_pre):
        y_pre = np.ones(X_pre.shape[0], dtype='object')
        for j, X in enumerate(X_pre):
            max_P = 0
            res_c = 0
            for i,c in enumerate(self.unique_y):
                res = 1
                temp_X = self.X[(self.y == c).ravel()]
                a = 0
                for x in range(X.shape[0]):
                    if x in self.con_col:
                        res *= self.get_con_P(temp_X, x, X[x])
                    else:
                        res *= (self.dis_P_all[i][a][X[x]])
                        a += 1
                res *= self.P_c[i]
                if res > max_P:
                    max_P = res
                    res_c = c
            y_pre[j] = res_c
        return y_pre[None].T

# 数据处理
data = pd.read_csv("d:/MachineLearning/ML/wBook/charpter7/西瓜数据集3.0.txt")
train_index = [0, 1,2,3,6,7,10,14,15,16]
test_index = [4,5,8,9,11,12,13]

train_data = data.loc[train_index]
test_data = data.loc[test_index]

X = np.array(train_data.iloc[:,:8])
y = np.array(train_data.iloc[:,8])[None].T
X_test = np.array(test_data.iloc[:,:8])
y_test = np.array(test_data.iloc[:,8])[None].T

#贝叶斯分类器
Bayes_machine = Bayes()
Bayes_machine.train(X, y, [6, 7])
print(Bayes_machine.dis_P_all)

test_X = np.array(['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460], dtype='object')

print(Bayes_machine.predict(X_test) == y_test)



unique = np.unique(X[:, 0])
X_em = np.array(data)[:, 1:]
y_em = np.array(data.iloc[:, 0])[None].T


y_test = unique[np.random.randint(unique.shape[0], size=y.shape[0])][None].T


while True:
    Bayes_machine.train(X_em, y_test, [5,6])
    new = Bayes_machine.predict(X_em)
    
    if (new == y_test).all():
        break        
    y_test = new

print(y_test)
