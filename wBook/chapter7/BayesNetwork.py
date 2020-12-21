import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st

def P(X, x, i, depend, con_col):
    raw = False
    if depend != []:
        for j in depend:
            raw = raw | (X[:,j] == x[j])
        X = X[raw]
    if i in con_col:
        return st.norm.pdf(x[i], loc=X[:,i].mean(), scale=X[:,i].std())
    return (X[:,i] == x[i]).sum() / X.shape[0]

def P_one(X, x, B, con_col):
    res = 1
    for i in range(x.shape[0]):
        res *= P(X, x, i, B[i], con_col)
    return res

def LL(X, B, con_col):
    sum = 0
    for i in range(X.shape[0]):
        sum += np.log(P_one(X, X[i], B, con_col))
    return sum

def BIC(X, B, m, con_col):
    return np.log(m)/2 * len(B) - LL(X, B, con_col)


# 数据处理
data = pd.read_csv("wBook/charpter7/西瓜数据集3.0.txt")
X = np.array(data.iloc[:,:8])
y = np.array(data.iloc[:,8])[None].T




B = [[]] * X.shape[1]
score = BIC(X, B, 2, [6,7])
for i in range(X.shape[1]):
    for j in range(i+1, X.shape[1]):
        B[i].append(j)
        res = BIC(X, B, 2, [6,7])
        if score > res:
            score = res
        else:
            B[i].pop()
print(B)
