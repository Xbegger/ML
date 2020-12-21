import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.stats


def k_means(data, k):
    m = data.shape[0]
    n = data.shape[1]
    C = np.zeros(m,dtype=int)
    D = np.zeros((m,k))
    
    rand = [random.randint(0,m-1) for i in range(k)]
    Mu = data[rand]
    flag = True

    while flag == True:
        for j in range(m):
            for i in range(k):
                D[j][i] = caldist_Ed(data[j], Mu[i])
                if D[j][i] < D[j][C[j]]:
                    C[j] = i
        
        Mu_bak = np.zeros((k, n))
        cnt = np.zeros((k,1))
        for j in range(m):
            Mu_bak[C[j]] += data[j]
            cnt[C[j]] += 1

        flag = False
        for i in range(k):
            Mu_bak[i] /= cnt[i]
            if (Mu_bak[i] != Mu[i]).all():
                flag = True
                Mu[i] = Mu_bak[i]
        # plot(X,C)
                
    return C
                

def LVQ(X, y, q, learningRate, epochs):
    #init prototype vector
    m = X.shape[0]
    n = X.shape[1]
    p = np.random.random(size=[q,n])
    yTag = np.unique(y)
    T = np.array([yTag[random.randint(0, len(yTag)-1)] for i in range(q)])

    for epoch in range(epochs):
        
        j = random.randint(0, m-1)
        i_hat = 0
        dist_hat = 10
        for i in range(q):
            dist = caldist_Ed(X[j], p[i])
            if dist < dist_hat:
                i_hat = i
                dist_hat = dist
        if y[j] == T[i_hat]:
            p[i_hat] = p[i_hat] + learningRate * (X[j] - p[i_hat])
        else:
            p[i_hat] = p[i_hat] - learningRate * (X[j] - p[i_hat])

    return p, T

def classifyAfterLVQ(X, p, T):
    m = data.shape[0]
    n = data.shape[1]
    k = p.shape[0]
    C = np.zeros(m,dtype=int)

    for i in range(m):
        dist = caldist_Ed(X[i], p[0])
        C[i] = T[0]
        for j in range(1, k):
            dist_ed = caldist_Ed(X[i], p[j])
            if dist_ed < dist:
                dist = dist_ed
                C[i] = T[j]
    return C
    
def MixGauss(X, k, epochs):
    m,n = X.shape
    Alpha = np.ones(k) / k
    Mu = np.random.random(size=[k, n])
    Sigma = random.random() * np.array([np.identity(n,dtype=float) for i in range(k)])
    gamma = np.zeros([k, m])
    p = np.zeros([m,k])

    for epoch in range(epochs):
        for j in range(m):
            for i in range(k):
                p[j][i] = Alpha[i] * Gaussian(Mu[i], Sigma[i], X[j])
            for i in range(k):
                gamma[i][j] = p[j][i] / np.sum(p[j])

        for i in range(k):
            Mu[i] = np.sum(gamma[i] @ X) / np.sum(gamma[i])
            bak_Sigma = np.zeros([n,n])
            for j in range(m):
                bak_Sigma += gamma[i][j] * (X[j]-Mu[i]).reshape(-1,1) @(X[j]-Mu[i]).reshape(1,-1)
                
            Sigma[i] = bak_Sigma / np.sum(gamma[i])
            Alpha[i] = np.sum(gamma[i]) / m
    
    C = np.zeros(m,dtype=int)
    for j in range(m):
        prob = Gaussian(Mu[0], Sigma[0], X[j])
        C[j] = 0
        for i in range(1, k):
            prob_hat = Gaussian(Mu[i], Sigma[i], X[j])
            if prob_hat > prob:
                prob = prob_hat
                C[j] = i
    return C

def Gaussian(mu, sigma, x):
    prob = scipy.stats.multivariate_normal(mu, sigma).pdf(x)
    return prob


def hierarchical(X, k):
    m,n = X.shape
    C = [list() for i in range(m)]
    for j in range(m):
        C[j].append(j)

    M = np.zeros([m,m])

    for i in range(m):
        for j in range(i+1, m):
            M[i][j] = caldist_max(X, C[i], C[j])
            M[j][i] = M[i][j]
    q = m

    while q > k:
        min_dis = M[0][1]
        hat_i = 0
        hat_j = 1
        for i in range(q):
            for j in range(i+1, q):
                if min_dis > M[i][j]:
                    min_dis = M[i][j]
                    hat_i = i
                    hat_j = j
        C[hat_i] += C[hat_j]

        for j in range(hat_j + 1, q):
            C[j-1] = C[j]
        M = np.delete(M, hat_j, axis=0)
        M = np.delete(M, hat_j, axis=1)

        for j in range(q-1):
            M[hat_i][j] = caldist_max(X, C[hat_i], C[j])
            M[j][hat_i] = M[hat_i][j]
        q -= 1
    
    C = C[0:k]
    ans = np.zeros(m, dtype=int)
    for i in range(len(C)):
        for j in C[i]:
            ans[j] = i
    
    return ans



def caldist_max(X, Ci, Cj):
    dis_max = 0
    for i in range(len(Ci)):
        for j in range(len(Cj)):
            dist = caldist_Ed(X[Ci[i]], X[Cj[j]])
            if dis_max < dist:
                dis_max = dist
    return dis_max



def plot(data, C, title):
    Color = ['b','c','g','k','m','r','y']
    Marker = ['*','o','v','^','p','+']
    K = dict()
    m = data.shape[0]

    for i in range(m):
        K.setdefault(C[i], [])
        K[C[i]].append(i)
    for key, value in K.items():
        plt.scatter(data[value,0], data[value, 1], marker=Marker[key % 6], c=Color[key % 7])
    plt.xlim([0,1.0])
    plt.ylim([0,1.0])
    plt.xlabel('Density')
    plt.ylabel('Sugar content')
    plt.title(title)
    plt.show()

    
def calMetric_JC(A, B):
    m = len(A)
    a = 0
    b = 0
    c = 0
    d = 0
    for i in range(m):
        for j in range(i+1, m):
            if A[i] == A[j]:
                if B[i] == B[j]:
                    a += 1
                else:
                    b += 1
            else:
                if B[i] == B[j]:
                    c += 1
                else:
                    d += 1
    JC = a / (a + b + c)
    return JC


def calMetric_FMI(A, B):
    m = len(A)
    a = 0
    b = 0
    c = 0
    d = 0
    for i in range(m):
        for j in range(i+1, m):
            if A[i] == A[j]:
                if B[i] == B[j]:
                    a += 1
                else:
                    b += 1
            else:
                if B[i] == B[j]:
                    c += 1
                else:
                    d += 1  
    FMI = np.sqrt( a / (a + b) * (a) / (a + c))
    return FMI

def calMetric_RI(A, B):
    m = len(A)
    a = 0
    b = 0
    c = 0
    d = 0
    for i in range(m):
        for j in range(i+1, m):
            if A[i] == A[j]:
                if B[i] == B[j]:
                    a += 1
                else:
                    b += 1
            else:
                if B[i] == B[j]:
                    c += 1
                else:
                    d += 1  
    RI = 2 * (a + b) / (m * (m -1))
    return RI



def calMetric_DBI(X, C):
    C_bak = C
    C = dict()

    for i in range(len(C_bak)):
        C.setdefault(C_bak[i], [])
        C[C_bak[i]].append(i)
    
    K = len(C.keys())
    DBI = 0

    for i in range(K):
        temp_max = 0
        for j in range(K):
            if j != i:
                temp = (cal_avg(X, C[i]) + cal_avg(X, C[j]))/ cal_cen(X, C[i], C[j])
                if temp > temp_max:
                    temp_max = temp
        DBI += temp_max
    DBI /= K
    return DBI

def calMetric_DI(X, C):
    C_bak = C
    C = dict()

    for i in range(len(C_bak)):
        C.setdefault(C_bak[i], [])
        C[C_bak[i]].append(i)
    
    K = len(C.keys())

    max_diam = cal_diam(X, C[0])
    for l in range(K):
        diam = cal_diam(X, C[l])
        if max_diam < diam:
            max_diam = diam
    min_dist_min = cal_min(X, C[0], C[1])
    for i in range(K):
        for j in range(K):
            if j != i:
                dist_min = cal_min(X, C[i], C[j])
                if min_dist_min > dist_min:
                    min_dist_min = dist_min
    DI = min_dist_min / max_diam
    return DI





def cal_avg(X, C):
    l = len(C)
    temp = 0
    for i in range(l):
        for j in range(i+1, l):
            temp += caldist_Ed(X[C[i]], X[C[j]])
    avg = 2 / (l * (l - 1)) * temp
    return avg

def cal_diam(X, C):
    l = len(C)
    dis_max = caldist_Ed(X[C[0]], X[C[1]])
    for i in range(l):
        for j in range(i+1, l):
            dist = caldist_Ed(X[C[i]], X[C[j]])
            if dis_max < dist:
                dis_max = dist
    return dis_max

def cal_min(X, A, B):
    dis_min = caldist_Ed(X[A[0]], X[B[0]])
    for a in A:
        for b in B:
            dist = caldist_Ed(X[a], X[b])
            if dis_min > dist:
                dis_min = dist
    return dis_min

def cal_cen(X, A, B):
    m, n = X.shape
    mu_a = np.zeros(n)
    mu_b = np.zeros(n)
    for a, b in zip(A,B):
        mu_a += X[a]
        mu_b += X[b]
    mu_a /= len(A)
    mu_b /= len(B)
    dis_cen = caldist_Ed(mu_a, mu_b)
    return dis_cen
    




def caldist_Ed(a, b):
    temp = a-b
    dist = np.sum(temp @ temp.T)
    dist_ed = np.sqrt(dist)
    return dist_ed



data = pd.read_csv("d:/MachineLearning/ML/wBook/charpter9/西瓜数据集4.0.txt")
Tag = np.zeros(30, dtype=int)
Tag[8:21] = 1


X = np.array(data)
C = [list() for i in range(4)]
Title = ["k_means","LVQ", "Gaussian", "hierarchical"]

plot(X, Tag, "orignal")

C[0] = k_means(X, 2)
plot(X, C[0], Title[0])
p,T = LVQ(X, Tag, 10, 0.8, 30)
C[1] = classifyAfterLVQ(X, p, T)
plot(X, C[1], Title[1])
C[2] = MixGauss(X, 2, 10)
plot(X, C[2], Title[2])

C[3] = hierarchical(X, 2)
plot(X, C[3], Title[3])

for c, title in zip(C, Title):
    scores = calMetric_JC(c, Tag)
    print(title, "JC:", scores)
    scores = calMetric_FMI(c, Tag)
    print(title, "FMI:", scores)
    scores = calMetric_RI(c, Tag)
    print(title, "RI:", scores)
    scores = calMetric_DBI(X, c)
    print(title, "DBI:", scores)
    scores = calMetric_DI(X, c)
    print(title, "DI:", scores)