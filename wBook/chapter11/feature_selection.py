import numpy as np
import matplotlib.pyplot as plt
import random


def filter(X, y, k):
    m,n = X.shape

    theta = np.zeros([n,1])

    for i in range(m):
        x_i = X[i]
        y_i = y[i]
        # find x_nh and x_nm
        x_nh = -1
        nh_dist = 0
        x_nm = -1
        nm_dist = 0
        for a in range(n):
            for j in range(m):
                if j != i:
                    x_j = X[j]
                    y_j = y[j]
                    
                    dist = calDist(x_i[a], x_j[a])
                    if y_j == y_i:
                        if x_nh == -1:
                            x_nh = j
                            nh_dist = dist
                        else:
                            if nh_dist > dist:
                                nh = j
                                nh_dist =  dist
                    else:
                        if x_nm == -1:
                            x_nm = j
                            nm_dist = dist
                        else:
                            if nm_dist > dist:
                                nm = j
                                nm_dist =  dist
        theta[a] += - nh_dist**2 + nm_dist**2
    ans = np.argsort(theta)
    ans = ans[0:k:1]
    return ans


def LVW(X, model, T):
    error = 100000000
    m,n = X.shape

    d = n
    t = 0
    A_orignal = [i for i in range(n)]
    while t < T:
        temp_A = random.sample(A_orignal, random.randint(0, n-1))
        temp_d = len(temp_A)
        model.fit(X[temp_A])
        temp_error = model.score(X[temp_A])
        if (error > temp_error) or (temp_error == error or temp_d < d):
            t = 0
            error = temp_error
            d = temp_d
            A = temp_A
        else:
            t += 1
    return A


def calDist(a, b, opt=1):
    if opt == 1:
        if a == b:
            return 0
        else:
            return 1
    else:
        return np.abs(a - b)
