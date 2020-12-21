import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets.samples_generator import make_blobs
from scipy.spatial.distance import cdist



def pca(dataMat, percentage=0.9, n_components=0):
    meanVals = np.mean(dataMat, axis=0)
    meanReoved = dataMat - meanVals
    covMat = np.cov(meanReoved, rowvar=0)

    # calcute eig value and eig vector
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))

    # cal the num of Principal Component
    if n_components == 0:
        k = eigValPct(eigVals, percentage)
    else:
        k = n_components
    eigVallnd = np.argsort(eigVals)
    eigVallnd = eigVallnd[:-(k+1):-1]

    redEigVects = eigVects[:,eigVallnd]

    lowDDataMat = meanReoved * redEigVects
    
    reconMat = (lowDDataMat *  redEigVects.T) + meanVals
    return lowDDataMat, reconMat

def eigValPct(eigVals, percentage):
    sortArray = np.sort(eigVals)
    sortArray = sortArray[-1::-1]
    arraySum = np.sum(sortArray)
    tempSum = 0
    num = 0
    for i in sortArray:
        tempSum += i
        num += 1
        if tempSum >= arraySum * percentage:
            return num

def mds(dataMat, lowD):
    m = dataMat.shape[0]
    dist = cdist(dataMat, dataMat)
    dist_square = dist * dist
    i_dist_square = np.sum(dist_square, axis=1) / m
    j_dist_square = np.sum(dist_square, axis=0) / m
    sum_dist_square = np.sum(dist_square) / (m * m)
    B = -1/2 * (dist_square - i_dist_square - j_dist_square + sum_dist_square)

    eigVals, eigVects = np.linalg.eig(np.mat(B))

    k = lowD

    eigVallnd = np.argsort(eigVals)
    eigVallnd = eigVallnd[:-(k+1):-1]

    redEigVects = eigVects[:,eigVallnd]
    diag = np.diag(eigVals[eigVallnd])
    lowDDataMat = (diag**(1/2)) * redEigVects.T

    return lowDDataMat.T


def plt3D(X):
    fig = plt.figure()
    plt.scatter(X[:,0], X[:,1], marker='o')
    plt.show()

# X -- sample feature;y -- sample class
X,Y = make_blobs(n_samples=1000,
                 n_features=3, 
                 centers=[[3,3,3],[0,0,0],[1,1,1],[2,2,2]],
                 cluster_std=[0.2, 0.1, 0.2, 0.2],
                 random_state=9)
fig = plt.figure()
ax = Axes3D(fig, rect=[0,0,1,1], elev=30, azim=20)
plt.scatter(X[:,0], X[:,1], X[:,2], marker='o')
plt.show()

lowDDataMat, reconMat = pca(X, n_components=2)
newX =  np.array(lowDDataMat)
plt3D(newX)

lowDDataMat = mds(X, 2)
newX =  np.array(lowDDataMat)
plt3D(newX)


