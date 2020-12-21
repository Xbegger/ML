import numpy as np
import pandas as pd
import matplotlib as plt

class ID3_DecisionTree():
    
    class Node():
        def __init__(self):
            self.next = {}
            self.is_genre = False

        def add_next_node(self, next_node, value, name):
            self.next_name = name
            self.next[value] = next_node
        def add_genre(self, genre):
            self.genre = int(genre)


    def __init__(self):
        self.head = ID3_DecisionTree.Node()
 

    def TreeGenerate(self, dataSet, A, now_node):
        #生成结点
        Y = dataSet[yTag]
        
        C = np.unique(Y)
        if C.shape[0] == 1:
            #标记为C类叶节点
            now_node.add_genre(C)
            now_node.is_genre = True
            return
        mostGenre = self.findMostGenre(Y)
        if not A or self.checkDataSame(dataSet, A):
            # 将node标记为叶节点，类别标记为D中样本数最多的类
            now_node.add_genre(mostGenre)
            now_node.is_genre = True
            return
        now_node.add_genre(mostGenre)
        # 从A中选择最优划分属性
        a = self.chooseBestAttri(dataSet, A)
        D_a = dataSet[a][:]
        tA = A.copy()
        tA.remove(a)
        
        for a_v in np.unique(D_a):
            # 为node生成分支
            D_a_v = dataSet[dataSet[a].isin([a_v])][:]
            if D_a_v.empty == True:
                # 将分支结点标记为叶节点，类别标记为D中样本最多的类
                now_node.is_genre = True
                now_node.genre = int(mostGenre)
                return
            else:
                next_node = self.Node()
                now_node.add_next_node(next_node, a_v, a)
                self.TreeGenerate(D_a_v, tA, next_node)
        return node
    
    def preTreeGenerate(self, dataSet, validationSet, A, now_node):
        #生成结点
        Y = dataSet[yTag]
        
        C = np.unique(Y)
        if C.shape[0] == 1:
            #标记为C类叶节点
            now_node.add_genre(C)
            now_node.is_genre = True
            return
        mostGenre = self.findMostGenre(Y)
        if A == None or self.checkDataSame(dataSet, A):
            # 将node标记为叶节点，类别标记为D中样本数最多的类
            now_node.add_genre(mostGenre)
            now_node.is_genre = True
            return
        now_node.add_genre(mostGenre)
        # 从A中选择最优划分属性
        a = self.chooseBestAttri(dataSet, A)
        D_a = dataSet[a][:]
        tA = A.copy()
        tA.remove(a)
        
        # 预剪枝
        # 验证集在当前决策树的分类结果
        old_cnt = 0
        Y_validationSet = validationSet[yTag]
        old_cnt = np.sum(Y_validationSet == mostGenre)
        # 验证集在属性a 分类下的分类结果
        validationSet_a = validationSet[a][:]
        new_cnt = 0
        for a_v in np.unique(D_a):
            # 为node生成分支
            D_a_v = dataSet[dataSet[a].isin([a_v])][:]
            Y_a_v = D_a_v[yTag]
            validationSetD_a_v = validationSet[validationSet[a].isin([a_v])][:]
            Y_a_v_validation = validationSetD_a_v[yTag]
            genreOfa_v = self.findMostGenre(Y_a_v)
            new_cnt += np.sum(Y_a_v_validation == genreOfa_v)
        if old_cnt > new_cnt:
            # 设置当前结点为叶节点
            now_node.add_genre(mostGenre)
            now_node.is_genre = True
            return 
        
        for a_v in np.unique(D_a):
            # 为node生成分支
            D_a_v = dataSet[dataSet[a].isin([a_v])][:]
            
            validationSetD_a_v = validationSet[validationSet[a].isin([a_v])][:]
            
            if D_a_v.empty == True:
                # 将分支结点标记为叶节点，类别标记为D中样本最多的类
                now_node.is_genre = True
                now_node.genre = int(mostGenre)

                return
            else:
                next_node = self.Node()
                now_node.add_next_node(next_node, a_v, a)
                self.preTreeGenerate(D_a_v,validationSetD_a_v, tA, next_node)
        

    def postpruning(self, validationSet, now_node):
        if now_node.is_genre == True:
            return
        
        # 将验证集分配到不同结点
        validationSet_a = validationSet[now_node.next_name]
        for a_v in now_node.next.keys():
            validationSet_a_v = validationSet[validationSet_a.isin([a_v])]
            # 不同结点递归调用本函数    
            self.postpruning(validationSet_a_v, now_node.next[a_v])
        # 结点判断是否剪枝
        if now_node.is_genre == False:
            Y = validationSet[yTag]
            old_genre = now_node.genre
            old_acc = np.sum(Y == old_genre)
            new_acc = 0
            for a_v, node in now_node.next.items():
                validationSet_a_v = validationSet[validationSet_a.isin([a_v])]
                Y_validationSet_a_v = validationSet_a_v[yTag]
                new_genre = node.genre
                new_acc += np.sum(Y_validationSet_a_v == new_genre)
            if old_acc > new_acc:
                now_node.is_genre = True

    # 计算准确率
    def calAccuItems(self, dataSet, genre):
        count = 0
        for value in dataSet:
            if value == genre:
                count += 1
        return count


    #从A中选择最优划分属性
    def chooseBestAttri(self, dataSet, A):
        tGain = -1
    
        for a in A:
            gain = self.calInfoGain(dataSet, a)            
            if gain > tGain:
                tGain = gain
                tAtrri = a
        return tAtrri

    # 判断数据集D中样本在A上取值是否相同
    def checkDataSame(self, dataSet, A):
        X = dataSet.iloc[:, :dataSet.shape[1] - 1]
        for a in A:
            length = len(np.unique(X[a]))
            if length > 1:
                return False
        return True

    # 找到数据集中类别最多的类
    def findMostGenre(self, Y):
        sum_Y = dict()
        temp = 0
        for y in Y.values:
            sum_Y.setdefault(y, 0)
            sum_Y[y] += 1
        
        for key, value in sum_Y.items():
            if value > temp:
                temp = value
                ans = key
        return ans

    def calInfoEntropy(self, Y):
        ans = 0
        sizeOfdata = Y.shape[0]

        sum_Y = dict()
        for y in Y.values:
            sum_Y.setdefault(y, 0)
            sum_Y[y] += 1
        for k in sum_Y.keys():
            p_k = sum_Y[k] / sizeOfdata
            ans += p_k * np.log2(p_k)
        return round(-ans, 3)

    def calInfoGain(self, data, a):
        Y = data[yTag][:]
        ent = self.calInfoEntropy(Y)
        D_a = data[a]

        ans = 0
        for v in np.unique(D_a):
            D_v = data[yTag][data[a].isin([v])]
            entD_v = self.calInfoEntropy(D_v)
            ans += D_v.shape[0] / data.shape[0] * entD_v
        
        ans =  round(ent -ans, 3)
        return ans

    def predict(self, X):
        y = []
        for i in range(X.shape[0]):
            temp = self.head
            x = X.iloc[i:i+1,:]
            while not temp.is_genre:
                key = x[temp.next_name].tolist()[0]
                if key in temp.next.keys():
                    temp = temp.next[key]
                else:
                    break
            y.append(temp.genre)
        return np.array(y)

def printTree(node, pre=''):
    pre += '   '
    if node.is_genre == False:
        print(pre, node.next_name)
    else:
        if node.genre == 1:
            print(pre + '好瓜')
        else:
            print(pre + '坏瓜')
    pre += '   '
    for t in node.next.keys():
        print(pre + '--' + str(t))
        if node.is_genre == False:
            printTree(node.next[t], pre)
        
def testGain(dataSet, tree):
    for a in dataSet.keys():
        print(a, tree.calInfoGain(dataSet, a))


def watermelonData():
    path = 'd:/MachineLearning/ML/wBook/charpter4/西瓜数据集3.0.txt'
    data = pd.read_csv(path)
    data['Good melon'][data['Good melon'].isin(['是'])] = 1
    data['Good melon'][data['Good melon'].isin(['否'])] = 0
    del data['Sugar content']
    del data['density']
    train_index = [1,2,3,6,7,10,14,15,16,17]
    test_index = [4,5,8,9,11,12,13]
    
    train = data.loc[train_index]
    test = data.loc[test_index]
    A = data.keys()[0:data.shape[1]-1].tolist()
    yTag = data.keys()[data.shape[1]-1]
    return train, test, A, yTag

def uciData():
    path = 'd:/MachineLearning/ML/wBook/charpter4/cmc.data'
    data = pd.read_csv(path, names=np.arange(10))
    data = data.drop(labels=[0, 3], axis=1)
    test_index = int(data.shape[0] * 0.25)
    test = data[0:test_index]
    train = data[test_index:]
    A = data.keys()[0:data.shape[1]-1].tolist()
    yTag = data.keys()[data.shape[1]-1]
    return train, test, A, yTag

train, test, A, yTag = uciData()
# train, test, A, yTag = watermelonData()

test_X = test.iloc[:, :test.shape[1]-1]
test_Y = test[yTag]


tree = ID3_DecisionTree()
node = ID3_DecisionTree.Node()



# X = test.iloc[:,:test.shape[1]-1]
# Y = test.iloc[:,test.shape[1]-1:]
# y_pre = tree.predict(X, Y)

tree.TreeGenerate(train, A, tree.head)
# printTree(tree.head)
print("未剪枝：" + str(np.sum(tree.predict(test_X) == test_Y )/ test_Y.shape[0]))
tree.postpruning(test, tree.head)
# printTree(tree.head)
print("后剪枝：" + str(np.sum(tree.predict(test_X) == test_Y )/ test_Y.shape[0]))

tree.preTreeGenerate(train, test, A, tree.head)
# printTree(tree.head)
print("预剪枝：" + str(np.sum(tree.predict(test_X) == test_Y )/ test_Y.shape[0]))

