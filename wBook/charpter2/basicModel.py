class BasicModel():
    def __init__(self, trainSet, testSet, learner):
        # 初始化模型
        # 初始化训练集 训练集[[特征], [标签]]
        self.trainSet = trainSet
        # 验证集
        self.validationSet = None
        # 初始化测试集
        self.testSet = testSet
        
        # 初始化学习器 一个学习器类
        self.learner = learner
