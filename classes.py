# 创建一个名为Person的结构体
class Factor:
    def __init__(self, U1, U2, U3, U4, D, M2):
        self.U1 = U1
        self.U2 = U2
        self.U3 = U3
        self.U4 = U4
        self.D = D
        self.M2 = M2


class Patch:
    def __init__(self, Indices, Rank, time):
        self.Indices = Indices
        self.Rank = Rank
        self.time = time

    def setPara(self, patsize, Pstep, nn):
        self.patsize = patsize
        self.Pstep = Pstep
        self.nn = nn

    def addY(self, y):
        self.curY = y

    def addM2(self, M2):
        self.M2 = M2

    def addX(self, X):
        self.X = X

    def addFactor(self, factor):
        self.factor = factor
