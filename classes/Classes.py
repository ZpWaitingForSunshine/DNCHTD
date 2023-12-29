class Factor:
    def __init__(self, U1, U2, U3, U4):
        self.U1 = U1
        self.U2 = U2
        self.U3 = U3
        self.U4 = U4
        # self.M2 = M2


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

    def addY2(self, Y2):
        self.Y2 = Y2

    def addZ2(self, Z2):
        self.Z2 = Z2

    def addLast(self, err):
        self.lasterr = err


class SparseTensor:
    def addOffset(self, row_offset):
        self.row_offset = row_offset

    def addIndices(self, col_indices):
        self.col_indices = col_indices

    def addData(self, data):
        self.data = data

class Parameters:
    def __init__(self):
        self.mu = 1e-4
        # % parameter
        self.tol = 1e-2
        self.lda = 100
        self.maxIter = 10
        self.minIter = 1

        self.patsize = 5 #
        self.Pstep = 1

        self.Vpatsize = 5
        self.VPstep = 1

        self.PN = 300
        self.Rank = 1
        self.ratio = 5

        self.KK = 80









