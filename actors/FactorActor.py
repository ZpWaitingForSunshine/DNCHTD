
import ray
import numpy as np
import time
import matplotlib.pyplot as plt

from utils.nonlocal_function import indices2Patch
from classes.Classes import Patch, Factor, SparseTensor
from utils.tensor_function import HT_recover, matricize, ktensor, ttm, calu3TTM, updateU1, updateU2, updateU3, updateU4, \
    updateB2, updateB1

from scipy.sparse import csr_matrix

# @ray.remote(num_cpus=2)
class FactorActor:
    def __init__(self, k1, Y, patsize, rows, cols, nn, rank):
        print("init factors")
        self.parDatalist = []
        # print(len(k1))
        patchList = [] #
        self.E_Img = np.zeros(nn)

        # self.M = np.zeros(nn)

        for i in range(len(k1)):
            patch = Patch(k1[i].astype(int), rank, 0)
            patchList.append(patch)

        for patch in patchList:
            ind = patch.Indices

            Ytt1 = indices2Patch(Y, ind, patsize, rows, cols)
            patch.addY2(np.linalg.norm(Ytt1))

            k = patch.Rank # rank

            R1, R2, R3, RB1 = k

            factor = Factor(R1, R2, R3, RB1)

            U1 = np.random.random([patsize * patsize, R1])
            diag_matrix = np.diag(np.reciprocal(np.sqrt(np.sum(U1 ** 2, axis=0))))
            U1 = np.dot(U1, diag_matrix)

            if R2 < len(ind):
                U2 = np.random.random([len(ind), R2])
            else:
                U2 = np.random.random([len(ind), len(ind)])
            diag_matrix = np.diag(np.reciprocal(np.sqrt(np.sum(U2 ** 2, axis=0))))
            U2 = np.dot(U2, diag_matrix)

            print(U2.shape)


            U3 = np.random.random([nn[2], R3])
            diag_matrix = np.diag(np.reciprocal(np.sqrt(np.sum(U3 ** 2, axis=0))))
            U3 = np.dot(U3, diag_matrix)

            # D
            U4 = np.random.random([Y.shape[2], R3])
            diag_matrix = np.diag(np.reciprocal(np.sqrt(np.sum(U4 ** 2, axis=0))))
            U4 = np.dot(U4, diag_matrix)


            # print(R1, len(ind))
            if R2 > len(ind):

                B1 = np.random.random([R1 * len(ind), RB1])
                diag_matrix = np.diag(np.reciprocal(np.sqrt(np.sum(B1 ** 2, axis=0))))
                B1 = np.dot(B1, diag_matrix)
                B1 = np.reshape(B1, (R1, len(ind), RB1))
                print(B1.shape)
            else:
                B1 = np.random.random([R1 * R2, RB1])
                diag_matrix = np.diag(np.reciprocal(np.sqrt(np.sum(B1 ** 2, axis=0))))
                B1 = np.dot(B1, diag_matrix)
                B1 = np.reshape(B1, (R1, R2, RB1))

            B2 = np.random.random([RB1, R3])
            diag_matrix = np.diag(np.reciprocal(np.sqrt(np.sum(B2 ** 2, axis=0))))
            B2 = np.dot(B2, diag_matrix)

            M2 = np.zeros(U4.shape)

            factor.setFactors(U1, U2, U3, U4, B1, B2, M2)

            patch.addFactor(factor)
            patch.addLast(10000)

            self.parDatalist.append(patch)

    def getparDatalist(self):
        return self.parDatalist

    def updateFactors(self, X, patsize, rows, cols, M, Y, lda, mu, R, nn):
        E_Img = np.zeros(nn)
        time_start = time.time()
        patchlist = self.parDatalist
        print("开始更新因子矩阵")
        curPatchlist = []
        theta = 0.0001
        for patch in patchlist:
            ind = patch.Indices
            # print(ind)
            # print(X.shape)
            tt1 = indices2Patch(X, ind, patsize, rows, cols)
            tt1 = np.transpose(tt1, (0, 2, 1))
            SO1 = matricize(tt1)

            Ytt1 = indices2Patch(Y, ind, patsize, rows, cols)
            Ytt1 = np.transpose(Ytt1, (0, 2, 1))
            YSO1 = matricize(Ytt1)

            Mtt1 = indices2Patch(M, ind, patsize, rows, cols)
            Mtt1 = np.transpose(Mtt1, (0, 2, 1))
            MM2 = matricize(Mtt1)

            indices = patch.Indices

            # 新更新
            # U1

            U1 = patch.factor.U1
            U2 = patch.factor.U2
            U3 = patch.factor.U3
            U4 = patch.factor.U4
            B1 = patch.factor.B1
            B2 = patch.factor.B2
            M2 = patch.factor.M2

            cur = HT_recover(U1, U2, U4, B1, B2)
            print(np.linalg.norm(cur - Ytt1))

            B2 = updateB2(U1, U2, U3, U4, B1, B2, YSO1[2], SO1[2], MM2[2], mu, lda, theta)
            cur = HT_recover(U1, U2, U4, B1, B2)
            print("Y", np.linalg.norm(cur - Ytt1))
            cur = HT_recover(U1, U2, U3, B1, B2)
            print("ZX", np.linalg.norm(cur - tt1))

            # B2[B2 < 0] = 0
            B1 = updateB1(U1, U2, U3, U4, B1, B2, Ytt1, tt1, Mtt1, mu, lda, theta)
            cur = HT_recover(U1, U2, U4, B1, B2)
            print(np.linalg.norm(cur - Ytt1))
            # B1[B1 < 0] = 0

            U2 = updateU2(U1, U2, U3, U4, B1, B2, YSO1[1], SO1[1], MM2[1], mu, lda, theta)
            cur = HT_recover(U1, U2, U4, B1, B2)
            print(np.linalg.norm(cur - Ytt1))


            U1 = updateU1(U1, U2, U3, U4, B1, B2, YSO1[0], SO1[0], MM2[0], mu, lda, theta)
            cur = HT_recover(U1, U2, U4, B1, B2)
            print(np.linalg.norm(cur - Ytt1))

            # U2


            U3 = updateU3(U1, U2, U3, U4, B1, B2, SO1[2], MM2[2], mu, theta, R, M2)
            cur = HT_recover(U1, U2, U4, B1, B2)
            print(np.linalg.norm(cur - Ytt1))

            U4 = updateU4(U1, U2, U3, U4, B1, B2, YSO1[2], mu, theta, lda, R, M2)
            cur = HT_recover(U1, U2, U4, B1, B2)
            print(np.linalg.norm(cur - Ytt1))









            print()

            M2 = M2 + mu * (U4 - np.dot(R, U3))


            patch.factor.setFactors(U1, U2, U3, U4, B1, B2, M2)
            patches = HT_recover(U1, U2, U3, B1, B2)
            patches = np.transpose(patches, (0, 2, 1))
            # patches = np.transpose(tt1, (0, 2, 1))
            for ind_cur, index in enumerate(indices):
                row = int(index % rows)
                col = int((index - row) / rows)
                # if(row == 0 and col == 0):
                #     print(patches[:, :, 1, ind_cur])
                E_Img[row: patsize + row, col: patsize + col, :] = \
                    E_Img[row: patsize + row, col: patsize + col, :] + \
                    np.reshape(patches[:, :, ind_cur], [patsize, patsize, nn[2]])

            curPatchlist.append(patch)
            # print("----end---")
        # print(len(curPatchlist))
        time_end = time.time()
        print("upate 分区更新完成，用时%f秒" % (time_end - time_start))

        # plt.imshow(E_Img[:, :, 1:4])
        # plt.show()
        # sparse

        sparseTensor = SparseTensor()
        #
        data = []
        # csr_first = csr_matrix(E_Img[:, :, 0])
        #
        for i in range(nn[2]):
            print(E_Img[:, :, i].shape)
            csr = csr_matrix(E_Img[:, :, i])
            # csr_M =
            sparseTensor.addIndices(csr.indices)
            sparseTensor.addOffset(csr.indptr)
            data.append(csr.data)


        sparseTensor.addData(data)
        #

        # print("零占比：", 1 - np.count_nonzero(E_Img[:, :, 0]) / nn[0] / nn[1])

        # self.E_Img = E_Img
        # return E_Img
        # self.E_Img = sparseTensor
        self.E_Img = E_Img
        return E_Img
    def reduce(self, data):
        self.E_Img = self.E_Img + data


    # def updateEimg(self, *parameterServer):
    #     parameterServer.put_HR_HSI.remote(self.E_Img)


    def getEimg(self):
        return self.E_Img
