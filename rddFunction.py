import time

import h5py
from pyspark import SparkConf, SparkContext, broadcast
import numpy as np
import sys
from function import knn, split_average, ktensor, cgsolve2
from classes import Patch, Factor
from function import indices2Patch, matricize
import scipy.linalg as sl
from scipy.linalg import solve_sylvester
def group(sc: SparkContext, nn, patsize, PN, num, Y_broadcast):
    rows = nn[0] - patsize + 1
    cols = nn[1] - patsize + 1
    t_start = time.time()
    # record the type
    indices = np.zeros((2, rows * cols)).astype('int')
    indices[0] = np.arange(rows * cols)
    indices[1] = sys.maxsize

    groups_edges = split_average(indices[0, :], PN, num)
    indices_set = []
    for i in range(num - 1):
        indices_split_arrays = np.array_split(indices, num, axis=1)

        indices_rdd = sc.parallelize(indices_split_arrays, 4)

        indices_list = indices_rdd.map(lambda x: knn(x, rows, cols, patsize, indices[0][0], Y_broadcast)).collect()

        # for k in len(indices_list):
        indices = np.concatenate(indices_list, axis=1)

        min_indices = find_min_indices(indices, len(groups_edges[i]))
        indices_set.append(indices[:, min_indices])

        indices = np.delete(indices, min_indices, axis=1)

    indices_set.append(indices)

    groupsRDD = sc.parallelize(indices_set, num)

    groups = groupsRDD.flatMap(lambda x: partitionGroup(x, rows, cols, patsize, Y_broadcast, PN)).collect()
    t_end = time.time()
    print("分组，用时%f秒" % (t_end - t_start))
    return groups


def partitionGroup(indices, rows, cols, patsize, Y_broadcast, PN):
    indices[1] = sys.maxsize
    num = int(np.ceil(indices.shape[1] / PN))
    indices_set = []
    for i in range(num - 1):
        indices = knn(indices, rows, cols, patsize, indices[0][0], Y_broadcast)
        min_indices = find_min_indices(indices, PN)
        indices_set.append(indices[:, min_indices][0])
        indices = np.delete(indices, min_indices, axis=1)
    indices_set.append(indices[0])
    return indices_set


def find_min_indices(arr, N):
    # 获取第二行的数据
    row = arr[1]

    # 使用argsort函数对第二行进行排序，并获取排序后的索引
    sorted_indices = np.argsort(row)

    # 取前N个最小值的索引
    min_indices = sorted_indices[:N]

    return min_indices


def initFactor(patchList, Y_broadcast: broadcast, patsize, Pstepsize,
               rows, cols, X_broadcast: broadcast, M2_broadcast: broadcast, nn):
    Y = Y_broadcast.value
    print("init factors")
    t1 = time.time()
    parDatalist = []
    for patch in patchList:
        ind = patch.Indices
        k = patch.Rank
        U1 = np.random.random([patsize, k])
        diag_matrix = np.diag(np.reciprocal(np.sqrt(np.sum(U1 ** 2, axis=0))))
        U1 = np.dot(U1, diag_matrix)

        U2 = np.random.random([patsize, k])
        diag_matrix = np.diag(np.reciprocal(np.sqrt(np.sum(U2 ** 2, axis=0))))
        U2 = np.dot(U2, diag_matrix)

        U3 = np.random.random([nn[2], k])
        diag_matrix = np.diag(np.reciprocal(np.sqrt(np.sum(U3 ** 2, axis=0))))
        U3 = np.dot(U3, diag_matrix)

        U4 = np.random.random([len(ind), k])
        diag_matrix = np.diag(np.reciprocal(np.sqrt(np.sum(U4 ** 2, axis=0))))
        U4 = np.dot(U4, diag_matrix)

        D = np.random.random([Y.shape[2], k])
        diag_matrix = np.diag(np.reciprocal(np.sqrt(np.sum(D ** 2, axis=0))))
        D = np.dot(D, diag_matrix)






        M2 = np.zeros([Y.shape[2], k])
        factor = Factor(U1, U2, U3, U4, D, M2)
        patch.addFactor(factor)
        parDatalist.append(patch)
    t2 = time.time()
    print("初始化，用时%f秒" % (t2 - t1))
    return parDatalist



def updateFactors(patchlist, X_broadcast: broadcast, patsize,
                  rows, cols, M2_broadcast: broadcast, Y_broadcast: broadcast, lda, mu, R):
    X = X_broadcast.value
    Y = Y_broadcast.value
    time_start = time.time()
    print("开始更新因子矩阵")
    M2 = M2_broadcast.value

    # patch.addM2(Mtt1)
    curPatchlist = []
    for patch in patchlist:
        ind = patch.Indices
        tt1 = indices2Patch(X, ind, patsize, rows, cols)
        SO1 = matricize(tt1)

        Ytt1 = indices2Patch(Y, ind, patsize, rows, cols)
        YSO1 = matricize(Ytt1)

        Mtt1 = indices2Patch(M2, ind, patsize, rows, cols)
        MM2 = matricize(Mtt1)

        # U1
        W1 = sl.khatri_rao(patch.factor.U4, patch.factor.U3)
        W1 = sl.khatri_rao(W1, patch.factor.U2)
        P1 = sl.khatri_rao(patch.factor.U4, patch.factor.D)
        P1 = sl.khatri_rao(P1, patch.factor.U2)

        G1 = np.dot(patch.factor.U4.T, patch.factor.U4) * np.dot(patch.factor.U3.T, patch.factor.U3) \
             * np.dot(patch.factor.U2.T, patch.factor.U2)
        PTP = np.dot(patch.factor.U4.T, patch.factor.U4) * np.dot(patch.factor.D.T, patch.factor.D) \
              * np.dot(patch.factor.U2.T, patch.factor.U2)
        t1 = mu * G1 + 2 * lda * PTP
        # np.dot(patch.M2[0].T, W1)
        patch.factor.U1 = np.dot(np.dot(MM2[0].T, W1) + 2 * lda * np.dot(YSO1[0].T, P1) +
                                 mu * np.dot(SO1[0].T, W1), np.linalg.inv(t1))

        # U2
        W2 = sl.khatri_rao(patch.factor.U4, patch.factor.U3)
        W2 = sl.khatri_rao(W2, patch.factor.U1)
        P2 = sl.khatri_rao(patch.factor.U4, patch.factor.D)
        P2 = sl.khatri_rao(P2, patch.factor.U1)
        G2 = np.dot(patch.factor.U4.T, patch.factor.U4) * np.dot(patch.factor.U3.T, patch.factor.U3) \
             * np.dot(patch.factor.U1.T, patch.factor.U1)
        P2TP2 = np.dot(patch.factor.U4.T, patch.factor.U4) * np.dot(patch.factor.D.T, patch.factor.D) \
                * np.dot(patch.factor.U1.T, patch.factor.U1)
        t2 = mu * G2 + 2 * lda * P2TP2
        patch.factor.U2 = np.dot(np.dot(MM2[1].T, W2) + 2 * lda * np.dot(YSO1[1].T, P2) + mu * np.dot(SO1[1].T, W2),
                                 np.linalg.inv(t2))

        # U3
        W3 = sl.khatri_rao(patch.factor.U4, patch.factor.U2)
        W3 = sl.khatri_rao(W3, patch.factor.U1)
        leftA = np.dot(R.T, R)
        rightA = np.dot(patch.factor.U4.T, patch.factor.U4) * np.dot(patch.factor.U2.T, patch.factor.U2) \
                 * np.dot(patch.factor.U1.T, patch.factor.U1)
        rightEqu = np.dot(SO1[2].T, W3) + np.dot(MM2[2].T, W3) / mu + np.dot(R.T, patch.factor.D
                                                                             + patch.factor.M2 / mu)
        patch.factor.U3 = solve_sylvester(leftA, rightA, rightEqu)

        # r_loss = loss(HH[id].U1, HH[id].U2, HH[id].U3, HH[id].D, Ytt1[id], lda, tt1, mu, Mtt1)
        # print(r_loss)

        t5 = 2 * lda * np.dot(W3.T, W3) + mu * np.eye(W3.shape[1])
        patch.factor.D = np.dot(2 * lda * np.dot(YSO1[2].T, W3) + mu * np.dot(R, patch.factor.U3)
                                - patch.factor.M2, np.linalg.inv(t5))

        W4 = sl.khatri_rao(patch.factor.U3, patch.factor.U2)
        W4 = sl.khatri_rao(W4, patch.factor.U1)
        P4 = sl.khatri_rao(patch.factor.D, patch.factor.U2)
        P4 = sl.khatri_rao(P4, patch.factor.U1)
        G4 = np.dot(patch.factor.U3.T, patch.factor.U3) * np.dot(patch.factor.U2.T, patch.factor.U2) \
             * np.dot(patch.factor.U1.T, patch.factor.U1)
        P4TP4 = np.dot(patch.factor.D.T, patch.factor.D) * np.dot(patch.factor.U2.T, patch.factor.U2) \
                * np.dot(patch.factor.U1.T, patch.factor.U1)
        t4 = mu * G4 + 2 * lda * P4TP4
        patch.factor.U4 = np.dot(np.dot(MM2[3].T, W4) + 2 * lda * np.dot(YSO1[3].T, P4)
                                 + mu * np.dot(SO1[3].T, W4), np.linalg.inv(t4))

        # r_loss = loss(HH[id].U1, HH[id].U2, HH[id].U3, HH[id].D, Ytt1[id], lda, tt1, mu, Mtt1)
        # print(r_loss)
        # tempPatch[id] = ktensor([HH[id].U1, HH[id].U2, HH[id].U3, HH[id].U4])
        # tempPatch[id] = np.reshape(tempPatch[id], [patsize * patsize, tempPatch[id].shape[2], tempPatch[id].shape[3]], order='F')
        patch.factor.M2 = patch.factor.M2 + mu * (patch.factor.D - np.dot(R, patch.factor.U3))

        curPatchlist.append(patch)
    # print(len(curPatchlist))
    time_end = time.time()
    print("upate 分区更新完成，用时%f秒" % (time_end - time_start))
    return curPatchlist


def X_fold(factors_set, nn, rows, cols, patsize):
    # print('collect')
    print("X_fold start")
    t_start = time.time()
    E_Img = np.zeros(nn)
    for patch in factors_set:
        indices = patch.Indices
        factor = patch.factor
        patches = ktensor([factor.U2, factor.U1, factor.U3, factor.U4])
        for ind_cur, index in enumerate(indices):
            row = int(index % rows)
            col = int((index - row) / rows)
            # if(row == 0 and col == 0):
            #     print(patches[:, :, 1, ind_cur])
            E_Img[row: patsize + row, col: patsize + col, :] = \
                E_Img[row: patsize + row, col: patsize + col, :] + patches[:, :, :, ind_cur]
    # print(E_Img.shape)
    t_end = time.time()

    print("X_fold 分区更新完成，用时%f秒" % (t_end - t_start))

    return [E_Img]
def cg(rr_list, mu, rate, s):
    print("分区开始运行CG")
    t1 = time.time()
    maxtrix = []
    for rr in rr_list:
        maxtrix.append(rr.T)
    rr_matrix = np.stack(maxtrix, axis=2)
    nn = rr_matrix.shape
    rr = np.reshape(rr_matrix, [nn[0] * nn[1] * nn[2]], order='F')
    t2 = time.time()
    print("分区更新完成，用时%d秒" % (t2 - t1))
    return [cgsolve2(rr, nn, mu, rate, s)]