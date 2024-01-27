import time

import pyspark
from pyspark import SparkConf, SparkContext
import numpy as np
import scipy.linalg as sl
from function import upsample, gaussian, graph, Im2Patch3D, matricize, ktensor, Patch2Im3D, cgsolve2, \
    knn, indices2Patch, getW_Imge_Matrix
import h5py
from rddFunction import group, updateFactors, X_fold, cg
from scipy.linalg import solve_sylvester
import sys
import matplotlib.pyplot as plt
from collections import namedtuple

import logging

from classes import Factor, Patch

from rddFunction import initFactor

logging.basicConfig(
    filemode='/data2/zp/nptcp.log',
    format='%(asctime)s - %(name)s - %(pathname)s - %(lineno)d - %(levelname)s - %(message)s'
)

# Factor = namedtuple('Factor', ['U1', 'U2', 'U3', 'D'])

# num is the number of partitions
def DP_NPTCP4(Ob, KK, Y, rate, PN, R, sc: SparkContext, num: int, s):

    print("start")

    time_nonlocal = 0
    time_updateX = 0
    time_group = 0

    t_start = time.time()
    logging.info("")
    max_HS = np.max(Ob)
    Ob = Ob / max_HS
    Y = Y / max_HS
    Ob = upsample(Ob, rate)

    # % parameter
    tol = 1e-2
    mu = 1e-4
    lda = 100
    maxIter = 10
    minIter = 1

    patsize = 5
    Pstep = 1

    # % initialize
    Z = Ob
    M1 = np.zeros(Z.shape)

    Ob = gaussian(Ob, s)

    Y_broadcast = sc.broadcast(Y)  # unfold Y
    X_broadcast = sc.broadcast(Ob)
    M1_broadcast = sc.broadcast(M1)

    nn = Ob.shape

    rows = nn[0] - patsize + 1
    cols = nn[1] - patsize + 1


    t1 = time.time()
    # group
    k1 = group(sc, nn, patsize, PN, num, Y_broadcast)
    t2 = time.time()

    time_group = t2 - t1

    patchList = []

    for i in range(len(k1)):
        patch = Patch(k1[i].astype(int), KK, 0)
        patchList.append(patch)

    indices_rdd = sc.parallelize(patchList, num)

    factor_rdd = indices_rdd.mapPartitions(lambda x: initFactor(x, Y_broadcast, patsize, patsize, rows, cols, X_broadcast, M1_broadcast, nn))\
        .cache()

    HT = Ob
    Z = Ob

    W_Img = getW_Imge_Matrix(nn, rows, cols, patsize)


    # patchlist, X_broadcast: broadcast, patsize,
    # rows, cols, M2_broadcast: broadcast, Y_broadcast: broadcast, lda, mu, R

    # res2 = updateFactors(factor_rdd, X_broadcast, patsize, Pstep, rows, cols, M2_broadcast, nn)
    for i in range(maxIter):
        HT_old = HT
        Z_old = Z


        t3 = time.time()
        factor_rdd = factor_rdd.mapPartitions(lambda x: updateFactors(x, X_broadcast, patsize,
                                                         rows, cols, M1_broadcast, Y_broadcast, lda, mu, R)).cache()
        HT = factor_rdd.mapPartitions(lambda x: X_fold(x, nn, rows, cols, patsize)).reduce(lambda x, y: x + y)
        for band in range(nn[2]):
            HT[:, :, band] = HT[:, :, band] / (W_Img + np.finfo(float).eps)

        t4 = time.time()

        time_nonlocal = time_nonlocal + t4 - t3

        # plt.imshow(HT[:, :, 1:4])
        # plt.show()


        t5 = time.time()

        rr = 2 * Ob + mu * HT - M1
        rr_rdd = sc.parallelize(rr.T, num)
        rr_rdd_set = rr_rdd.mapPartitions(lambda x: cg(x, mu, rate, s)).collect()
        Z = np.concatenate(rr_rdd_set, axis=2)

        t6 = time.time()

        time_updateX = time_updateX + t6 - t5

        M1 = M1 + mu * (Z - HT)
        mu = 1.01 * mu
        stopCond = np.linalg.norm(HT - HT_old) / np.linalg.norm(HT_old)
        stopCond2 = np.linalg.norm(Z - Z_old) / np.linalg.norm(Z_old)
        print('the %d iter\n', i)

        # plt.imshow(Z[:, :, 1:4])
        # plt.show()

        res1 = Z - HT
        RZ = Z * max_HS
        ReChange = np.linalg.norm(res1) / np.linalg.norm(Z)
        print('%10.5f\t%10.5f\t%10.5f\t\n', ReChange, stopCond, stopCond2)
        if ReChange < tol and stopCond < tol and stopCond2 < tol and i > (minIter - 1):
            break

        X_broadcast = sc.broadcast(Z)
        M1_broadcast = sc.broadcast(M1)

    Out = Z
    Out = Out * max_HS

    logging.info("group time: %ds\t, nonlocal time: %ds\t, updateX time: %ds", time_group, time_nonlocal, time_updateX )
    return Out
