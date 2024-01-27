import time

import numpy as np
import scipy.linalg as sl
from function import upsample, gaussian, graph, Im2Patch3D, matricize, ktensor, Patch2Im3D, cgsolve2, graphKNN
import h5py
from scipy.linalg import solve_sylvester
import matplotlib.pyplot as plt
from collections import namedtuple
# 创建一个名为Person的结构体
from PIL import Image
class Factor:
    def __init__(self, U1, U2, U3, U4, D):
        self.U1 = U1
        self.U2 = U2
        self.U3 = U3
        self.U4 = U4
        self.D = D


# Factor = namedtuple('Factor', ['U1', 'U2', 'U3', 'D'])

def NPTCP4(Ob, KK, Y, rate, PN, R, s):

    print("start")
    max_HS = np.max(Ob)
    Ob = Ob / max_HS
    Y = Y / max_HS
    Ob = upsample(Ob, rate)

    # % parameter
    tol = 1e-2
    mu = 1e-4
    lda = 100
    maxIter= 10
    minIter= 1

    nn = Ob.shape
    patsize = 5
    Pstep = 1
    rows = nn[0] - patsize + 1
    cols = nn[1] - patsize + 1
    # % initialize
    Z = Ob
    M1 = np.zeros(Z.shape)

    Ob = gaussian(Ob, s)

    # TY = cat(3, Y, Y(:,:, end));
    Npatch2 = Im2Patch3D(Y, patsize, Pstep)

    unfoldPatch = np.reshape(Npatch2, [int(Npatch2.shape[0] * Npatch2.shape[1]), Npatch2.shape[2]], order='F')

    k1 = graph(unfoldPatch, patsize, Pstep, nn, PN)
    #
    # image = Image.new("RGB", (rows, cols))
    # pixels = image.load()
    # pic = np.zeros([rows, cols])
    # for (index, group) in enumerate(k1):
    #     for i in group:
    #         row = int(i % rows)
    #         col = int((i - row) / rows)
    #         pixels[row, col] = (int(index / 2), int(255 - index / 2), int(index / 2))
    #         pic[row][col] = index
    #     print()
    # image.show()

    # k1 = graphKNN(rows, cols, patsize, Y, PN, 4)

    Y = unfoldPatch

    # hf = h5py.File("./data/k1.mat", 'r')
    # k1 = np.array(hf["k1"])
    # k1 = k1.flatten()
    # k1 = np.array([1] * PN)
    L = int(np.ceil(Y.shape[1] / PN))
    ind = []
    in_k = []
    HH = []
    Ytt1 = []
    YSO1 = []
    M2 = []




    # hf = h5py.File("./data/rr.mat", 'r')
    # rr = np.array(hf["rr"]).T
    # rr = np.reshape(rr, [nn[0] * nn[1] * nn[2]], order='F')
    # Z = cgsolve2(rr, nn, mu, rate)

    tempPatch = [0] * L
    for i in range(L):
        # ind.append(np.where(k1 == i + 1)[0])
        ind.append(k1[i])
        in_k.append(KK)
        k = in_k[i]
        # U1 = np.random.random([patsize, k])
        # diag_matrix = np.diag(np.reciprocal(np.sqrt(np.sum(U1 ** 2, axis=0))))
        # U1 = np.dot(U1, diag_matrix)
        #
        #
        #
        # U2 = np.random.random([patsize, k])
        # diag_matrix = np.diag(np.reciprocal(np.sqrt(np.sum(U2 ** 2, axis=0))))
        # U2 = np.dot(U2, diag_matrix)
        #
        # U3 = np.random.random([Ob.shape[2], k])
        # diag_matrix = np.diag(np.reciprocal(np.sqrt(np.sum(U3 ** 2, axis=0))))
        # U3 = np.dot(U3, diag_matrix)
        #
        # U4 = np.random.random([len(ind[i]), k])
        # diag_matrix = np.diag(np.reciprocal(np.sqrt(np.sum(U4 ** 2, axis=0))))
        # U4 = np.dot(U4, diag_matrix)
        #
        # D = np.random.random([Npatch2.shape[1], k])
        # diag_matrix = np.diag(np.reciprocal(np.sqrt(np.sum(D ** 2, axis=0))))
        # D = np.dot(D, diag_matrix)

        hf = h5py.File("./data/U1.mat", 'r')
        U1 = np.array(hf["U1"]).T

        hf = h5py.File("./data/U2.mat", 'r')
        U2 = np.array(hf["U2"]).T

        hf = h5py.File("./data/U3.mat", 'r')
        U3 = np.array(hf["U3"]).T

        hf = h5py.File("./data/U4.mat", 'r')
        U4 = np.array(hf["U4"]).T
        U4 = U4[0: len(ind[i]), :]

        hf = h5py.File("./data/D.mat", 'r')
        D = np.array(hf["D"]).T

        Ytt1.append(np.reshape(Npatch2[:, :, ind[i]],
                               [patsize, patsize, Npatch2.shape[1], len(ind[i])], order='F'))
        # Ytt1[i] = np.transpose(Ytt1[i], (0, 1, 3, 2))
        YSO1.append(matricize(Ytt1[i]))
        # if len()
        factor = Factor(U1, U2, U3, U4, D)
        M2.append(np.zeros(D.shape))
        HH.append(factor)

    HT = Ob
    Z = Ob
    for i in range(maxIter):
        HT_old = HT
        Z_old = Z
        Curpatch = Im2Patch3D(Z, patsize, Pstep)
        MCurpatch = Im2Patch3D(M1, patsize, Pstep)
        Ncur = np.zeros(Curpatch.shape)
        W = np.ones((Curpatch.shape[0], Curpatch.shape[2]))

        # hf = h5py.File("./data/D.mat", 'r')
        # D = np.array(hf["D"]).T

        # HT = Patch2Im3D(Ncur, W, patsize, Pstep, nn)

        for id in range(L):
            tt1 = Curpatch[:, :, ind[id]]
            tt1 = np.reshape(tt1, [patsize, patsize, Curpatch.shape[1], len(ind[id])], order='F')
            # tt1 = np.transpose(tt1, (0, 1, 3, 2))
            Mtt1 = MCurpatch[:, :, ind[id]]
            Mtt1 = np.reshape(Mtt1, [patsize, patsize, MCurpatch.shape[1], len(ind[id])], order='F')
            # Mtt1 = np.transpose(Mtt1, (0, 1, 3, 2))
            SO1 = matricize(tt1)

            # hf = h5py.File("./data/U2.mat", 'r')
            # HH[id].U2 = np.array(hf["U2"]).T
            #
            # hf = h5py.File("./data/U3.mat", 'r')
            # HH[id].U3 = np.array(hf["U3"]).T
            MM = matricize(Mtt1)


            W1 = sl.khatri_rao(HH[id].U4, HH[id].U3)
            W1 = sl.khatri_rao(W1, HH[id].U2)
            P1 = sl.khatri_rao(HH[id].U4, HH[id].D)
            P1 = sl.khatri_rao(P1, HH[id].U2)

            G1 = np.dot(HH[id].U4.T, HH[id].U4) * np.dot(HH[id].U3.T, HH[id].U3) \
                 * np.dot(HH[id].U2.T, HH[id].U2)
            PTP = np.dot(HH[id].U4.T, HH[id].U4) * np.dot(HH[id].D.T, HH[id].D) \
                  * np.dot(HH[id].U2.T, HH[id].U2)
            t1 = mu * G1 + 2 * lda * PTP



            HH[id].U1 = np.dot(np.dot(MM[0].T, W1) + 2 * lda * np.dot(YSO1[id][0].T, P1) + mu * np.dot(SO1[0].T, W1),
                               np.linalg.inv(t1))

            # def loss(U1, U2, U3, U4, Y, lda, X, mu, M):
            # r_loss = loss(HH[id].U1, HH[id].U2, HH[id].U3, HH[id].D, Ytt1[id], lda, tt1, mu, Mtt1)
            # print(r_loss)

            W2 = sl.khatri_rao(HH[id].U4, HH[id].U3)
            W2 = sl.khatri_rao(W2, HH[id].U1)
            P2 = sl.khatri_rao(HH[id].U4, HH[id].D)
            P2 = sl.khatri_rao(P2, HH[id].U1)
            G2 = np.dot(HH[id].U4.T, HH[id].U4) * np.dot(HH[id].U3.T, HH[id].U3) \
                 * np.dot(HH[id].U1.T, HH[id].U1)
            P2TP2 = np.dot(HH[id].U4.T, HH[id].U4) * np.dot(HH[id].D.T, HH[id].D) \
                    * np.dot(HH[id].U1.T, HH[id].U1)
            t2 = mu * G2 + 2 * lda * P2TP2
            HH[id].U2 = np.dot(np.dot(MM[1].T, W2) + 2 * lda * np.dot(YSO1[id][1].T, P2) + mu * np.dot(SO1[1].T, W2),
                               np.linalg.inv(t2))
            # r_loss = loss(HH[id].U1, HH[id].U2, HH[id].U3, HH[id].D, Ytt1[id], lda, tt1, mu, Mtt1)
            # print(r_loss)



            W3 = sl.khatri_rao(HH[id].U4, HH[id].U2)
            W3 = sl.khatri_rao(W3, HH[id].U1)
            leftA = np.dot(R.T, R)
            rightA = np.dot(HH[id].U4.T, HH[id].U4) * np.dot(HH[id].U2.T, HH[id].U2) \
                     * np.dot(HH[id].U1.T, HH[id].U1)
            rightEqu = np.dot(SO1[2].T, W3) + np.dot(MM[2].T, W3) / mu + np.dot(R.T, HH[id].D
                                                                                + M2[id] / mu)
            HH[id].U3 = solve_sylvester(leftA, rightA, rightEqu)

            # r_loss = loss(HH[id].U1, HH[id].U2, HH[id].U3, HH[id].D, Ytt1[id], lda, tt1, mu, Mtt1)
            # print(r_loss)

            t5 = 2 * lda * np.dot(W3.T, W3) + mu * np.eye(W3.shape[1])
            HH[id].D = np.dot(2 * lda * np.dot(YSO1[id][2].T, W3) + mu * np.dot(R, HH[id].U3)
                              - M2[id], np.linalg.inv(t5))


            W4 = sl.khatri_rao(HH[id].U3, HH[id].U2)
            W4 = sl.khatri_rao(W4, HH[id].U1)
            P4 = sl.khatri_rao(HH[id].D, HH[id].U2)
            P4 = sl.khatri_rao(P4, HH[id].U1)
            G4 = np.dot(HH[id].U3.T, HH[id].U3) * np.dot(HH[id].U2.T, HH[id].U2) \
                 * np.dot(HH[id].U1.T, HH[id].U1)
            P4TP4 = np.dot(HH[id].D.T, HH[id].D) * np.dot(HH[id].U2.T, HH[id].U2) \
                    * np.dot(HH[id].U1.T, HH[id].U1)
            t4 = mu * G4 + 2 * lda * P4TP4
            HH[id].U4 = np.dot(np.dot(MM[3].T, W4) + 2 * lda * np.dot(YSO1[id][3].T, P4)
                               + mu * np.dot(SO1[3].T, W4), np.linalg.inv(t4))


            # r_loss = loss(HH[id].U1, HH[id].U2, HH[id].U3, HH[id].D, Ytt1[id], lda, tt1, mu, Mtt1)
            # print(r_loss)
            time_ktensor = time.time()
            tempPatch[id] = ktensor([HH[id].U1, HH[id].U2, HH[id].U3, HH[id].U4])
            tempPatch[id] = np.reshape(tempPatch[id], [patsize * patsize, tempPatch[id].shape[2], tempPatch[id].shape[3]], order='F')
            time_ktensor_end = time.time()
            print(time_ktensor_end - time_ktensor)
            M2[id] = M2[id] + mu * (HH[id].D - np.dot(R, HH[id].U3))


        for ii in range(L):
            Ncur[:, :, ind[ii]] = Ncur[:, :, ind[ii]] + tempPatch[ii]

        HT = Patch2Im3D(Ncur, W, patsize, Pstep, nn)

        # plt.imshow(HT[:, :, 1:4])
        # plt.show()

        rr = np.reshape(2 * Ob + mu * HT - M1, [nn[0] * nn[1] * nn[2]], order='F')

        # hf = h5py.File("./data/rr.mat", 'r')
        # rr = np.array(hf["rr"]).T
        # rr = np.reshape(rr, [nn[0] * nn[1] * nn[2]], order='F')
        Z = cgsolve2(rr, nn, mu, rate, s)

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
    Out = Z
    Out = Out * max_HS
    return Out

