
import numpy as np
import scipy.linalg as sl
from function import upsample, gaussian, graph, Im2Patch3D, matricize, ktensor, Patch2Im3D, cgsolve2
import h5py
from scipy.linalg import solve_sylvester
import matplotlib.pyplot as plt
import tensorly as tl
from collections import namedtuple

# 创建一个名为Person的结构体
class Factor:


    def setFactors(self,U1, U2, U3, U4, B1, B2, M2):
        self.U1 = U1
        self.U2 = U2
        self.U3 = U3
        self.B1 = B1
        self.U4 = U4
        self.B2 = B2
        self.M2 = M2

    def __init__(self, R1, R2, R3, RB1):
        self.R1 = R1
        self.R2 = R2
        self.R3 = R3
        self.RB1 = RB1

# Factor = namedtuple('Factor', ['U1', 'U2', 'U3', 'D'])

def NPTCP(Ob, KK, Y, rate, PN, R, s):
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

    # % initialize
    Z = Ob
    M1 = np.zeros(Z.shape)

    Ob = gaussian(Ob, s)

    # TY = cat(3, Y, Y(:,:, end));
    Npatch2 = Im2Patch3D(Y, patsize, Pstep)

    unfoldPatch = np.reshape(Npatch2, [int(Npatch2.shape[0] * Npatch2.shape[1]), Npatch2.shape[2]], order='F')

    k1 = graph(unfoldPatch, patsize, Pstep, nn, PN)

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
        R1 = 24
        R2 = 50
        R3 = 35
        RB1 = 35
        factor = Factor(R1, R2, R3, RB1)
        # k = patch.Rank  # rank
        #
        # R1, R2, R3, RB1 = k


        U1 = np.random.random([patsize * patsize, R1])
        diag_matrix = np.diag(np.reciprocal(np.sqrt(np.sum(U1 ** 2, axis=0))))
        U1 = np.dot(U1, diag_matrix)

        # U2 = np.random.random([len(ind[i]), k])
        # diag_matrix = np.diag(np.reciprocal(np.sqrt(np.sum(U2 ** 2, axis=0))))
        # U2 = np.dot(U2, diag_matrix)

        if R2 < len(ind[i]):
            U2 = np.random.random([len(ind[i]), R2])
            diag_matrix = np.diag(np.reciprocal(np.sqrt(np.sum(U2 ** 2, axis=0))))
            U2 = np.dot(U2, diag_matrix)
        else:
            U2 = np.random.random([len(ind[i]), len(ind[i])])
            diag_matrix = np.diag(np.reciprocal(np.sqrt(np.sum(U2 ** 2, axis=0))))
            U2 = np.dot(U2, diag_matrix)

        U3 = np.random.random([Ob.shape[2], R3])
        diag_matrix = np.diag(np.reciprocal(np.sqrt(np.sum(U3 ** 2, axis=0))))
        U3 = np.dot(U3, diag_matrix)

        D = np.random.random([Npatch2.shape[1], R3])
        diag_matrix = np.diag(np.reciprocal(np.sqrt(np.sum(D ** 2, axis=0))))
        D = np.dot(D, diag_matrix)

        if R2 > len(ind[i]):

            B1 = np.random.random([R1 * len(ind[i]), RB1])
            diag_matrix = np.diag(np.reciprocal(np.sqrt(np.sum(B1 ** 2, axis=0))))
            B1 = np.dot(B1, diag_matrix)
            B1 = np.reshape(B1, (R1, len(ind[i]), RB1))

        else:
            B1 = np.random.random([R1 * R2, RB1])
            diag_matrix = np.diag(np.reciprocal(np.sqrt(np.sum(B1 ** 2, axis=0))))
            B1 = np.dot(B1, diag_matrix)
            B1 = np.reshape(B1, (R1, R2, RB1))

        B2 = np.random.random([RB1, R3])
        diag_matrix = np.diag(np.reciprocal(np.sqrt(np.sum(B2 ** 2, axis=0))))
        B2 = np.dot(B2, diag_matrix)

        Ytt1.append(Npatch2[:, :, ind[i]])
        Ytt1[i] = np.transpose(Ytt1[i], (0, 2, 1))
        YSO1.append(matricize(Ytt1[i]))
        # if len()
        # factor = Factor(U1, U2, U3, D)
        M2.append(np.zeros(D.shape))
        factor.setFactors(U1, U2, U3, D, B1, B2, M2)
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
        theta = 0.0001
        for id in range(L):
            tt1 = Curpatch[:, :, ind[id]]
            tt1 = np.transpose(tt1, (0, 2, 1))
            Mtt1 = MCurpatch[:, :, ind[id]]
            Mtt1 = np.transpose(Mtt1, (0, 2, 1))
            SO1 = matricize(tt1)

            # hf = h5py.File("./data/U2.mat", 'r')
            # HH[id].U2 = np.array(hf["U2"]).T
            #
            # hf = h5py.File("./data/U3.mat", 'r')
            # HH[id].U3 = np.array(hf["U3"]).T

            MM = matricize(Mtt1)



            HH[id].B2 = updateB2(HH[id].U1, HH[id].U2, HH[id].U3, HH[id].U4, HH[id].B1, HH[id].B2, YSO1[id][2].T, SO1[2].T, MM[2].T, mu, lda, theta)

            HH[id].B1 = updateB1(HH[id].U1, HH[id].U2, HH[id].U3, HH[id].U4, HH[id].B1, HH[id].B2, Ytt1[id], tt1, Mtt1, mu, lda, theta)

            # r_loss = loss(HH[id].U1, HH[id].U2, HH[id].U3, HH[id].D, Ytt1[id], lda, tt1, mu, Mtt1)
            # print(r_loss)
            HH[id].U2 = updateU2(HH[id].U1, HH[id].U2, HH[id].U3, HH[id].U4, HH[id].B1, HH[id].B2, YSO1[id][1].T, SO1[1].T, MM[1].T, mu, lda, theta)

            HH[id].U1 = updateU1(HH[id].U1, HH[id].U2, HH[id].U3, HH[id].U4, HH[id].B1, HH[id].B2, YSO1[id][0].T, SO1[0].T, MM[0].T, mu, lda, theta)

            HH[id].U3 = updateU3(HH[id].U1, HH[id].U2, HH[id].U3, HH[id].U4, HH[id].B1, HH[id].B2, SO1[2].T, MM[2].T, mu, theta, R, M2[id])

            HH[id].U4 = updateU4(HH[id].U1, HH[id].U2, HH[id].U3, HH[id].U4, HH[id].B1, HH[id].B2, YSO1[id][2].T, mu, theta, lda, R, M2[id])

            tempPatch[id] = HT_recover(HH[id].U1, HH[id].U2, HH[id].U3, HH[id].B1, HH[id].B2)
            # tempPatch[id] = ktensor([HH[id].U1, HH[id].U3, HH[id].U2])
            tempPatch[id] = np.transpose(tempPatch[id], (0, 2, 1))

            M2[id] = M2[id] + mu * (HH[id].U4 - np.dot(R, HH[id].U3))



        for ii in range(L):
            Ncur[:, :, ind[ii]] = Ncur[:, :, ind[ii]] + tempPatch[ii]

        HT = Patch2Im3D(Ncur, W, patsize, Pstep, nn)

        plt.imshow(HT[:, :, 1:4])
        plt.show()

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

        plt.imshow(Z[:, :, 1:4])
        plt.show()

        res1 = Z - HT
        RZ = Z * max_HS
        ReChange = np.linalg.norm(res1) / np.linalg.norm(Z)
        print('%10.5f\t%10.5f\t%10.5f\t\n', ReChange, stopCond, stopCond2)
        if ReChange < tol and stopCond < tol and stopCond2 < tol and i > (minIter - 1):
            break
    Out = Z
    Out = Out * max_HS
    return Out
def HT_recover(U1, U2, U3, B1, B2):
    a = tl.tenalg.mode_dot(B1, U1, mode=0)
    a = tl.tenalg.mode_dot(a, U2, mode=1)
    a = tl.tenalg.mode_dot(a, B2.T, mode=2)
    a = tl.tenalg.mode_dot(a, U3, mode=2)
    return a

def updateU1(U1, U2, U3, U4, B1, B2, Y, X, M, mu, lda, theta):
    # update U1
    Core = tl.tenalg.mode_dot(B1, U2, mode=1)
    Core = tl.tenalg.mode_dot(Core, B2.T, mode=2)
    # tensor_mode_unfolding(tensor, mode=i)
    A = tl.tenalg.mode_dot(Core, U4, mode=2)
    A = tensor_mode_unfolding(A, mode=0)
    B = tl.tenalg.mode_dot(Core, U3, mode=2)
    B = tensor_mode_unfolding(B, mode=0)


    right = 2 * lda * np.dot(Y, A.T) + np.dot(mu * X + M, B.T) + \
              theta * U1


    left = 2 * lda * np.dot(A, A.T) + mu * np.dot(B, B.T) + \
                theta * np.eye(B.shape[0])
    U1 = np.dot(right, np.linalg.inv(left))
    return U1
    # cur = HT_recover(U1, U2, U3, B1, B2)
    # print(np.linalg.norm(cur - tt1))

def updateU2(U1, U2, U3, U4, B1, B2, Y, X, M, mu, lda, theta):
    Core = tl.tenalg.mode_dot(B1, U1, mode=0)
    Core = tl.tenalg.mode_dot(Core, B2.T, mode=2)


    A = tl.tenalg.mode_dot(Core, U4, mode=2)
    A = tensor_mode_unfolding(A, mode=1)

    B = tl.tenalg.mode_dot(Core, U3, mode=2)
    B = tensor_mode_unfolding(B, mode=1)

    left_U2 = 2 * lda * np.dot(Y, A.T) + np.dot(mu * X + M, B.T) + \
              theta * U2
    right_U2 = 2 * lda * np.dot(A, A.T) + mu * np.dot(B, B.T) + \
               theta * np.eye(B.shape[0])

    U2 = np.dot(left_U2, np.linalg.pinv(right_U2))
    return U2
    # cur = HT_recover(U1, U2, U3, B1, B2)
    # print(np.linalg.norm(cur - tt1))

def updateU3(U1, U2, U3, U4, B1, B2, X, M, mu, theta, R, F):
    # U3

    core = tl.tenalg.mode_dot(B1, B2.T, mode=2)
    core = tl.tenalg.mode_dot(core, U2, mode=1)
    core = tl.tenalg.mode_dot(core, U1, mode=0)
    A = tensor_mode_unfolding(core, mode=2)
    # A = tl.unfold(core, mode=2)

    right1 = np.dot(R.T, R) * mu
    right2 = mu * np.dot(A, A.T) + 2 * theta * np.eye(A.shape[0])
    left = mu * np.dot(X + M / mu, A.T) + mu * np.dot(R.T, U4 + F / mu) + 2 * theta * U3

    U3 = sl.solve_sylvester(right1, right2, left)


    # Core = tl.tenalg.mode_dot(B1, U2, mode=1)
    # Core = tl.tenalg.mode_dot(Core, U1, mode=0)
    # Core = tl.tenalg.mode_dot(Core, B2.T, mode=2)
    # # b = tl.tenalg.mode_dot(b, U3, mode=2)
    # Core = tensor_mode_unfolding(Core, mode=2)
    # right_U3 = mu * np.dot(X + M / mu, Core.T) + \
    #            mu * np.dot(R.T, U4 + F / mu) + 2 * theta * U3
    # left_U3 = mu * np.dot(R.T, R) + 2 * theta * np.eye(R.shape[1])
    # mid_U3 = np.dot(Core, Core.T)
    #
    # U3 = sl.solve_sylvester(left_U3, mid_U3, right_U3)
    return U3


def updateU4(U1, U2, U3, U4, B1, B2, Y, mu, theta, lda, R, F):
    Core = tl.tenalg.mode_dot(B1, U2, mode=1)
    Core = tl.tenalg.mode_dot(Core, U1, mode=0)
    Core = tl.tenalg.mode_dot(Core, B2.T, mode=2)
    # Core = tl.tenalg.mode_dot(Core, U4, mode=2)
    Core = tensor_mode_unfolding(Core, mode=2)
    left_U4 = 2 * lda * np.dot(Y, Core.T) + \
              mu * np.dot(R, U3) - F + 2 * theta * U4
    right_U4 = 2 * lda * np.dot(Core, Core.T) + \
               (mu + 2 * theta) * np.eye(U3.shape[1])

    U4 = np.dot(left_U4, np.linalg.pinv(right_U4))

    return U4




def updateB2(U1, U2, U3, U4, B1, B2, Y, X, M, mu, lda, theta):
    Core = tl.tenalg.mode_dot(B1, U2, mode=1)
    Core = tl.tenalg.mode_dot(Core, U1, mode=0)
    Core = tensor_mode_unfolding(Core, mode=2)
    A = np.dot(Core, Core.T)
    B = 2 * lda * np.dot(U4.T, U4)
    C = mu * np.dot(U3.T, U3)
    D = 2 * lda * np.dot(np.dot(Core, Y.T), U4) + mu * np.dot(np.dot(Core, M.T / mu + X.T), U3) + 2 * theta * B2
    temp = np.linalg.solve(np.kron(B.T + C.T, A) + theta * np.eye(B.shape[0] * A.shape[0]), tl.tensor_to_vec(D.T))
    return np.reshape(temp, B2.shape).T


def updateB1(U1, U2, U3, U4, B1, B2, Y, X, M, mu, lda, theta):
    mu_ = 0.01
    C2 = B1
    C1 = C2
    V1 = np.zeros(C1.shape)
    for i in range(6):
        C2 = updateCore(np.dot(U3, B2.T), U2, U1, X + M / mu, C2, V1, mu / 2, mu_)
        C1 = updateCore(np.dot(U4, B2.T), U2, U1, Y, C1, V1, lda, mu_)
        V1 = V1 - (C1 - C2)
    B1 = C1
    return B1

def updateCore(S, H, W, HSI, C, V2, lda, mu):
    C = tl.tensor_to_vec(C.T)
    V2 = tl.tensor_to_vec(V2.T)
    STS = np.dot(S.T, S)
    S1S, S1U, S1S_ = np.linalg.svd(STS)
    # SU = np.diag(S1U)

    HS, HU, W_ = np.linalg.svd(np.dot(H.T, H))
    # H1U = np.diag(HU)

    WS, WU, W_ = np.linalg.svd(np.dot(W.T, W))
    # W1U = np.diag(WU)

    temp1 = np.kron(S1U, HU)
    temp1 = np.kron(temp1, WU)
    #

    # D1Y = calu3TTM(HSI, W.T, H.T, S.T)
    # right =  tl.tensor_to_vec(D1Y.T) + mu * C - mu * V2
    mid = lda * temp1 + mu * np.ones(temp1.shape)
    mid = np.reciprocal(mid)

    D2Y = calu3TTM(HSI, W.T, H.T, S.T)
    D2Y_ = tl.tensor_to_vec(D2Y.T)

    right = np.reshape(lda * D2Y_ + mu * C - mu * V2, [D2Y.shape[2], D2Y.shape[1], D2Y.shape[0]]).T

    temp1 = calu3TTM(right, WS.T, HS.T, S1S.T)
    temp1 = tl.tensor_to_vec(temp1.T)
    temp1 = temp1 * mid


    left = np.reshape(temp1, [S1S.shape[0], HS.shape[0], WS.shape[0]]).T

    C2 = calu3TTM(left, WS, HS, S1S)

    C2 = tl.tensor_to_vec(C2.T)

    C2 = np.reshape(C2, [S.shape[1], H.shape[1], W.shape[1]]).T

    return C2




    #
    #
    # temp2 = calu3TTM(right, WS.T, HS.T, S1S.T)
    #
    #
    # temp = np.kron(S1U, HU)
    # temp = np.kron(temp, WU)
    # mid = temp + mu * np.ones(temp.shape)
    # mid = np.reciprocal(mid)
    #
    #
    # # print("kron mid: ", t2 - t1)
    #
    # mid = np.reshape(mid, temp2.shape, order='F')
    # temp3 = temp2 * mid
    #
    # C1 = calu3TTM(temp3, WS, HS, S1S)
    #
    # return C1
    #
def tensor_mode_unfolding(tensor, mode):
    """Performs mode unfolding of a tensor along a given mode in row-major order."""
    shape = tensor.shape
    mode_unfolded = np.moveaxis(tensor, mode, 0)
    mode_unfolded = np.reshape(mode_unfolded, (shape[mode], -1), order='F')
    return mode_unfolded
    # return tl.unfold(tensor, mode)

def calu3TTM(G, U1, U2, U3):
    G = ttm(G, U1, mode=0)
    G = ttm(G, U2, mode=1)
    G = ttm(G, U3, mode=2)
    return G

def ttm(T, M, mode):
    return tl.tenalg.mode_dot(T, M, mode=mode)
