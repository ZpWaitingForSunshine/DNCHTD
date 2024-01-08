import numpy as np
import tensorly as tl
import scipy.linalg as sl
def tensor_mode_unfolding(tensor, mode):
    """Performs mode unfolding of a tensor along a given mode in row-major order."""
    shape = tensor.shape
    mode_unfolded = np.moveaxis(tensor, mode, 0)
    mode_unfolded = np.reshape(mode_unfolded, (shape[mode], -1), order='F')
    return mode_unfolded
    # return tl.unfold(tensor, mode)


def matricize(tensor):
    dim = np.ndim(tensor)
    data = []
    for i in range(dim):
        data.append(tensor_mode_unfolding(tensor, mode=i))
    return data


def ktensor(Us):
    R = Us[0].shape[1]
    cp_tensor = (np.ones(R), Us)
    return tl.cp_to_tensor(cp_tensor)


def ttm(T, M, mode):
    return tl.tenalg.mode_dot(T, M, mode=mode)

# def ttm(tensor, matrix, mode):
#     """
#     Tensor times matrix (ttm) on specified mode.
#     Mimics MATLAB's ttm function behavior.
#     :param tensor: Input tensor (3D array).
#     :param matrix: Matrix to multiply with the tensor.
#     :param mode: The mode on which to perform the multiplication.
#     :return: Resulting tensor after ttm.
#     """
#     # 对张量进行转置以匹配 MATLAB 的列优先行为
#     if mode == 0:
#         tensor_permuted = tensor
#     elif mode == 1:
#         tensor_permuted = np.transpose(tensor, (1, 0, 2))
#     else:  # mode == 2
#         tensor_permuted = np.transpose(tensor, (2, 1, 0))
#
#     # 获取转置后张量的形状
#     I, J, K = tensor_permuted.shape
#
#     # 根据模态进行重塑和乘法运算
#     if mode == 0:
#         reshaped_tensor = tensor_permuted.reshape(I, J * K)
#         result = matrix @ reshaped_tensor
#         result = result.reshape(matrix.shape[0], J, K)
#     elif mode == 1:
#         reshaped_tensor = tensor_permuted.reshape(I, J * K)
#         result = matrix @ reshaped_tensor
#         result = result.reshape(matrix.shape[0], J, K)
#         result = np.transpose(result, (1, 0, 2))
#     else:  # mode == 2
#         reshaped_tensor = tensor_permuted.reshape(I, J * K)
#         result = matrix @ reshaped_tensor
#         result = result.reshape(matrix.shape[0], J, K)
#         result = np.transpose(result, (2, 1, 0))
#
#     return result


def calu3TTM(G, U1, U2, U3):
    G = ttm(G, U1, mode=0)
    G = ttm(G, U2, mode=1)
    G = ttm(G, U3, mode=2)
    return G

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
    for i in range(3):
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
