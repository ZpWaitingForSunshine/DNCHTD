import numpy as np
from scipy.ndimage import convolve
import h5py
from scipy.signal import convolve2d
from scipy import signal, ndimage
import tensorly as tl
import ctypes
from numpy.ctypeslib import ndpointer
import sys
import time

import logging



# from rddFunction import find_min_indices

logging.basicConfig(
    filemode='/data2/zp/nptcp.log',
    format='%(asctime)s - %(name)s - %(pathname)s - %(lineno)d - %(levelname)s - %(message)s'
)

def find_min_indices(arr, N):
    # 获取第二行的数据
    row = arr[1]

    # 使用argsort函数对第二行进行排序，并获取排序后的索引
    sorted_indices = np.argsort(row)

    # 取前N个最小值的索引
    min_indices = sorted_indices[:N]

    return min_indices

def gaussian(X, s):
    X_copy = X.copy()
    # hf = h5py.File("./data/hrow.mat", 'r')
    # hrow = np.array(hf["hrow"]).T
    # hf = h5py.File("./data/hcol.mat", 'r')
    # hcol = np.array(hf["hcol"]).T
    # hf = h5py.File("./data/s.mat", 'r')
    # s = np.array(hf["s"]).T
    # ratio = 5
    # sigma = (1 / (2 * 2.7725887 / ratio ** 2)) ** 0.5
    for i in range(X.shape[2]):
        X_copy[:, :, i] = convolve(X[:, :, i], s, mode='reflect')
        # temp = np.pad(X[:, :, i], 4, mode='symmetric')
        # temp = convolve2d(temp, hcol, mode='valid')
        # temp = convolve2d(temp, hrow, mode='valid')
        # # 创建卷积核矩阵
        # kernel = np.outer(hcol, hrow)
        #
        # # 执行有效卷积
        # # temp = np.convolve(temp, kernel, mode='valid')
        #
        # temp = convolve2d(temp, kernel, mode='valid')

        # hf = h5py.File("./data/rr.mat", 'r')
        # rr = np.array(hf["rr"]).T
        # z = rr - temp
        # X[:, :, i] = gaussian_filter(X[:, :, i], sigma=sigma, mode='wrap')

    return X_copy


def downsample(inMatrix, rate):
    inMatrix_ = inMatrix.copy()
    return inMatrix_[::rate, 0::rate, :]

def upsample(I_Interpolated, ratio):
    L = 45
    [r, c, b] = I_Interpolated.shape
    kernel = ratio * signal.firwin(L, 1 / ratio)
    kernel = np.reshape(kernel, [1, len(kernel)])
    I1LRU = np.zeros([ratio * r, ratio * c, b])
    I1LRU[0::ratio, 0::ratio, :] = I_Interpolated.copy()
    for ii in range(b):
        t = I1LRU[:, :, ii]
        t = convolve(t.T, kernel, mode='wrap')
        I1LRU[:, :, ii] = convolve(t.T, kernel, mode='wrap')
    return I1LRU

def Im2Patch3D(Video, patsize:int, step:int):
    TotalPatNum = int((np.floor((Video.shape[0] - patsize) / step) + 1) * \
                  (np.floor((Video.shape[1] - patsize) / step) + 1))
    Y = np.zeros((int(patsize * patsize), Video.shape[2], TotalPatNum))
    k = 0
    for i in range(patsize):
        for j in range(patsize):
            tempPatch = Video[i: Video.shape[0] - patsize + i + 1: step, j: Video.shape[1] - patsize + j + 1: step, :]
            Y[k, :, :] = tensor_mode_unfolding(tempPatch, mode=2)
            k = k + 1
    return Y

def tensor_mode_unfolding(tensor, mode):
    """Performs mode unfolding of a tensor along a given mode in row-major order."""
    shape = tensor.shape
    mode_unfolded = np.moveaxis(tensor, mode, 0)
    mode_unfolded = np.reshape(mode_unfolded, (shape[mode], -1), order='F')
    return mode_unfolded


# #
# def graph(Npatch2, patsize, Pstep, nn, NP):
#     # unfoldPatch = np.reshape(Npatch2, [nn[0] * nn[1], nn[2]], order='F')
#     Y = Npatch2
#     total: int = int(((nn[0] - patsize) / Pstep + 1) * ((nn[1] - patsize) / Pstep + 1))
#     rows: int = int((nn[0] - patsize) / Pstep + 1)
#     cols: int = int((nn[1] - patsize) / Pstep + 1)
#
#     cur_rows = np.tile(np.arange(0, rows), cols)
#     cur_cols = np.array(np.repeat(np.arange(0, rows), cols))
#     ixMat = np.array([cur_rows, cur_cols])
#     x_restMat = np.vstack([Y, np.ones((1, Y.shape[1]))])
#     x_restMat = np.vstack([x_restMat, -0.5 * np.sum(Y ** 2, axis=0)])
#     x_1Mat = np.vstack([Y, -0.5 * np.sum(Y ** 2, axis=0)])
#     x_1Mat = np.vstack([x_1Mat, np.ones((1, Y.shape[1]))])
#
#     x_1Vec = np.reshape(x_1Mat, x_1Mat.shape[0] * x_1Mat.shape[1], order='F')
#     x_restVec = np.reshape(x_restMat, x_restMat.shape[0] * x_restMat.shape[1], order='F')
#     ixVec = np.reshape(ixMat, ixMat.shape[0] * ixMat.shape[1], order='F').astype(np.float64)
#
#
#     N = Y.shape[0] + 2
#     pointsNum = Y.shape[1]
#     randStartIx = 1
#     regGrid = 1
#
#     B = 81
#     eps = 10 ** 0.01
#     randVec = np.random.rand(Y.shape[1])
#
#     mylib = ctypes.cdll.LoadLibrary('./DSP.dll')
#     mylib.mexFunction.argtypes = [ndpointer(dtype=np.float64, ndim=1), ndpointer(dtype=np.float64, ndim=1),ndpointer(dtype=np.float64, ndim=1),
#                                  ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
#                                  ctypes.c_double, ndpointer(dtype=np.float64, ndim=1), ctypes.c_double,
#                                  ctypes.c_int]
#     mylib.mexFunction.restype = ctypes.POINTER(ctypes.c_int32)
#     content = mylib.mexFunction(
#         x_1Vec,
#         x_restVec,
#         ixVec,
#         N, pointsNum, rows, cols, randStartIx,
#         B, randVec, eps,
#         regGrid
#     )
#
#     sortedIndex = np.ctypeslib.as_array(content, shape=(pointsNum,)) - 1
#     indices = split_given_size(sortedIndex, NP)
#     for row in indices:
#         row.sort()
#     return indices

def matricize(tensor):
    dim = np.ndim(tensor)
    data = []
    for i in range(dim):
        data.append(tensor_mode_unfolding(tensor, mode=i).T)
    return data


def ktensor(Us):
    R = Us[0].shape[1]
    cp_tensor = (np.ones(R), Us)
    return tl.cp_to_tensor(cp_tensor)

def Patch2Im3D(ImPat, WPat, patsize, step, sizeV):
    TempR = int(np.floor((sizeV[0] - patsize) / step) + 1)
    TempC = int(np.floor((sizeV[1] - patsize) / step) + 1)
    TempOffsetR = np.arange(0, (TempR - 1) * step + 1, step)
    TempOffsetC = np.arange(0, (TempC - 1) * step + 1, step)
    # ImPat[:, :, 1: ImPat.shape[2]] = 0
    E_Img = np.zeros(sizeV)
    W_Img = np.zeros(sizeV)
    k = 0

    for i in range(patsize):
        for j in range(patsize):
            aa = np.tile(WPat[k, :], (sizeV[2], 1))
            E_Img[i: TempR + i, j: TempC + j, :] = E_Img[i: TempR + i, j: TempC + j, :] + \
                np.reshape(ImPat[k, :, :].T, [TempR, TempC, sizeV[2]], order='F')
            W_Img[i: TempR + i, j: TempC + j, :] = W_Img[i: TempR + i, j: TempC + j, :] + \
                np.reshape(np.tile(WPat[k, :].T, (sizeV[2], 1)), [TempR, TempC, sizeV[2]], order='F')
            k = k + 1
    E_Img = E_Img / (W_Img + np.finfo(float).eps)
    return E_Img


def ParitionsPatch2Im3D(ImPat, WPat, patsize, step, sizeV):
    TempR = int(np.floor((sizeV[0] - patsize) / step) + 1)
    TempC = int(np.floor((sizeV[1] - patsize) / step) + 1)
    TempOffsetR = np.arange(0, (TempR - 1) * step + 1, step)
    TempOffsetC = np.arange(0, (TempC - 1) * step + 1, step)

    E_Img = np.zeros(sizeV)
    W_Img = np.zeros(sizeV)
    k = 0

    for i in range(patsize):
        for j in range(patsize):
            aa = np.tile(WPat[k, :], (sizeV[2], 1))
            E_Img[i: TempR + i, j: TempC + j, :] = E_Img[i: TempR + i, j: TempC + j, :] + \
                np.reshape(ImPat[k, :, :].T, [TempR, TempC, sizeV[2]], order='F')
            W_Img[i: TempR + i, j: TempC + j, :] = W_Img[i: TempR + i, j: TempC + j, :] + \
                np.reshape(np.tile(WPat[k, :].T, (sizeV[2], 1)), [TempR, TempC, sizeV[2]], order='F')
            k = k + 1
    E_Img = E_Img / (W_Img + np.finfo(float).eps)
    return E_Img
def myfun2(X, mu, rate, nn, s):
    X_ = np.reshape(X, nn, order='F')
    ours = gaussian(X_, s)
    ours = downsample(ours, rate)
    ours = upsample(ours, rate)
    ours = gaussian(ours, s)
    re = 2 * ours + mu * X_
    return np.reshape(re, re.shape[0] * re.shape[1] * re.shape[2], order='F')


def cgsolve2(b, nn, mu, rate, s):
    # b = np.reshape(b, b.shape[0], order='F')
    n = len(b)
    maxiters = 50
    normb = np.linalg.norm(b)
    x = np.zeros(n)
    r = b.copy()
    rtr = np.dot(r.T, r)
    d = r
    niters = 0
    while np.sqrt(rtr) / normb > 2e-6 and niters < maxiters:
        niters = niters + 1
        Ad = myfun2(d, mu, rate, nn, s)
        alpha = rtr / np.dot(d.T, Ad)
        x = x + alpha * d
        r = r - alpha * Ad
        rtrold = rtr
        rtr = np.dot(r, r)
        beta = rtr / rtrold
        d = r + beta * d

    return np.reshape(x, nn, order='F')


def CC(ref, tar):
    rows, cols, bands = tar.shape
    out = np.zeros(bands)

    for i in range(bands):
        tar_tmp = tar[:, :, i]
        ref_tmp = ref[:, :, i]
        cc = np.corrcoef(tar_tmp.ravel(), ref_tmp.ravel())
        out[i] = cc[0, 1]

    return np.mean(out)

def SAM(ref, tar):
    rows, cols, bands = tar.shape
    prod_scal = np.sum(ref * tar, axis=2)
    norm_orig = np.sum(ref * ref, axis=2)
    norm_fusa = np.sum(tar * tar, axis=2)
    prod_norm = np.sqrt(norm_orig * norm_fusa)
    prod_map = prod_norm.copy()
    prod_map[prod_map == 0] = np.finfo(float).eps
    # map = np.arccos(prod_scal / prod_map)
    prod_scal = prod_scal.ravel()
    prod_norm = prod_norm.ravel()
    z = np.where(prod_norm == 0)
    prod_scal = np.delete(prod_scal, z)
    prod_norm = np.delete(prod_norm, z)
    angle_SAM = np.sum(np.arccos(prod_scal / prod_norm)) * (180 / np.pi) / len(prod_norm)

    return angle_SAM

def RMSE(ref, tar):
    rows, cols, bands = ref.shape
    out = np.sqrt(np.sum(np.sum(np.sum((tar - ref) ** 2))) / (rows * cols * bands))
    return out


def ERGAS(I, I_Fus, Resize_fact):
    I = I.astype(np.float64)
    I_Fus = I_Fus.astype(np.float64)

    Err = I - I_Fus
    ERGAS = 0
    for iLR in range(Err.shape[2]):
        ERGAS += np.mean(Err[:, :, iLR] ** 2) / np.mean(I[:, :, iLR]) ** 2

    ERGAS = (100 / Resize_fact) * np.sqrt((1 / Err.shape[2]) * ERGAS)

    return ERGAS


def PSNR3D(imagery1, imagery2):
    m, n, k = imagery1.shape
    mm, nn, kk = imagery2.shape
    m = min(m, mm)
    n = min(n, nn)
    k = min(k, kk)
    imagery1 = imagery1[0:m, 0:n, 0:k]
    imagery2 = imagery2[0:m, 0:n, 0:k]
    psnr = 0
    for i in range(k):
        mse = np.mean((imagery1[:, :, i] - imagery2[:, :, i]) ** 2)
        psnr += 10 * np.log10(255 ** 2 / mse)
    psnr /= k

    return psnr


def loss(U1, U2, U3, U4, Y, lda, X, mu, M):
    L1 = lda * np.linalg.norm(ktensor([U1, U2, U4]) - Y)
    L2 = mu / 2 * np.linalg.norm(ktensor([U1, U2, U3]) - X - M/mu)
    return L1 + L2

def QualityIndices(I_HS, I_REF, ratio):
    rows, cols, bands = I_REF.shape
    I_HS = I_HS[ratio: rows - ratio, ratio: cols - ratio,:]
    I_REF = I_REF[ratio :rows - ratio, ratio: cols - ratio,:]
    cc = CC(I_HS, I_REF)
    logging.info("cc: ", cc)
    print("cc: ", cc)
    sam = SAM(I_HS, I_REF)
    print('sam: ', sam)
    logging.info('sam: ', sam)


    rmse = RMSE(I_HS, I_REF)
    print('rmse: ', rmse)
    logging.info('rmse: ', rmse)

    ergas = ERGAS(I_HS, I_REF, ratio)
    print('ERGAS: ', ergas)
    logging.info('ERGAS: ', ergas)




def graph(Npatch2, patsize, Pstep, nn, NP):
    # unfoldPatch = np.reshape(Npatch2, [nn[0] * nn[1], nn[2]], order='F')
    Y = Npatch2
    total: int = int(((nn[0] - patsize) / Pstep + 1) * ((nn[1] - patsize) / Pstep + 1))
    rows: int = int((nn[0] - patsize) / Pstep + 1)
    cols: int = int((nn[1] - patsize) / Pstep + 1)

    cur_rows = np.tile(np.arange(0, rows), cols)
    cur_cols = np.array(np.repeat(np.arange(0, rows), cols))
    ixMat = np.array([cur_rows, cur_cols])
    x_restMat = np.vstack([Y, np.ones((1, Y.shape[1]))])
    x_restMat = np.vstack([x_restMat, -0.5 * np.sum(Y ** 2, axis=0)])
    x_1Mat = np.vstack([Y, -0.5 * np.sum(Y ** 2, axis=0)])
    x_1Mat = np.vstack([x_1Mat, np.ones((1, Y.shape[1]))])

    x_1Vec = np.reshape(x_1Mat, x_1Mat.shape[0] * x_1Mat.shape[1], order='F')
    x_restVec = np.reshape(x_restMat, x_restMat.shape[0] * x_restMat.shape[1], order='F')
    ixVec = np.reshape(ixMat, ixMat.shape[0] * ixMat.shape[1], order='F').astype(np.float64)


    N = Y.shape[0] + 2
    pointsNum = Y.shape[1]
    randStartIx = 1
    regGrid = 1

    B = 81
    eps = 10 ** 0.01
    randVec = np.random.rand(Y.shape[1])

    import platform
    system = platform.system()
    if system == "Windows":
        mylib = ctypes.cdll.LoadLibrary('./DSP.dll')
    else:
        mylib = ctypes.cdll.LoadLibrary('./DSP.so')
    mylib.mexFunction.argtypes = [ndpointer(dtype=np.float64, ndim=1), ndpointer(dtype=np.float64, ndim=1),ndpointer(dtype=np.float64, ndim=1),
                                 ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                 ctypes.c_double, ndpointer(dtype=np.float64, ndim=1), ctypes.c_double,
                                 ctypes.c_int]
    mylib.mexFunction.restype = ctypes.POINTER(ctypes.c_int32)
    content = mylib.mexFunction(
        x_1Vec,
        x_restVec,
        ixVec,
        N, pointsNum, rows, cols, randStartIx,
        B, randVec, eps,
        regGrid
    )

    sortedIndex = np.ctypeslib.as_array(content, shape=(pointsNum,)) - 1
    indices = np.split(sortedIndex, np.arange(NP, len(sortedIndex), NP))
        # split_given_size(sortedIndex, NP)
    for row in indices:
        row.sort()
    return indices

def graphKNN(rows, cols, Pstepsize, Y, PN, num):
    t_start = time.time()
    # record the type
    indices = np.zeros((2, rows * cols)).astype('int')
    indices[0] = np.arange(rows * cols)
    indices[1] = sys.maxsize
    indices_set = []

    groups_edges = split_average(indices[0, :], PN, num)

    groups = []

    for i in range(num - 1):
        indices = knn2(indices, rows, cols, Pstepsize, indices[0][0], Y)
        min_indices = find_min_indices(indices, len(groups_edges[i]))
        indices_set.append(indices[:, min_indices])
        indices = np.delete(indices, min_indices, axis=1)
    indices_set.append(indices)

    for item in indices_set:
        for i in range(int(np.ceil(item.shape[1] / PN))):
            item = knn2(item, rows, cols, Pstepsize, item[0][0], Y)
            min_indices = find_min_indices(item, PN)
            groups.append(item[:, min_indices][0, :])
            item = np.delete(item, min_indices, axis=1)

    return groups


    # groups_edges = split_average(indices[0, :], PN, num)
    # indices_set = []
    # for i in range(num - 1):
    #     indices_split_arrays = np.array_split(indices, num, axis=1)
    #
    #     indices_rdd = sc.parallelize(indices_split_arrays, num)
    #
    #     indices_list = indices_rdd.map(lambda x: knn(x, rows, cols, patsize, indices[0][0], Y_broadcast)).collect()
    #
    #     # for k in len(indices_list):
    #     indices = np.concatenate(indices_list, axis=1)
    #
    #     min_indices = find_min_indices(indices, len(groups_edges[i]))
    #     indices_set.append(indices[:, min_indices])
    #
    #     indices = np.delete(indices, min_indices, axis=1)
    #
    # indices_set.append(indices)
    #
    # groupsRDD = sc.parallelize(indices_set, num)
    #
    # groups = groupsRDD.flatMap(lambda x: partitionGroup(x, rows, cols, patsize, Y_broadcast, PN)).collect()
    # t_end = time.time()
    # print("分组，用时%f秒" % (t_end - t_start))
    # return groups

def calEuclidean(x, y):
    dist = np.sqrt(np.sum(np.square(x-y)))   # 注意：np.array 类型的数据可以直接进行向量、矩阵加减运算。np.square 是对每个元素求平均~~~~
    return dist

#indices [][3],
def knn(indices, rows, cols, Pstepsize, index, Y_broadcast):
    # print('knn+ =')
    # extract thte patches
    # get the vectoer
    Y = Y_broadcast.value
    row = int(index % rows)
    col = int((index - row) / rows)
    print(row, col, index)
    patch = Y[row: row + Pstepsize, col: col + Pstepsize, :]
    nn = patch.shape
    vec = np.reshape(patch, [nn[0] * nn[1] * nn[2]], order='F')

    for i in range(indices.shape[1]):
        row = int(indices[0][i] % rows)
        col = int((indices[0][i] - row) / rows)
        patch = Y[row: row + Pstepsize, col: col + Pstepsize, :]
        nn = patch.shape
        cur_vector = np.reshape(patch, [nn[0] * nn[1] * nn[2]], order='F')
        # print(cur_vector.shape)
        # print(vec.shape)
        distance = calEuclidean(cur_vector, vec)
        # print(distance.shape)
        indices[1][i] = distance

    # 找到第三列不是-1的元素的索引
    # column_indices = np.where(indices[2, :] == -1)[0]
    return indices


def knn2(indices, rows, cols, Pstepsize, index, Y):
    row = int(index % rows)
    col = int((index - row) / rows)
    print(row, col, index)
    patch = Y[row: row + Pstepsize, col: col + Pstepsize, :]
    nn = patch.shape
    vec = np.reshape(patch, [nn[0] * nn[1] * nn[2]], order='F')

    for i in range(indices.shape[1]):
        row = int(indices[0][i] % rows)
        col = int((indices[0][i] - row) / rows)
        patch = Y[row: row + Pstepsize, col: col + Pstepsize, :]
        nn = patch.shape
        cur_vector = np.reshape(patch, [nn[0] * nn[1] * nn[2]], order='F')
        # print(cur_vector.shape)
        # print(vec.shape)
        distance = calEuclidean(cur_vector, vec)
        # print(distance.shape)
        indices[1][i] = distance

    # 找到第三列不是-1的元素的索引
    # column_indices = np.where(indices[2, :] == -1)[0]
    return indices

# arr is an 1xN array
def split_average(arr, PN, num):
    # arr = np.arange(121)
    # PN = 11
    # num = 4

    # the number of Patch group
    num_groups = np.ceil(len(arr) / PN)
    # each partition have num_partitions patch
    num_partitions = int(np.ceil(num_groups / num))
    indices_spilt = []
    for i in range(num):
        start = int(i * PN * num_partitions)
        end = int((i + 1) * PN * num_partitions)
        # print(start, end)
        indices_spilt.append(arr[start: end])
    # print(num_partitions)
    return indices_spilt


def indices2Patch(img, indices, Pstepsize, rows, cols):
    W = img.shape[0] # 获取列宽
    Patch = np.zeros((Pstepsize, Pstepsize, img.shape[2], len(indices)))
    for i in range(len(indices)):
        # col = int(np.floor(indices[i] / (W - patchsize + 1)))
        # row = int(indices[i] - (W - patchsize + 1) * col)
        # cube = img[row: row + patchsize, col: col + patchsize, :]
        row = int(indices[i] % rows)
        col = int((indices[i] - row) / rows)
        patch = img[row: row + Pstepsize, col: col + Pstepsize, :]
        cube = np.transpose(patch, [1, 0, 2])
        Patch[:, :, :, i] = cube
        # if (indices[i] == 100):
        #     print("indices to patch")
        #     print(cube)
    return Patch


def getW_Imge_Matrix(nn, rows, cols, patsize):
    W_Img = np.zeros([nn[0], nn[1]])
    for index in range(int(rows * cols)):
        row = int(index % rows)
        col = int((index - row) / rows)
        W_Img[row: patsize + row, col: patsize + col] = \
            W_Img[row: patsize + row, col: patsize + col] + np.ones((patsize, patsize))
    return W_Img