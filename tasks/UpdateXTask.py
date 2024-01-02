import time

import ray
import numpy as np
import sys

from utils.nonlocal_function import find_min_indices, knn2
from utils.tools import calEuclidean, cgsolve2

@ray.remote(num_cpus=1)
def cg(indices, rr, mu, rate, s):
    print("分区开始运行CG")
    t1 = time.time()
    # maxtrix = []
    # for rr in rr_list:
    #     maxtrix.append(rr.T)
    rr_matrix = rr[:, :, indices[0]: indices[-1] + 1]
    # rr_matrix = np.stack(maxtrix, axis=2)
    nn = rr_matrix.shape
    rr = np.reshape(rr_matrix, [nn[0] * nn[1] * nn[2]], order='F')
    res = cgsolve2(rr, nn, mu, rate, s)

    t2 = time.time()
    print("分区更新完成，用时%d秒" % (t2 - t1))
    return res
