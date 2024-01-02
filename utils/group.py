import sys
import time

import pickle
import numpy as np
import ray
import matplotlib.pyplot as plt

from actors.FactorActor import FactorActor
from utils.nonlocal_function import find_min_indices, split_average, getW_Imge_Matrix
from utils.tools import upsample, gaussian
from tasks.GroupTask import knn, partitions_group


def group(PN, num, rows, cols, patsize, Y_ref):
    indices = np.zeros((2, rows * cols)).astype('int')
    indices[0] = np.arange(rows * cols)
    indices[1] = 1000000
    groups_edges = split_average(indices[0, :], PN, num)
    indices_set = []
    # Y = ray.get(Y_ref)
    indices_set = []
    for i in range(num - 1):
        indices_split_arrays = np.array_split(indices, num, axis=1)
        task_ids = [knn.remote(item, rows, cols, patsize, indices[0][0], Y_ref) for item in indices_split_arrays]
        indices_list = ray.get(task_ids)
        indices = np.concatenate(indices_list, axis=1)
        min_indices = find_min_indices(indices, len(groups_edges[i]))
        indices_set.append(indices[:, min_indices])
        indices = np.delete(indices, min_indices, axis=1)
    indices_set.append(indices)
    # 创建任务
    task_ids = [partitions_group.remote(item, rows, cols, patsize, Y_ref, PN) for item in indices_set]
    groups = ray.get(task_ids)
    return groups