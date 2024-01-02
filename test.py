import tensorly as tl
import numpy as np
from utils.tensor_function import ttm
# 创建一个 3x3x3 的张量
# dim1, dim2, dim3 = 3, 3, 3
#
# # 使用 arange 生成连续的值，然后将其重塑为三维张量
# tensor = np.arange(dim1 * dim2 * dim3).reshape(dim1, dim2, dim3)
# matrix = np.arange(dim1 * dim2).reshape(dim1, dim2)
#
# # print(tensor)
# # print(matrix)
#
# res = ttm(tensor, matrix, 0)
# # print(tensor)
# # print(matrix)
# print(res)
#
# G = np.random.random([10, 40, 20])
# U1 = np.random.random([100, 10])
# U2 = np.random.random([100, 40])
#
# G = tl.tenalg.mode_dot(G, U1, mode=0)
# G = tl.tenalg.mode_dot(G, U2, mode=1)
#
# print(G.shape)


# T = np.random.random([100, 100, 10])
# B = np.random.random([5, 100])
# C = np.random.random([6, 100])
#
# T1 = tl.tenalg.mode_dot(T, B, mode=0)
# T1 = tl.tenalg.mode_dot(T1, C, mode=1)
# T1 = tl.unfold(T1, mode=2)
# T2 = np.dot(tl.unfold(T, mode=2), np.kron(B, C).T)
U1 = np.random.random([110, 10])
U2 = np.random.random([120, 20])
U3 = np.random.random([130, 30])
B1 = np.random.random([10, 20, 40])
B2 = np.random.random([40, 30])

a = tl.tenalg.mode_dot(B1, U1, mode=0)
a = tl.tenalg.mode_dot(a, U2, mode=1)
a = tl.tenalg.mode_dot(a, B2.T, mode=2)
a = tl.tenalg.mode_dot(a, U3, mode=2)
c = a


a = tl.tenalg.mode_dot(B1, U1, mode=0)
a = tl.tenalg.mode_dot(a, U2, mode=1)
a = tl.tenalg.mode_dot(a, np.dot(B2, U3.T).T, mode=2)

b = tl.tenalg.mode_dot(B1, U2, mode=1)
b = tl.tenalg.mode_dot(b, B2.T, mode=2)
b = tl.tenalg.mode_dot(b, U3, mode=2)
b = tl.tenalg.mode_dot(b, U1, mode=0)





c = tl.unfold(a, mode=0)
d = np.dot(U1, tl.unfold(b, mode=0))




print()












