
import findspark
findspark.init()
import pyspark
from pyspark import SparkConf, SparkContext
import numpy as np
import tensorly as tl
conf = SparkConf().setMaster("local")
conf.set("spark.executor.memory", "4g")
conf.set("spark.driver.memory", "10g")  # 这里是增加jvm的内存
conf.set("spark.driver.maxResultSize", "10g")  # 这里是最大显示结果，这里是提示我改的。

sc = SparkContext(conf=conf)

import random
def change(x, y_broadcast):
    x[0] = 2
    print(y_broadcast.value)
    return x
main_list = []

for i in range(100):
    main_list.append(i)

def add(iter):
    t = np.random.random([10, 10])
    t = tl.tensor(t)
    return [t]
# def tt(x):
#     array = np.array([])
#     for i in x:
#         array.
#


m = np.random.random([20, 10, 5])
rdd = sc.parallelize(m.T, 4)
rdd.map(lambda x: print(x.shape)).collect()



x = rdd.mapPartitions(lambda x: add(x)).fold(0, lambda x, y: x + y)
print(x)

# for _ in range(10):  # 初始化10个元素
#     random_list = random.sample(range(100), 3)  # 从0到99的范围内选择3个随机数
#     print(random_list)
#     main_list.append(random_list)
# y_broadcast = sc.broadcast(3)
# print('************************************************')
# rdd = sc.parallelize(main_list, 4)
# rdd2 = rdd.map(lambda x: change(x, y_broadcast)).cache()
# res = rdd2.map(lambda x: x[2]).collect()
# y_broadcast = sc.broadcast(4)
# print(res)
# print('************************************************')
# rdd2.map(lambda x: print(x, y_broadcast.value)).collect()

