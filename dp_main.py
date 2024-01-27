import findspark
findspark.init()

from pyspark import SparkConf, SparkContext

import sys

# This is a sample Python script.
import time

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from dataReader import readData, loads
from NPTCP import NPTCP
from DP_NPTCP import DP_NPTCP4
import h5py
import numpy as np
import pickle
from function import QualityIndices, PSNR3D
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

import logging
logging.basicConfig(
    filemode='/data2/zp/nptcp.log',
    format='%(asctime)s - %(name)s - %(pathname)s - %(lineno)d - %(levelname)s - %(message)s'
)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    t_start = time.time()

    conf = SparkConf().setMaster("spark://com1:7077")
    # conf = SparkConf().setMaster("local")
    # conf.set("spark.executor.memory", "10g")
    # conf.set("spark.driver.memory", "20g")  # 这里是增加jvm的内存
    # conf.set("spark.driver.maxResultSize", "20g")  # 这里是最大显示结果，这里是提示我改的。
    # #
    sc = SparkContext(conf=conf)

    # PN = 300
    # Rank = 80
    # ratio = 5
    # num = 4

    PN = int(sys.argv[1])
    Rank = int(sys.argv[2])
    ratio = int(sys.argv[3])
    num = int(sys.argv[4])
    file = sys.argv[5]

    #
    I_REF, MSI, HSI, R = readData(file)

    s = loads()

    t1 = time.time()

    # I_CTD = NPTCP(HSI, Rank, MSI, ratio, PN, R)
    I_CTD = DP_NPTCP4(HSI, Rank, MSI, ratio, PN, R, sc, num, s)

    t_end = time.time()

    logging.info("all time: %d", t_end - t_start)

    # hf = h5py.File("./data/I_CTD.mat", 'r')
    # I_CTD = np.array(hf["I_CTD"]).T

    # hf = h5py.File("./data/I_REF.mat", 'r')
    # I_REF = np.array(hf["I_REF"]).T

    QualityIndices(I_CTD, I_REF, ratio)
    AM = np.max(I_REF)
    psnr = PSNR3D(I_CTD * 255 / AM, I_REF * 255 / AM)
    logging.info('psnr: ', psnr)
    print('psnr: ', psnr)

    t2 = time.time()
    print('time(s): ', t2 - t1)
    logging.info('time(s): ', t2 - t1)
    filename = str(time.time())

    with open(filename, 'wb') as file:
        pickle.dump(I_CTD, file)

