# This is a sample Python script.
import time

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from dataReader import readData, loads
from NPTCP import NPTCP
from NPTCP4 import NPTCP4
import h5py
import numpy as np
import pickle
from function import QualityIndices, PSNR3D
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #  读取数据
    I_REF, MSI, HSI, R = readData('DC50')

    PN = 60
    Rank = 80
    ratio = 5
    s = loads()
    t1 = time.time()

    I_CTD = NPTCP(HSI, Rank, MSI, ratio, PN, R, s)
    # I_CTD = NPTCP4(HSI, Rank, MSI, ratio, PN, R, s)

    # hf = h5py.File("./data/I_CTD.mat", 'r')
    # I_CTD = np.array(hf["I_CTD"]).T

    # hf = h5py.File("./data/I_REF.mat", 'r')
    # I_REF = np.array(hf["I_REF"]).T

    QualityIndices(I_CTD, I_REF, ratio)
    AM = np.max(I_REF)
    psnr = PSNR3D(I_CTD * 255 / AM, I_REF * 255 / AM)
    print('psnr: ', psnr)

    t2 = time.time()
    print('time(s): ', t2 - t1)
    filename = "x_3order.pkl"

    with open(filename, 'wb') as file:
        pickle.dump(I_CTD, file)

    # print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
