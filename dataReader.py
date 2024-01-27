import numpy as np
import scipy.io as sio
import h5py

from function import gaussian, downsample


# read data from /data
def readData(filename):
    R = loadR()
    s = loads()
    if(filename == "DC50"):
        hf = h5py.File("./data/dc50.mat", 'r')
        HH = np.array(hf["I_REF"]).T
        nn = HH.shape
    elif filename == 'DC':
        hf = h5py.File("./data/I_REF.mat", 'r')
        HH = np.array(hf["I_REF"]).T
        nn = HH.shape
    elif (filename == "P"):
        hf = h5py.File("./data/Pavia_HH.mat", 'r')
        HH = np.array(hf["I_REF"]).T
        nn = HH.shape
        R = R[:, 0: nn[2]]
    else:
        print("filename must be DC50, DC, and P")
        return
    I_temp = np.reshape(HH, [nn[0] * nn[1], nn[2]], order='F')
    I_ms = np.dot(R, I_temp.T)
    MSI = np.reshape(I_ms.T, [nn[0], nn[1], R.shape[0]], order='F')

    I_HSn = gaussian(HH, s)
    HSI = downsample(I_HSn, 5)

    return HH, MSI, HSI, R


def loadR():
    hf = h5py.File("./data/R.mat", 'r')
    R = np.array(hf["R"])
    return R.T


def loads():
    hf = h5py.File("./data/s.mat", 'r')
    R = np.array(hf["s"])
    return R.T