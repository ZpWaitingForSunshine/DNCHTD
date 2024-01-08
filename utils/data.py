import numpy as np
import scipy.io as sio
import h5py
import platform
from utils.tools import gaussian, downsample

dir = "/data2/data/"
system = platform.system()
print(system)
if system == "Windows":
    dir = "./data/"

if system == "Darwin":
    dir = "./data/"

# read data from /data
def readData(filename):
    R = loadR()
    s = loads()

    frames = 10

    if(filename == "DC50"):
        hf = h5py.File(dir + "dc50.mat", 'r')
        HH = np.array(hf["I_REF"]).T
        HH = HH[:, 0:40, :]
        nn = HH.shape
    elif filename == 'DC':
        hf = h5py.File(dir + "I_REF.mat", 'r')
        HH = np.array(hf["I_REF"]).T
        nn = HH.shape
    elif (filename == "P"):
        hf = h5py.File(dir + "pavia.mat", 'r')
        HH = np.array(hf["I_REF"]).T
        nn = HH.shape
        R = R[:, 0: nn[2]]
    elif (filename == "M"):
        hf = h5py.File(dir + "m.mat", 'r')
        HH = np.array(hf["I_REF"]).T
        HH = HH[0: 1860, 0: 680, :]
        hf = h5py.File(dir + "R_186.mat", 'r')
        R = np.array(hf["R"]).T
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

    # HH = np.repeat(HH[..., np.newaxis], frames, axis=3)
    # HSI = np.repeat(HSI[..., np.newaxis], frames, axis=3)
    # MSI = np.repeat(MSI[..., np.newaxis], frames, axis=3)

    return HH, MSI, HSI, R


def loadR():
    hf = h5py.File(dir + "R.mat", 'r')
    R = np.array(hf["R"])
    return R.T


def loads():
    hf = h5py.File(dir + "s.mat", 'r')
    R = np.array(hf["s"])
    return R.T