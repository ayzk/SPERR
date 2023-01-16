import numpy as np
import pywt
import pickle
import time

structure = None


def dwt(data, wave_type):
    start = time.time()
    c = pywt.wavedecn(data, wave_type, mode="periodization")
    d = pywt.coeffs_to_array(c)
    end = time.time()
    print('pure python time without c++ binding', end - start)
    global shape, structure
    structure = pickle.dumps(d[1])
    return d[0].astype(np.float32)

def dwt_structure():
    global structure
    return structure


def idwt(data, wave_structure, wave_type, ori_shape):
    start = time.time()
    structure = pickle.loads(wave_structure)
    dc_c = pywt.array_to_coeffs(data, structure)
    b = pywt.waverecn(dc_c, wave_type, mode="periodization")
    if b.shape != ori_shape:
        if len(ori_shape) == 2:
            b = b[:ori_shape[0], :ori_shape[1]]
        else:
            b = b[:ori_shape[0], :ori_shape[1], :ori_shape[2]]
    end = time.time()
    print('pure python time without c++ binding', end - start)
    return b.astype(np.float32)
