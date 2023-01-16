import numpy as np
import pywt
import pickle
import time


def dwt(data, shape, wave_type):
    start = time.time()
    x = np.array(data, copy=False).astype(np.float32).reshape(shape)
    c = pywt.wavedecn(x, wave_type, mode="periodization")
    d = pywt.coeffs_to_array(c)
    end = time.time()
    print('pure python time without c++ binding', end - start)
    return {
        'shape': d[0].shape,
        'structure': pickle.dumps(d[1]),
        'data': np.ravel(d[0].astype(np.float32)).tolist()
    }


def idwt(data, shape, wave_structure, wave_type, ori_shape):
    data = np.array(data, copy=False).astype(np.float32).reshape(shape)
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
    return np.ravel(b.astype(np.float32)).tolist()
