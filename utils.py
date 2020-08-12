#!/usr/bin/python

import numpy as np
from scipy.cluster.vq import whiten


def loadData(fname):
    data = np.loadtxt(fname, dtype=np.float32)

    x = data[:, 1:]
    y = data[:, 0]
    '''
    # normalization
    mean = np.mean(x, 0)
    std  = np.std(x, 0)
    mean[3], mean[5], mean[6] = 0., 0., 0.
    std[3], std[5], std[6] = 1., 1., 1.
    print(mean.shape)
    mean = np.expand_dims(mean, 0)
    std = np.expand_dims(std, 0)
    x = (x - mean) / std
    '''
    # x = whiten(x)
    n = y.shape[0]
    labels = np.zeros((n), dtype=np.int32)
    boundary = [0, 15, 30, 45, 60, 600]
    multiclass_labels = np.zeros((n, 4))
    for i in range(len(boundary)-1):
        idx = np.logical_and(y > boundary[i], y <= boundary[i+1])
        labels[idx] = i
    for i in range(n):
        multiclass_labels[i, labels[i]:] = 1
    log_igfr = np.log(y)
    return x, log_igfr, multiclass_labels
