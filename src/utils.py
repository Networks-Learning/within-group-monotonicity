"""
Utils for Subset Selection
"""
import numpy as np
from sklearn.metrics import mean_squared_error,accuracy_score


def transform_except_last_dim(data, scaler):
    return np.concatenate((scaler.transform(data[:, :-1]), data[:, -1:]), axis=1)


def calculate_expected_qualified(s, y, m):
    return np.sum(s * y) * 1. * m / y.size


def calculate_expected_selected(s, y, m):
    return np.sum(s) * 1. * m / y.size

def list_maker(val,n):
    return [val]*n

Z_map = {
    0: list_maker(0,25) + list_maker(1,25) + list_maker(2,25) + list_maker(3,25),
    1: list_maker(0,16)+list_maker(1,4)+list_maker(2,2)+list_maker(3,3),
    2: [0,0,0,0,1],
    4: [0,1],
    6: [0,1,2,2,3],
    14: [0,1],
    15: [0] + [1] + list_maker(2,3) + list_maker(3,4)
}

ks = [5,10,15]


