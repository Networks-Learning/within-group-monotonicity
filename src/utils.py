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


