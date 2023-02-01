"""
Generate Data For Each Run
"""
import argparse
import pickle
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import numpy as np
from exp_utils import transform_except_last_dim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train", type=int, help="number of training examples")
    parser.add_argument("--n_cal", type=int, help="number of calibration examples")
    parser.add_argument("--train_cal_raw_path", type=str, help="raw data for sampling train and calibration data")
    parser.add_argument("--train_data_path", type=str, help="the path for saving the training data")
    parser.add_argument("--cal_data_path", type=str, help="the path for saving the calibration data")
    parser.add_argument("--scaler_path", type=str, help="the path for saving the scaler")

    args = parser.parse_args()
    n_train = args.n_train
    n_cal = args.n_cal
    n = n_train + n_cal
    with open(args.train_cal_raw_path, 'rb') as f:
        X, y = pickle.load(f)

    X, y = shuffle(X, y)
    X, y = X[:n], y[:n]
    scaler = StandardScaler()
    X_train, y_train = X[:n_train], y[:n_train]
    X_cal, y_cal = X[n_train:], y[n_train:]
    X_train = np.concatenate((scaler.fit_transform(X_train[:, :-1]), X_train[:, -1:]), axis=1)
    X_cal = transform_except_last_dim(X_cal, scaler)

    with open(args.train_data_path, "wb") as f:
        pickle.dump([X_train, y_train], f)
    with open(args.cal_data_path, "wb") as f:
        pickle.dump([X_cal, y_cal], f)
    with open(args.scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
