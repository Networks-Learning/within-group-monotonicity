"""
Train the Logistic Regression Classifier, f_{lr}, from Training Data
"""
import argparse
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from utils import calculate_expected_qualified, calculate_expected_selected, transform_except_last_dim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str, help="the input training data path")
    parser.add_argument("--cal_data_path", type=str, help="the input calibration data path")
    parser.add_argument("--test_raw_path", type=str, help="the raw test data for sampling test data")
    parser.add_argument("--lbd", type=float, help="L2 regularization parameter")
    parser.add_argument("--Z_indices", type=str, default="", help="features defining the group membership")
    parser.add_argument("--classifier_path", type=str, help="the output classifier path")
    parser.add_argument("--scaler_path", type=str, help="the path for the scaler")

    args = parser.parse_args()
    Z_indices = [int(index) for index in args.Z_indices.split('_')]

    with open(args.train_data_path, "rb") as f:
        X, y = pickle.load(f)
        available_features = np.setdiff1d(np.arange(X.shape[1]), Z_indices)
        X = X[:, available_features]
        n = y.shape[0]
        C = 1 / (args.lbd * n)

    with open(args.cal_data_path, 'rb') as f:
        X_cal, y_cal = pickle.load(f)
        X_cal = X_cal[:, available_features]

    classifier = LogisticRegression(C=C).fit(X, y)

    with open(args.classifier_path, "wb") as f:
        pickle.dump(classifier, f)
