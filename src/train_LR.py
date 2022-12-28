"""
Train a Noisy Logistic Regression Classifier from Training Data
"""
import argparse
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error,accuracy_score
from utils import calculate_expected_qualified, calculate_expected_selected, transform_except_last_dim



class NoisyLR(LogisticRegression):
    def set_noise_ratio(self, noise_ratio=None):
        self.noise_ratio = noise_ratio

    def predict_proba(self, X):
        proba = super().predict_proba(X)
        if self.noise_ratio is not None:
            for i in range(proba.shape[0]):
                # print(X[i, -1])
                if int(X[i, -1]) == 1:
                    noise_or_not = np.random.binomial(1, self.noise_ratio["maj"])
                else:
                    noise_or_not = np.random.binomial(1, self.noise_ratio["min"])
                if noise_or_not:
                    noise = np.random.beta(1, 4)
                    proba[i, 0] = 1. - noise
                    proba[i, 1] = noise
        return proba


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str, help="the input training data path")
    parser.add_argument("--cal_data_path", type=str, help="the input calibration data path")
    parser.add_argument("--test_raw_path", type=str, help="the raw test data for sampling test data")
    parser.add_argument("--lbd", type=float, help="L2 regularization parameter")
    # parser.add_argument("--Z_indices", nargs="*", default = [], help="features defining the group membership")
    parser.add_argument("--Z_indices", type=str, default="", help="features defining the group membership")
    # parser.add_argument("--C", type=float, help="L2 regularization parameter")
    parser.add_argument("--classifier_path", type=str, help="the output classifier path")
    parser.add_argument('--noise_ratio_maj', type=float, default=0., help="noise ratio of majority group")
    parser.add_argument('--noise_ratio_min', type=float, default=-1., help="noise ratio of minority group")
    parser.add_argument("--scaler_path", type=str, help="the path for the scaler")

    args = parser.parse_args()
    Z_indices = [int(index) for index in args.Z_indices.split('_')]
    # print(args.Z_indices)
    # print(Z_indices)
    # print(args.Z_indices.split(','))
    # Z_indices = args.Z_indices.replace(',',' ')
    # print(Z_in)


    with open(args.train_data_path, "rb") as f:
        X, y = pickle.load(f)
        available_features = np.setdiff1d(np.arange(X.shape[1]),Z_indices)
        X = X[:,available_features]
        # X = X[:,:-1]
        # print(X.shape)
        n = y.shape[0]
        C = 1 / (args.lbd * n)
        # C = args.C

    with open(args.cal_data_path, 'rb') as f:
        X_cal, y_cal = pickle.load(f)
        X_cal = X_cal[:,available_features]
        # print(X_cal.shape)


    if args.noise_ratio_min < 0.:
        classifier = LogisticRegression(C=C).fit(X, y)

        print("---calibration---")
        # print("----MSE")
        # print(mean_squared_error(classifier.predict_proba(X_cal)[:,1], y_cal * 1.))
        print("----Accuracy")
        print(accuracy_score(classifier.predict(X_cal), y_cal))

    else:
        classifier = NoisyLR(C=C).fit(X, y)
        noise_ratio = {}
        noise_ratio["maj"] = args.noise_ratio_maj
        noise_ratio["min"] = args.noise_ratio_min
        classifier.set_noise_ratio(noise_ratio)
        print("---Train---")
        print("----MSE")
        print(mean_squared_error(classifier.predict_proba(X)[:, 1], y * 1.))
        print("----Accuracy")
        print(accuracy_score(classifier.predict(X), y))


        print("---calibration---")
        print("----MSE")
        print(mean_squared_error(classifier.predict_proba(X_cal)[:, 1], y_cal * 1.))
        print("----Accuracy")
        print(accuracy_score(classifier.predict(X_cal), y_cal))


    with open(args.classifier_path, "wb") as f:
        pickle.dump(classifier, f)
