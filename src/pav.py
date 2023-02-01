"""
Recalibrates a given calibrated classifier so that it is within-group monotone using Algorithm 1
"""
import argparse
import pickle
import numpy as np
# from umb_ss import UMBSelect
from partition import BinPartition
from utils import *
from sklearn.metrics import mean_squared_error, accuracy_score


class PAV(BinPartition):
    def __init__(self, n_bins, Z_indices, groups, Z_map, alpha):
        super().__init__(n_bins, Z_indices, groups, Z_map, alpha)

    def _find_potential_merges(self):
        S = np.ones(shape=(self.n_bins, self.n_bins, self.n_bins))
        for l in range(1, self.n_bins):
            for r in range(l, self.n_bins):
                lr_positives, lr_total, lr_group_positives, lr_group_total, lr_group_rho = self._get_merged_statistics(
                    l, r)
                for k in range(l):
                    kl_positives, kl_total, kl_group_positives, kl_group_total, kl_group_rho = self._get_merged_statistics(
                        k, l - 1)
                    if np.where(np.logical_and(lr_group_rho, kl_group_rho),
                                np.greater(kl_group_positives * lr_group_total, lr_group_positives * kl_group_total), \
                                np.zeros(shape=kl_group_rho.shape)).any():  # can be adjacent
                        S[l][r][k] = 0
        return S

    def recalibrate(self):
        S = self._find_potential_merges()
        mid_point = np.repeat(-2, self.n_bins).reshape(self.n_bins)
        mid_point[0] = -1

        for r in range(1, self.n_bins):
            l = r - 1
            while l >= 0:
                assert mid_point[l] != -2, f"{l, r}"
                if not S[l + 1, r, mid_point[l] + 1]:
                    if mid_point[l] != -1:
                        l = mid_point[l]
                    else:
                        mid_point[r] = -1
                        break
                else:
                    mid_point[r] = l
                    break

        return mid_point

    def get_optimal_partition(self, r):
        assert (self.mid_point is not None), "not yet recalibrated"
        if r == -1:
            return []

        assert (self.mid_point[r] != -2)
        return self.get_optimal_partition(self.mid_point[r]) + [self.mid_point[r] + 1]

    def fit(self, X_est, y_score, y, m):

        # fit the umb
        super().fit(X_est, y_score, y, m)

        # recalibrate using algorithm 2
        self.mid_point = self.recalibrate()

        # get optimal partition
        self.optimal_partition = self.get_optimal_partition(self.mid_point[self.n_bins - 1])

        self.recal_n_bins = len(self.optimal_partition)

        # get new upper edges
        recal_bin_assignment, self.recal_num_positives_in_bin, self.recal_num_in_bin, self.recal_group_num_positives_in_bin, \
        self.recal_group_num_in_bin = self.get_recal_bin_points(y_score)

        group_assignment = self.group_points(X_est)
        assert np.sum([recal_bin_assignment == i for i in range(self.recal_n_bins)]) == X_est.shape[
            0], f"{np.sum([recal_bin_assignment == i for i in range(self.recal_n_bins)]), X_est.shape[0]}"
        assert np.sum(group_assignment) == X_est.shape[0]

        self.recal_bin_rho = self.recal_num_in_bin / self.num_examples
        self.recal_bin_values = self.recal_num_positives_in_bin / self.recal_num_in_bin

        positive_group_rho = np.greater(self.recal_group_num_in_bin, np.zeros(shape=self.recal_group_num_in_bin.shape))
        self.recal_group_rho = np.where(positive_group_rho,
                                        (self.recal_group_num_in_bin) / (self.recal_num_in_bin)[:, np.newaxis],
                                        np.zeros(shape=self.recal_group_num_in_bin.shape))
        self.recal_group_bin_values = np.where(positive_group_rho,
                                               self.recal_group_num_positives_in_bin / self.recal_group_num_in_bin,
                                               np.zeros(shape=self.recal_group_num_in_bin.shape))

        self.find_discriminations()
        assert np.sum(self.recal_discriminated_against) == 0

        self.sanity_check()

        # find threshold bin and theta
        self.recal_b, self.recal_theta = self.get_recal_threshold(m)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cal_data_path", type=str, help="the input calibration data path")
    parser.add_argument("--test_raw_path", type=str, help="the raw test data for sampling test data")
    parser.add_argument("--classifier_path", type=str, help="the input classifier path")
    parser.add_argument("--Z_indices", type=str, default="", help="features defining the group membership")
    parser.add_argument("--pav_path", type=str, help="the output pav path")
    parser.add_argument("--result_path", type=str, help="the output selection result path")
    parser.add_argument("--k", type=float, help="the target expected number of qualified candidates")
    parser.add_argument("--m", type=float, help="the expected number of incoming candidates")
    parser.add_argument("--alpha", type=float, default=0.1, help="the failure probability")
    parser.add_argument("--B", type=int, help="the number of bins")
    parser.add_argument("--scaler_path", type=str, help="the path for the scaler")
    parser.add_argument("--n_runs_test", type=int, help="the number of tests for estimating the expectation")

    args = parser.parse_args()
    # k = args.k
    m = args.m
    alpha = args.alpha

    args = parser.parse_args()
    Z_indices = [int(index) for index in args.Z_indices.split('_')]


    def list_maker(val, n):
        return [val] * n


    Z_map = {
        15: [0] + [1] + list_maker(2, 3) + list_maker(3, 4),
        1: list_maker(0, 16) + list_maker(1, 4) + list_maker(2, 2) + list_maker(3, 3),
        0: list_maker(0, 25) + list_maker(1, 25) + list_maker(2, 25) + list_maker(3, 25),
        14: [0, 1],
        4: [0, 1],
        6: [0, 1, 2, 2, 3],
        2: [0, 0, 0, 0, 1]
    }

    with open(args.cal_data_path, 'rb') as f:
        X_cal_all_features, y_cal = pickle.load(f)
        available_features = np.setdiff1d(np.arange(X_cal_all_features.shape[1]), Z_indices)
        X_cal = X_cal_all_features[:, available_features]

    groups = np.unique(X_cal_all_features[:, Z_indices])

    with open(args.classifier_path, "rb") as f:
        classifier = pickle.load(f)

    n = y_cal.size
    scores_cal = classifier.predict_proba(X_cal)[:, 1]

    pav = PAV(args.B, Z_indices, groups, Z_map, alpha)
    pav.fit(X_cal_all_features, scores_cal, y_cal, m)

    # test
    with open(args.test_raw_path, "rb") as f:
        X_test_all_features, y_test_raw = pickle.load(f)
    with open(args.scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    X_test_all_features = transform_except_last_dim(X_test_all_features, scaler)
    X_test_raw = X_test_all_features[:, available_features]
    scores_test_raw = classifier.predict_proba(X_test_raw)[:, 1]

    accuracy = np.empty(len(ks))
    f1score = np.empty(len(ks))
    for k_idx, k in enumerate(ks):
        accuracy[k_idx], f1score[k_idx] = pav.get_accuracy(scores_test_raw, y_test_raw)

    # simulating pools of candidates
    num_selected = np.empty(shape=(len(ks), args.n_runs_test))
    num_qualified = np.empty(shape=(len(ks), args.n_runs_test))
    for i in range(args.n_runs_test):
        indexes = np.random.choice(list(range(y_test_raw.size)), int(m))
        y_test = y_test_raw[indexes]
        scores_test = scores_test_raw[indexes]
        for k_idx, k in enumerate(ks):
            test_selected = pav.recal_select(scores_test, k_idx)
            num_selected[k_idx][i] = calculate_expected_selected(test_selected, y_test, m)
            num_qualified[k_idx][i] = calculate_expected_qualified(test_selected, y_test, m)

    performance_metrics = {}
    performance_metrics["num_qualified"] = np.mean(num_qualified, axis=1)
    performance_metrics["num_selected"] = np.mean(num_selected, axis=1)
    assert (performance_metrics["num_qualified"].shape[0] == len(ks) and performance_metrics["num_selected"].shape[
        0] == len(ks))

    performance_metrics["accuracy"] = accuracy
    performance_metrics["f1_score"] = f1score
    performance_metrics["num_positives_in_bin"] = pav.recal_num_positives_in_bin
    performance_metrics["num_in_bin"] = pav.recal_num_in_bin
    performance_metrics["bin_values"] = pav.recal_bin_values
    performance_metrics["group_num_positives_in_bin"] = pav.recal_group_num_positives_in_bin
    performance_metrics["group_num_in_bin"] = pav.recal_group_num_in_bin
    performance_metrics["group_bin_values"] = pav.recal_group_bin_values
    performance_metrics["group_rho"] = pav.recal_group_rho
    performance_metrics["bin_rho"] = pav.recal_bin_rho
    performance_metrics["groups"] = pav.groups
    performance_metrics["num_groups"] = pav.num_groups
    performance_metrics["n_bins"] = pav.recal_n_bins
    performance_metrics["discriminated_against"] = pav.recal_discriminated_against
    performance_metrics["alpha"] = 0

    with open(args.result_path, 'wb') as f:
        pickle.dump(performance_metrics, f)

    with open(args.pav_path, "wb") as f:
        pickle.dump(pav, f)
