"""
Recalibrates a given calibrated classifier so that it is within-group calibrated using Algorithm 3.
"""
import argparse
import pickle
import numpy as np
# from umb_ss import UMBSelect
from partition import BinPartition
from utils import *
from sklearn.metrics import mean_squared_error, accuracy_score


class WGC(BinPartition):
    def __init__(self, n_bins, Z_indices, groups, Z_map, alpha):
        super().__init__(n_bins, Z_indices, groups, Z_map, alpha)
        self.eps = None

    def _find_potential_merges(self, eps):
        S = np.ones(shape=(self.n_bins, self.n_bins))
        for l in range(self.n_bins):
            for r in range(l, self.n_bins):
                lr_positives, lr_total, lr_group_positives, lr_group_total, lr_group_rho = self._get_merged_statistics(
                    l, r)
                lr_positives = np.repeat(lr_positives, lr_group_positives.shape[0]).reshape(lr_group_positives.shape)
                lr_total = np.repeat(lr_total, lr_group_positives.shape[0]).reshape(lr_group_positives.shape)
                if np.where(lr_group_rho,
                            (np.abs(lr_group_positives / lr_group_total - lr_positives / lr_total) > eps),
                            np.zeros(shape=lr_group_rho.shape)).any():  # can be adjacent
                    S[l][r] = 0
        return S

    def recalibrate(self):
        low = 0
        high = 100
        eps = (low + high) // 2
        final_dp = np.zeros(shape=(self.n_bins))
        final_mid_point = np.repeat(-2, self.n_bins).reshape(self.n_bins)
        while low <= high:
            mid = (low + high) // 2
            S = self._find_potential_merges(mid / 100)
            dp = np.zeros(shape=(self.n_bins))
            mid_point = np.repeat(-2, self.n_bins).reshape(self.n_bins)

            for r in range(self.n_bins):
                if S[0][0]:
                    dp[0] = 1
                    mid_point[0] = -1

            for l in range(1, self.n_bins):
                for r in range(l, self.n_bins):
                    # for k in range(l):
                    # print(f"{l =} { r =} {k = } {np.sum(S[l][r])}")
                    if S[l][r] and dp[l - 1] != 0 and dp[l - 1] + 1 > dp[r]:
                        dp[r] = dp[l - 1] + 1
                        mid_point[r] = l - 1
                        # print(f"{l, r}")
            if mid_point[self.n_bins - 1] != -2:
                final_dp = dp
                final_mid_point = mid_point
                eps = mid
                high = mid - 1
            else:
                low = mid + 1

        return final_dp, final_mid_point, eps / 100

    def get_optimal_partition(self, r):
        assert (self.dp is not None and self.mid_point is not None), "not yet recalibrated"
        if r == -1:
            return []

        assert (self.mid_point[r] != -2)

        return self.get_optimal_partition(self.mid_point[r]) + [self.mid_point[r] + 1]

    def fit(self, X_est, y_score, y, m):

        # fit the umb
        super().fit(X_est, y_score, y, m)

        # recalibrate using algorithm 2
        self.dp, self.mid_point, self.eps = self.recalibrate()

        # get optimal partition
        self.optimal_partition = self.get_optimal_partition(self.n_bins - 1)
        # print(self.optimal_partition)

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

        # sanity check
        self.sanity_check()

        # find threshold bin and theta
        self.recal_b, self.recal_theta = self.get_recal_threshold(m)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cal_data_path", type=str, help="the input calibration data path")
    parser.add_argument("--test_raw_path", type=str, help="the raw test data for sampling test data")
    parser.add_argument("--classifier_path", type=str, help="the input classifier path")
    parser.add_argument("--Z_indices", type=str, default="", help="features defining the group membership")
    parser.add_argument("--wgc_path", type=str, help="the output wgc classifier path")
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

    wgc = WGC(args.B, Z_indices, groups, Z_map, alpha)
    wgc.fit(X_cal_all_features, scores_cal, y_cal, m)

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
        accuracy[k_idx], f1score[k_idx] = wgc.get_accuracy(scores_test_raw, y_test_raw)

    # simulating pools of candidates
    num_selected = np.empty(shape=(len(ks), args.n_runs_test))
    num_qualified = np.empty(shape=(len(ks), args.n_runs_test))
    for i in range(args.n_runs_test):
        indexes = np.random.choice(list(range(y_test_raw.size)), int(m))
        y_test = y_test_raw[indexes]
        scores_test = scores_test_raw[indexes]
        for k_idx, k in enumerate(ks):
            test_selected = wgc.recal_select(scores_test, k_idx)
            num_selected[k_idx][i] = calculate_expected_selected(test_selected, y_test, m)
            num_qualified[k_idx][i] = calculate_expected_qualified(test_selected, y_test, m)

    performance_metrics = {}
    performance_metrics["num_qualified"] = np.mean(num_qualified, axis=1)
    performance_metrics["num_selected"] = np.mean(num_selected, axis=1)
    assert (performance_metrics["num_qualified"].shape[0] == len(ks) and performance_metrics["num_selected"].shape[
        0] == len(ks))
    performance_metrics["accuracy"] = accuracy
    performance_metrics["f1_score"] = f1score
    performance_metrics["num_positives_in_bin"] = wgc.recal_num_positives_in_bin
    performance_metrics["num_in_bin"] = wgc.recal_num_in_bin
    performance_metrics["bin_values"] = wgc.recal_bin_values
    performance_metrics["group_num_positives_in_bin"] = wgc.recal_group_num_positives_in_bin
    performance_metrics["group_num_in_bin"] = wgc.recal_group_num_in_bin
    performance_metrics["group_bin_values"] = wgc.recal_group_bin_values
    performance_metrics["group_rho"] = wgc.recal_group_rho
    performance_metrics["bin_rho"] = wgc.recal_bin_rho
    performance_metrics["groups"] = wgc.groups
    performance_metrics["num_groups"] = wgc.num_groups
    performance_metrics["n_bins"] = wgc.recal_n_bins
    performance_metrics["discriminated_against"] = wgc.recal_discriminated_against
    performance_metrics["alpha"] = wgc.eps

    with open(args.result_path, 'wb') as f:
        pickle.dump(performance_metrics, f)

    with open(args.wgc_path, "wb") as f:
        pickle.dump(wgc, f)
