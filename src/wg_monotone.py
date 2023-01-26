"""
Recalibrates a given calibrated classifier so that it is within-group monotone using Algorithm2.
"""
import argparse
import pickle
import numpy as np
from umb import UMBSelect
from partition import BinPartition
from utils import *
from sklearn.metrics import mean_squared_error,accuracy_score,roc_curve, roc_auc_score



class WGM(BinPartition):
    def __init__(self, n_bins, Z_indices, groups, Z_map,alpha):
        super().__init__(n_bins, Z_indices, groups, Z_map,alpha)


    def _find_potential_merges(self):
        S = np.ones(shape=(self.n_bins,self.n_bins,self.n_bins))
        for l in range(1,self.n_bins):
            for r in range(l,self.n_bins):
                lr_positives, lr_total, lr_group_positives, lr_group_total, lr_group_rho = self._get_merged_statistics(l,r)
                for k in range(l):
                    kl_positives, kl_total, kl_group_positives, kl_group_total, kl_group_rho = self._get_merged_statistics(k, l-1)
                    if np.where(np.logical_and(lr_group_rho,kl_group_rho), np.greater(kl_group_positives*lr_group_total,lr_group_positives * kl_group_total),\
                                np.zeros(shape=kl_group_rho.shape)).any():  #can be adjacent
                        S[l][r][k] = 0
        return S


    def recalibrate(self):
        S = self._find_potential_merges()
        dp = np.zeros(shape = (self.n_bins,self.n_bins))
        mid_point = np.repeat(-2,self.n_bins*self.n_bins).reshape(self.n_bins,self.n_bins)

        for r in range(self.n_bins):
            dp[0][r] = 1
            mid_point[0][r] = -1

        for l in range(1,self.n_bins):
            for r in range(l,self.n_bins):
                for k in range(l):
                    # print(f"{l =} { r =} {k = } {np.sum(S[l][r])}")
                    if S[l][r][k] and mid_point[k][l-1]!=-2 and dp[k][l-1]+1>dp[l][r]:
                        dp[l][r] = dp[k][l-1] + 1
                        mid_point[l][r] = k

        return dp, mid_point

    def get_optimal_partition(self,l,r):
        assert (self.dp is not None and self.mid_point is not None), "not yet recalibrated"
        if l == 0:
            return [0]

        assert(self.mid_point[l][r] != -2), f"{self.mid_point[l][r]}"

        return self.get_optimal_partition(self.mid_point[l][r],l-1) + [l]


    def fit(self, X_est, y_score, y, m):

        #fit the umb
        super().fit(X_est, y_score, y, m)

        #recalibrate using algorithm 2
        self.dp, self.mid_point = self.recalibrate()

        #get optimal partition
        recal_n_bins = -1
        l = -1
        for i in range(self.n_bins):
            if self.mid_point[i][self.n_bins-1]!=-2:
                candidate_partition = self.get_optimal_partition(i,self.n_bins-1)
                assert(len(candidate_partition)>0), f"{candidate_partition}"
                if len(candidate_partition)>recal_n_bins:
                    recal_n_bins = len(candidate_partition)
                    l = i
        self.optimal_partition = self.get_optimal_partition(l,self.n_bins-1)

        self.recal_n_bins = len(self.optimal_partition)

        #get new upper edges
        recal_bin_assignment, self.recal_num_positives_in_bin, self.recal_num_in_bin, self.recal_group_num_positives_in_bin,\
        self.recal_group_num_in_bin = self.get_recal_bin_points(y_score)

        group_assignment = self.group_points(X_est)
        assert np.sum([recal_bin_assignment==i for i in range(self.recal_n_bins)])==X_est.shape[0]
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
        assert np.sum(self.recal_discriminated_against) == 0 ,f"{self.recal_discriminated_against}"

        self.sanity_check()


        # find threshold bin and theta
        self.recal_b, self.recal_theta = self.get_recal_threshold(m)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cal_data_path", type=str, help="the input calibration data path")
    parser.add_argument("--test_raw_path", type=str, help="the raw test data for sampling test data")
    parser.add_argument("--classifier_path", type=str, help="the input classifier path")
    parser.add_argument("--Z_indices", type=str, default="", help="features defining the group membership")
    parser.add_argument("--wgm_path", type=str, help="the output wgm classifier path")
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


    with open(args.cal_data_path, 'rb') as f:
        X_cal_all_features, y_cal = pickle.load(f)
        available_features = np.setdiff1d(np.arange(X_cal_all_features.shape[1]), Z_indices)
        X_cal = X_cal_all_features[:,available_features]

    groups = np.unique(X_cal_all_features[:, Z_indices])

    with open(args.classifier_path, "rb") as f:
        classifier = pickle.load(f)

    n = y_cal.size
    scores_cal = classifier.predict_proba(X_cal)[:, 1]

    wgm = WGM(args.B,Z_indices,groups,Z_map,alpha)
    wgm.fit(X_cal_all_features,scores_cal, y_cal, m)

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
        accuracy[k_idx],f1score[k_idx] = wgm.get_accuracy(scores_test_raw, y_test_raw)


    num_selected = np.empty(shape=(len(ks), args.n_runs_test))
    num_qualified = np.empty(shape=(len(ks), args.n_runs_test))
    for i in range(args.n_runs_test):
        indexes = np.random.choice(list(range(y_test_raw.size)), int(m))
        y_test = y_test_raw[indexes]
        scores_test = scores_test_raw[indexes]
        for k_idx, k in enumerate(ks):
            test_selected = wgm.recal_select(scores_test,k_idx)
            num_selected[k_idx][i] = calculate_expected_selected(test_selected, y_test, m)
            num_qualified[k_idx][i] = calculate_expected_qualified(test_selected, y_test, m)

    performance_metrics = {}
    performance_metrics["num_qualified"] = np.mean(num_qualified,axis=1)
    performance_metrics["num_selected"] = np.mean(num_selected,axis=1)
    assert (performance_metrics["num_qualified"].shape[0] == len(ks) and performance_metrics["num_selected"].shape[
        0] == len(ks))
    performance_metrics["accuracy"] = accuracy
    performance_metrics["f1_score"] = f1score
    performance_metrics["num_positives_in_bin"] = wgm.recal_num_positives_in_bin
    performance_metrics["num_in_bin"] = wgm.recal_num_in_bin
    performance_metrics["bin_values"] = wgm.recal_bin_values
    performance_metrics["group_num_positives_in_bin"] = wgm.recal_group_num_positives_in_bin
    performance_metrics["group_num_in_bin"] = wgm.recal_group_num_in_bin
    performance_metrics["group_bin_values"] = wgm.recal_group_bin_values
    performance_metrics["group_rho"] = wgm.recal_group_rho
    performance_metrics["bin_rho"] = wgm.recal_bin_rho
    performance_metrics["groups"] = wgm.groups
    performance_metrics["num_groups"] = wgm.num_groups
    performance_metrics["n_bins"] = wgm.recal_n_bins
    performance_metrics["discriminated_against"] = wgm.recal_discriminated_against
    performance_metrics["alpha"] = 0

    with open(args.result_path, 'wb') as f:
        pickle.dump(performance_metrics, f)

    with open(args.wgm_path, "wb") as f:
        pickle.dump(wgm, f)
