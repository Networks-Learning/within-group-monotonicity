"""
Recalibrates a given calibrated classifier so that it is within-group calibrated.
"""
import argparse
import pickle
import numpy as np
from train_LR import NoisyLR
# from umb_ss import UMBSelect
from partition import BinPartition
from utils import calculate_expected_selected, calculate_expected_qualified, transform_except_last_dim
from sklearn.metrics import mean_squared_error,accuracy_score

class WGC(BinPartition):
    def __init__(self, n_bins, Z_indices, groups, Z_map):
        super().__init__(n_bins, Z_indices, groups, Z_map)
        self.alpha = None
        # Parameters to be learned
        # self.dp = None
        # self.prev_point = None
        # self.optimal_partition = None
        # self.recal_n_bins = None
        # self.recal_bin_upper_edges = None
        # self.recal_num_positives_in_bin = None
        # self.recal_num_in_bin = None
        # self.recal_bin_values = None
        # self.recal_bin_rho = None
        # self.num_examples = None
        # self.epsilon = None
        # self.b = None
        # self.theta = None
        # # self.num_groups = None
        # self.recal_group_num_positives_in_bin = None
        # self.recal_group_num_in_bin = None
        # self.recal_group_rho = None
        # self.recal_group_bin_values = None
        # self.recal_b = None
        # self.recal_theta = None
        # self.recal_discriminated_against = None

    # def _get_merged_statistics(self, l, r):
    #     positives = np.sum(self.num_positives_in_bin[l:r+1])
    #     total = np.sum(self.num_in_bin[l:r+1])
    #     group_positives = np.sum(self.group_num_positives_in_bin[l:r+1],axis=0)
    #     group_total = np.sum(self.group_num_in_bin[l:r+1],axis=0)
    #     group_rho = np.average(self.group_rho[l:r+1],axis=0)
    #
    #     assert(np.sum(group_positives) == positives and np.sum(group_total)==total and np.sum(group_rho)-1<1e-2)
    #
    #     return positives, total, group_positives, group_total, group_rho


    # def get_recal_upper_edges(self):
    #     assert (self.recal_n_bins is not None and self.optimal_partition is not None), "not yet recalibrated"
    #     recal_bin_upper_edges = []
    #     for i in range(len(self.optimal_partition)):
    #         recal_bin_upper_edges.append(self.bin_upper_edges[self.optimal_partition[i]])
    #
    #     assert len(recal_bin_upper_edges) == len(self.optimal_partition)
    #
    #
    #     return np.array(recal_bin_upper_edges)


    # def get_recal_bin_points(self, scores):
    #     assert self.recal_n_bins is not None and self.optimal_partition is not None
    #     recal_bin_assignment = np.empty(shape=scores.size)
    #     bin_assignment = self._bin_points(scores)
    #     in_bin = np.empty(shape=(self.recal_n_bins, scores.size))
    #
    #     recal_num_in_bin = np.empty(self.recal_n_bins)
    #     recal_num_positives_in_bin = np.empty(self.recal_n_bins)
    #     recal_group_num_positives_in_bin = np.empty(shape=(self.recal_n_bins, self.num_groups))
    #     recal_group_num_in_bin = np.empty(shape=(self.recal_n_bins, self.num_groups))
    #     optimal_partition = np.append(self.optimal_partition, self.n_bins)
    #
    #     for i in range(self.recal_n_bins):
    #         in_bin[i] = (np.logical_and(bin_assignment >= optimal_partition[i], bin_assignment < optimal_partition[i+1]))
    #         recal_bin_assignment[in_bin[i].astype(bool)] = i
    #         recal_num_positives_in_bin[i] = np.sum(self.num_positives_in_bin[optimal_partition[i]:optimal_partition[i+1]])
    #         recal_num_in_bin[i] = np.sum(self.num_in_bin[optimal_partition[i]:optimal_partition[i+1]])
    #         for j in range(self.num_groups):
    #             recal_group_num_positives_in_bin[i][j] = np.sum(self.group_num_positives_in_bin[optimal_partition[i]:optimal_partition[i+1],j])
    #             recal_group_num_in_bin[i][j] = np.sum(
    #                 self.group_num_in_bin[optimal_partition[i]:optimal_partition[i + 1],j])
    #
    #
    #     assert (np.sum(in_bin) == scores.size) ,f"{np.sum(in_bin), scores.size}"
    #     return recal_bin_assignment, recal_num_positives_in_bin, recal_num_in_bin, recal_group_num_positives_in_bin, recal_group_num_in_bin

    def _find_potential_merges(self,alpha):
        S = np.ones(shape=(self.n_bins, self.n_bins))
        for l in range(self.n_bins):
            for r in range(l, self.n_bins):
                lr_positives, lr_total, lr_group_positives, lr_group_total, lr_group_rho = self._get_merged_statistics(
                    l, r)
                lr_positives = np.repeat(lr_positives,lr_group_positives.shape[0]).reshape(lr_group_positives.shape)
                lr_total = np.repeat(lr_total, lr_group_positives.shape[0]).reshape(lr_group_positives.shape)
                # if l==0 and r==0:
                #     print(f"{l,r,np.where(lr_group_rho,lr_group_positives/lr_group_total-lr_positives/lr_total,np.zeros(lr_group_rho.shape))}")
                if np.where(lr_group_rho,
                            (np.abs(lr_group_positives/lr_group_total - lr_positives/lr_total) > alpha),
                            np.zeros(shape=lr_group_rho.shape)).any():  # can be adjacent
                    S[l][r] = 0
        # print(S)
        return S

    def recalibrate(self):
        low = 0
        high = 100
        alpha = (low + high)//2
        final_dp = np.zeros(shape=(self.n_bins))
        final_mid_point = np.repeat(-2, self.n_bins).reshape(self.n_bins)
        while low<=high:
            mid = (low + high)//2
            S = self._find_potential_merges(mid/100)
            dp = np.zeros(shape = (self.n_bins))
            mid_point = np.repeat(-2,self.n_bins).reshape(self.n_bins)

            for r in range(self.n_bins):
                if S[0][0]:
                    dp[0] = 1
                    mid_point[0] = -1

            for l in range(1,self.n_bins):
                for r in range(l,self.n_bins):
                    # for k in range(l):
                        # print(f"{l =} { r =} {k = } {np.sum(S[l][r])}")
                        if S[l][r] and dp[l-1]!=0 and dp[l-1]+1>dp[r]:
                            dp[r] = dp[l-1] + 1
                            mid_point[r] = l-1
                            # print(f"{l, r}")
            if mid_point[self.n_bins-1] != -2:
                final_dp = dp
                final_mid_point = mid_point
                alpha = mid
                high = mid - 1
            else:
                low = mid + 1

        return final_dp, final_mid_point,alpha/100

    def get_optimal_partition(self,r):
        assert (self.dp is not None and self.mid_point is not None), "not yet recalibrated"
        # print(f"{r,self.prev_point[r]}")
        if r == -1:
            return []

        assert(self.mid_point[r] != -2)

        return self.get_optimal_partition(self.mid_point[r]) + [self.mid_point[r]+1]


    def fit(self, X_est, y_score, y, m, k):

        #fit the umb
        super().fit(X_est, y_score, y, m, k)

        #recalibrate using algorithm 2
        self.dp, self.mid_point,self.alpha = self.recalibrate()
        # print(f"{self.alpha =}")

        #get optimal partition

        self.optimal_partition = self.get_optimal_partition(self.n_bins-1)
        # print(self.optimal_partition)

        self.recal_n_bins = len(self.optimal_partition)

        #get new upper edges
        recal_bin_assignment, self.recal_num_positives_in_bin, self.recal_num_in_bin, self.recal_group_num_positives_in_bin,\
        self.recal_group_num_in_bin = self.get_recal_bin_points(y_score)


        # for i in range(self.recal_n_bins):
        #     assert (self.recal_num_positives_in_bin[i] == np.sum(self.recal_group_num_positives_in_bin[
        #                                                        i])), f"{self.num_positives_in_bin[i]},{np.sum(self.group_num_positives_in_bin[i]) + self.num_groups}"
        #     assert (self.recal_num_in_bin[i] == np.sum(self.recal_group_num_in_bin[i]))
        #     assert (self.recal_num_in_bin[i] > 0).all()

        group_assignment = self.group_points(X_est)
        assert np.sum([recal_bin_assignment==i for i in range(self.recal_n_bins)])==X_est.shape[0], f"{np.sum([recal_bin_assignment==i for i in range(self.recal_n_bins)]), X_est.shape[0]}"
        assert np.sum(group_assignment) == X_est.shape[0]

        self.recal_bin_rho = self.recal_num_in_bin / self.num_examples
        self.recal_bin_values = self.recal_num_positives_in_bin / self.recal_num_in_bin

        # for i in range(self.recal_n_bins - 1):
        #     assert (self.recal_bin_values[i] <= self.recal_bin_values[i + 1])

        positive_group_rho = np.greater(self.recal_group_num_in_bin,np.zeros(shape=self.recal_group_num_in_bin.shape))
        self.recal_group_rho = np.where(positive_group_rho,(self.recal_group_num_in_bin) / (self.recal_num_in_bin)[:,np.newaxis],np.zeros(shape=self.recal_group_num_in_bin.shape))
        self.recal_group_bin_values = np.where(positive_group_rho, self.recal_group_num_positives_in_bin / self.recal_group_num_in_bin, np.zeros(shape=self.recal_group_num_in_bin.shape))

        self.find_discriminations()

        # sanity check
        self.sanity_check()
        # assert np.sum(self.recal_discriminated_against) == 0  # within-group monotonicity
        # for i in range(self.recal_n_bins):
        #     assert (np.sum(self.recal_group_rho[i] * self.recal_group_bin_values[i]) - self.recal_bin_values[i] < 1e-3), f"{self.recal_group_rho[i], self.recal_group_bin_values[i], self.recal_bin_values[i]}"
        #     assert (np.sum(self.recal_group_rho[i]) - 1.0 < 1e-2)
        #     for j in range(self.num_groups):
        #         if i < self.recal_n_bins-1:
        #             if i < self.recal_n_bins - 1 and positive_group_rho[i][j]:
        #                 self.recal_discriminated_against[i][j] = np.greater(
        #                     self.recal_group_num_positives_in_bin[i][j] * self.recal_group_num_in_bin[i + 1][j],
        #                     self.recal_group_num_positives_in_bin[i + 1][j] * self.recal_group_num_in_bin[i][j])
        #         if self.recal_group_num_in_bin[i][j] == 0:
        #             assert self.recal_group_rho[i][j] == 0
        #         else:
        #             assert self.recal_group_rho[i][j] * self.recal_num_in_bin[i] - self.recal_group_num_in_bin[i][j]< 1e-2
        #
        #         if self.recal_group_num_positives_in_bin[i][j]==0:
        #             assert self.recal_group_bin_values[i][j] ==0
        #         else:
        #             assert self.recal_group_bin_values[i][j]*self.recal_group_num_in_bin[i][j] - self.recal_group_num_positives_in_bin[i][j]< 1e-2

        # find threshold bin and theta
        self.recal_b, self.recal_theta = self.get_recal_threshold(m, k)
        # recal_sum_scores = 0
        # recal_b = 0  # bin on the threshold
        # recal_theta = 1.
        # for i in reversed(range(self.recal_n_bins)):
        #     recal_sum_scores += m * (self.recal_num_positives_in_bin[i] / self.num_examples - self.epsilon)
        #     if recal_sum_scores >= k:
        #         recal_sum_scores -= m * (self.recal_num_positives_in_bin[i] / self.num_examples - self.epsilon)
        #         recal_b = i
        #         recal_theta = (k - recal_sum_scores) / (
        #                     m * (self.recal_num_positives_in_bin[i] / self.num_examples
        #                          - self.epsilon))
        #         break
        # self.recal_b = recal_b
        # self.recal_theta = recal_theta
        # print(f"{self.recal_b=}")
        # print(f"{self.recal_theta=}")

    # def recal_select(self, scores):
    #     scores = scores.squeeze()
    #     size = scores.size
    #
    #     # assign test data to bins
    #     test_bins,_,_,_,_ = self.get_recal_bin_points(scores)
    #
    #     # make decisions
    #     s = np.zeros(size, dtype=bool)
    #     for i in range(size):
    #         if test_bins[i] > self.recal_b:
    #             s[i] = True
    #         elif test_bins[i] == self.recal_b:
    #             s[i] = bool(np.random.binomial(1, self.recal_theta))
    #         else:
    #             s[i] = False
    #     return s


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
    k = args.k
    m = args.m
    alpha = args.alpha

    args = parser.parse_args()
    Z_indices = [int(index) for index in args.Z_indices.split('_')]
    def list_maker(val,n):
        return [val]*n
    Z_map = {
        15: [0] + [1] + list_maker(2,3) + list_maker(3,4),
        1: list_maker(0,16)+list_maker(1,4)+list_maker(2,2)+list_maker(3,3),
        0: list_maker(0,25) + list_maker(1,25) + list_maker(2,25) + list_maker(3,25),
        14: [0,1],
        4: [0,1],
        6: [0,1,2,2,3],
        2: [0,0,0,0,1]
    }

    with open(args.cal_data_path, 'rb') as f:
        X_cal_all_features, y_cal = pickle.load(f)
        available_features = np.setdiff1d(np.arange(X_cal_all_features.shape[1]), Z_indices)
        X_cal = X_cal_all_features[:,available_features]
        # print(X_cal.shape,X_cal_est.shape)

    groups = np.unique(X_cal_all_features[:, Z_indices])


    with open(args.classifier_path, "rb") as f:
        classifier = pickle.load(f)

    # with open(args.calibrated_classifier_path, "rb") as f:
    #     calibrated_classifier = pickle.load(f)

    n = y_cal.size
    scores_cal = classifier.predict_proba(X_cal)[:, 1]

    wgc = WGC(args.B,Z_indices,groups,Z_map)
    wgc.fit(X_cal_all_features,scores_cal, y_cal, m, k)

    # test
    with open(args.test_raw_path, "rb") as f:
        X_test_all_features, y_test_raw = pickle.load(f)
    with open(args.scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    X_test_all_features = transform_except_last_dim(X_test_all_features, scaler)
    X_test_raw = X_test_all_features[:, available_features]
    scores_test_raw = classifier.predict_proba(X_test_raw)[:, 1]
    total_test_selected = wgc.recal_select(scores_test_raw)
    fpr, tpr = wgc.recal_get_test_roc(X_test_all_features, scores_test_raw, y_test_raw)
    # accuracy = wgc.get_accuracy(total_test_selected, y_test_raw)
    # group_accuracy = wgc.get_group_accuracy(total_test_selected, X_test_all_features, y_test_raw)

    #simulating pools of candidates
    num_selected = []
    num_qualified = []
    for i in range(args.n_runs_test):
        indexes = np.random.choice(list(range(y_test_raw.size)), int(m))
        y_test = y_test_raw[indexes]
        scores_test = scores_test_raw[indexes]
        recal_test_selected = wgc.recal_select(scores_test)
        num_selected.append(calculate_expected_selected(recal_test_selected, y_test, m))
        num_qualified.append(calculate_expected_qualified(recal_test_selected, y_test, m))


    performance_metrics = {}
    performance_metrics["num_qualified"] = np.mean(num_qualified)
    performance_metrics["num_selected"] = np.mean(num_selected)
    performance_metrics["constraint_satisfied"] = True if performance_metrics["num_qualified"] >= k else False
    performance_metrics["fpr"] = fpr
    performance_metrics["tpr"] = tpr
    # performance_metrics["group_fpr"] = group_fpr
    # performance_metrics["group_tpr"] = group_tpr
    # performance_metrics["accuracy"] = accuracy
    # performance_metrics["group_accuracy"] = group_accuracy
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
    performance_metrics["alpha"] = wgc.alpha

    with open(args.result_path, 'wb') as f:
        pickle.dump(performance_metrics, f)

    with open(args.wgc_path, "wb") as f:
        pickle.dump(wgc, f)
