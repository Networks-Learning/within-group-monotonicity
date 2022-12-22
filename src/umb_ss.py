"""
Select a Shortlist of Applicants Based on Uniform Mass Binning (details in the paper).
"""
import argparse
import pickle
import numpy as np
from train_LR import NoisyLR
from utils import calculate_expected_selected, calculate_expected_qualified, transform_except_last_dim, get_mean_calibration_error
from sklearn.metrics import mean_squared_error,accuracy_score



class UMBSelect(object):
    def __init__(self, n_bins, alpha, Z_indices, groups, Z_map):
        # Hyper-parameters
        self.n_bins = n_bins
        self.delta = 0#1e-10
        self.alpha = alpha
        self.groups = groups
        self.Z_indices = Z_indices
        self.Z_map = Z_map

        # Parameters to be learned
        self.bin_upper_edges = None
        self.num_positives_in_bin = None
        self.num_in_bin = None
        self.bin_values = None
        self.num_examples = None
        self.epsilon = None
        self.b = None
        self.theta = None
        self.num_groups = None
        self.group_num_positives_in_bin = None
        self.group_num_in_bin = None
        self.group_rho = None
        self.group_bin_values = None
        # Internal variables
        self.fitted = False


    def _get_uniform_mass_bins(self, scores):
        assert (scores.size >= 2 * self.n_bins), "Fewer points than 2 * number of bins"

        scores_sorted = np.sort(scores)

        # split scores into groups of approx equal size
        groups = np.array_split(scores_sorted, self.n_bins)
        bin_upper_edges = list()

        for cur_group in range(self.n_bins - 1):
            bin_upper_edges += [max(groups[cur_group])]
        bin_upper_edges.append(np.inf)

        return np.array(bin_upper_edges)

    def _bin_points(self, scores):
        assert (self.bin_upper_edges is not None), "Bins have not been defined"
        scores = scores.squeeze()
        assert (np.size(scores.shape) < 2), "scores should be a 1D vector or singleton"
        scores = np.reshape(scores, (scores.size, 1))
        bin_edges = np.reshape(self.bin_upper_edges, (1, self.bin_upper_edges.size))
        return np.sum(scores > bin_edges, axis=1)

    def _group_points(self, X_est):
        assert (self.num_groups is not None), "Group number not set"
        group_index = np.empty(shape=(self.num_groups, X_est.shape[0]))
        for i, group in enumerate(self.groups):
            group = np.repeat(group, X_est.shape[0]).reshape(X_est.shape[0], 1)
            group_index[Z_map[Z_indices[0]][i]] = np.logical_or(group_index[Z_map[Z_indices[0]][i]],
                                                                np.equal(X_est[:, self.Z_indices], group).squeeze())

        return group_index

    def get_bin_statistics(self,y_score, y):
        bin_assignment = self._bin_points(y_score)
        # compute statistics of each bin
        num_in_bin = np.empty(self.n_bins)
        positives_in_bin = np.empty(self.n_bins)
        for i in range(self.n_bins):
            bin_idx = (bin_assignment == i)
            num_in_bin[i] = np.sum(bin_idx)
            positives_in_bin[i] = np.sum(np.logical_and(bin_idx, y))
        return num_in_bin, positives_in_bin

    # def get_positives_in_bin(self,y_score,y):
    #     bin_assignment = self._bin_points(y_score)
    #     # compute statistics of each bin
    #     positives_in_bin = np.empty(self.n_bins)
    #     for i in range(self.n_bins):
    #         bin_idx = (bin_assignment == i)
    #         positives_in_bin[i] = np.sum(np.logical_and(bin_idx, y))
    #     return positives_in_bin
    #
    # def get_num_in_bin(self,y_score):
    #     bin_assignment = self._bin_points(y_score)
    #     # compute statistics of each bin
    #     num_in_bin = np.empty(self.n_bins)
    #     for i in range(self.n_bins):
    #         bin_idx = (bin_assignment == i)
    #         num_in_bin[i] = np.sum(bin_idx)
    #     return num_in_bin

    # def get_bin_value(self):
    #     return self.num_positives_in_bin/self.num_in_bin

    def get_group_statistics(self, X_est,y_score,y):
        bin_assignment = self._bin_points(y_score)
        group_assignment = self._group_points(X_est)
        assert (np.sum(group_assignment) == X_est.shape[0])
        num_in_bin = np.empty(self.n_bins)
        num_positives_in_bin = np.empty(self.n_bins)
        group_num_positives_in_bin = np.empty(shape=(self.n_bins, self.num_groups))
        group_num_in_bin = np.empty(shape=(self.n_bins, self.num_groups))

        for i in range(self.n_bins):
            bin_idx = (bin_assignment == i)
            num_positives_in_bin[i] = np.sum(np.logical_and(bin_idx, y))
            num_in_bin[i] = np.sum(bin_idx)
            for j in range(self.num_groups):
                group_num_positives_in_bin[i][j] = np.sum(
                    np.logical_and(np.logical_and(bin_idx, y), group_assignment[j]))
                group_num_in_bin[i][j] = np.sum(np.logical_and(bin_idx, group_assignment[j]))

            assert (num_in_bin[i] == np.sum(group_num_in_bin[i]))
            assert (num_positives_in_bin[i] == np.sum(group_num_positives_in_bin[i]))

        group_rho = group_num_in_bin / self.num_in_bin.reshape(self.num_in_bin.shape[0], 1)

        # smoothing for the case where there is no sample of a particular group in some bin
        smooth_alpha = np.ones(shape=group_num_positives_in_bin.shape)
        smooth_d = np.repeat(1.0 / self.bin_values, group_num_in_bin.shape[1]).reshape(group_num_in_bin.shape)
        group_bin_values = (group_num_positives_in_bin + smooth_alpha) / (group_num_in_bin + smooth_d)

        # sanity check
        for i in range(self.n_bins):
            assert (np.sum(group_rho[i] * group_bin_values[i]) - self.bin_values[i] < 1e-3)   # for test set there can be assertion error if cal and test data vary significantly
            assert (np.sum(group_rho[i]) - 1.0 < 1e-4)

        return num_positives_in_bin, num_in_bin, group_num_positives_in_bin, group_num_in_bin, group_bin_values, group_rho

    # def get_group_num_positives_in_bin(self, X_est, y_score,y):
    #     assert(len(Z_map[Z_indices[0]])==groups.shape[0])
    #     bin_assignment = self._bin_points(y_score)
    #     group_num_positives_in_bin = np.empty(shape=(self.n_bins,self.num_groups))
    #     bin_index = np.empty(shape=(self.n_bins, bin_assignment.shape[0]))
    #     group_index = np.empty(shape=(self.num_groups, X_est.shape[0]))
    #     for i in range(self.n_bins):
    #         bin_index[i] = (bin_assignment == i)
    #     for i, group in enumerate(groups):
    #         group = np.repeat(group,X_est.shape[0]).reshape(X_est.shape[0],1)
    #         group_index[Z_map[Z_indices[0]][i]] = np.logical_or(group_index[Z_map[Z_indices[0]][i]],np.equal(X_est[:, self.Z_indices], group).squeeze())
    #     for i in range(self.n_bins):
    #         for j in range(self.num_groups):
    #             # bin_idx = (bin_assignment == i)
    #             # group_idx = np.array([np.array_equal(X_est[_,self.Z_indices],group) for _ in range(X_est.shape[0])])
    #             group_num_positives_in_bin[i][j] = np.sum(np.logical_and(np.logical_and(bin_index[i], y),group_index[j]))
    #     # for i in range(self.n_bins):
    #         assert(self.num_positives_in_bin[i]==np.sum(group_num_positives_in_bin[i]))
    #     return group_num_positives_in_bin
    #
    # def get_group_num_in_bin(self,X_est, y_score):
    #     bin_assignment = self._bin_points(y_score)
    #     group_assignment = self._group_points(X_est)
    #     group_num_in_bin = np.empty(shape=(self.n_bins,self.num_groups))
    #
    #     bin_index = np.empty(shape=(self.n_bins,bin_assignment.shape[0]))
    #     # group_index = np.empty(shape=(self.groups.shape[0],X_est.shape[0]))
    #     for i in range(self.n_bins):
    #         bin_index[i] = (bin_assignment == i)
    #     # for i,group in enumerate(groups):
    #     #     group_index[i] = np.logical_or(group_index[Z_indices[0]],np.array([np.array_equal(X_est[_, self.Z_indices], group) for _ in range(X_est.shape[0])]))
    #     for i in range(self.n_bins):
    #         for j, group in range(self.num_groups):
    #             # bin_idx = (bin_assignment == i)
    #             # group_index = np.array([np.array_equal(X_est[_,self.Z_indices],group) for _ in range(X_est.shape[0])])
    #             group_num_in_bin[i][j] = np.sum(np.logical_and(bin_index[i],group_assignment[j]))
    #     # for i in range(self.n_bins):
    #         assert self.num_in_bin[i] == np.sum(group_num_in_bin[i])
    #     return group_num_in_bin


    # def get_group_bin_value(self,y_score):
    #     alpha = np.ones(shape=self.group_num_positives_in_bin.shape)
    #     bin_value = self.get_bin_value()
    #     bin_assignment = self._bin_points(y_score)
    #     d = np.repeat(1.0/bin_value,self.group_num_in_bin.shape[1]).reshape(self.group_num_in_bin.shape)
    #     print(d.shape)
    #     group_bin_value = (self.group_num_positives_in_bin + alpha)/(self.group_num_in_bin + d)
    #     #sanity check
    #     group_rho = self.get_group_rho()
    #     bin_value = self.get_bin_value()
    #     for i in range(self.n_bins):
    #         assert(np.sum(group_rho[i]*group_bin_value[i])-bin_value[i]<1e-2)
    #
    #     return group_bin_value

    # def get_group_rho(self):
    #     group_rho = self.group_num_in_bin/self.num_in_bin.reshape(self.num_in_bin.shape[0],1)
    #     # sanity check
    #     for i in range(self.n_bins):
    #         assert (np.sum(group_rho[i]) - 1.0 < 1e-4)
    #
    #     return group_rho

    def _nudge(self, matrix):
        return ((matrix + np.random.uniform(low=0,
                                            high=self.delta,
                                            size=matrix.shape)) / (1 + self.delta))

    def fit(self, X_est, y_score, y, m, k):
        assert (self.n_bins is not None), "Number of bins has to be specified"
        y_score = y_score.squeeze()
        y = y.squeeze()
        assert (y_score.size == y.size), "Check dimensions of input matrices"
        assert (y.size >= 2 * self.n_bins), "Number of bins should be less than two " \
                                            "times the number of calibration points"

        # All required (hyper-)parameters have been passed correctly
        # Uniform-mass binning/histogram binning code starts below
        self.num_examples = y_score.size
        # grouping based on values of the chosen featue, see Z_map
        self.num_groups = np.unique(Z_map[Z_indices[0]]).shape[0]
        # self.epsilon = np.sqrt(2 * np.log(2 / alpha) / n)
        self.epsilon = 0

        # delta-randomization
        y_score = self._nudge(y_score)

        # compute uniform-mass-bins using calibration data
        self.bin_upper_edges = self._get_uniform_mass_bins(y_score)

        # assign calibration data to bins
        bin_assignment = self._bin_points(y_score)
        group_assignment = self._group_points(X_est)
        assert(np.sum(group_assignment)==X_est.shape[0])

        # compute statistics of each bin
        self.num_in_bin = np.empty(self.n_bins)
        self.num_positives_in_bin = np.empty(self.n_bins)
        self.group_num_positives_in_bin = np.empty(shape=(self.n_bins,self.num_groups))
        self.group_num_in_bin = np.empty(shape=(self.n_bins, self.num_groups))

        for i in range(self.n_bins):
            bin_idx = (bin_assignment == i)
            self.num_positives_in_bin[i] = np.sum(np.logical_and(bin_idx, y))
            self.num_in_bin[i] = np.sum(bin_idx)
            for j in range(self.num_groups):
                self.group_num_positives_in_bin[i][j] = np.sum(np.logical_and(np.logical_and(bin_idx, y),group_assignment[j]))
                self.group_num_in_bin[i][j] = np.sum(np.logical_and(bin_idx, group_assignment[j]))

            assert(self.num_in_bin[i]==np.sum(self.group_num_in_bin[i]))
            assert(self.num_positives_in_bin[i]==np.sum(self.group_num_positives_in_bin[i]))

        self.bin_values = self.num_positives_in_bin / self.num_in_bin

        self.group_rho = self.group_num_in_bin / self.num_in_bin.reshape(umb_select.num_in_bin.shape[0], 1)

        # smoothing for the case where there is no sample of a particular group in some bin
        smooth_alpha = np.ones(shape=self.group_num_positives_in_bin.shape)
        smooth_d = np.repeat(1.0 / self.bin_values, self.group_num_in_bin.shape[1]).reshape(self.group_num_in_bin.shape)
        self.group_bin_values = (self.group_num_positives_in_bin + smooth_alpha) / (self.group_num_in_bin + smooth_d)


        # sanity check
        for i in range(self.n_bins):
            assert (np.sum(self.group_rho[i] * self.group_bin_values[i]) - self.bin_values[i] < 1e-3)
            assert (np.sum(self.group_rho[i]) - 1.0 < 1e-4)

        # find threshold bin and theta
        sum_scores = 0
        b = 0  # bin on the threshold
        theta = 1.
        for i in reversed(range(self.n_bins)):
            sum_scores += m * (self.num_positives_in_bin[i] / self.num_examples - self.epsilon)
            if sum_scores >= k:
                sum_scores -= m * (self.num_positives_in_bin[i] / self.num_examples - self.epsilon)
                b = i
                theta = (k - sum_scores) / (m * (self.num_positives_in_bin[i] / self.num_examples
                                                 - self.epsilon))
                break
        self.b = b
        self.theta = theta



        # histogram binning done
        self.fitted = True

    def select(self, scores):
        scores = scores.squeeze()
        size = scores.size
        print(size)

        # delta-randomization
        scores = self._nudge(scores)

        # assign test data to bins
        test_bins = self._bin_points(scores)

        # make decisions
        s = np.zeros(size, dtype=bool)
        for i in range(size):
            if test_bins[i] > self.b:
                s[i] = True
            elif test_bins[i] == self.b:
                s[i] = bool(np.random.binomial(1, self.theta))
            else:
                s[i] = False
        return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cal_data_path", type=str, help="the input calibration data path")
    parser.add_argument("--test_raw_path", type=str, help="the raw test data for sampling test data")
    parser.add_argument("--classifier_path", type=str, help="the input classifier path")
    parser.add_argument("--umb_path", type=str, help="the output umb path")
    parser.add_argument("--result_path", type=str, help="the output selection result path")
    parser.add_argument("--Z_indices", type=str, default="", help="features defining the group membership")
    parser.add_argument("--k", type=float, help="the target expected number of qualified candidates")
    parser.add_argument("--m", type=float, help="the expected number of incoming candidates")
    parser.add_argument("--alpha", type=float, default=0.1, help="the failure probability")
    parser.add_argument("--B", type=int, help="the number of bins")
    parser.add_argument("--scaler_path", type=str, help="the path for the scaler")

    args = parser.parse_args()
    k = args.k
    m = args.m
    alpha = args.alpha

    args = parser.parse_args()
    Z_indices = [int(index) for index in args.Z_indices.split('_')]
    Z_map = {
        15: [0, 1, 1, 2, 2, 2, 3, 3, 3]
    }


    with open(args.cal_data_path, 'rb') as f:
        X_cal_est, y_cal = pickle.load(f)
        available_features = np.setdiff1d(np.arange(X_cal_est.shape[1]), Z_indices)
        X_cal = X_cal_est[:,available_features]
        # print(X_cal.shape,X_cal_est.shape)

    groups = np.unique(X_cal_est[:, Z_indices])
    print(groups)


    with open(args.classifier_path, "rb") as f:
        classifier = pickle.load(f)

    # test
    with open(args.test_raw_path, "rb") as f:
        X_test_est, y_test_raw = pickle.load(f)
    with open(args.scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    X_test_est = transform_except_last_dim(X_test_est, scaler)
    X_test_raw = X_test_est[:, available_features]
    scores_test_raw = classifier.predict_proba(X_test_raw)[:, 1]

    # print(X_test_raw.shape)
    # print("---test---")
    # print("----MSE")
    # print(mean_squared_error(classifier.predict_proba(X_test_raw)[:, 1], y_test_raw * 1.))
    # print("----Accuracy")
    # print(accuracy_score(classifier.predict(X_test_raw), y_test_raw))

    # calibration
    # with open(args.cal_data_path, 'rb') as f:
    #     X_cal, y_cal = pickle.load(f)
    # with open(args.classifier_path, "rb") as f:
    #     classifier = pickle.load(f)
    n = y_cal.size
    scores_cal = classifier.predict_proba(X_cal)[:, 1]

    umb_select = UMBSelect(args.B, alpha,Z_indices,groups,Z_map)
    umb_select.fit(X_cal_est,scores_cal, y_cal, m, k)

    s_test_raw = umb_select.select(scores_test_raw)
    performance_metrics = {}
    performance_metrics["num_qualified"] = calculate_expected_qualified(s_test_raw, y_test_raw, m)
    performance_metrics["num_selected"] = calculate_expected_selected(s_test_raw, y_test_raw, m)
    performance_metrics["constraint_satisfied"] = True if performance_metrics["num_qualified"] >= k else False
    performance_metrics["num_positives_in_bin"] = umb_select.num_positives_in_bin
    performance_metrics["num_in_bin"] = umb_select.num_in_bin
    performance_metrics["bin_values"] = umb_select.bin_values
    performance_metrics["group_num_positives_in_bin"] = umb_select.group_num_positives_in_bin
    performance_metrics["group_num_in_bin"] = umb_select.group_num_in_bin
    performance_metrics["group_bin_values"] = umb_select.group_bin_values
    performance_metrics["group_rho"] = umb_select.group_rho
    performance_metrics["groups"] = umb_select.groups
    performance_metrics["num_groups"] = umb_select.num_groups
    # print(performance_metrics["group_bin_value"])
    with open(args.result_path, 'wb') as f:
        pickle.dump(performance_metrics, f)

    with open(args.umb_path, "wb") as f:
        pickle.dump(umb_select, f)
