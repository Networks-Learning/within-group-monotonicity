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
    def __init__(self, n_bins, Z_indices, groups, Z_map):
        # Hyper-parameters
        self.n_bins = n_bins
        self.delta = 0#1e-10
        # self.alpha = alpha
        self.groups = groups
        self.Z_indices = Z_indices
        self.Z_map = Z_map
        self.num_groups = np.unique(Z_map[Z_indices[0]]).shape[0]  # grouping based on values of the chosen feature, see Z_map

        # Parameters to be learned
        self.bin_upper_edges = None
        self.bin_lower_edges = None
        self.num_positives_in_bin = None
        self.num_in_bin = None
        self.bin_values = None
        self.bin_rho = None
        self.num_examples = None
        self.epsilon = None
        self.b = None
        self.theta = None
        # self.num_groups = None
        self.group_num_positives_in_bin = None
        self.group_num_in_bin = None
        self.group_rho = None
        self.group_bin_values = None
        # self.sorted = None
        # Internal variables
        self.fitted = False


    def _get_uniform_mass_bins(self, scores, y):
        assert (scores.size >= 2 * self.n_bins), "Fewer points than 2 * number of bins"

        scores_sorted = np.sort(scores)

        # split scores into groups of approx equal size
        groups = np.array_split(scores_sorted, self.n_bins)
        bin_upper_edges = list()
        bin_lower_edges = list()
        bin_upper_edges += [max(groups[0])]
        bin_lower_edges += [-np.inf]
        # bin_upper_edges.append(-np.inf)
        for cur_group in range(1,self.n_bins - 1):
            bin_upper_edges += [max(groups[cur_group])]
            bin_lower_edges += [max(groups[cur_group-1])]
            # bin_lower_edges += [min(groups[cur_group])]
        bin_upper_edges += [np.inf]
        bin_lower_edges += [max(groups[self.n_bins - 2])]
        scores = scores.squeeze()
        assert (np.size(scores.shape) < 2), "scores should be a 1D vector or singleton"
        num_positives_in_bin = np.empty(self.n_bins)
        num_in_bin = np.empty(self.n_bins)
        in_bin = np.empty(shape=(self.n_bins,scores.size))
        for i in range(self.n_bins):
            lower_edge = np.repeat(bin_lower_edges[i],scores.size)
            upper_edge = np.repeat(bin_upper_edges[i],scores.size)
            in_bin[i] = (np.logical_and(scores > lower_edge,scores <= upper_edge))
            num_in_bin[i] = np.sum(in_bin[i])
            num_positives_in_bin[i] = np.sum(np.logical_and(in_bin[i], y))

        assert(np.sum(in_bin)==scores.size), f"{np.sum(in_bin), scores.size}"
        # bin_upper_edges = bin_upper_edges[1:]
        sorted = np.argsort(num_positives_in_bin/num_in_bin)
        assert(len(bin_upper_edges)==len(sorted))
        num_positives_in_bin = num_positives_in_bin[sorted]
        num_in_bin = num_in_bin[sorted]
        for i in range(self.n_bins-1):
            assert(num_positives_in_bin[i]/num_in_bin[i]<=num_positives_in_bin[i+1]/num_in_bin[i+1])
        bin_upper_edges = np.array(bin_upper_edges)
        bin_upper_edges = bin_upper_edges[sorted]

        bin_lower_edges = np.array(bin_lower_edges)
        bin_lower_edges = bin_lower_edges[sorted]
        return bin_upper_edges, bin_lower_edges

    def _bin_points(self, scores):
        assert (self.bin_upper_edges is not None and self.bin_lower_edges is not None), "Bins have not been defined"
        scores = scores.squeeze()
        assert (np.size(scores.shape) < 2), "scores should be a 1D vector or singleton"
        bin_assignment = np.empty(shape=scores.size)
        # bin_upper_edges = [-np.inf] + [upper_edge for upper_edge in self.bin_upper_edges]
        # bin_lower_edges = [-np.inf] + [upper_edge for upper_edge in self.bin_upper_edges]
        in_bin = np.empty(shape=(self.n_bins, scores.size))

        for i in range(self.n_bins):
            lower_edge = np.repeat(self.bin_lower_edges[i], scores.size)
            upper_edge = np.repeat(self.bin_upper_edges[i], scores.size)
            in_bin[i] = (np.logical_and(scores > lower_edge,scores <= upper_edge))
            bin_assignment[in_bin[i].astype(bool)] = i
        assert(np.sum(in_bin) == scores.size), f"{np.sum(in_bin), scores.size}"
        assert(bin_assignment>=0).all()
        # print(bin_assignment[bin_assignment.size-1])
        # for ba in bin_assignment:
        #     if not (ba>=0 or ba<self.n_bins):
                # print(f"{ba = }")
        assert (bin_assignment < self.n_bins).all(), f"{bin_assignment}"
        return bin_assignment

    def group_points(self, X_est):
        assert (self.num_groups is not None), "Group number not set"
        group_index = np.zeros(shape=(self.num_groups, X_est.shape[0]))
        for i, group in enumerate(self.groups):
            # print(group)
            group = np.repeat(group, X_est.shape[0]).reshape(X_est.shape[0], 1)
            group_index[self.Z_map[self.Z_indices[0]][i]] = np.logical_or(group_index[self.Z_map[self.Z_indices[0]][i]],
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


    def get_group_statistics(self, X_est,y_score,y):
        bin_assignment = self._bin_points(y_score,y)
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


        # smoothing for the case where there is no sample of a particular group in some bin
        smooth_alpha = np.ones(shape=group_num_positives_in_bin.shape)
        smooth_d = np.repeat(1.0 / self.bin_values, group_num_in_bin.shape[1]).reshape(group_num_in_bin.shape)
        rho_smooth_d = np.repeat(self.num_groups, num_in_bin.shape[0]).reshape(num_in_bin.shape)

        group_rho = (group_num_in_bin + smooth_alpha) / (num_in_bin + rho_smooth_d).reshape(num_in_bin.shape[0],1)
        group_bin_values = (group_num_positives_in_bin + smooth_alpha) / (group_num_in_bin + smooth_d)

        # sanity check
        for i in range(self.n_bins):
            assert (np.sum(group_rho[i] * group_bin_values[i]) - self.bin_values[i] < 1e-3)   # for test set there can be assertion error if cal and test data vary significantly
            assert (np.sum(group_rho[i]) - 1.0 < 1e-4)

        return num_positives_in_bin, num_in_bin, group_num_positives_in_bin, group_num_in_bin, group_bin_values, group_rho

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
        # self.num_groups = np.unique(Z_map[Z_indices[0]]).shape[0]
        # self.epsilon = np.sqrt(2 * np.log(2 / alpha) / n)
        self.epsilon = 0

        # delta-randomization
        y_score = self._nudge(y_score)

        # compute uniform-mass-bins using calibration data
        self.bin_upper_edges, self.bin_lower_edges= self._get_uniform_mass_bins(y_score,y)

        # assign calibration data to bins
        bin_assignment = self._bin_points(y_score)
        group_assignment = self.group_points(X_est)
        assert(np.sum(group_assignment)==X_est.shape[0]), f"{np.sum(group_assignment),X_est.shape[0]}"

        # compute statistics of each bin
        self.num_in_bin = np.empty(self.n_bins)
        self.num_positives_in_bin = np.empty(self.n_bins)
        self.group_num_positives_in_bin = np.empty(shape=(self.n_bins,self.num_groups))
        self.group_num_in_bin = np.empty(shape=(self.n_bins, self.num_groups))

        for i in range(self.n_bins):
            bin_idx = (bin_assignment == i)
            self.num_positives_in_bin[i] = np.sum(np.logical_and(bin_idx, y)) + self.num_groups
            self.num_in_bin[i] = np.sum(bin_idx)
            for j in range(self.num_groups):
                self.group_num_positives_in_bin[i][j] = np.sum(np.logical_and(np.logical_and(bin_idx, y),group_assignment[j])) + 1
                self.group_num_in_bin[i][j] = np.sum(np.logical_and(bin_idx, group_assignment[j])) + np.ceil(self.num_in_bin[i]/self.num_positives_in_bin[i])


            self.num_in_bin[i] += self.num_groups * np.ceil(self.num_in_bin[i]/self.num_positives_in_bin[i])
            # self.num_positives_in_bin[i] += self.num_groups
            assert(self.num_positives_in_bin[i]==np.sum(self.group_num_positives_in_bin[i])),  f"{self.num_positives_in_bin[i]},{np.sum(self.group_num_positives_in_bin[i]) + self.num_groups}"
            assert(self.num_in_bin[i]==np.sum(self.group_num_in_bin[i])), f"{self.num_in_bin[i],np.sum(self.group_num_in_bin[i])}"
            assert(self.group_num_in_bin[i]>0).all(), f"{self.group_num_in_bin}"
            assert(self.group_num_positives_in_bin[i]>0).all()
            assert(self.num_positives_in_bin[i]>0).all()
            assert(self.num_in_bin[i]>0).all()



        self.bin_rho = self.num_in_bin / self.num_examples
        self.bin_values = self.num_positives_in_bin / self.num_in_bin

        # self.sorted = np.argsort(self.bin_values)

        # for i in range(self.n_bins-1):
            # assert (self.bin_values[i]<=self.bin_values[i+1]), f"{self.bin_values[i], self.bin_values[i+1]}"
            # assert (self.num_positives_in_bin[i] * self.num_in_bin[i+1]<= self.num_positives_in_bin[i+1] * self.num_in_bin[i]), \
            #     f"{self.num_positives_in_bin[i], self.num_in_bin[i+1], self.num_positives_in_bin[i+1], self.num_in_bin[i]}"
            # if (self.bin_values[i]<self.bin_values[i+1]):
            #     print("--------------not monotone---------------------")

        # smoothing for the case where there is no sample of a particular group in some bin
        smooth_alpha = np.ones(shape=self.group_num_positives_in_bin.shape)
        smooth_d = np.repeat(1.0 / self.bin_values, self.group_num_in_bin.shape[1]).reshape(self.group_num_in_bin.shape)
        rho_smooth_d = np.repeat(self.num_groups, self.num_in_bin.shape[0]).reshape(self.num_in_bin.shape)

        # self.group_rho = (self.group_num_in_bin + smooth_alpha) / (self.num_in_bin + rho_smooth_d).reshape(self.num_in_bin.shape[0],1)
        # self.group_bin_values = (self.group_num_positives_in_bin + smooth_alpha) / (self.group_num_in_bin + smooth_d)

        self.group_rho = (self.group_num_in_bin) / (self.num_in_bin ).reshape(
            self.num_in_bin.shape[0], 1)
        self.group_bin_values = (self.group_num_positives_in_bin) / (self.group_num_in_bin)

        # sanity check
        for i in range(self.n_bins):
            assert (np.sum(self.group_rho[i] * self.group_bin_values[i]) - self.bin_values[i] < 1e-1), f"{self.num_in_bin ,self.group_rho[i],self.group_bin_values[i] , self.bin_values[i]}"
            assert (np.sum(self.group_rho[i]) - 1.0 < 1e-4)
            for j in range(self.num_groups):
                # if self.group_num_in_bin[i][j] == 0:
                #     assert (self.group_rho[i][j] - 1.0 / self.num_groups < 1e-3)
                # else:
                assert self.group_rho[i][j] - (
                            self.group_num_in_bin[i][j] / self.num_in_bin[i]) < 1e-3

                # if self.group_num_positives_in_bin[i][j] == 0:
                #     assert self.group_bin_values[i][j] - self.bin_values[i] < 1e-2
                # else:
                assert self.group_bin_values[i][j] - (
                            self.group_num_positives_in_bin[i][j] / self.group_num_in_bin[i][j]) < 1e-1

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

        # delta-randomization
        # scores = self._nudge(scores)

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
    # alpha = args.alpha

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
        X_cal_est, y_cal = pickle.load(f)
        available_features = np.setdiff1d(np.arange(X_cal_est.shape[1]), Z_indices)
        X_cal = X_cal_est[:,available_features]
        # print(X_cal.shape,X_cal_est.shape)

    groups = np.unique(X_cal_est[:, Z_indices])
    # print(f"{groups = }")


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

    umb_select = UMBSelect(args.B,Z_indices,groups,Z_map)
    umb_select.fit(X_cal_est,scores_cal, y_cal, m, k)

    s_test_raw = umb_select.select(scores_test_raw)
    # num_positives_in_bin, num_in_bin, group_num_positives_in_bin, group_num_in_bin, group_bin_values, group_rho = umb_select.get_group_statistics(X_cal_est, scores_cal,y_cal)
    # assert(umb_select.num_positives_in_bin == num_positives_in_bin).all()
    # assert(umb_select.num_in_bin == num_in_bin).all()
    # assert(umb_select.group_num_in_bin == group_num_in_bin).all()
    # assert(umb_select.group_num_positives_in_bin==group_num_positives_in_bin).all()
    # assert(umb_select.group_bin_values==group_bin_values).all()
    # assert(umb_select.group_rho==group_rho).all()

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
    performance_metrics["n_bins"] = umb_select.n_bins
    # print(performance_metrics["group_bin_value"])
    with open(args.result_path, 'wb') as f:
        pickle.dump(performance_metrics, f)

    with open(args.umb_path, "wb") as f:
        pickle.dump(umb_select, f)
