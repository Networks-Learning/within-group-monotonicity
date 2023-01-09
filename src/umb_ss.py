"""
Select a Shortlist of Applicants Based on Uniform Mass Binning (details in the paper).
"""
import argparse
import pickle
import numpy as np
from utils import calculate_expected_selected, calculate_expected_qualified, transform_except_last_dim
from sklearn.metrics import mean_squared_error,accuracy_score,roc_curve, roc_auc_score,log_loss,f1_score,precision_score


import warnings
warnings.filterwarnings(action='ignore',
                        category=RuntimeWarning)  # setting ignore as a parameter and further adding category

class UMBSelect(object):
    def __init__(self, n_bins, Z_indices, groups, Z_map):
        # Hyper-parameters
        self.n_bins = n_bins
        self.groups = groups
        self.Z_indices = Z_indices
        self.Z_map = Z_map
        self.num_groups = np.unique(Z_map[Z_indices[0]]).shape[0]  # grouping based on values of the chosen feature, see Z_map
        self.alpha = 0

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
        self.discriminated_against = None
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
        for cur_group in range(1,self.n_bins - 1):
            bin_upper_edges += [max(groups[cur_group])]
            bin_lower_edges += [max(groups[cur_group-1])]
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
        sorted = np.argsort(num_positives_in_bin/num_in_bin)
        assert(len(bin_upper_edges)==len(sorted))
        num_positives_in_bin = num_positives_in_bin[sorted]
        num_in_bin = num_in_bin[sorted]
        for i in range(self.n_bins-1):
            assert(num_positives_in_bin[i]*num_in_bin[i+1]<=num_positives_in_bin[i+1]*num_in_bin[i])   #monotonicity
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

        in_bin = np.empty(shape=(self.n_bins, scores.size))

        for i in range(self.n_bins):
            lower_edge = np.repeat(self.bin_lower_edges[i], scores.size)
            upper_edge = np.repeat(self.bin_upper_edges[i], scores.size)
            in_bin[i] = (np.logical_and(scores > lower_edge,scores <= upper_edge))
            bin_assignment[in_bin[i].astype(bool)] = i
        assert(np.sum(in_bin) == scores.size), f"{np.sum(in_bin), scores.size}"
        assert(bin_assignment>=0).all()
        assert (bin_assignment < self.n_bins).all(), f"{bin_assignment}"
        return bin_assignment.astype(int)

    def group_points(self, X_est):
        assert (self.num_groups is not None), "Group number not set"
        group_index = np.zeros(shape=(self.num_groups, X_est.shape[0]))
        for i, group in enumerate(self.groups):
            group = np.repeat(group, X_est.shape[0]).reshape(X_est.shape[0], 1)
            group_index[self.Z_map[self.Z_indices[0]][i]] = np.logical_or(group_index[self.Z_map[self.Z_indices[0]][i]],
                                                                np.equal(X_est[:, self.Z_indices], group).squeeze())

        return group_index


    # def _nudge(self, matrix):
    #     return ((matrix + np.random.uniform(low=0,
    #                                         high=self.delta,
    #                                         size=matrix.shape)) / (1 + self.delta))

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
        self.epsilon = 0

        # delta-randomization
        # y_score = self._nudge(y_score)

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
            self.num_positives_in_bin[i] = np.sum(np.logical_and(bin_idx, y)) #+ self.num_groups
            self.num_in_bin[i] = np.sum(bin_idx)
            for j in range(self.num_groups):
                self.group_num_positives_in_bin[i][j] = np.sum(np.logical_and(np.logical_and(bin_idx, y),group_assignment[j])) #+ 1
                self.group_num_in_bin[i][j] = np.sum(np.logical_and(bin_idx, group_assignment[j])) #+ np.ceil(self.num_in_bin[i]/self.num_positives_in_bin[i])

            assert(self.num_positives_in_bin[i]==np.sum(self.group_num_positives_in_bin[i])),  f"{self.num_positives_in_bin[i]},{np.sum(self.group_num_positives_in_bin[i]) + self.num_groups}"
            assert(self.num_in_bin[i]==np.sum(self.group_num_in_bin[i])), f"{self.num_in_bin[i],np.sum(self.group_num_in_bin[i])}"
            assert(self.num_in_bin[i]>0).all()



        self.bin_rho = self.num_in_bin / self.num_examples
        self.bin_values = self.num_positives_in_bin / self.num_in_bin

        # self.sorted = np.argsort(self.bin_values)

        for i in range(self.n_bins-1):
            # assert (self.bin_values[i]<=self.bin_values[i+1]), f"{self.bin_values[i], self.bin_values[i+1]}"
            assert (self.num_positives_in_bin[i] * self.num_in_bin[i+1]<= self.num_positives_in_bin[i+1] * self.num_in_bin[i]), \
                f"{self.num_positives_in_bin[i], self.num_in_bin[i+1], self.num_positives_in_bin[i+1], self.num_in_bin[i]}"

        # smoothing for the case where there is no sample of a particular group in some bin
        # smooth_alpha = np.ones(shape=self.group_num_positives_in_bin.shape)
        # smooth_d = np.repeat(1.0 / self.bin_values, self.group_num_in_bin.shape[1]).reshape(self.group_num_in_bin.shape)
        # rho_smooth_d = np.repeat(self.num_groups, self.num_in_bin.shape[0]).reshape(self.num_in_bin.shape)

        # self.group_rho = (self.group_num_in_bin + smooth_alpha) / (self.num_in_bin + rho_smooth_d).reshape(self.num_in_bin.shape[0],1)
        # self.group_bin_values = (self.group_num_positives_in_bin + smooth_alpha) / (self.group_num_in_bin + smooth_d)

        positive_group_rho = np.greater(self.group_num_in_bin,np.zeros(shape=self.group_num_in_bin.shape))
        assert positive_group_rho.shape==self.group_num_in_bin.shape
        self.group_rho = np.where(positive_group_rho,(self.group_num_in_bin) / (self.num_in_bin )[:,np.newaxis],np.zeros(shape=self.group_num_in_bin.shape))
        self.group_bin_values = np.where(positive_group_rho,(self.group_num_positives_in_bin) / (self.group_num_in_bin),np.zeros(shape=self.group_num_in_bin.shape))
        self.discriminated_against = np.zeros(shape=self.group_num_in_bin.shape)
        #sanity check
        for i in range(self.n_bins):
            assert (np.sum(self.group_rho[i] * self.group_bin_values[i]) - self.bin_values[i] < 1e-2), f"{self.num_in_bin ,self.group_rho[i],self.group_bin_values[i] , self.bin_values[i]}"
            assert (np.sum(self.group_rho[i]) - 1.0 < 1e-2)
            for j in range(self.num_groups):
                if i < self.n_bins - 1 and positive_group_rho[i][j]:
                    self.discriminated_against[i][j] =  np.greater(self.group_num_positives_in_bin[i][j] * self.group_num_in_bin[i + 1][j],
                            self.group_num_positives_in_bin[i + 1][j] * self.group_num_in_bin[i][j])
                if self.group_num_in_bin[i][j] == 0:
                    assert (self.group_rho[i][j] ==0)
                else:
                    assert self.group_rho[i][j] * self.num_in_bin[i] - self.group_num_in_bin[i][j] < 1e-2

                if self.group_num_positives_in_bin[i][j] == 0:
                    assert self.group_bin_values[i][j] == 0
                else:
                    assert self.group_bin_values[i][j] * self.group_num_in_bin[i][j] - self.group_num_positives_in_bin[i][j] < 1e-2

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
        # print(f"{self.b=}")
        # print(f"{self.theta=}")



        # histogram binning done
        self.fitted = True

    def select(self, scores):
        scores = scores.squeeze()
        size = scores.size
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

    def global_select(self,scores):
        scores = scores.squeeze()
        size = scores.size
        # assign test data to bins
        test_bins = self._bin_points(scores)
        # make decisions
        return self.bin_values[test_bins]>0.5


    def get_test_roc(self, X, scores, y):
        scores = scores.squeeze()
        # assign test data to bins
        test_bins = self._bin_points(scores)
        y_prob = self.bin_values[test_bins]
        fpr, tpr, _ = roc_curve(y,y_prob)

        test_group_assignment = self.group_points(X).astype(bool)

        group_fpr = np.zeros(shape = (self.num_groups,self.n_bins+1))
        group_tpr = np.zeros(shape = (self.num_groups,self.n_bins+1))
        #
        # for j in range(self.num_groups):
        #     group_fpr[j], group_tpr[j], thresholds = roc_curve(y[test_group_assignment[j]],y_prob[test_group_assignment[j]],drop_intermediate=False)
            # print(f"{thresholds,self.bin_values}")
        return fpr, tpr#, group_fpr, group_tpr

    def get_shortlist_group_accuracy(self,selection, X, y):
        test_group_assignment = self.group_points(X).astype(bool)
        shortlist_group_accuracy = np.zeros(self.num_groups)
        for j in range(self.num_groups):
            # print(f"{np.sum(test_group_assignment[j])=}")
            shortlist_group_accuracy[j] = accuracy_score(selection[test_group_assignment[j]],y[test_group_assignment[j]])
            # group_accuracy[j] = np.average(selection[test_group_assignment[j]])
        # print(f"{group_accuracy = }")
        return shortlist_group_accuracy

    def get_accuracy(self,selection, y):
        # scores = scores.squeeze()
        # assign test data to bins
        # test_bins = self._bin_points(scores)
        # y_prob = self.bin_values[test_bins]
        # y_pred = y_prob>self.theta
        return accuracy_score(y,selection),precision_score(y,selection)

    def get_group_accuracy(self,X, scores, y):
        scores = scores.squeeze()
        # assign test data to bins
        test_group_assignment = self.group_points(X).astype(bool)
        group_accuracy = np.zeros(self.num_groups)
        test_bins = self._bin_points(scores)
        y_prob = self.bin_values[test_bins]
        y_pred = y_prob>0.5
        for grp in range(self.num_groups):
            group_accuracy[grp] = accuracy_score(y[test_group_assignment[grp]],y_pred[test_group_assignment[grp]])

        return group_accuracy

    def get_calibration_curve(self,scores,y):

        sorted_indexes = np.argsort(scores)
        y = y[sorted_indexes]
        scores = scores[sorted_indexes]
        # split scores into groups of approx equal size
        split_size = 30
        groups = np.array_split(sorted_indexes, split_size)
        scores = scores.squeeze()
        # assign test data to bins
        test_bins = self._bin_points(scores)
        prob_pred = np.zeros(split_size)
        prob_true = np.zeros(split_size)
        ECE = np.zeros(split_size)

        for i,group in enumerate(groups):
            # print(group)
            prob_true[i] = np.sum(y[group])
            prob_pred[i] = np.sum(self.bin_values[test_bins[group]])
            ECE[i]= abs(prob_true[i]-prob_pred[i])
        return prob_true, prob_pred, np.sum(ECE)/scores.shape[0]


    def get_ECE(self,scores,y):
        from sklearn.calibration import calibration_curve
        scores = scores.squeeze()
        test_bins = self._bin_points(scores)
        y_pred = self.bin_values[test_bins]
        prob_true, prob_pred = calibration_curve(y,y_pred,n_bins=self.n_bins,strategy='quantile')
        return np.average(np.abs(prob_true - prob_pred))

    def get_sharpness(self,scores,y):
        # sorted_indexes = np.argsort(scores)
        # scores = scores[sorted_indexes]
        # split scores into groups of approx equal size
        # groups = np.array_split(sorted_indexes, self.n_bins)
        # split_size = int(scores.shape[0]/self.n_bins)
        scores = scores.squeeze()
        # assign test data to bins
        test_bins = self._bin_points(scores)
        var = np.zeros(self.n_bins)

        for i in range(self.n_bins):
            in_bin_i = (test_bins==i)
            var[i] = np.var(y[in_bin_i])

        return np.average(var)


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
    parser.add_argument("--n_runs_test", type=int, help="the number of tests for estimating the expectation")


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
        X_cal_all_features, y_cal = pickle.load(f)
        available_features = np.setdiff1d(np.arange(X_cal_all_features.shape[1]), Z_indices)
        X_cal = X_cal_all_features[:,available_features]
        # print(X_cal.shape,X_cal_est.shape)

    groups = np.unique(X_cal_all_features[:, Z_indices])

    with open(args.classifier_path, "rb") as f:
        classifier = pickle.load(f)

    n = y_cal.size
    scores_cal = classifier.predict_proba(X_cal)[:, 1]

    umb_select = UMBSelect(args.B, Z_indices, groups, Z_map)
    umb_select.fit(X_cal_all_features, scores_cal, y_cal, m, k)


    #test
    with open(args.test_raw_path, "rb") as f:
        X_test_all_features, y_test_raw = pickle.load(f)
    with open(args.scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    X_test_all_features = transform_except_last_dim(X_test_all_features, scaler)
    X_test_raw = X_test_all_features[:, available_features]
    scores_test_raw = classifier.predict_proba(X_test_raw)[:, 1]
    total_test_selected = umb_select.select(scores_test_raw)
    fpr, tpr = umb_select.get_test_roc(X_test_all_features,scores_test_raw,y_test_raw)
    accuracy,f1score = umb_select.get_accuracy(total_test_selected,y_test_raw)
    group_accuracy = umb_select.get_group_accuracy(X_test_all_features,scores_test_raw,y_test_raw)
    # prob_true, prob_pred, ECE = umb_select.get_calibration_curve(scores_cal,y_cal)
    # ECE = umb_select.get_ECE(scores_cal,y_cal)
    sharpness = umb_select.get_sharpness(scores_cal,y_cal)
    # group_accuracy = umb_select.get_group_accuracy(total_test_selected,X_test_all_features,y_test_raw)

    # simulating pools of candidates
    num_selected = []
    num_qualified = []
    for i in range(args.n_runs_test):
        indexes = np.random.choice(list(range(y_test_raw.size)), int(m))
        y_test = y_test_raw[indexes]
        scores_test = scores_test_raw[indexes]
        test_selected = umb_select.select(scores_test)
        num_selected.append(calculate_expected_selected(test_selected, y_test, m))
        num_qualified.append(calculate_expected_qualified(test_selected, y_test, m))

    performance_metrics = {}
    performance_metrics["num_qualified"] = np.mean(num_qualified)
    performance_metrics["num_selected"] = np.mean(num_selected)
    performance_metrics["constraint_satisfied"] = True if performance_metrics["num_qualified"] >= k else False
    performance_metrics["fpr"] = fpr
    performance_metrics["tpr"] = tpr
    # performance_metrics["group_fpr"] = group_fpr
    # performance_metrics["group_tpr"] = group_tpr
    performance_metrics["accuracy"] = accuracy
    performance_metrics["f1_score"] = f1score
    # performance_metrics["prob_true"] = prob_true
    # performance_metrics["prob_pred"] = prob_pred
    # performance_metrics["ECE"] = ECE
    performance_metrics["sharpness"] = sharpness
    # performance_metrics["MSE"] = MSE
    performance_metrics["group_accuracy"] = group_accuracy
    performance_metrics["num_positives_in_bin"] = umb_select.num_positives_in_bin
    performance_metrics["num_in_bin"] = umb_select.num_in_bin
    performance_metrics["bin_values"] = umb_select.bin_values
    performance_metrics["group_num_positives_in_bin"] = umb_select.group_num_positives_in_bin
    performance_metrics["group_num_in_bin"] = umb_select.group_num_in_bin
    performance_metrics["group_bin_values"] = umb_select.group_bin_values
    performance_metrics["group_rho"] = umb_select.group_rho
    performance_metrics["bin_rho"] = umb_select.bin_rho
    performance_metrics["groups"] = umb_select.groups
    performance_metrics["num_groups"] = umb_select.num_groups
    performance_metrics["n_bins"] = umb_select.n_bins
    performance_metrics["discriminated_against"] = umb_select.discriminated_against
    performance_metrics["alpha"] = 0
    # print(performance_metrics["group_bin_value"])
    with open(args.result_path, 'wb') as f:
        pickle.dump(performance_metrics, f)

    with open(args.umb_path, "wb") as f:
        pickle.dump(umb_select, f)
