"""
Utils for Subset Selection
"""
import numpy as np
from sklearn.metrics import mean_squared_error,accuracy_score


def transform_except_last_dim(data, scaler):
    return np.concatenate((scaler.transform(data[:, :-1]), data[:, -1:]), axis=1)


def calculate_expected_qualified(s, y, m):
    return np.sum(s * y) * 1. * m / y.size


def calculate_expected_selected(s, y, m):
    return np.sum(s) * 1. * m / y.size

def list_maker(val,n):
    return [val]*n

def save_results(classifier,result_path):
    # test
    with open(args.test_raw_path, "rb") as f:
        X_test_all_features, y_test_raw = pickle.load(f)
    with open(args.scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    X_test_all_features = transform_except_last_dim(X_test_all_features, scaler)
    X_test_raw = X_test_all_features[:, available_features]
    scores_test_raw = classifier.predict_proba(X_test_raw)[:, 1]
    total_test_selected = wgm.recal_global_select(scores_test_raw)
    fpr, tpr = wgm.recal_get_test_roc(X_test_all_features, scores_test_raw, y_test_raw)
    accuracy, f1score = wgm.get_accuracy(total_test_selected, y_test_raw)
    group_accuracy = wgm.recal_get_group_accuracy(X_test_all_features, scores_test_raw, y_test_raw)
    # prob_true, prob_pred, ECE = wgm.recal_get_calibration_curve(scores_cal, y_cal)
    # ECE = wgm.recal_get_ECE(scores_cal,y_cal)
    sharpness = wgm.recal_get_sharpness(scores_cal, y_cal)
    # group_accuracy = wgm.get_group_accuracy(total_test_selected, X_test_all_features, y_test_raw)

    # simulating pools of candidates
    num_selected = []
    num_qualified = []
    for i in range(args.n_runs_test):
        indexes = np.random.choice(list(range(y_test_raw.size)), int(m))
        y_test = y_test_raw[indexes]
        scores_test = scores_test_raw[indexes]
        recal_test_selected = wgm.recal_select(scores_test)
        num_selected.append(calculate_expected_selected(recal_test_selected, y_test, m))
        num_qualified.append(calculate_expected_qualified(recal_test_selected, y_test, m))

    performance_metrics = {}
    performance_metrics["num_qualified"] = np.mean(num_qualified)
    performance_metrics["num_selected"] = np.mean(num_selected)
    performance_metrics["constraint_satisfied"] = True if performance_metrics["num_qualified"] >= k else False
    performance_metrics["accuracy"] = accuracy
    performance_metrics["f1_score"] = f1score
    # performance_metrics["prob_true"] = prob_true
    # performance_metrics["prob_pred"] = prob_pred
    # performance_metrics["ECE"] = ECE
    performance_metrics["sharpness"] = sharpness
    performance_metrics["fpr"] = fpr
    performance_metrics["tpr"] = tpr
    # performance_metrics["group_fpr"] = group_fpr
    # performance_metrics["group_tpr"] = group_tpr
    performance_metrics["group_accuracy"] = group_accuracy
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


Z_map = {
    0: list_maker(0,25) + list_maker(1,25) + list_maker(2,25) + list_maker(3,25),
    1: list_maker(0,16)+list_maker(1,4)+list_maker(2,2)+list_maker(3,3),
    2: [0,0,0,0,1],
    4: [0,1],
    6: [0,1,2,2,3],
    14: [0,1],
    15: [0] + [1] + list_maker(2,3) + list_maker(3,4)
}


