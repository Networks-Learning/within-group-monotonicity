"""
Prepare data
"""
import argparse
import pickle
from sklearn.utils import shuffle
import numpy as np

from folktables import ACSDataSource, ACSEmployment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_cal_raw_path", type=str, help="the path for saving the raw data for "
                                                               "sampling train and calibration data")
    parser.add_argument("--test_raw_path", type=str, help="the path for saving the raw data for sampling test data")
    parser.add_argument("--q_ratio", type=float, default=0.2, help="percentage of qualified applicants")
    parser.add_argument("--test_ratio", type=float, default=0.5, help="percentage data as test data")
    args = parser.parse_args()
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    # definition_df = data_source.get_definitions(download=True)
    # categories = data_source.generate_categories(features=ACSEmployment.features, definition_df=definition_df)

    acs_data = data_source.get_data(download=True)
    X, y, _ = ACSEmployment.df_to_numpy(acs_data)
    X, y = shuffle(X, y)
    # print(X[:10],y[:10])
    print(np.average(y))
    print(y.sum()/y.shape[0])
    print(y.shape)
    print(type(y))
    print(y[0])

    test_raw_size = int(y.size * args.test_ratio)
    print(y.shape)
    print(type(y))
    print(y[0])
    with open(args.test_raw_path, "wb") as f:
        pickle.dump([X[:test_raw_size], y[:test_raw_size]], f)
    with open(args.train_cal_raw_path, "wb") as f:
        pickle.dump([X[test_raw_size:], y[test_raw_size:]], f)
