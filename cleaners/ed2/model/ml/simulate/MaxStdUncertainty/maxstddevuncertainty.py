import numpy as np
from ml.simulate.common.utils import calc_total_f1

def get_all_certainty_stddev(x, featurenames):
    ids = []
    for feature_i in range(len(featurenames)):
        if 'certainty_stddev' == featurenames[feature_i]:
            ids.append(feature_i)

    return x[:,ids]

def select_by_max_stddev_certainty_all(tensor_run, column_states, current_f_list, col_list, matrix_all_certainty_stddev, steps, use_sum=True):

    for s in range(steps):
        max_uncertainty_stddev = 0.0
        max_uncertainty_id = -1
        for col_i in range(len(column_states)):
            if matrix_all_certainty_stddev[col_i][column_states[col_i]] > max_uncertainty_stddev:
                max_uncertainty_stddev = matrix_all_certainty_stddev[col_i][column_states[col_i]]
                max_uncertainty_id = col_i

        column_states[max_uncertainty_id] += 1
        col_list.append(max_uncertainty_id)
        current_f_list.append(calc_total_f1(tensor_run, column_states))

    return current_f_list, col_list
