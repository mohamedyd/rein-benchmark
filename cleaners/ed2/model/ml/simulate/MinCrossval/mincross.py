import numpy as np
from ml.simulate.common.utils import calc_total_f1

def get_cross_val_sum(x, featurenames):
    ids = []
    for feature_i in range(len(featurenames)):
        if 'icross_val' in featurenames[feature_i]:
            ids.append(feature_i)

    return np.mean(x[:,ids], axis=1)

def select_by_min_cross_val(tensor_run, column_states, current_f_list, col_list, matrix_crossval, steps, use_sum=True):

    for s in range(steps):
        min_cross = 10000000.0
        min_cross_id = -1
        for col_i in range(len(column_states)):
            if matrix_crossval[col_i][column_states[col_i]] < min_cross:
                min_cross = matrix_crossval[col_i][column_states[col_i]]
                min_cross_id = col_i

        column_states[min_cross_id] += 1
        col_list.append(min_cross_id)
        current_f_list.append(calc_total_f1(tensor_run, column_states))

    return current_f_list, col_list