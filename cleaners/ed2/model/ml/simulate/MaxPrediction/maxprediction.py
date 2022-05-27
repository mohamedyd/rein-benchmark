import numpy as np
from ml.simulate.common.utils import calc_total_f1

def get_prediction_change(x, featurenames):
    #prediction_change[column_id] = change_0_to_1 + change_1_to_0

    ids = []
    for feature_i in range(len(featurenames)):
        if 'change_0_to_1' in featurenames[feature_i] or 'change_1_to_0' in featurenames[feature_i]:
            ids.append(feature_i)

    return np.sum(x[:, ids], axis=1)

def select_by_max_prediction_change(tensor_run, column_states, current_f_list, col_list, matrix_change_sum, steps, use_sum=True):

    for s in range(steps):
        max_change_sum = -1.0
        max_change_id = -1
        for col_i in range(len(column_states)):
            if matrix_change_sum[col_i][column_states[col_i]] > max_change_sum:
                max_change_sum = matrix_change_sum[col_i][column_states[col_i]]
                max_change_id = col_i

        column_states[max_change_id] += 1
        col_list.append(max_change_id)
        current_f_list.append(calc_total_f1(tensor_run, column_states))

    return current_f_list, col_list
