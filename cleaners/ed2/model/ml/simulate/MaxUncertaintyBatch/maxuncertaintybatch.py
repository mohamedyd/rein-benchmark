import numpy as np
from ml.simulate.common.utils import calc_total_f1

def get_batch_certainty_sum(x, featurenames):
    ids = []
    for feature_i in range(len(featurenames)):
        if 'batch_certainty' in featurenames[feature_i]:
            ids.append(feature_i)

    return np.sum(x[:,ids], axis=1)

def select_by_max_uncertainty_batch(tensor_run, column_states, current_f_list, col_list, matrix_batch_certainty_sum, steps, use_sum=True):

    for s in range(steps):
        max_uncertainty_sum = 10000000.0
        max_uncertainty_id = -1
        for col_i in range(len(column_states)):
            if matrix_batch_certainty_sum[col_i][column_states[col_i]] < max_uncertainty_sum:
                max_uncertainty_sum = matrix_batch_certainty_sum[col_i][column_states[col_i]]
                max_uncertainty_id = col_i

        column_states[max_uncertainty_id] += 1
        col_list.append(max_uncertainty_id)
        current_f_list.append(calc_total_f1(tensor_run, column_states))

    return current_f_list, col_list
