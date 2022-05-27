import numpy as np
from ml.simulate.common.utils import calc_total_f1
from ml.simulate.common.utils import calc_total_precision
from ml.simulate.common.utils import calc_total_recall

def get_all_certainty_sum(x, featurenames):
    ids = []
    for feature_i in range(len(featurenames)):
        if 'certainty' == featurenames[feature_i]:
            ids.append(feature_i)

    return x[:,ids]

def select_by_max_uncertainty_all(tensor_run, column_states, current_f_list, col_list, matrix_all_certainty_sum, steps, use_sum=True):

    for s in range(steps):
        max_uncertainty_sum = 10000000.0
        max_uncertainty_id = -1
        for col_i in range(len(column_states)):
            if matrix_all_certainty_sum[col_i][column_states[col_i]] < max_uncertainty_sum:
                max_uncertainty_sum = matrix_all_certainty_sum[col_i][column_states[col_i]]
                max_uncertainty_id = col_i

        column_states[max_uncertainty_id] += 1
        col_list.append(max_uncertainty_id)
        current_f_list.append(calc_total_f1(tensor_run, column_states))

    return current_f_list, col_list



def select_by_max_uncertainty_all_metrics(tensor_run, column_states, current_f_list, current_prec_list, current_rec_list, col_list, matrix_all_certainty_sum, steps, use_sum=True):

    for s in range(steps):
        max_uncertainty_sum = 10000000.0
        max_uncertainty_id = -1
        for col_i in range(len(column_states)):
            if matrix_all_certainty_sum[col_i][column_states[col_i]] < max_uncertainty_sum:
                max_uncertainty_sum = matrix_all_certainty_sum[col_i][column_states[col_i]]
                max_uncertainty_id = col_i

        column_states[max_uncertainty_id] += 1
        col_list.append(max_uncertainty_id)
        current_f_list.append(calc_total_f1(tensor_run, column_states))
        current_prec_list.append(calc_total_precision(tensor_run, column_states))
        current_rec_list.append(calc_total_recall(tensor_run, column_states))

    return current_f_list, current_prec_list, current_rec_list, col_list



def select_by_max_uncertainty_all_prob(tensor_run, column_states, current_f_list, col_list, matrix_all_certainty_sum, steps, use_sum=True):

    for s in range(steps):
        certainties = np.zeros(len(column_states))
        for col_i in range(len(column_states)):
            certainties[col_i] = matrix_all_certainty_sum[col_i][column_states[col_i]]

        certainties -= np.ones(len(column_states))

        #certainties = np.square(certainties)
        certainties = certainties / np.sum(certainties)

        max_uncertainty_id = np.random.choice(len(certainties), 1, p=certainties)[0]
        #max_uncertainty_id = np.argmax(certainties)

        column_states[max_uncertainty_id] += 1
        col_list.append(max_uncertainty_id)

        print column_states
        print tensor_run.shape

        current_f_list.append(calc_total_f1(tensor_run, column_states))

    return current_f_list, col_list
