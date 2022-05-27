import numpy as np
from ml.simulate.common.utils import calc_total_f1
from ml.simulate.common.utils import calc_total_precision
from ml.simulate.common.utils import calc_total_recall

def select_by_round_robin(tensor_run, column_states, current_f_list, col_list, steps, use_sum=True):

    for s in range(steps):
        col_id = int(np.min(np.argmin(column_states)))

        column_states[col_id] += 1
        col_list.append(col_id)
        current_f_list.append(calc_total_f1(tensor_run, column_states))

    return current_f_list, col_list


def select_by_round_robin_all_measures(tensor_run, column_states, current_f_list, current_prec_list, current_rec_list, col_list, steps, use_sum=True):

    for s in range(steps):
        col_id = int(np.min(np.argmin(column_states)))

        column_states[col_id] += 1
        col_list.append(col_id)
        #current_f_list.append(calc_total_f1(tensor_run, column_states))
        p = calc_total_precision(tensor_run, column_states)
        r = calc_total_recall(tensor_run, column_states)
        current_f_list.append((2 *(p*r)/(p+r)) )
        current_prec_list.append(p)
        current_rec_list.append(r)

    return current_f_list, current_prec_list, current_rec_list, col_list
