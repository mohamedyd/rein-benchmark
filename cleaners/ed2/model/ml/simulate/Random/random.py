import numpy as np
from ml.simulate.common.utils import calc_total_f1

def select_by_random(tensor_run, column_states, current_f_list, col_list, steps, use_sum=True):

    for s in range(steps):
        col_id = np.random.randint(len(column_states))

        column_states[col_id] += 1
        col_list.append(col_id)
        current_f_list.append(calc_total_f1(tensor_run, column_states))

    return current_f_list, col_list
