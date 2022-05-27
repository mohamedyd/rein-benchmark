import numpy as np
from ml.simulate.common.utils import calc_total_f1
from ml.simulate.common.utils import calc_column_f1

def find_max_total_f(tensor_run, column_states, current_f_list, col_list, steps, use_sum=True):
    if steps > 0:
        all_list = []
        all_col_list = []
        all_sum = []

        for col in range(len(column_states)):
            if (column_states[col] + 1) < len(tensor_run[col]):
                new_column_states = column_states.copy()
                new_column_states[col] += 1
                new_current_f_list = list(current_f_list)
                new_current_f_list.append(calc_total_f1(tensor_run, new_column_states))
                new_col_list = list(col_list)
                new_col_list.append(col)
                # print new_current_f_list

                result_list, result_col_list = find_max_total_f(tensor_run, new_column_states, new_current_f_list,
                                                                new_col_list, steps - 1)
                all_list.append(result_list)
                all_col_list.append(result_col_list)
                if use_sum:
                    all_sum.append(np.sum(result_list))
                else:
                    all_sum.append(np.max(result_list))

        if len(all_sum) > 0:
            argmax = np.argmax(all_sum)
            return all_list[argmax], all_col_list[argmax]
        else:
            return current_f_list, col_list
    else:
        return current_f_list, col_list


def find_max_total_f_new(tensor_run, column_states, current_f_list, col_list, steps, done={}, use_sum=True):
    if steps > 0:
        all_list = []
        all_col_list = []
        all_sum = []

        for col in range(len(column_states)):
            if (column_states[col] + 1) < len(tensor_run[col]) and not col in done:
                new_column_states = column_states.copy()
                new_column_states[col] += 1
                new_current_f_list = list(current_f_list)
                new_current_f_list.append(calc_total_f1(tensor_run, new_column_states))
                new_col_list = list(col_list)
                new_col_list.append(col)

                column_f1 = calc_column_f1(tensor_run, new_column_states, col)
                new_done = dict(done)
                if column_f1 == 1.0:
                    new_done[col] = True
                # print new_current_f_list

                result_list, result_col_list = find_max_total_f_new(tensor_run, new_column_states, new_current_f_list,
                                                                new_col_list, steps - 1, new_done)
                all_list.append(result_list)
                all_col_list.append(result_col_list)
                if use_sum:
                    all_sum.append(np.sum(result_list))
                else:
                    all_sum.append(np.max(result_list))

        if len(all_sum) > 0:
            argmax = np.argmax(all_sum)
            return all_list[argmax], all_col_list[argmax]
        else:
            return current_f_list, col_list
    else:
        return current_f_list, col_list