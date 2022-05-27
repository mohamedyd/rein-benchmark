def calc_total_f1(tensor_run, column_states):
    tp_all_sum = 0.0
    fp_all_sum = 0.0
    fn_all_sum = 0.0

    f_p = 0
    f_n = 1
    t_p = 2

    for col in range(len(column_states)):
        if column_states[col] == -1:
            fp_all_sum += 0.0
            fn_all_sum += tensor_run[col, 0, t_p] + tensor_run[col, 0, f_n]
            tp_all_sum += 0.0
        else:
            fp_all_sum += tensor_run[col, column_states[col], f_p]
            fn_all_sum += tensor_run[col, column_states[col], f_n]
            tp_all_sum += tensor_run[col, column_states[col], t_p]

    total_fscore = (2 * tp_all_sum) / ((2 * tp_all_sum) + (fp_all_sum + fn_all_sum))

    return total_fscore


def calc_total_precision(tensor_run, column_states):
    tp_all_sum = 0.0
    fp_all_sum = 0.0

    f_p = 0
    t_p = 2

    for col in range(len(column_states)):
        if column_states[col] == -1:
            fp_all_sum += 0.0
            tp_all_sum += 0.0
        else:
            fp_all_sum += tensor_run[col, column_states[col], f_p]
            tp_all_sum += tensor_run[col, column_states[col], t_p]

    total_precision = tp_all_sum / (tp_all_sum + fp_all_sum)

    return total_precision

def calc_total_recall(tensor_run, column_states):
    tp_all_sum = 0.0
    fn_all_sum = 0.0

    f_n = 1
    t_p = 2

    for col in range(len(column_states)):
        if column_states[col] == -1:
            fn_all_sum += tensor_run[col, 0, t_p] + tensor_run[col, 0, f_n]
            tp_all_sum += 0.0
        else:
            fn_all_sum += tensor_run[col, column_states[col], f_n]
            tp_all_sum += tensor_run[col, column_states[col], t_p]

    total_recall = tp_all_sum / (tp_all_sum + fn_all_sum)

    return total_recall


def calc_column_f1(tensor_run, column_states, col):
    f_p = 0
    f_n = 1
    t_p = 2

    total_fscore = (2 * tensor_run[col, column_states[col], t_p]) / ((2 * tensor_run[col, column_states[col], t_p]) + (tensor_run[col, column_states[col], f_p] + tensor_run[col, column_states[col], f_n]))

    return total_fscore


def calc_total_f1_new(tensor_run, column_states):
    tp_all_sum = 0.0
    fp_all_sum = 0.0
    fn_all_sum = 0.0

    f_p = 0
    f_n = 1
    t_p = 2

    for col in range(len(column_states)):
        fp_all_sum += tensor_run[col, column_states[col], f_p]
        fn_all_sum += tensor_run[col, column_states[col], f_n]
        tp_all_sum += tensor_run[col, column_states[col], t_p]

    precision = tp_all_sum / (tp_all_sum + fp_all_sum)
    recall = tp_all_sum / (tp_all_sum + fn_all_sum)

    total_fscore = (2 * precision * recall) / (precision + recall)

    return total_fscore