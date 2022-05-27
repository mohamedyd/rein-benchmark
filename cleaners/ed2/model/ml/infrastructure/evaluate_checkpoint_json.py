import json
from pprint import pprint
import numpy as np

def evaluate_check_point_json(checkpoint_json_file):


    with open(checkpoint_json_file) as data_file:
        data = json.load(data_file)

    loss_history = data['val_loss_history']
    checkpoint_pointer = data['val_loss_history_it']

    best_i = np.argmin(loss_history)

    return loss_history[best_i], checkpoint_pointer[best_i]


print evaluate_check_point_json('/home/felix/SequentialPatternErrorDetection/checkpoint_json/checkpoint_1800.json')