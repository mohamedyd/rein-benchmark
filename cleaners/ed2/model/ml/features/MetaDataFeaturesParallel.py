from scipy.sparse import lil_matrix
from scipy.sparse import hstack
import numpy as np
import re
import multiprocessing as mp

from typing import Dict, Any, Union


def get_number_of_occurrences_fit(data, column_id):
    unique_value_counts = {}  # type: Dict[Any, int]
    for i in range(data.shape[0]):
        value = data[i, column_id]
        if value in unique_value_counts:
            unique_value_counts[value] = unique_value_counts[value] + 1
        else:
            unique_value_counts[value] = 1
    return unique_value_counts


def get_number_of_occurrences_transform(data, column_id, model):
    unique_value_counts = model[column_id][get_number_of_occurrences_fit]
    feature = lil_matrix((data.shape[0], 1))
    for i in range(data.shape[0]):
        value = data[i, column_id]
        if value in unique_value_counts:
            feature[i] = unique_value_counts[value]

    return feature, 'occurrence_count'


def is_number(value, model=None):
    try:
        float(value)
        return True
    except ValueError:
        return False


def is_numerical(data, column_id, model=None):
    feature = np.zeros((data.shape[0], 1), dtype=bool)
    for i in range(data.shape[0]):
        value = data[i, column_id]
        feature[i] = is_number(value)

    return feature, 'is_numerical'


def string_length(data, column_id, model=None):
    feature = np.zeros((data.shape[0], 1))
    for i in range(data.shape[0]):
        value = data[i, column_id]
        try:
            feature[i] = len(str(value.encode('utf-8')))
        except:
            feature[i] = len(str(value))
    return feature, 'string_length'


def is_alphabetical(data, column_id, model=None):
    feature = np.zeros((data.shape[0], 1), dtype=bool)
    for i in range(data.shape[0]):
        value = data[i, column_id]
        feature[i, 0] = re.findall("^[A-Za-z_]+$", value)

    return feature, 'is_alphabetical'


def extract_number(data, column_id, model=None):
    feature = lil_matrix((data.shape[0], 1))
    for i in range(data.shape[0]):
        value = data[i, column_id]
        try:
            feature[i] = float(value)
        except ValueError:
            pass
    return feature, 'extracted_number'


def run_multi_transform(params):
    results = {}
    results['feature'], results['name'] = params['method'](params['data'], params['column_id'], params['model'])
    return results

def run_multi_fit(params):
    return params['method'](params['data'], params['column_id'])


class MetaDataFeatures:
    def __init__(self, processes=4):
        self.unique_value_counts = {}
        self.processes = processes

    def fit(self, data):
        self.model = {}


        feature_methods_fit = [get_number_of_occurrences_fit]

        my_array = []
        for c in range(data.shape[1]):
            self.model[c] = {}
            for feature_method in feature_methods_fit:
                my_params = {}
                my_params['method'] = feature_method
                my_params['data'] = data
                my_params['column_id'] = c
                my_array.append(my_params)
                self.model[c][feature_method] = {}

        my_pool = mp.Pool(self.processes)
        results = my_pool.map(run_multi_fit, my_array)
        my_pool.close()
        my_pool.join()
        for r_i in range(len(results)):
            self.model[my_array[r_i]['column_id']][my_array[r_i]['method']] = results[r_i]


    def transform(self, data, columns):
        feature_methods_transform = [get_number_of_occurrences_transform,
                           string_length,
                           is_numerical,
                           is_alphabetical,
                           extract_number]

        my_array = []
        for c in range(data.shape[1]):
            for feature_method in feature_methods_transform:
                my_params = {}
                my_params['method'] = feature_method
                my_params['data'] = data
                my_params['column_id'] = c
                my_params['model'] = self.model
                my_array.append(my_params)

        feature_list = []
        feature_name_list = []

        my_pool = mp.Pool(self.processes)
        results = my_pool.map(run_multi_transform, my_array)
        my_pool.close()
        my_pool.join()
        for r_i in range(len(results)):
            new_feature = results[r_i]['feature']
            feature_name = results[r_i]['name']
            c = my_array[r_i]['column_id']
            feature_list.append(new_feature)
            feature_name_list.append(str(columns[c]) + "_" + feature_name)
            print("Finished " + feature_name + " -> " + str(new_feature.shape))

        metadata_features = hstack(feature_list)

        return metadata_features, feature_name_list

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

