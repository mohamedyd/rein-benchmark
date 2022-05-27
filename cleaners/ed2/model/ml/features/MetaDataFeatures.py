from scipy.sparse import lil_matrix
from scipy.sparse import hstack
import numpy as np
import re

class MetaDataFeatures:
    def __init__(self):
        self.unique_value_counts = {}

    def get_number_of_occurrences_fit(self, data, column_id):
        if not column_id in self.unique_value_counts:
            self.unique_value_counts[column_id] = {}
        for i in range(data.shape[0]):
            value = data[i, column_id]
            if value in self.unique_value_counts[column_id]:
                self.unique_value_counts[column_id][value] = self.unique_value_counts[column_id][value] + 1
            else:
                self.unique_value_counts[column_id][value] = 1

    def get_number_of_occurrences_transform(self, data, column_id):
        feature = lil_matrix((data.shape[0], 1))
        for i in range(data.shape[0]):
            value = data[i, column_id]
            if value in self.unique_value_counts[column_id]:
                feature[i] = self.unique_value_counts[column_id][value]

        return feature, 'occurrence_count'

    def is_number(self,value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    def is_numerical(self, data, column_id):
        feature = np.zeros((data.shape[0],1), dtype=bool)
        for i in range(data.shape[0]):
            value = data[i, column_id]
            feature[i] = self.is_number(value)

        return feature, 'is_numerical'


    def string_length(self, data, column_id):
        feature = np.zeros((data.shape[0],1))
        for i in range(data.shape[0]):
            value = data[i, column_id]
            try:
                feature[i] = len(str(value.encode('utf-8')))
            except:
                feature[i] = len(str(value))
        return feature, 'string_length'

    def is_alphabetical(self, data, column_id):
        feature = np.zeros((data.shape[0],1), dtype=bool)
        for i in range(data.shape[0]):
            value = data[i, column_id]
            feature[i,0] = re.findall("^[A-Za-z_]+$", value)

        return feature, 'is_alphabetical'

    def extract_number(self, data, column_id):
        feature = lil_matrix((data.shape[0],1))
        for i in range(data.shape[0]):
            value = data[i, column_id]
            try:
                feature[i] = float(value)
            except ValueError:
                pass
        return feature, 'extracted_number'

    def fit(self, data):
        feature_methods_fit = [self.get_number_of_occurrences_fit]

        for c in range(data.shape[1]):
            for feature_method in feature_methods_fit:
                feature_method(data, c)

    def transform(self, data, columns):
        feature_methods_transform = [self.get_number_of_occurrences_transform,
                           self.string_length,
                           self.is_numerical,
                           self.is_alphabetical,
                           self.extract_number]

        feature_list = []
        feature_name_list = []

        for c in range(data.shape[1]):
            for feature_method in feature_methods_transform:
                new_feature, feature_name = feature_method(data, c)
                feature_list.append(new_feature)
                feature_name_list.append(str(columns[c]) + "_" + feature_name)
                print("Finished " + feature_name + " -> " + str(new_feature.shape))

        metadata_features = hstack(feature_list)

        return metadata_features, feature_name_list

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

