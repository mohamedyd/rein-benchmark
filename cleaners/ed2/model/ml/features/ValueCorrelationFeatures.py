import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from scipy.sparse import hstack

class ValueCorrelationFeatures():
    def __init__(self):
        pass

    def add_features(self, dataSet,
                              train_indices, test_indices,
                              all_matrix_train, all_matrix_test,
                              feature_name_list,
                              features_only):

        data_train = dataSet.dirty_pd.values[train_indices, :]
        data_test = dataSet.dirty_pd.values[test_indices, :]

        self.fit(data_train)

        features_train = self.transform(data_train)
        names = self.get_feature_names(dataSet)

        if features_only:
            all_features_train_new = features_train
            feature_name_list = names
            all_features_test_new = all_matrix_test

        else:
            all_features_train_new = hstack((all_matrix_train, features_train)).tocsr()
            feature_name_list.extend(names)

            if data_test.shape[0] > 0:
                features_test = self.transform(data_test)
                all_features_test_new = hstack((all_matrix_test, features_test)).tocsr()
            else:
                all_features_test_new = all_matrix_test

        return all_features_train_new, all_features_test_new, feature_name_list


    def get_feature_names(self, dataSet):
        feature_names = []
        for col_a in range(dataSet.shape[1]):
            for col_b in range(dataSet.shape[1]):
                if col_a != col_b:
                    feature_names.append('P(' + dataSet.clean_pd.columns[col_a] + ' | ' + dataSet.clean_pd.columns[col_b] + ')')
        return feature_names


    def fit(self, data):
        self.column_dicts = [{}] * data.shape[1]

        for row_i in range(data.shape[0]):
            for col_a in range(data.shape[1]):
                for col_b in range(data.shape[1]):
                    if col_a != col_b:
                        other_col_value = data[row_i, col_b] + "_col" + str(col_b)
                        own_value = data[row_i, col_a]

                        if not other_col_value in self.column_dicts[col_a]:
                            self.column_dicts[col_a][other_col_value] = {}
                        if not own_value in self.column_dicts[col_a][other_col_value]:
                            self.column_dicts[col_a][other_col_value][own_value] = 1
                        else:
                            self.column_dicts[col_a][other_col_value][own_value] += 1


    def transform(self, data):
        result = np.zeros((data.shape[0], data.shape[1] * (data.shape[1]-1)))
        for row_i in range(data.shape[0]):
            f_counter = 0
            for col_a in range(data.shape[1]):
                for col_b in range(data.shape[1]):
                    if col_a != col_b:
                        other_col_value = data[row_i, col_b] + "_col" + str(col_b)
                        own_value = data[row_i, col_a]
                        result[row_i, f_counter] = self.column_dicts[col_a][other_col_value][own_value] / float(len(self.column_dicts[col_a][other_col_value]))
                        f_counter += 1

        return result


if __name__ == '__main__':
    from ml.datasets.flights.FlightHoloClean import FlightHoloClean

    data = FlightHoloClean()

    f = ValueCorrelationFeatures()
    f.fit(data.dirty_pd.values)
    print(f.transform(data.dirty_pd.values))
    print(f.get_feature_names(data))
    print(len(f.get_feature_names(data)))