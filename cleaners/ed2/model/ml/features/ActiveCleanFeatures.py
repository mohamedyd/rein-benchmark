import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
import unicodedata

class ActiveCleanFeatures():
    def __init__(self):
        pass

    def add_features(self, dataSet,
                              train_indices, test_indices,
                              all_matrix_train, all_matrix_test,
                              feature_name_list,
                              only):

        data_train = dataSet.dirty_pd.values[train_indices, :]
        data_test = dataSet.dirty_pd.values[test_indices, :]

        self.fit(data_train)

        features_train = self.transform(data_train)
        names = self.get_feature_names(dataSet)

        if only:
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


    def get_feature_names(self, dataset=None):
        names = []
        for fname in self.vectorizer.get_feature_names():
                names.append("record" + "_ActiveClean_" + str(fname))
        return names



    def matrix_to_str(self, data_matrix):
        data = []
        for row_i in range(len(data_matrix)):
            row_str = ""
            for col_i in range(data_matrix.shape[1]):
                row_str += data_matrix[row_i, col_i] + ' '
            data.append(row_str)

        try:
            text = [unicodedata.normalize('NFKD', unicode(row_txt, errors='replace')).encode('ascii', 'ignore')
                    for row_txt in data]
        except TypeError:
            text = [unicodedata.normalize('NFKD', row_txt).encode('ascii', 'ignore')
                    for row_txt in data]

        return text

    def fit(self, data_matrix):
        self.vectorizer = TfidfVectorizer(min_df=1, stop_words='english')

        text = self.matrix_to_str(data_matrix)
        self.vectorizer.fit(text)

    def transform(self, data_matrix):
        text = self.matrix_to_str(data_matrix)
        return self.vectorizer.transform(text)



if __name__ == '__main__':
    #from ml.datasets.flights.FlightHoloClean import FlightHoloClean
    #data = FlightHoloClean()
    from ml.datasets.blackOak.BlackOakDataSetUppercase import BlackOakDataSetUppercase
    data = BlackOakDataSetUppercase()

    f = ActiveCleanFeatures()
    f.fit(data.dirty_pd.values)
    print(f.transform(data.dirty_pd.values))

    for f_n in f.get_feature_names():
        print(f_n)

    print(len(f.get_feature_names()))