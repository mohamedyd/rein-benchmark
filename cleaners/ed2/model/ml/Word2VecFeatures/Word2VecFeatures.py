import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from scipy.sparse import hstack

class Word2VecFeatures():
    def __init__(self, vector_size=100, epochs=10):
        self.vector_size = vector_size
        self.epochs = epochs

    def add_word2vec_features(self, dataSet,
                              train_indices, test_indices,
                              all_matrix_train, all_matrix_test,
                              feature_name_list,
                              word2vec_only):

        data_train = dataSet.dirty_pd.values[train_indices, :]
        data_test = dataSet.dirty_pd.values[test_indices, :]

        self.fit(data_train)

        features_train = self.transform(data_train)
        names = self.get_feature_names(dataSet)

        if word2vec_only:
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
        names = []
        for col_i in range(dataSet.shape[1]):
            for vec_i in range(self.vector_size):
                names.append(dataSet.dirty_pd.columns[col_i] + "_word2vec_" + str(vec_i))
        return names


    def fit(self, data):
        # create dictionary: val -> word
        self.column_dictionaries = []
        words = np.zeros((data.shape[0], data.shape[1]), dtype=object)
        for column_i in range(data.shape[1]):
            col_val2word = {}
            for row_i in range(data.shape[0]):
                val = data[row_i, column_i]
                if not val in col_val2word:
                    col_val2word[val] = 'col' + str(column_i) + "_" + str(len(col_val2word))
                words[row_i, column_i] = col_val2word[val]
            self.column_dictionaries.append(col_val2word)

        # train word2vec
        self.model = Word2Vec(words.tolist(), size=self.vector_size, window=words.shape[1] * 2, min_count=1, workers=4, negative=0, hs=1)
        self.model.train(words.tolist(), total_examples=words.shape[0], epochs=self.epochs)

    def transform(self, data):
        words = np.zeros((data.shape[0], data.shape[1]), dtype=object)

        for column_i in range(data.shape[1]):
            for row_i in range(data.shape[0]):
                val = data[row_i, column_i]
                if val in self.column_dictionaries[column_i]:
                    words[row_i, column_i] = self.column_dictionaries[column_i][val]

        final_matrix = np.zeros((words.shape[0], self.vector_size * words.shape[1]))
        for column_i in range(words.shape[1]):
            for row_i in range(words.shape[0]):
                try:
                    final_matrix[row_i, column_i * self.vector_size:(column_i + 1) * self.vector_size] = self.model[words[row_i, column_i]]
                except:
                    pass

        return final_matrix


if __name__ == '__main__':
    from ml.datasets.flights.FlightHoloClean import FlightHoloClean

    data = FlightHoloClean()

    f = Word2VecFeatures()
    f.fit(data.dirty_pd.values)
    print(f.transform(data.dirty_pd.values))