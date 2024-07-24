from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import random
import numpy as np
'''
according to SFS this are the index of top 100
(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 96, 128, 129, 130, 131, 132, 133, 134, 171)
'''
class EEG_Loader(object):
    def __init__(self):
        pass

    def load(self):
        data = pd.read_csv("./datasets/EEG_data_Epileptic_Seizure_Recognition.csv")
        y = data.y.values
        data = data.drop(['Unnamed: 0'], axis=1)

        del data["y"]  # remove rings from data, so we can convert all the dataframe to a numpy 2D array.
        data2=data.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
        X = data2.values.astype(np.float)


        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size= 0.086956522 , random_state=4)

        print(x_test.shape)



        return x_train, x_test, y_train, y_test



