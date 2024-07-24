from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import random
import numpy as np

class Abalone_loader(object):
    def __init__(self):
        self.name = "Abalone_loader"
        pass
    def load(self):
        column_names = ["sex", "length", "diameter", "height", "whole weight",
                        "shucked weight", "viscera weight", "shell weight", "rings"]
        data = pd.read_csv("./datasets/abalone2.data", names=column_names)
        # print("Number of samples: %d" % len(data))

        # for more complicated cases use sklearn.feature_extraction.DictVectorizer
        for label in "MFI":
            data[label] = data["sex"] == label
        del data["sex"]
        # print(data.head())

        data["M"] = pd.Series(np.where(data.M.values == True, int(1), int(0)), data.index)
        data["F"] = pd.Series(np.where(data.F.values == True, int(1), int(0)), data.index)
        data["I"] = pd.Series(np.where(data.I.values == True, int(1), int(0)), data.index)

        y = data.rings.values

        del data["rings"]  # remove rings from data, so we can convert all the dataframe to a numpy 2D array.
        X = data.values.astype(np.float)

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=4)
        print("the Abalone dataset has been loaded \n")



        return x_train, x_test, y_train, y_test
