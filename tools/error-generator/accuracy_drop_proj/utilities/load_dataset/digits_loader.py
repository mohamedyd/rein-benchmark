import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class Digits_Loader(object):
    def __init__(self):
        self.name = "Digits_Loader"
        pass
    def load(self):
        
        
        #prepare X
        data_x = pd.read_csv("./datasets/digits/digits_train.data", header=None, delimiter=" ")
        data_x = data_x.drop(1568,axis=1)
        xx = []

        for row in data_x.values:
            xx.append([row][0])

        X = np.asarray(xx, dtype=np.float32)
        print("Number of samples: %d" % len(X))
        #prepare y

        data_y = pd.read_csv("./datasets/digits/digits_train.solution", header=None, delimiter=" ")

        yy = []

        for rows in data_y.values:
            itemindex = np.where(rows == 1)[0]
            yy.append(itemindex[0])

        y = np.asarray(yy, dtype=np.float32)

        X=X[0:10]
        y=y[0:10]

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)

        print("the Digits dataset has been loaded \n")

        return x_train, x_test, y_train, y_test