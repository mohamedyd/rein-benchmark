from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class Iris_Loader(object):
    def __init__(self):
        self.name ="Iris_Loader"
        pass

    def load(self):
        # save "bunch" object containing iris dataset and its attributes
        iris = load_iris()


        # print the iris data
        # print(iris.data)

        # print the iris data
        # print(iris.data)


        # print the names of the four features
        # print(iris.feature_names)


        # print integers representing the species of each observation# print
        # print(iris.target)

        # store feature matrix in "X"# store
        X = iris.data

        # store response vector in "y"
        y = iris.target

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=4)
        print("the Iris dataset has been loaded \n")
        return x_train, x_test, y_train, y_test