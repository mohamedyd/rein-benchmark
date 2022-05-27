from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import cross_val_score
import numpy as np
import graphviz

class MyDecisionTreeClassifier(object):
    name = 'DecisionTree'

    def __init__(self, X_train, X_test):
        self.params = {}
        self.model = {}

        self.error_fraction = {}

        self.name = MyDecisionTreeClassifier.name


    def run_cross_validation(self, train, train_target, folds, column_id):
        cv_params = {'max_depth': range(2,4), 'criterion':['gini','entropy'], 'min_samples_split': range(2, 403, 10)}
        ind_params = {'random_state': 0 }

        optimized_GBM = GridSearchCV(DecisionTreeClassifier(**ind_params), cv_params, scoring='f1', cv=folds, n_jobs=4, verbose=0)

        print train.shape

        optimized_GBM.fit(train, train_target)

        our_params = ind_params.copy()
        our_params.update(optimized_GBM.best_params_)

        self.params[column_id] = our_params

    def train_predict(self, x, y, column_id):

        param = dict(self.params[column_id])  # or orig.copy()

        self.model[column_id] = DecisionTreeClassifier(**param)
        self.model[column_id].fit(x, y)


        # predict
        probability_prediction_all = self.model[column_id].predict_proba(self.X_train)

        if self.model[column_id].classes_[1] == True:
            probability_prediction = probability_prediction_all[:, 1]
        else:
            probability_prediction = probability_prediction_all[:, 0]
        class_prediction = probability_prediction > 0.5

        return probability_prediction, class_prediction

    def train_predict_all(self, x, y, column_id, x_all, feature_names=None, column_names=None):

        param = dict(self.params[column_id])  # or orig.copy()

        self.model[column_id] = DecisionTreeClassifier(**param)
        self.model[column_id].fit(x, y)

        if feature_names != None:


            dot_data = tree.export_graphviz(self.model[column_id], out_file=None,
                                            feature_names=feature_names,
                                            class_names=['clean', 'dirty'],
                                            filled=True, rounded=True,
                                            special_characters=True)
            graph = graphviz.Source(dot_data)
            graph.render('out/' + str(column_id) + "_" + column_names[column_id])


            # predict
        probability_prediction_all = self.model[column_id].predict_proba(x_all)

        if self.model[column_id].classes_[1] == True:
            probability_prediction = probability_prediction_all[:, 1]
        else:
            probability_prediction = probability_prediction_all[:, 0]
        class_prediction = probability_prediction > 0.5

        return probability_prediction, class_prediction

    def predict(self, column_id):
        probability_prediction_all = self.model[column_id].predict_proba(self.X_test)

        if self.model[column_id].classes_[1] == True:
            probability_prediction = probability_prediction_all[:, 1]
        else:
            probability_prediction = probability_prediction_all[:, 0]
        class_prediction = probability_prediction > 0.5

        return probability_prediction, class_prediction

    def run_cross_validation_eval(self, train, train_target, folds, column_id):
        dummy = np.zeros(folds)
        return dummy