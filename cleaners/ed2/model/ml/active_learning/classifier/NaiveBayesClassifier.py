from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
#from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np

class NaiveBayesClassifier(object):
    name = 'NaiveBayes'

    def __init__(self, X_train, X_test):
        self.params = {}
        self.model = {}

        self.error_fraction = {}

        self.name = NaiveBayesClassifier.name

        for data_i in range(len(X_train.data)):
            if not np.isfinite(X_train.data[data_i]):
                X_train.data[data_i] = 0.0


        self.scaler = StandardScaler(with_mean=False, copy=True)

        self.scaler.fit(np.abs(X_train))
        '''

        self.X_train = self.scaler.transform(X_train)

        if X_test != None:
            self.X_test = self.scaler.transform(X_test)
        '''

    def run_cross_validation(self, train, train_target, folds, column_id):
        cv_params = {'alpha': [0.0, 0.1, 0.01, 0.001, 0.0001]}
        ind_params = { }

        optimized_GBM = GridSearchCV(MultinomialNB(**ind_params),
                                     cv_params,
                                     scoring='f1', cv=folds, n_jobs=1, verbose=0)

        print train.shape

        optimized_GBM.fit(self.scaler.transform(np.abs(train)), train_target)

        our_params = ind_params.copy()
        our_params.update(optimized_GBM.best_params_)

        self.params[column_id] = our_params

    def train_predict(self, x, y, column_id):

        param = dict(self.params[column_id])  # or orig.copy()

        self.model[column_id] = MultinomialNB(**param)
        self.scaler = StandardScaler(with_mean=False, copy=True)
        self.scaler.fit(x)
        self.model[column_id].fit(self.scaler.transform(np.abs(x)), y)
        # predict
        probability_prediction_all = self.model[column_id].predict_proba(self.scaler.transform(np.abs(self.X_train)))

        if self.model[column_id].classes_[1] == True:
            probability_prediction = probability_prediction_all[:, 1]
        else:
            probability_prediction = probability_prediction_all[:, 0]
        class_prediction = probability_prediction > 0.5

        return probability_prediction, class_prediction

    def train_predict_all(self, x, y, column_id, x_all):

        param = dict(self.params[column_id])  # or orig.copy()

        self.model[column_id] = MultinomialNB(**param)
        self.scaler = StandardScaler(with_mean=False, copy=True)
        self.scaler.fit(x)

        self.model[column_id].fit(self.scaler.transform(np.abs(x)), y)
        # predict
        probability_prediction_all = self.model[column_id].predict_proba(self.scaler.transform(np.abs(x_all)))

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