from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

class RBFSVMClassifier(object):
    def __init__(self, X_train, X_test):
        self.params = {}
        self.model = {}

        self.error_fraction = {}

        self.scaler = StandardScaler(with_mean=False, copy=True)

        self.scaler.fit(X_train)

        self.X_train = self.scaler.transform(X_train)

        if X_test != None:
            self.X_test = self.scaler.transform(X_test)

    def run_cross_validation(self, train, train_target, folds, column_id):
        cv_params = {'C': [10000.0, 1000.0, 100.0, 10.0, 1.0, 0.1, 0.01, 0.001, 0.0001],
                     'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 0.5]}
        ind_params = {
            'probability': True,
            'kernel': 'rbf'
        }

        optimized_GBM = GridSearchCV(SVC(**ind_params),
                                     cv_params,
                                     scoring='f1', cv=folds, n_jobs=1, verbose=0)

        print train.shape

        optimized_GBM.fit(self.scaler.transform(train), train_target)

        our_params = ind_params.copy()
        our_params.update(optimized_GBM.best_params_)

        self.params[column_id] = our_params

    def train_predict(self, x, y, column_id):

        param = dict(self.params[column_id])  # or orig.copy()

        self.model[column_id] = SVC(**param)

        self.model[column_id].fit(self.scaler.transform(x), y)
        # predict
        probability_prediction_all = self.model[column_id].predict_proba(self.X_train)

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
        scores = cross_val_score(SVC(**self.params[column_id]), self.scaler.transform(train), train_target, cv=folds, scoring='f1')
        return scores