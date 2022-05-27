
from rein.auxiliaries.configurations import *

from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier, LinearRegression, BayesianRidge, Ridge
from sklearn.linear_model import RANSACRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from autosklearn.classification import AutoSklearnClassifier
from tpot import TPOTClassifier
from tpot import TPOTRegressor
from autosklearn.regression import AutoSklearnRegressor
#from hpsklearn import HyperoptEstimator
#from hpsklearn import any_classifier
#from hpsklearn import any_preprocessing
#from hyperopt import tpe
from sklearn.metrics import mean_absolute_error
#from hpsklearn import any_regressor
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, AffinityPropagation, AgglomerativeClustering, OPTICS, Birch
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
import optuna
from sklearn.metrics import f1_score

# Define constant to be returned if Optuna cannot be executed
skip_optimization = -1

class Hyperoptimization:
    """Class contains objective functions for each ML model"""

    def __init__(self, data_prepared, model_name):
        """ Constructor defines the necessary variables """
        self.x_train = data_prepared[0]
        self.y_train = data_prepared[1]
        self.x_test = data_prepared[2]
        self.y_test = data_prepared[3]

        self.model_name = model_name

        trial = optuna.create_study(direction="maximize")

    def __call__(self, trial):

        # ================ Regression Algorithms =================

        if self.model_name == 'forest_reg':
            # Define the space of the hyperparameters
            criterion = trial.suggest_categorical("criterion", ["mse", "mae"])
            max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
            n_estimators = trial.suggest_int("n_estimators", 100, 500)
            # Define the ML model
            rf_reg = RandomForestRegressor(criterion=criterion, max_depth=max_depth, n_estimators=n_estimators)
            train_score = cross_val_score(rf_reg, self.x_train, self.y_train, n_jobs=-1, cv=5).mean()

        elif self.model_name == 'tree_reg':
            # Define the space of the hyperparameters
            max_depth = trial.suggest_int("max_depth", 2, 150, log=True)
            # Define the ML model
            tree_reg = DecisionTreeRegressor(max_depth=max_depth)
            train_score = cross_val_score(tree_reg, self.x_train, self.y_train, n_jobs=-1, cv=5).mean()

        elif self.model_name == 'adaboost_reg':
            # Define the space of the hyperparameters
            loss = trial.suggest_categorical("loss", ["linear", "square", "exponential"])
            n_estimators = trial.suggest_int("n_estimators", 50, 200, log=True)
            learning_rate = trial.suggest_float("learning_rate", 0.1, 1)
            # Define the ML model
            ada_reg = AdaBoostRegressor(loss=loss, n_estimators=n_estimators, learning_rate=learning_rate)
            train_score = cross_val_score(ada_reg, self.x_train, self.y_train, n_jobs=-1, cv=5).mean()

        elif self.model_name == 'svm_reg':
            # Define the space of the hyperparameters
            kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", 'sigmoid'])
            C = trial.suggest_float("C", 0.1, 100)
            # Define the ML model
            svm_reg = SVR(kernel=kernel, C=C)
            train_score = cross_val_score(svm_reg, self.x_train, self.y_train, n_jobs=-1, cv=5).mean()

        elif self.model_name == 'mlp_reg':
            # Define the space of the hyperparameters
            activation = trial.suggest_categorical("activation", ["identity", "logistic", "tanh", "relu"])
            learning_rate = trial.suggest_categorical("learning_rate", ["constant", "invscaling", "adaptive"])
            # Define the ML model
            mlp_reg = MLPRegressor(activation=activation, learning_rate=learning_rate, max_iter=1000)
            train_score = cross_val_score(mlp_reg, self.x_train, self.y_train, n_jobs=-1, cv=5).mean()

        elif self.model_name == 'ridge_reg':
            # Define the space of the hyperparameters
            alpha = trial.suggest_float("alpha", 0.0001, 1)
            # Define the ML model
            ridge_reg = Ridge(alpha=alpha)
            train_score = cross_val_score(ridge_reg, self.x_train, self.y_train, n_jobs=-1, cv=5).mean()

        elif self.model_name == 'knn_reg':
            # Define the space of the hyperparameters
            metric = trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"])
            n_neighbors = trial.suggest_int("n_neighbors", 1, 100)
            weights = trial.suggest_categorical("weights", ["uniform", "distance"])
            # Define the ML model
            knn_reg = KNeighborsRegressor(metric=metric, weights=weights, n_neighbors=n_neighbors)
            train_score = cross_val_score(knn_reg, self.x_train, self.y_train, n_jobs=-1, cv=5).mean()

        # ================= Classifiers =========================

        elif self.model_name == 'forest_clf':
            # Define the space of the hyperparameters
            criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
            max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
            n_estimators = trial.suggest_int("n_estimators", 100, 500)
            # Define the ML model
            rf_clf = RandomForestClassifier(criterion=criterion, max_depth=max_depth, n_estimators=n_estimators)
            train_score = cross_val_score(rf_clf, self.x_train, self.y_train, n_jobs=-1, cv=5).mean()

        elif self.model_name == 'logit_clf':
            # Define the space of the hyperparameters
            C = trial.suggest_float("C", 0.1, 100)
            penalty = trial.suggest_categorical("penalty", ['none', 'l2'])
            # Define the ML model
            logit_clf = LogisticRegression(C=C, penalty=penalty, max_iter=5000, multi_class='auto')
            train_score = cross_val_score(logit_clf, self.x_train, self.y_train, n_jobs=-1, cv=5).mean()

        elif self.model_name == 'svc_clf':
            # Define the space of the hyperparameters
            C = trial.suggest_float("C", 0.1, 100)
            kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", 'sigmoid'])
            degree = trial.suggest_int("degree", 3, 6)
            # Define the ML model
            svc_clf = SVC(C=C, kernel=kernel, degree=degree, cache_size=7000)
            train_score = cross_val_score(svc_clf, self.x_train, self.y_train, n_jobs=-1, cv=5).mean()

        elif self.model_name == 'sgd_svc_clf':
            # Define the space of the hyperparameters
            penalty = trial.suggest_categorical("penalty", ['l1', 'l2', 'elasticnet'])
            # Define the ML model
            sgd_svc_clf = SGDClassifier(penalty=penalty, max_iter=1000, tol=1e-3)
            train_score = cross_val_score(sgd_svc_clf, self.x_train, self.y_train, n_jobs=-1, cv=5).mean()

        elif self.model_name == 'knn_clf':
            # Define the space of the hyperparameters
            metric = trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"])
            n_neighbors = trial.suggest_int("n_neighbors", 1, 100)
            weights = trial.suggest_categorical("weights", ["uniform", "distance"])
            # Define the ML model
            knn_clf = KNeighborsClassifier(metric=metric, weights=weights, n_neighbors=n_neighbors)
            train_score = cross_val_score(knn_clf, self.x_train, self.y_train, n_jobs=-1, cv=5).mean()

        elif self.model_name == 'tree_clf':
            # Define the space of the hyperparameters
            criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
            max_features = trial.suggest_categorical("max_features", ["sqrt", "auto", "log2"])
            # Define the ML model
            tree_clf = DecisionTreeClassifier(criterion=criterion, max_features=max_features)
            train_score = cross_val_score(tree_clf, self.x_train, self.y_train, n_jobs=-1, cv=5).mean()

        elif self.model_name == 'adaboost_clf':
            # Define the space of the hyperparameters
            n_estimators = trial.suggest_int("n_estimators", 50, 200, log=True)
            learning_rate = trial.suggest_float("learning_rate", 0.1, 1)
            # Define the ML model
            ada_clf = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
            train_score = cross_val_score(ada_clf, self.x_train, self.y_train, n_jobs=-1, cv=5).mean()

        elif self.model_name == 'xgboost_clf':
            # Define the space of the hyperparameters
            max_depth = trial.suggest_int("max_depth", 1, 100)
            gamma = trial.suggest_float("gamma", 0.1, 5)
            # Define the ML model
            xgboost_clf = XGBClassifier(max_depth=max_depth, gamma=gamma, eval_metric="logloss", use_label_encoder=False)
            train_score = cross_val_score(xgboost_clf, self.x_train, self.y_train, n_jobs=-1, cv=5).mean()

        elif self.model_name == 'ridge_clf':
            # Define the space of the hyperparameters
            alpha = trial.suggest_float("alpha", 0.0001, 1)
            # Define the ML model
            ridge_clf = RidgeClassifier(alpha=alpha)
            train_score = cross_val_score(ridge_clf, self.x_train, self.y_train, n_jobs=-1, cv=5).mean()

        elif self.model_name == 'mlp_clf':
            # Define the space of the hyperparameters
            alpha = trial.suggest_float("alpha", 0.0001, 1)
            activation = trial.suggest_categorical("activation", ["identity", "logistic", "tanh", "relu"])
            learning_rate = trial.suggest_categorical("learning_rate", ["constant", "invscaling", "adaptive"])
            # Define the ML model
            mlp_clf = MLPClassifier(activation=activation, learning_rate=learning_rate, alpha=alpha, max_iter=1000)
            train_score = cross_val_score(mlp_clf, self.x_train, self.y_train, n_jobs=-1, cv=5).mean()

        else:
            logging.info("Optuna skipped: {} does not have hyperparameters".format(self.model_name))
            train_score = skip_optimization

        return train_score


