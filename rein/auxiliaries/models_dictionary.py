####################################################
# Benchmark: Dictionary of ML methods
# Authors: Mohamed Abdelaal
# Date: March 2021
# Software AG
# All Rights Reserved
####################################################

from rein.auxiliaries.hyperopt_functions import *

models_dictionary = {

    # ========================= Common Classification Models ===================
    "classification": {
        "forest_clf": {
            "name": "forest_clf",
            "fn": RandomForestClassifier,
            'hyperopt': Hyperoptimization,
            "fixed_params": {},
            "hyperparams": {
                # "max_depth": [1, 20, 40, 80, 100, 150, 200],
                "n_estimators": [10, 100, 1000, 1500, 2000]}
        },
        "logit_clf": {
            "name": "logit_clf",
            "fn": LogisticRegression,
            'hyperopt': Hyperoptimization,
            "fixed_params": {"max_iter": 5000, "multi_class": 'auto'},
            "hyperparams": [{"C": [100, 10, 1.0, 0.1, 0.01], "penalty": ["none", "l2"]},
                            {"solver": ["liblinear"], "penalty": ["l1"]}]
        },
        "svc_clf": {
            "name": "svc_clf",
            "fn": SVC,
            'hyperopt': Hyperoptimization,
            "fixed_params": {"cache_size": 7000},
            "hyperparams": {
                "C": [100, 10, 1.0, 0.1, 0.001],
                "kernel": ["linear",  "poly", "rbf", "sigmoid"],
                "degree": [3, 4, 5, 6]}
        },
        "sgd_svc_clf": {
            "name": "sgd_svc_clf",
            "fn": SGDClassifier,
            "fixed_params": {"max_iter": 1000, "tol": 1e-3},
            "hyperparams": {"penalty": ["l2", "l1", "elasticnet"]}
        },
        "knn_clf": {
            "name": "knn_clf",
            "fn": KNeighborsClassifier,
            "fixed_params": {},
            "hyperparams": {
                "n_neighbors": [1, 5, 10, 20, 40, 60, 80, 95],
                "metric": ["euclidean", "manhattan", "minkowski"],
                "weights": ["uniform", "distance"]}
        },
        "tree_clf": {
            "name": "tree_clf",
            "fn": DecisionTreeClassifier,
            "fixed_params": {},
            "hyperparams": {"criterion": ["gini", "entropy"],
                            "max_features": ["sqrt", "auto", "log2"]}
        },
        "adaboost_clf": {
            "name": "adaboost_clf",
            "fn": AdaBoostClassifier,
            "fixed_params": {"n_estimators": 200},
            "hyperparams": {"learning_rate": [0.01, 0.1, 0.2, 0.4, 0.8, 1]}
        },
        "gaussian_nb_clf": {
            "name": "gaussian_nb_clf",
            "fn": GaussianNB,
            "fixed_params": {},
            "hyperparams": {}
        },
        "xgboost_clf": {
            "name": "xgboost_clf",
            "fn": XGBClassifier,
            "fixed_params": {"eval_metric": "logloss"},
            "hyperparams": {
                "max_depth": [1, 5, 10, 20, 50, 100],
                "gamma": [0.5, 1, 1.5, 2, 5]}
        },
        "ridge_clf": {
            "name": "ridge_clf",
            "fn": RidgeClassifier,
            "fixed_params": {},
            "hyperparams": {"alpha": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
        },
        "mlp_clf": {
            "name": "mlp_clf",
            "fn": MLPClassifier,
            "fixed_params": {},
            "hyperparams": {
                "alpha": [0.01, 0.1, 0.5, 1],
                "activation": ["identity", "logistic", "tanh", "relu"],
                "learning_rate": ["constant", "invscaling", "adaptive"]}
        },
        'mn_clf': {
            'name': 'mn_clf',
            'fn': MultinomialNB,
            'fixed_params': {},
            'hyperparams': {}
        },
        'autosklearn_clf':{
            'name': 'autosklearn_clf',
            'fn': AutoSklearnClassifier,
            'fixed_params': {'time_left_for_this_task':180}
        },
        'tpot_clf': {
            'name': 'tpot_clf',
            'fn': TPOTClassifier,
            'fixed_params': {'generations':5, 'cv':5, 'population_size':50, 'scoring': 'accuracy', 'n_jobs':-1},
            'note': 'A TPOT.fit call may fail when there are outlier minority classes. '
                    'Because of TPOT internals, the small minority classes may cause an error when optimizing towards log loss'
        },
        # 'hyperopt_clf': {
        #     'name': 'hyperopt_clf',
        #     'fn': HyperoptEstimator,
        #     'fixed_params': {'classifier': any_classifier('clf'), 'algo':tpe.suggest, 'trial_timeout':300, 'max_evals':50}
        # }
    },

    # ====================== Common Regression Models ==========================
    "regression": {
        "lin_reg": {
            "name": "lin_reg",
            "fn": LinearRegression,
            "fixed_params": {},
            "hyperparams": {}
        },
        "tree_reg": {
            "name": "tree_reg",
            "fn": DecisionTreeRegressor,
            "fixed_params": {},
            "hyperparams": {"max_depth": [1, 50, 100, 200]}
        },
        "forest_reg": {
            "name": "forest_reg",
            "fn": RandomForestRegressor,
            "fixed_params": {},
            'hyperopt': Hyperoptimization,
            "hyperparams": {
                "max_depth": [1, 50, 100, 200],
                "n_estimators": [3, 10, 30],
                "max_features": [2, 4, 6, 8]}
        },
        "adaboost_reg": {
            "name": "adaboost_reg",
            "fn": AdaBoostRegressor,
            "fixed_params": {},
            "hyperparams": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.05, 0.1, 0.3, 1],
                "loss": ["linear", "square", "exponential"]}
        },
        "svm_reg": {
            "name": "svm_reg",
            "fn": SVR,
            "fixed_params": {},
            "hyperparams": {
                "kernel": ["linear", "poly", "rbf", "sigmoid"],
                "gamma": ["auto", "scale"],
                "C": [100, 10, 1.0, 0.1, 0.01]}
        },
        "bayes_ridge_reg": {
            "name": "bayes_ridge_reg",
            "fn": BayesianRidge,
            "fixed_params": {},
            "hyperparams": {}
       },
        "xgboost_reg": {
            "name": "xgboost_reg",
            "fn": XGBRegressor,
            "fixed_params": {},
            "hyperparams": {}
        },
        "mlp_reg": {
            "name": "mlp_reg",
            "fn": MLPRegressor,
            "fixed_params": {"max_iter": 1000},
            "hyperparams": {
                "activation": ["identity", "logistic", "tanh", "relu"],
                "learning_rate": ["constant", "invscaling", "adaptive"]}
        },
        "ridge_reg": {
            "name": "ridge_reg",
            "fn": Ridge,
            "fixed_params": {},
            "hyperparams": {"alpha": [1, 0.1, 0.01, 0.001, 0.0001]}
        },
        "knn_reg": {
            "name": "knn_reg",
            "fn": KNeighborsRegressor,
            "fixed_params": {},
            "hyperparams": {
                "n_neighbors": [1, 5, 10, 20, 40, 60, 80, 95],
                "metric": ["euclidean", "manhattan", "minkowski"],
                "weights": ["uniform", "distance"]}
        },
        "ransac_reg": {
            "name": "ransac_reg",
            "fn": RANSACRegressor,
            "fixed_params": {'min_samples': 10},
            "hyperparams": {}
        },
        'autosklearn_reg':{
            'name': 'autosklearn_reg',
            'fn': AutoSklearnRegressor,
            'fixed_params': {'time_left_for_this_task':180}
        },
        'tpot_reg': {
            'name': 'tpot_reg',
            'fn': TPOTRegressor,
            'fixed_params': {'generations': 5, 'cv': 5, 'population_size':50, 'n_jobs': -1},
            'note': 'A TPOT.fit call may fail when there are outlier minority classes. '
                    'Because of TPOT internals, the small minority classes may cause an error when optimizing towards log loss'
        },
        # 'hyperopt_reg': {
        #     'name': 'hyperopt_reg',
        #     'fn': HyperoptEstimator,
        #     'fixed_params': {'regressor': any_regressor('reg'), 'algo': tpe.suggest, 'trial_timeout': 300,
        #                      'max_evals': 50, 'loss_fn': mean_absolute_error}
        # }
    },

    # =========================== Common Clustering Methods =====================
    "clustering": {
        "gm_cls": {
            "name": "gm_cls",
            "fn": GaussianMixture,
            "fixed_params": {'n_components': 2},
            "hyperparams": {
                "covariance_type": ["full", "tied", "diag", "spherical"],
                "init_params": ["kmeans", "random"]}
        },
        "kmeans_cls": {
            "name": "kmeans_cls",
            "fn": KMeans,
            "fixed_params": {},
            "hyperparams": {
                "n_clusters": [2, 4, 8, 16, 32, 64],
                "init": ["k-means++", "random"]}
        },
        "affinity_cls": {
            "name": "affinity_cls",
            "fn": AffinityPropagation,
            "fixed_params": {"max_iter": 1000,
                             "random_state": None,
                             "verbose": True,
                             'damping': 0.9},
            "hyperparams": {}
        },
        "hierarchical_cls": {
            "name": "hierarchical_cls",
            "fn": AgglomerativeClustering,
            "fixed_params": {},
            "hyperparams": {
                "n_clusters": [2, 4, 8, 16, 32, 64],
                "affinity": ["euclidean", "l1", "l2", "manhattan", "cosine"],
                "linkage": ["ward", "complete", "average", "single"]}
        },
        "optics_cls": {
            "name": "optics_cls",
            "fn": OPTICS,
            "fixed_params": {},
            "hyperparams": {
                "min_samples": [3, 5, 10, 20]}
        },
        "birch_cls": {
            "name": "birch_cls",
            "fn": Birch,
            "fixed_params": {},
            "hyperparams": {
                "n_clusters": [3, 5, 8, 12, 16, 20],
                "threshold": [0.01, 0.1, 0.2, 0.4, 0.8, 1]}
        }
    }
}

models = ["classification", "regression", "clustering"]




if __name__ == "__main__":
    print("===== Testing Models Dictionary ========")
    for model, model_details in tqdm(models_dictionary["clustering"].items()):
        print(model, ": ", model_details["fn"].__name__)
