import numpy as np
import xgboost as xgb
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.multioutput import MultiOutputRegressor

from ml.features.CompressedDeepFeatures import read_compressed_deep_features


def run_cross_validation(train, train_target, folds):
    cv_params = {'estimator__min_child_weight': [1, 3, 5],
                 'estimator__subsample': [0.7, 0.8, 0.9],
                 'estimator__learning_rate': [0.1, 0.01],
                 'estimator__max_depth': [3, 5, 7],
                 'estimator__n_estimators': [1000] #try 100
                 }
    ind_params = {'colsample_bytree': 0.8,
                  'silent': 1,
                  'seed': 0,
                  'objective': 'reg:linear'}

    pipeline = MultiOutputRegressor(xgb.XGBRegressor(**ind_params))

    print sorted(pipeline.get_params().keys())

    optimized_GBM = GridSearchCV(pipeline,
                                 cv_params,
                                 scoring='r2', cv=folds, n_jobs=1, verbose=4)

    optimized_GBM.fit(train, train_target)

    print "best scores: " + str(optimized_GBM.grid_scores_)

    our_params = ind_params.copy()
    our_params.update(optimized_GBM.best_params_)

    return our_params

def run_cross_validation_linear(train, train_target, folds):
    scores = cross_val_score(linear_model.LinearRegression(), train, train_target, cv=folds, n_jobs=-1)
    return scores

def run_cross_validation_eval(train, train_target, folds, our_params):
    scores = cross_val_score(MultiOutputRegressor(xgb.XGBRegressor(**our_params)), train, train_target, cv = folds, scoring = 'f1')
    return scores

data = read_compressed_deep_features("/home/felix/SequentialPatternErrorDetection/deepfeatures/BlackOak/last_state/")

neurons = 128

for column_id in range(data.shape[1] / neurons):

    x_id = []
    y_id = []

    for i in range(data.shape[1]):
        if i >= column_id * neurons and i < (column_id+1) * neurons:
            x_id.append(i)
        else:
            y_id.append(i)



    x = data[:, x_id]
    y = data[:, y_id]

    print x.shape
    print y.shape

    print "column: " + str(column_id)

    #scores = run_cross_validation_linear(x, y, 3)

    clf = linear_model.LinearRegression(n_jobs=-1)
    clf.fit(x,y)
    y_pred = clf.predict(x)

    print np.mean(y_pred-y)
    #print euclidean_distances(y, y_pred)


#our_params = run_cross_validation(x, y, 10)
#our_params = {'colsample_bytree': 0.8, 'silent': 1, 'learning_rate': 0.1, 'min_child_weight': 1, 'n_estimators': 1000, 'subsample': 0.8, 'seed': 0, 'objective': 'reg:linear', 'max_depth': 5}
#print run_cross_validation_eval(x, y, 2, our_params)

