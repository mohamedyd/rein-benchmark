import pandas as pd
import numpy as np

data = pd.read_csv("/home/felix/elisa/data.csv", header=0, delimiter='\t')

print data

X = np.matrix(data.values[:,0:9], dtype=float)
y = data.values[:,-1]

print X

'''
from sklearn import tree
from sklearn.model_selection import cross_val_score
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X, y)

#print cross_val_score(clf, X, y, cv=10)

print data.columns[0:9]
print clf.feature_importances_

import matplotlib.pyplot as plt

inds = clf.feature_importances_.argsort()

fig, ax = plt.subplots()
plt.bar(np.arange(9), clf.feature_importances_[inds])
plt.xticks(np.arange(9), data.columns[0:9][inds])
plt.show()

'''

import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import operator

def run_cross_validation(train, train_target, folds, scoring):
    cv_params = {'min_child_weight': [1, 3, 5, 7],
                 'subsample': [0.5, 0.7, 0.8, 0.9],
                 'learning_rate': [0.01],
                 'max_depth': [3, 5, 7],
                 'n_estimators': [10, 100] #try 100
                 }
    ind_params = {'colsample_bytree': 0.8,
                  'silent': 1,
                  'seed': 0,
                  'objective': 'reg:linear'}

    optimized_GBM = GridSearchCV(xgb.XGBRegressor(**ind_params),
                                 cv_params,
                                 scoring=scoring, cv=folds, n_jobs=1, verbose=4)

    optimized_GBM.fit(train, train_target)

    print "best scores: " + str(optimized_GBM.grid_scores_)

    our_params = ind_params.copy()
    our_params.update(optimized_GBM.best_params_)

    return our_params

our_params = run_cross_validation(X, y, 10, scoring='neg_mean_squared_error')

print our_params

feature_names=list(data.columns[0:9])

print feature_names

mat = xgb.DMatrix(X, y, feature_names=feature_names)
final = xgb.train(our_params, mat, num_boost_round=3000, verbose_eval=False)

enable_plotting = True

if enable_plotting:
    try:
        import os
        import webbrowser
        from eli5 import show_weights
        from eli5.formatters import format_as_text
        from eli5 import explain_weights
        import jinja2

        path = '/home/felix/SequentialPatternErrorDetection/html/fpredict/model.html'
        url = 'file://' + path
        html = show_weights(final, feature_names=feature_names, importance_type="gain").data

        with open(path, 'w') as webf:
            webf.write(html)
        webf.close()
        # webbrowser.open(url)
    except jinja2.exceptions.UndefinedError:
        print format_as_text(explain_weights(final, feature_names=feature_names))


importances = final.get_score(importance_type='gain')
print importances

sorted_x = sorted(importances.items(), key=operator.itemgetter(1), reverse=True)
print sorted_x

labels = []
score = []
t = 0
for key, value in sorted_x:
    labels.append(key)
    score.append(value)
    t +=1
    if t == 25:
        break

if enable_plotting:
    ind = np.arange(len(score))
    plt.barh(ind, score, align='center', alpha=0.5)
    plt.yticks(ind, labels)
    plt.show()



