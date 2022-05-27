import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import operator
from eli5 import show_weights
from eli5.formatters import format_as_text
from eli5 import explain_weights
import jinja2

def plot(y, y_pred):
    fig = plt.figure()
    ax = plt.subplot(111)

    ax.plot(range(len(y)), y, label="actual")
    ax.plot(range(len(y)), y_pred, label="predicted")

    ax.set_ylabel('fscore')
    ax.set_xlabel('round')

    ax.legend(loc=4)

    plt.show()

def read_csv1(path, header):
    data = pd.read_csv(path, header=header)

    print data.shape

    x = data[data.columns[0:(data.shape[1]-1)]].values
    y = data[data.columns[data.shape[1] - 1]].values

    print x.shape
    print y.shape

    return x,y

def predict(clf, x):
    predicted = clf.predict(x)
    predicted[predicted > 1.0] = 1.0
    return predicted

def predict_tree(clf, test_x, feature_names):
    mat = xgb.DMatrix(test_x, feature_names=feature_names)
    predicted = clf.predict(mat)
    #predicted[predicted > 1.0] = 1.0
    return predicted

def run_cross_validation(train, train_target, folds):
    cv_params = {'min_child_weight': [1, 3, 5],
                 'subsample': [0.7, 0.8, 0.9],
                 'max_depth': [3, 5, 7],
                 'n_estimators': [100, 1000]}
    ind_params = {#'min_child_weight': 1, # we could optimize this: 'min_child_weight': [1, 3, 5]
                  'learning_rate': 0.1, # we could optimize this: 'learning_rate': [0.1, 0.01]
                  #'n_estimators': 1000, # we choose default 100
                  'colsample_bytree': 0.8,
                  'silent': 1,
                  'seed': 0,
                  'objective': 'binary:logistic'}

    optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params),
                                 cv_params,
                                 scoring='f1', cv=folds, n_jobs=1, verbose=0)

    print train.shape

    optimized_GBM.fit(train, train_target)

    #print "best scores: " + str(optimized_GBM.grid_scores_)

    our_params = ind_params.copy()
    our_params.update(optimized_GBM.best_params_)

    return our_params



def add_history(x, y, nr_columns):
    x_with_history = np.hstack((x[nr_columns:len(x),:], x[0:len(x)-nr_columns,:]))
    y_with_history = y[nr_columns:len(x)]

    return x_with_history, y_with_history
'''
def add_history(x, y, nr_columns):
    x_with_history = x[nr_columns:len(x),:]
    y_with_history = y[nr_columns:len(x)]

    return x_with_history, y_with_history
'''



train_x, train_y = read_csv1("/home/felix/SequentialPatternErrorDetection/progress_log_data/distinct/log_progress_hospital.csv", None)
train_x1, train_y1 = read_csv1("/home/felix/SequentialPatternErrorDetection/progress_log_data/distinct/log_progress_blackoak.csv", None)
#train_x1, train_y1 = read_csv1("/home/felix/SequentialPatternErrorDetection/progress_log_data/with switches/log_progress_flight.csv", None)

#train_x2, train_y2 = read_csv1("/home/felix/SequentialPatternErrorDetection/log_progress_test.csv", None)


train_x, train_y = add_history(train_x,train_y, 17)
train_x1, train_y1 = add_history(train_x1,train_y1, 10)
#train_x1, train_y1 = add_history(train_x1,train_y1, 4)


#train_x2, train_y2 = add_history(train_x2,train_y2, 17)

train_x_n = np.vstack((train_x, train_x1))
train_y_n = np.concatenate((train_y,train_y1))

train_y_n_bool = train_y_n > 0.99

print train_x_n.shape

#our_params = run_cross_validation(train_x_n, train_y_n_bool, 5)
our_params = {'colsample_bytree': 0.8, 'silent': 1, 'learning_rate': 0.1, 'min_child_weight': 1, 'n_estimators': 100, 'subsample': 0.8, 'seed': 0, 'objective': 'binary:logistic', 'max_depth': 3}


print our_params


feature_names = ['distinct_values_fraction','labels','certainty','minimum_certainty']

for i in range(100):
    feature_names.append('certainty_histogram' + str(i))

for i in range(7):
    feature_names.append('cross_val' + str(i))

feature_names.append('mean_cross_val')

feature_names.append('no_change_0')
feature_names.append('no_change_1')
feature_names.append('change_0_to_1')
feature_names.append('change_1_to_0')

print feature_names

size = len(feature_names)
for s in range(size):
    feature_names.append(feature_names[s] + "_old")



print train_x_n.shape

mat = xgb.DMatrix(train_x_n, train_y_n_bool, feature_names=feature_names)
final = xgb.train(our_params, mat, num_boost_round=3000, verbose_eval=False)

try:
    import os
    import webbrowser

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

ind = np.arange(len(score))
plt.barh(ind, score, align='center', alpha=0.5)
plt.yticks(ind, labels)
plt.show()

y_pred = final.predict(mat)

#nr_columns = 17
#nr_columns = 10
nr_columns = 4

#t_x, t_y = read_csv1("/home/felix/SequentialPatternErrorDetection/progress_log_data/with switches/log_progress_hospital.csv", None)
#t_x, t_y = read_csv1("/home/felix/SequentialPatternErrorDetection/progress_log_data/with switches/log_progress_blackoak.csv", None)
t_x, t_y = read_csv1("/home/felix/SequentialPatternErrorDetection/progress_log_data/distinct/log_progress_flight.csv", None)

t_x, t_y = add_history(t_x, t_y, nr_columns)

t_y_bool = t_y > 0.99

test_mat = xgb.DMatrix(t_x, feature_names=feature_names)
t_y_pred_b = final.predict(test_mat)
res_new = (t_y_pred_b > 0.5)

print t_y_pred_b

print f1_score(t_y_bool, res_new)
print precision_score(t_y_bool, res_new)
print recall_score(t_y_bool, res_new)

#np.savetxt('/home/felix/SequentialPatternErrorDetection/progress_log_data/with distinct values/log_progress_flight_prediction.csv', t_y_pred)
#np.savetxt('/home/felix/SequentialPatternErrorDetection/progress_log_data/with distinct values/log_progress-BlackOak_prediction.csv', t_y_pred)
#np.savetxt('/home/felix/SequentialPatternErrorDetection/progress_log_data/with distinct values/log_progress_hospital_prediction.csv', t_y_pred)

plot(t_y_bool, res_new)

print t_y_bool
print res_new