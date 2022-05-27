import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import operator
from eli5 import show_weights
from eli5.formatters import format_as_text
from eli5 import explain_weights
import jinja2
import pickle

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
                 'learning_rate': [0.01],
                 'max_depth': [3, 5, 7],
                 'n_estimators': [100,1000] #try 100
                 }
    ind_params = {'colsample_bytree': 0.8,
                  'silent': 1,
                  'seed': 0,
                  'objective': 'reg:linear'}

    optimized_GBM = GridSearchCV(xgb.XGBRegressor(**ind_params),
                                 cv_params,
                                 scoring='r2', cv=folds, n_jobs=1, verbose=4)

    optimized_GBM.fit(train, train_target)

    print "best scores: " + str(optimized_GBM.grid_scores_)

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


use_absolute_difference = True # False == Squared / True == Absolute

#log_directory = "/home/felix/SequentialPatternErrorDetection/progress_log_data/distinct"
log_directory = "/home/felix/SequentialPatternErrorDetection/progress_log_data/metadata_new"

#train_x, train_y = read_csv1(log_directory + "/log_progress_hospital.csv", None)
train_x, train_y = read_csv1(log_directory + "/log_progress_blackoak.csv", None)
train_x1, train_y1 = read_csv1(log_directory + "/log_progress_flight.csv", None)

#train_x2, train_y2 = read_csv1("/home/felix/SequentialPatternErrorDetection/log_progress_test.csv", None)

train_x = train_x[:, 0:train_x.shape[1]-4]
train_x1 = train_x1[:, 0:train_x1.shape[1]-4]




print train_x.shape

#col1 = 17
col1 = 7
col2 = 4

endf1 = np.zeros(col1)
endf2 = np.zeros(col2)

for i in range(col1):
    endf1[i] = train_y[len(train_y) - col1 + i]

for i in range(col2):
    endf2[i] = train_y1[len(train_y1) - col2 + i]

for i in range(len(train_y)):
    if use_absolute_difference:
        train_y[i] = endf1[i % col1] - train_y[i]
    else:
        train_y[i] = np.square(endf1[i % col1] - train_y[i])

for i in range(len(train_y1)):
    if use_absolute_difference:
        train_y1[i] = endf2[i % col2] - train_y1[i]
    else:
        train_y1[i] = np.square(endf2[i % col2] - train_y1[i])


train_x, train_y = add_history(train_x,train_y, col1)
train_x1, train_y1 = add_history(train_x1,train_y1, col2)

train_x_n = np.vstack((train_x, train_x1))
train_y_n = np.concatenate((train_y,train_y1))

print "real shape: " + str(train_x_n.shape)

our_params = run_cross_validation(train_x_n, train_y_n, 5)
#our_params = {'colsample_bytree': 0.8, 'silent': 1, 'learning_rate': 0.01, 'min_child_weight': 1, 'n_estimators': 1000, 'subsample': 0.8, 'seed': 0, 'objective': 'reg:linear', 'max_depth': 7}
#our_params = {'colsample_bytree': 0.8, 'silent': 1, 'learning_rate': 0.01, 'min_child_weight': 1, 'n_estimators': 1000, 'subsample': 0.7, 'seed': 0, 'objective': 'reg:linear', 'max_depth': 7}

#absolute potential
#our_params = {'colsample_bytree': 0.8, 'silent': 1, 'learning_rate': 0.01, 'min_child_weight': 3, 'n_estimators': 1000, 'subsample': 0.9, 'seed': 0, 'objective': 'reg:linear', 'max_depth': 7}

#squared potential
#our_params = {'colsample_bytree': 0.8, 'silent': 1, 'learning_rate': 0.01, 'min_child_weight': 3, 'n_estimators': 1000, 'subsample': 0.8, 'seed': 0, 'objective': 'reg:linear', 'max_depth': 3}

#hospital - no change - squared
#our_params = {'colsample_bytree': 0.8, 'silent': 1, 'learning_rate': 0.01, 'min_child_weight': 1, 'n_estimators': 1000, 'subsample': 0.8, 'seed': 0, 'objective': 'reg:linear', 'max_depth': 7}

#flights - no change
#our_params = {'colsample_bytree': 0.8, 'silent': 1, 'learning_rate': 0.01, 'min_child_weight': 3, 'n_estimators': 1000, 'subsample': 0.9, 'seed': 0, 'objective': 'reg:linear', 'max_depth': 3}

#blackoak - no change
#our_params = {'colsample_bytree': 0.8, 'silent': 1, 'learning_rate': 0.01, 'min_child_weight': 5, 'n_estimators': 1000, 'subsample': 0.7, 'seed': 0, 'objective': 'reg:linear', 'max_depth': 5}


print our_params


feature_names = ['distinct_values_fraction','labels','certainty','minimum_certainty']

for i in range(100):
    feature_names.append('certainty_histogram' + str(i))

for i in range(7):
    feature_names.append('cross_val' + str(i))

feature_names.append('mean_cross_val')

'''
feature_names.append('no_change_0')
feature_names.append('no_change_1')
feature_names.append('change_0_to_1')
feature_names.append('change_1_to_0')
'''

print feature_names

size = len(feature_names)
for s in range(size):
    feature_names.append(feature_names[s] + "_old")



print train_x_n.shape

mat = xgb.DMatrix(train_x_n, train_y_n, feature_names=feature_names)
final = xgb.train(our_params, mat, num_boost_round=3000, verbose_eval=False)


fileObject = open("/tmp/model.p", "wb")
pickle.dump(final, fileObject)

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

nr_columns = 17
#nr_columns = 7
#nr_columns = 4

t_x, t_y = read_csv1(log_directory + "/log_progress_hospital.csv", None)
#t_x, t_y = read_csv1(log_directory + "/log_progress_blackoak.csv", None)
#t_x, t_y = read_csv1(log_directory + "/log_progress_flight.csv", None)

t_x = t_x[:,0:t_x.shape[1]-4]

print t_x.shape

endfnew = np.zeros(nr_columns)

for i in range(nr_columns):
    endfnew[i] = t_y[len(t_y) - nr_columns + i]

for i in range(len(t_y)):
    if use_absolute_difference:
        t_y[i] = endfnew[i % nr_columns] - t_y[i]
    else:
        t_y[i] = np.square(endfnew[i % nr_columns] - t_y[i])

t_x, t_y = add_history(t_x, t_y, nr_columns)

t_y_pred = predict_tree(final, t_x, feature_names)

print r2_score(t_y, t_y_pred)

#np.savetxt('/home/felix/SequentialPatternErrorDetection/progress_log_data/with distinct values/log_progress_flight_prediction.csv', t_y_pred)
#np.savetxt('/home/felix/SequentialPatternErrorDetection/progress_log_data/with distinct values/log_progress-BlackOak_prediction.csv', t_y_pred)
#np.savetxt('/home/felix/SequentialPatternErrorDetection/progress_log_data/with distinct values/log_progress_hospital_prediction.csv', t_y_pred)

plot(t_y, t_y_pred)


f_avg = np.zeros(nr_columns)
f_avg_pred = np.zeros(nr_columns)

f_all = []
f_all_pred = []

for i in range(len(t_y)):
    current_column = i % nr_columns
    f_avg[current_column] = t_y[i]
    f_avg_pred[current_column] = t_y_pred[i]

    f_all.append(np.mean(f_avg))
    f_all_pred.append(np.mean(f_avg_pred))

plot(f_all, f_all_pred)