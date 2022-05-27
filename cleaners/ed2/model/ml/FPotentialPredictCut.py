import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import operator
import pickle

from ml.active_learning.classifier.XGBoostClassifier import XGBoostClassifier

from ml.datasets.blackOak.BlackOakDataSetUppercase import BlackOakDataSetUppercase
from ml.datasets.flights.FlightHoloClean import FlightHoloClean
from ml.datasets.hospital.HospitalHoloClean import HospitalHoloClean
from ml.datasets.luna.book.Book import Book
from ml.datasets.salary_data.Salary import Salary
from ml.datasets.luna.restaurant.Restaurant import Restaurant

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

def run_cross_validation(train, train_target, folds, scoring='r2'):
    cv_params = {'min_child_weight': [1, 3, 5],
                 'subsample': [0.7, 0.8, 0.9],
                 'learning_rate': [0.01],
                 'max_depth': [3, 5, 7],
                 'n_estimators': [100,1000] #try 100
                 }
    ind_params = {'colsample_bytree': 0.8,
                  'silent': 1,
                  'seed': 0,
                  'objective': 'reg:logistic'} # logistic or linear

    optimized_GBM = GridSearchCV(xgb.XGBRegressor(**ind_params),
                                 cv_params,
                                 scoring=scoring, cv=folds, n_jobs=1, verbose=4)

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


feature_names = ['distinct_values_fraction','labels','certainty','minimum_certainty']

for i in range(100):
    feature_names.append('certainty_histogram' + str(i))

for i in range(7):
    feature_names.append('icross_val' + str(i))

feature_names.append('mean_cross_val')

for i in range(100):
    feature_names.append('change_histogram' + str(i))

#feature_names.append('mean_squared_certainty_change')

feature_names.append('no_change_0')
feature_names.append('no_change_1')
feature_names.append('change_0_to_1')
feature_names.append('change_1_to_0')


print feature_names

size = len(feature_names)
for s in range(size):
    feature_names.append(feature_names[s] + "_old")


which_features_to_use = []
for feature_index in range(len(feature_names)):
    if not 'histogram' in feature_names[feature_index] \
            and not 'mean_squared_certainty_change' in feature_names[feature_index]:
        which_features_to_use.append(feature_index)
print which_features_to_use

feature_names = [i for j, i in enumerate(feature_names) if j in which_features_to_use]


use_absolute_difference = True # False == Squared / True == Absolute

use_change_features = True

enable_plotting = True

classifier_log_paths = {}
#classifier_log_paths[XGBoostClassifier.name] = "/home/felix/SequentialPatternErrorDetection/progress_log_data/log_newer/xgboost"
#classifier_log_paths[LinearSVMClassifier.name] = "/home/felix/SequentialPatternErrorDetection/progress_log_data/log_newer/linearsvm"
#classifier_log_paths[NaiveBayesClassifier.name] = "/home/felix/SequentialPatternErrorDetection/progress_log_data/log_newer/naivebayes"

classifier_log_paths[XGBoostClassifier.name] = "/home/felix/SequentialPatternErrorDetection/progress_log_data/neweat_backup"#"/home/felix/ExampleDrivenErrorDetection/progress_log_data/new_mean_certainty_change_all"#hist_change"



dataset_log_files = {}
dataset_log_files[HospitalHoloClean().name] = "hospital"
dataset_log_files[BlackOakDataSetUppercase().name] = "blackoak"
dataset_log_files[FlightHoloClean().name] = "flight"
dataset_log_files[Book().name] = "book"
dataset_log_files[Salary().name] = "salaries"
dataset_log_files[Restaurant().name] = "restaurant"


classifier_to_use = XGBoostClassifier
model_for_dataset = HospitalHoloClean()

datasets = [HospitalHoloClean(), BlackOakDataSetUppercase(), FlightHoloClean(), Book(), Salary(), Restaurant()]

for i in range(len(datasets)):
    if datasets[i].name == model_for_dataset.name:
        datasets.pop(i)
        break

print "datasets used for training:"
for i in range(len(datasets)):
    print datasets[i]


train_x = {}
train_y = {}
endf = {}


for d in range(len(datasets)):
    train_x[d], train_y[d] = read_csv1(classifier_log_paths[classifier_to_use.name] + "/log_progress_" +  dataset_log_files[datasets[d].name] + ".csv", None)

    n = datasets[d].get_number_dirty_columns()

    # retrieve change
    change = train_x[d][:, train_x[d].shape[1]-2:train_x[d].shape[1]]
    sum_change = np.sum(change, axis=1)
    print sum_change[-n:len(sum_change)]


    #determine convergence point
    endf[d] = np.zeros(n)
    for i in range(n):
        endf[d][i] = train_y[d][len(train_y[d]) - n + i]

    #calculate column potential
    for i in range(len(train_y[d])):
        if use_absolute_difference:
            train_y[d][i] = endf[d][i % n] - train_y[d][i]
        else:
            train_y[d][i] = np.square(endf[d][i % n] - train_y[d][i])

    train_x[d], train_y[d] = add_history(train_x[d],train_y[d], n)

    print "before cut: " + str(train_x[d].shape)

    '''

    #cut zero change data
    sum_change = sum_change[n:len(sum_change)]
    is_converged = np.ones(n, dtype=bool)

    new_list_x = []
    new_list_y = []

    for i in range(train_x[d].shape[0] - 1, -1, -1):
        if sum_change[i] > 0.0:
            is_converged[i % n] = False
        if not is_converged[i % n]:
            new_list_x.append(train_x[d][i])
            new_list_y.append(train_y[d][i])

    train_x[d] = np.matrix(new_list_x)
    train_y[d] = np.array(new_list_y)

    print "after cut: " + str(train_x[d].shape)
    '''




X = []
y = []
for i in range(len(datasets)):
    X.append(train_x[i])
    y.append(train_y[i])

train_x_n = np.vstack(X)
train_y_n = np.concatenate(y)

train_y_n[train_y_n < 0]= 0.0


train_x_n = train_x_n[:, which_features_to_use]

our_params = run_cross_validation(train_x_n, train_y_n, 5, scoring='neg_mean_squared_error')
#our_params = run_cross_validation(train_x_n, train_y_n, 5, scoring='r2') #worse
print our_params






print train_x_n.shape

mat = xgb.DMatrix(train_x_n, train_y_n, feature_names=feature_names)
final = xgb.train(our_params, mat, num_boost_round=3000, verbose_eval=False)




fileObject = open("/tmp/model" + dataset_log_files[model_for_dataset.name] + "_" + classifier_to_use.name + ".p", "wb")
pickle.dump(final, fileObject)

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

y_pred = final.predict(mat)

nr_columns = model_for_dataset.get_number_dirty_columns()
t_x, t_y = read_csv1(
    classifier_log_paths[classifier_to_use.name] +
    "/log_progress_" + dataset_log_files[model_for_dataset.name] + ".csv",
    None)

if not use_change_features:
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

t_x = t_x[:, which_features_to_use]

t_y_pred = predict_tree(final, t_x, feature_names)

print "R2: " + str(r2_score(t_y, t_y_pred))
print "MSE: " + str(mean_squared_error(t_y, t_y_pred))




if enable_plotting:
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

if enable_plotting:
    plot(f_all, f_all_pred)