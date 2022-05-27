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


feature_names = ['distinct_values_fraction','labels','certainty','certainty_stddev','minimum_certainty']

for i in range(100):
    feature_names.append('certainty_histogram' + str(i))

for i in range(7):
    feature_names.append('icross_val' + str(i))

feature_names.append('mean_cross_val')
feature_names.append('stddev_cross_val')

for i in range(100):
    feature_names.append('change_histogram' + str(i))

feature_names.append('mean_squared_certainty_change')
feature_names.append('stddev_squared_certainty_change')

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
    if not 'histogram' in feature_names[feature_index]:
        which_features_to_use.append(feature_index)
print which_features_to_use

feature_names = [i for j, i in enumerate(feature_names) if j in which_features_to_use]


use_absolute_difference = True # False == Squared / True == Absolute

enable_plotting = True

classifier_log_paths = {}
#classifier_log_paths[XGBoostClassifier.name] = "/home/felix/SequentialPatternErrorDetection/progress_log_data/log_newer/xgboost"
#classifier_log_paths[LinearSVMClassifier.name] = "/home/felix/SequentialPatternErrorDetection/progress_log_data/log_newer/linearsvm"
#classifier_log_paths[NaiveBayesClassifier.name] = "/home/felix/SequentialPatternErrorDetection/progress_log_data/log_newer/naivebayes"

classifier_log_paths[XGBoostClassifier.name] = "/home/felix/ExampleDrivenErrorDetection/progress_log_data/7"



dataset_log_files = {}
dataset_log_files[HospitalHoloClean().name] = "hospital"
dataset_log_files[BlackOakDataSetUppercase().name] = "blackoak"
dataset_log_files[FlightHoloClean().name] = "flight"
dataset_log_files[Book().name] = "book"
dataset_log_files[Salary().name] = "salaries"
dataset_log_files[Restaurant().name] = "restaurant"


classifier_to_use = XGBoostClassifier
model_for_dataset = BlackOakDataSetUppercase()

datasets = [HospitalHoloClean(), BlackOakDataSetUppercase(), FlightHoloClean(), Book(), Salary(), Restaurant()]

for i in range(len(datasets)):
    if datasets[i].name == model_for_dataset.name:
        datasets.pop(i)
        break

print "datasets used for training:"
for i in range(len(datasets)):
    print datasets[i]

N_datasets = 7


X = []
y = []
for d in range(len(datasets)):
    for ndata in [1]:
        file_path = "/home/felix/ExampleDrivenErrorDetection/progress_log_data/7/log_progress_"+ datasets[d].name +"_" + str(ndata)  +".csv"
        train_x, train_y = read_csv1(file_path, None)

        n = datasets[d].get_number_dirty_columns()

        #determine convergence point
        endf = np.zeros(n)
        for i in range(n):
            endf[i] = train_y[len(train_y) - n + i]

        #calculate column potential
        for i in range(len(train_y)):
            if use_absolute_difference:
                train_y[i] = endf[i % n] - train_y[i]
            else:
                train_y[i] = np.square(endf[i % n] - train_y[i])

        train_x, train_y = add_history(train_x,train_y, n)

        print "size: " + str(train_x.shape)


        X.append(np.copy(train_x))
        y.append(np.copy(train_y))

train_x_n = np.vstack(X)
train_y_n = np.concatenate(y)

train_y_n[train_y_n < 0]= 0.0


train_x_n = train_x_n[:, which_features_to_use]


print "size: training : " + str(train_x_n.shape)

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


N_datasets_test = 7
X_test = []
y_test = []
pred_test = []

for ndata in range(N_datasets_test):
    file_path_test = "/home/felix/ExampleDrivenErrorDetection/progress_log_data/7/log_progress_" + model_for_dataset.name + "_" + str(ndata) + ".csv"

    t_x, t_y = read_csv1(file_path_test, None)

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

    X_test.append(t_x)
    y_test.append(t_y)
    pred_test.append(t_y_pred)


train_x_test = np.vstack(X_test)
train_y_test = np.concatenate(y_test)
pred_y_test = np.concatenate(pred_test)

print len(pred_y_test)
print len(train_y_test)

print "R2: " + str(r2_score(train_y_test, pred_y_test))
print "MSE: " + str(mean_squared_error(train_y_test, pred_y_test))




if enable_plotting:
    plot(train_y_test, pred_y_test)


f_avg = np.zeros(nr_columns)
f_avg_pred = np.zeros(nr_columns)

f_all = []
f_all_pred = []

for i in range(len(train_y_test)):
    current_column = i % nr_columns
    f_avg[current_column] = train_y_test[i]
    f_avg_pred[current_column] = pred_y_test[i]

    f_all.append(np.mean(f_avg))
    f_all_pred.append(np.mean(f_avg_pred))

if enable_plotting:
    plot(f_all, f_all_pred)