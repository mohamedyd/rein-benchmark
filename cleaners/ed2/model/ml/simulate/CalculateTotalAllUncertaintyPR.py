import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import operator
import pickle
from ml.simulate.MaxUncertaintyAll.maxuncertaintyall import select_by_max_uncertainty_all_metrics
from ml.simulate.MaxUncertaintyAll.maxuncertaintyall import select_by_max_uncertainty_all_prob
from ml.simulate.MaxUncertaintyAll.maxuncertaintyall import get_all_certainty_sum
from ml.simulate.RoundRobin.round import select_by_round_robin_all_measures

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

    mapping = {}
    mapping['f1'] = 0
    mapping['fp'] = 1
    mapping['fn'] = 2
    mapping['tp'] = 3

    x = data[data.columns[0:(data.shape[1]-4)]].values
    y = data[data.columns[(data.shape[1] - 4):data.shape[1]]].values

    f1 = y[:, mapping['f1']]
    fp = y[:, mapping['fp']]
    fn = y[:, mapping['fn']]
    tp = y[:, mapping['tp']]

    #y = fp + fn
    #y = fp

    print x.shape
    print y.shape

    return x, fp, fn, tp

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
                 'n_estimators': [100,1000]
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


use_history = False


feature_names = ['distinct_values_fraction','labels','certainty','certainty_stddev','minimum_certainty']

for i in range(100):
    feature_names.append('certainty_histogram' + str(i))

feature_names.append('predicted_error_fraction')

for i in range(7):
    feature_names.append('icross_val' + str(i))

feature_names.append('mean_cross_val')
feature_names.append('stddev_cross_val')

feature_names.append('training_error_fraction')

for i in range(100):
    feature_names.append('change_histogram' + str(i))

feature_names.append('mean_squared_certainty_change')
feature_names.append('stddev_squared_certainty_change')

for i in range(10):
    feature_names.append('batch_certainty' + str(i))

feature_names.append('no_change_0')
feature_names.append('no_change_1')
feature_names.append('change_0_to_1')
feature_names.append('change_1_to_0')

all_features = len(feature_names)

print feature_names

if use_history:
    size = len(feature_names)
    for s in range(size):
        feature_names.append(feature_names[s] + "_old")


which_features_to_use = []
for feature_index in range(len(feature_names)):
    if True:#not 'histogram' in feature_names[feature_index]:
        which_features_to_use.append(feature_index)
print which_features_to_use

feature_names = [i for j, i in enumerate(feature_names) if j in which_features_to_use]


use_absolute_difference = True # False == Squared / True == Absolute

enable_plotting = True

cutting = True

use_potential = False



classifier_log_paths = {}
#classifier_log_paths[XGBoostClassifier.name] = "/home/felix/SequentialPatternErrorDetection/progress_log_data/log_newer/xgboost"
#classifier_log_paths[LinearSVMClassifier.name] = "/home/felix/SequentialPatternErrorDetection/progress_log_data/log_newer/linearsvm"
#classifier_log_paths[NaiveBayesClassifier.name] = "/home/felix/SequentialPatternErrorDetection/progress_log_data/log_newer/naivebayes"

#classifier_log_paths[XGBoostClassifier.name] = "/home/felix/ExampleDrivenErrorDetection/progress_log_data/unique"


'''
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

N_datasets = 7
'''



dataset = FlightHoloClean()#FlightHoloClean()#BlackOakDataSetUppercase()#HospitalHoloClean() #BlackOakDataSetUppercase()

#future_steps = 60 #BlackOak = 7, Flights = 9



def getConfig(dataset):
    path = None
    future_steps = -1
    if type(dataset) == type(FlightHoloClean()):
        path = '/home/felix/phd/round_robin_part/flights'
        future_steps = 20

    return path, future_steps

mypath, future_steps = getConfig(dataset)


n = dataset.get_number_dirty_columns()

best_sum_total_f = {}
best_col_seq  = {}
recall_list ={}
precision_list ={}



for d in range(10):
    file_path = mypath + "/label_log_progress_" + dataset.name + "_" + str(
        d) + ".csv"
    x, fp, fn, tp = read_csv1(file_path, None)

    certainty_sum = get_all_certainty_sum(x, feature_names)

    #print certainty_sum

    print "train: " + str(x.shape[0])
    print "features: " + str(all_features)
    assert x.shape[1] == all_features



    print "number dirty attributes: " + str(n)


    runs = 71
    tensor_run = np.zeros((n, runs, 3))

    matrix_certainty_sum = np.zeros((n, runs))

    f_p = 0
    f_n = 1
    t_p = 2

    for run in range(runs):
        for col in range(n):
            tensor_run[col, run, f_p] = fp[col + n * run]
            tensor_run[col, run, f_n] = fn[col + n * run]
            tensor_run[col, run, t_p] = tp[col + n * run]
            matrix_certainty_sum[col, run] = certainty_sum[col + n * run]

    '''
    for col in range(n):
        fig = plt.figure()
        ax = plt.subplot(111)

        ax.plot(range(41), matrix_certainty_sum[col, :], label="certainty")

        ax.set_ylabel('certainty')
        ax.set_xlabel('iteration')

        ax.legend(loc=4)

        plt.show()
    '''

    # print tensor_run
    best_sum_total_f[d], precision_list[d], recall_list[d], best_col_seq[d] = select_by_round_robin_all_measures(
        tensor_run, np.ones(n, dtype=int) * -1, [], [], [], [],
        n * 2,
        True)  # Flight = 9, Blackoak 7, Hospital=5

    best_sum_total_f[d], precision_list[d], recall_list[d], best_col_seq[d] = select_by_max_uncertainty_all_metrics(tensor_run, np.ones(n, dtype=int), best_sum_total_f[d],precision_list[d], recall_list[d], best_col_seq[d], matrix_certainty_sum, future_steps, True) #Flight = 9, Blackoak 7, Hospital=5

print best_col_seq

print list(best_sum_total_f.values())

average_best = np.sum(best_sum_total_f.values(), axis=0) / float(len(best_sum_total_f))
average_recall = np.sum(recall_list.values(), axis=0) / float(len(recall_list))
average_precision = np.sum(precision_list.values(), axis=0) / float(len(precision_list))

labels = []

start = 0
for ii in range(n):
    start +=4
    labels.append(start)

while len(labels) < len(average_best):
    labels.append(labels[-1]+10)


fig = plt.figure()
ax = plt.subplot(111)

ax.plot(labels, average_best, label="actual")
#ax.plot(range(len(y)), y_pred, label="predicted")

ax.set_ylabel('total fscore')
ax.set_ylim((0.0, 1.0))
ax.set_xlabel('round')

ax.legend(loc=4)

plt.show()

print "Fscore:"
print list(average_best)
print "Recall:"
print list(average_recall)
print "Precision:"
print list(average_precision)
print "labels"
print labels


latex = '\n\n\n'
latex += "\\addplot+[mark=none] coordinates{"

for c in range(len(average_best)):
    if np.isnan(average_best[c]):
        latex += "(" + str(labels[c]) + "," + str(0.0) + ")"
    else:
        latex += "(" + str(labels[c]) + "," + str(average_best[c]) + ")"
latex += "};\n"

print latex


