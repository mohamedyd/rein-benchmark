import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import operator
from sklearn.multioutput import MultiOutputRegressor

def plot(y, y_pred):
    fig = plt.figure()
    ax = plt.subplot(111)

    ax.plot(range(len(y)), y, label="actual")
    ax.plot(range(len(y)), y_pred, label="predicted")

    ax.set_ylabel('fscore')
    ax.set_xlabel('round')

    ax.legend(loc=4)

    plt.show()


def run_cross_validation_multi(train, train_target, folds):
    cv_params = {'min_child_weight': [1, 3, 5],
                 'subsample': [0.7, 0.8, 0.9],
                 'learning_rate': [0.1, 0.01],
                 'max_depth': [3, 5, 7],
                 'n_estimators': [1000] #try 100
                 }
    ind_params = {'colsample_bytree': 0.8,
                  'silent': 1,
                  'seed': 0,
                  'objective': 'reg:linear'}

    grid_search = GridSearchCV(xgb.XGBRegressor(**ind_params), cv_params, verbose=10)
    multi_grid_search = MultiOutputRegressor(grid_search)
    multi_grid_search.fit(train, train_target)

    params = [estimator.best_params_ for estimator in multi_grid_search.estimators_]

    print params

    #our_params = ind_params.copy()
    #our_params.update(optimized_GBM.best_params_)

    #return our_params

def read_csv1(path, header):
    data = pd.read_csv(path, header=header)

    print data.shape

    x = data[data.columns[0:(data.shape[1]-3)]].values
    y = data[data.columns[(data.shape[1]-3):(data.shape[1])]].values

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
    predicted[predicted > 1.0] = 1.0
    return predicted


def add_history(x, y, nr_columns):
    x_with_history = np.hstack((x[nr_columns:len(x),:], x[0:len(x)-nr_columns,:]))
    y_with_history = y[nr_columns:len(x),:]

    return x_with_history, y_with_history

def add_history(x, y, nr_columns):
    x_with_history = np.hstack((x[nr_columns:len(x),:], x[0:len(x)-nr_columns,:]))
    y_with_history = y[nr_columns:len(x),:]

    return x_with_history, y_with_history
'''
def add_history(x, y, nr_columns):
    x_with_history = x[nr_columns:len(x),:]
    y_with_history = y[nr_columns:len(x)]

    return x_with_history, y_with_history
'''


train_x, train_y = read_csv1("/home/felix/SequentialPatternErrorDetection/progress_log_data/confusion_matrix/log_progress_hospital.csv", None)
train_x1, train_y1 = read_csv1("/home/felix/SequentialPatternErrorDetection/progress_log_data/confusion_matrix/log_progress_blackoak.csv", None)
#train_x1, train_y1 = read_csv1("/home/felix/SequentialPatternErrorDetection/progress_log_data/confusion_matrix/log_progress_flight.csv", None)


train_x, train_y = add_history(train_x,train_y, 17)
train_x1, train_y1 = add_history(train_x1,train_y1, 10)

train_x_n = np.vstack((train_x, train_x1))
train_y_n = np.concatenate((train_y,train_y1))


print train_x_n.shape

#our_params = run_cross_validation_multi(train_x_n, train_y_n, 5)
our_params = [{'n_estimators': 1000, 'subsample': 0.8, 'learning_rate': 0.1, 'max_depth': 7, 'min_child_weight': 1},
{'n_estimators': 1000, 'subsample': 0.9, 'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 5},
{'n_estimators': 1000, 'subsample': 0.8, 'learning_rate': 0.01, 'max_depth': 3, 'min_child_weight': 1}]

print our_params


feature_names = ['distinct_values_fraction','labels','certainty','minimum_certainty']

for i in range(100):
    feature_names.append('certainty_histogram' + str(i))

for i in range(7*3):
    feature_names.append('cross_val' + str(i))

for i in range(3):
    feature_names.append('mean_cross_val' + str(i))

size = len(feature_names)
for s in range(size):
    feature_names.append(feature_names[s] + "_old")



print train_x_n.shape

t_x, t_y = read_csv1(
    "/home/felix/SequentialPatternErrorDetection/progress_log_data/confusion_matrix/log_progress_flight.csv", None)
t_x, t_y = add_history(t_x, t_y, 4)

t_y_pred = []

cu = 2
for cu in range(3):
    mat = xgb.DMatrix(train_x_n, train_y_n[:,cu], feature_names=feature_names)
    final = xgb.train(our_params[cu], mat, num_boost_round=3000, verbose_eval=False)

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
        if t == 20:
            break

    ind = np.arange(len(score))
    plt.barh(ind, score, align='center', alpha=0.5)
    plt.yticks(ind, labels)
    plt.show()

    #t_x, t_y = read_csv1("/home/felix/SequentialPatternErrorDetection/progress_log_data/with distinct values/log_progress-BlackOak.csv", None)
    #t_x, t_y = read_csv1("/home/felix/SequentialPatternErrorDetection/progress_log_data/with crossval/log_progress_hospital.csv", None)

    t_y_pred.append(predict_tree(final, t_x, feature_names))

    #print r2_score(t_y[:,cu], t_y_pred)

    #np.savetxt('/home/felix/SequentialPatternErrorDetection/progress_log_data/with distinct values/log_progress_flight_prediction.csv', t_y_pred)
    #np.savetxt('/home/felix/SequentialPatternErrorDetection/progress_log_data/with distinct values/log_progress-BlackOak_prediction.csv', t_y_pred)
    #np.savetxt('/home/felix/SequentialPatternErrorDetection/progress_log_data/with distinct values/log_progress_hospital_prediction.csv', t_y_pred)

    #plot(t_y[:,cu], t_y_pred)


def precision(tp, fp):
    return tp / (tp + fp)

def recall(tp, fn):
    return tp / (tp + fn)

def fscore(tp, fp, fn):
    prec = precision(tp, fp)
    rec = recall(tp, fn)

    f = (2 * prec * rec) / (prec + rec)

    return f

fn =t_y[:,0]
tp =t_y[:,1]
fp =t_y[:,2]

fn_pred =t_y_pred[0]
tp_pred =t_y_pred[1]
fp_pred =t_y_pred[2]

print r2_score(precision(tp, fp), precision(tp_pred, fp_pred))
print r2_score(recall(tp, fn), recall(tp_pred, fn_pred))

print r2_score(fscore(tp, fp, fn), fscore(tp_pred, fp_pred, fn_pred))