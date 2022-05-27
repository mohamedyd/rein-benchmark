import operator
import time
import warnings
from sets import Set

import jinja2
import numpy as np
import xgboost as xgb
from eli5 import explain_weights
from eli5 import show_weights
from eli5.formatters import format_as_text
from scipy.sparse import hstack
from scipy.sparse import vstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from ml.datasets.blackOak import BlackOakDataSet

warnings.filterwarnings('ignore')


total_start_time = time.time()


def create_user_start_data(feature_matrix, target, num_errors=2):
    error_ids = np.where(target == True)[0]
    correct_ids = np.where(target == False)[0]

    if (len(error_ids) == 0 or len(correct_ids) == 0):
        return None,None

    error_select_ids = range(len(error_ids))
    np.random.shuffle(error_select_ids)
    error_select_ids = error_select_ids[0:num_errors]

    correct_select_ids = range(len(correct_ids))
    np.random.shuffle(correct_select_ids)
    correct_select_ids = correct_select_ids[0:num_errors]

    list_ids = []
    list_ids.extend(error_ids[error_select_ids])
    list_ids.extend(correct_ids[correct_select_ids])

    train = feature_matrix[list_ids, :]
    train_target = target[list_ids]
    print train_target


    return train, train_target


def create_next_data(train, train_target, feature_matrix, target, y_pred, n, data, column_id):
    diff = np.absolute(y_pred - 0.5)
    sorted_ids = np.argsort(diff)

    certainty = (np.sum(diff) / len(diff)) * 2

    if certainty == 1.0:
        return train, train_target, 1.0

    #plt.hist(diff)
    #plt.show()

    trainl = []

    current_set = Set()
    i = 0
    while len(current_set) < 10:
        if not data.dirty_pd.values[sorted_ids[i], column_id] in current_set:
            current_set.add(data.dirty_pd.values[sorted_ids[i], column_id])
            print data.dirty_pd.values[sorted_ids[i], column_id]

            trainl.append(feature_matrix[sorted_ids[i]])
            train = vstack((train, feature_matrix[sorted_ids[i]]))
            train_target = np.append(train_target, [target[sorted_ids[i]]])
        i += 1

    return train, train_target, certainty

def run_cross_validation(train, train_target, folds):
    cv_params = {'min_child_weight': [1, 3, 5],
                 'subsample': [0.7, 0.8, 0.9]}
    ind_params = {#'min_child_weight': 1, # we could optimize this: 'min_child_weight': [1, 3, 5]
                  'learning_rate': 0.1, # we could optimize this: 'learning_rate': [0.1, 0.01]
                  'max_depth': 3, # we could optimize this: 'max_depth': [3, 5, 7]
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

def run_cross_validation_eval(train, train_target, folds, our_params):
    scores = cross_val_score(xgb.XGBClassifier(**our_params), train, train_target, cv = folds, scoring = 'f1')
    return scores

def print_stats(target, res):
    print "F-Score: " + str(f1_score(target, res))
    print "Precision: " + str(precision_score(target, res))
    print "Recall: " + str(recall_score(target, res))

def print_stats_whole(target, res, label):
    print label + " F-Score: " + str(f1_score(target.flatten(), res.flatten()))
    print label + " Precision: " + str(precision_score(target.flatten(), res.flatten()))
    print label + " Recall: " + str(recall_score(target.flatten(), res.flatten()))

def go_to_next_column(column_id, min_certainties):
    if True:#round_robin:
        column_id = column_id + 1
        if column_id == dataSet.shape[1]:
            column_id = 0
        return column_id
    else:
        min_certainty = 1.0
        min_certainty_index = -1

        print min_certainties

        for key, value in min_certainties.iteritems():
            if min_certainty > value:
                min_certainty = value
                min_certainty_index = key

        print min_certainty_index

        return min_certainty_index


#input

start_time = time.time()

dataSet = BlackOakDataSet()
#dataSet = FlightLarysa()
#from ml.flights.FlightHoloClean import FlightHoloClean
#dataSet = FlightHoloClean()
#dataSet = HospitalHoloClean()
#dataSet = IQContest()

print("read: %s seconds ---" % (time.time() - start_time))

start_time = time.time()

train_fraction = 1.0
ngrams = 2
runSVD = True
replace = True
svd_dimensions = 100

def create_features(dataSet, train_fraction = 0.8, ngrams=2, runSVD=False, replace = True, svd_dimensions = None):
    split_id = int(dataSet.shape[0] * train_fraction)

    feature_name_list = []
    feature_list_train = []
    feature_list_test = []

    pipeline = Pipeline([('vect', CountVectorizer(analyzer='char', lowercase=False, ngram_range=(1, ngrams))),
                         ('tfidf', TfidfTransformer())
                         ])

    #create features
    for column_id in range(dataSet.shape[1]):
        data_column_train = dataSet.dirty_pd.values[0:split_id, column_id]
        data_column_test = dataSet.dirty_pd.values[split_id:dataSet.shape[0], column_id]

        # bag of characters
        pipeline.fit(data_column_train)

        feature_matrix_train = pipeline.transform(data_column_train).astype(float)
        if len(data_column_test) > 0:
            feature_matrix_test = pipeline.transform(data_column_test).astype(float)

        if not replace:
            listed_tuples = sorted(pipeline.named_steps['vect'].vocabulary_.items(), key=operator.itemgetter(1))
            feature_name_list.extend([str(dataSet.clean_pd.columns[column_id]) + "_letter_" + tuple_dict_sorted[0] for tuple_dict_sorted in listed_tuples])

        # correlations
        if runSVD:
            if svd_dimensions == None:
                svd_dimensions = (feature_matrix_train.shape[1] - 1)
            else:
                svd_dimensions = min((feature_matrix_train.shape[1] - 1), svd_dimensions)
            svd = TruncatedSVD(n_components=svd_dimensions, n_iter=10, random_state=42)
            svd.fit(feature_matrix_train)
            correlated_matrix_train = svd.transform(feature_matrix_train)

            feature_name_list.extend([dataSet.clean_pd.columns[column_id] + "_svd_" + str(svd_id) for svd_id in range(svd_dimensions)])

            if replace:
                feature_matrix_train = np.matrix(correlated_matrix_train)
            else:
                feature_matrix_train = hstack((feature_matrix_train, correlated_matrix_train))

            if len(data_column_test) > 0:
                correlated_matrix_test = svd.transform(feature_matrix_test)

                if replace:
                    feature_matrix_test = np.matrix(correlated_matrix_test)
                else:
                    feature_matrix_test = hstack((feature_matrix_test, correlated_matrix_test)).tocsr()


        print feature_matrix_train.shape

        feature_list_train.append(feature_matrix_train)

        if len(data_column_test) > 0:
            feature_list_test.append(feature_matrix_test)


    all_matrix_train = np.hstack(feature_list_train)
    if len(data_column_test) > 0:
        all_matrix_test = np.matrix(np.hstack(feature_list_test))
    else:
        all_matrix_test = None

    return np.matrix(all_matrix_train), all_matrix_test , feature_name_list

def getTarget(dataSet, column_id, train_fraction = 0.8):
    split_id = int(dataSet.shape[0] * train_fraction)

    train = dataSet.matrix_is_error[0:split_id, column_id]
    test = dataSet.matrix_is_error[split_id:dataSet.shape[0], column_id]

    return train, test

def compare_change(old_prediction, new_prediction):
    size = float(len(old_prediction))

    no_change_1 = len((np.where(np.logical_and(old_prediction == True, new_prediction == True)))[0]) / size
    no_change_0 = len((np.where(np.logical_and(old_prediction == False, new_prediction == False)))[0]) / size
    change_1_to_0 = len((np.where(np.logical_and(old_prediction == True, new_prediction == False)))[0]) / size
    change_0_to_1 = len((np.where(np.logical_and(old_prediction == False, new_prediction == True)))[0]) / size

    return no_change_0, no_change_1, change_0_to_1, change_1_to_0

all_matrix_train, all_matrix_test, feature_name_list = create_features(dataSet, train_fraction, ngrams, runSVD, replace, svd_dimensions)


print("features: %s seconds ---" % (time.time() - start_time))

checkN = 1


f = open('/home/felix/SequentialPatternErrorDetection/log_progress.csv', 'w+')

print dataSet.shape[0] * dataSet.shape[1]

split_id = int(dataSet.shape[0] * train_fraction)

for check_this in range(checkN):

    data_result = []

    column_id = 0

    feature_matrix = all_matrix_train
    testdmat = xgb.DMatrix(all_matrix_train)
    general_test_mat = xgb.DMatrix(all_matrix_test)

    all_error_status = np.zeros((all_matrix_train.shape[0], dataSet.shape[1]), dtype=bool)
    if all_matrix_test != None:
        all_error_status_test = np.zeros((all_matrix_test.shape[0], dataSet.shape[1]), dtype=bool)


    save_fscore = []
    save_labels = []
    save_certainty = []
    save_fscore_general = []
    save_time = []

    our_params = {}
    train = {}
    train_target = {}
    y_pred = {}
    certainty = {}
    min_certainty = {}
    final_gb = {}
    res = {}

    for round in range(6 * dataSet.shape[1]):
        print "round: " + str(round)

        #switch to column
        target_run, target_test = getTarget(dataSet, column_id, train_fraction)


        if round < dataSet.shape[1]:
            start_time = time.time()

            num_errors = 2
            train[column_id], train_target[column_id] = create_user_start_data(feature_matrix, target_run, num_errors)
            if train[column_id] == None:
                certainty[column_id] = 1.0
                column_id = go_to_next_column(column_id, min_certainty)
                continue

            print "Number of errors in training: " + str(np.sum(train_target[column_id]))
            print("clustering: %s seconds ---" % (time.time() - start_time))

            #cross-validation
            start_time = time.time()
            our_params[column_id] = run_cross_validation(train[column_id], train_target[column_id], num_errors)
            print("cv: %s seconds ---" % (time.time() - start_time))

            min_certainty[column_id] = 0.0

        else:
            if train[column_id] == None:
                column_id = go_to_next_column(column_id, min_certainty)
                continue

            if column_id in certainty:
                min_certainty[column_id] = np.min(np.absolute(y_pred[column_id] - 0.5))
            else:
                min_certainty[column_id] = 0.0


            '''
            # important stop criterion
            diff = np.absolute(y_pred[column_id] - 0.5)
            print "min: " + str(np.min(diff))
            if np.min(diff) >= 0.4: #and certainty[column_id] > 0.9:
                column_id = go_to_next_column(column_id, min_certainty)
                continue
            '''


            train[column_id], train_target[column_id], certainty[column_id] = create_next_data(train[column_id], train_target[column_id], feature_matrix, target_run, y_pred[column_id], 10, dataSet, column_id)

            print "column: " + str(column_id) + " - current certainty: " + str(certainty[column_id])

            # think about a different stoppping criteria
            # e.g. we can introduce again > 0.4 -> 1.0
            # or certainty > 0.9
            # introduce threshold!!
            #if certainty[column_id] > 0.9:
            #    column_id = go_to_next_column(column_id, round, certainty, old_certainty)
            #    continue

            #start_time = time.time()
            # cross-validation
            if round < dataSet.shape[1]*2:
                our_params[column_id] = run_cross_validation(train[column_id], train_target[column_id], num_errors)
            #print("cv: %s seconds ---" % (time.time() - start_time))

            eval_scores = run_cross_validation_eval(train[column_id], train_target[column_id], 7, our_params[column_id])


        start_time = time.time()
        #train
        xgdmat = xgb.DMatrix(train[column_id], train_target[column_id])
        final_gb[column_id] = xgb.train(our_params[column_id], xgdmat, num_boost_round=3000, verbose_eval=False)
        #predict
        y_pred[column_id] = final_gb[column_id].predict(testdmat)
        res_new = (y_pred[column_id] > 0.5)

        if column_id in res:
            no_change_0, no_change_1, change_0_to_1, change_1_to_0 = compare_change(res[column_id], res_new)

            print "no change 0: " + str(no_change_0) + " no change 1: " + str(no_change_1) + " sum no change: " + str(no_change_0 + no_change_1)
            print "change 0 ->1: " + str(change_0_to_1) + " change 1->0: " + str(change_1_to_0) + " sum change: " + str(change_0_to_1 + change_1_to_0)

        res[column_id] = res_new
        all_error_status[:, column_id] = res[column_id]
        print("train & predict: %s seconds ---" % (time.time() - start_time))

        if all_matrix_test != None:
            y_pred_test = final_gb[column_id].predict(general_test_mat)
            res_gen = (y_pred_test > 0.5)
            all_error_status_test[:, column_id] = res_gen


        try:
            column_name = dataSet.clean_pd.columns[column_id]

            path = '/home/felix/SequentialPatternErrorDetection/html/Flights/' + column_name + '.html'
            url = 'file://' + path
            html = "<h1>" + column_name + "</h1>"
            html += "<h2>number of labels: " + str(train[column_id].shape[0]) + "</h2>"
            html += "<h2>F-Score: " + str(f1_score(target_run, res[column_id])) + "</h2>"
            html += show_weights(final_gb[column_id], feature_names=feature_name_list, importance_type="gain").data

            with open(path, 'w') as webf:
                webf.write(html)
            webf.close()
            #webbrowser.open(url)
        except jinja2.exceptions.UndefinedError:
            print format_as_text(explain_weights(final_gb[column_id], feature_names=feature_name_list))


        print "current train shape: " + str(train[column_id].shape)

        print "column: " + str(column_id)
        print_stats(target_run, res[column_id])
        print_stats_whole(dataSet.matrix_is_error[0:split_id,:], all_error_status, "run all")
        if all_matrix_test != None:
            print_stats_whole(dataSet.matrix_is_error[split_id:dataSet.shape[0], :], all_error_status_test, "test general")

        number_samples = 0
        for key, value in train.iteritems():
            if value != None:
                number_samples += value.shape[0]
        print "total labels: " + str(number_samples) + " in %: " + str(float(number_samples)/(dataSet.shape[0]*dataSet.shape[1]))

        sum_certainty = 0.0
        for key, value in certainty.iteritems():
            if value != None:
                sum_certainty += value
        sum_certainty /= dataSet.shape[1]
        print "total certainty: " + str(sum_certainty)

        save_fscore.append(f1_score(dataSet.matrix_is_error[0:split_id,:].flatten(), all_error_status.flatten()))
        if all_matrix_test != None:
            save_fscore_general.append(f1_score(dataSet.matrix_is_error[split_id:dataSet.shape[0],:].flatten(), all_error_status_test.flatten()))
        save_labels.append(number_samples)
        save_certainty.append(sum_certainty)


        if round >= dataSet.shape[1]:
            num_hist_bin = 100

            diff = np.absolute(y_pred[column_id] - 0.5)

            str_hist = ""

            for i in range(num_hist_bin):
                str_hist += "," + str(float(len(diff[np.logical_and(diff >= i * (0.5 / num_hist_bin), diff < (i+1) * (0.5 / num_hist_bin))])) / len(diff))

            for score in eval_scores:
                str_hist += "," + str(score)

            str_hist += "," + str(np.mean(eval_scores))

            str_hist += "," + str(no_change_0)
            str_hist += "," + str(no_change_1)
            str_hist += "," + str(change_0_to_1)
            str_hist += "," + str(change_1_to_0)

            distinct_values_fraction = float(len(dataSet.dirty_pd[dataSet.dirty_pd.columns[column_id]].unique())) / float(dataSet.shape[0])


            f.write(str(distinct_values_fraction) + ',' + str(train[column_id].shape[0]) + ',' + str(certainty[column_id]) + ',' + str(np.min(np.absolute(y_pred[column_id] - 0.5))) + str_hist + ',' + str(f1_score(target_run, res[column_id])) + '\n')


        column_id = go_to_next_column(column_id, min_certainty)

        current_runtime = (time.time() - total_start_time)
        print("iteration end: %s seconds ---" % current_runtime)
        save_time.append(current_runtime)

    print save_fscore
    print save_fscore_general
    print save_labels
    print save_certainty
    print save_time