import math
import operator
import time
import warnings
from sets import Set

import jinja2
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from eli5 import explain_weights
from eli5 import show_weights
from eli5.formatters import format_as_text
from scipy.sparse import hstack
from scipy.sparse import vstack
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

warnings.filterwarnings('ignore')


def create_clustered_data(feature_matrix, target, train_size=0.00001):
    train_len = train_size

    if (train_size <= 1):
        train_len = int(math.ceil(feature_matrix.shape[0] * train_size))

    kmeans = MiniBatchKMeans(n_clusters=train_len, init='k-means++', random_state=0).fit(feature_matrix)
    cluster_ids = kmeans.predict(feature_matrix)

    print "number clusters: " + str(len(kmeans.cluster_centers_))

    n = len(cluster_ids)

    shuffled = np.arange(n)
    np.random.shuffle(shuffled)

    sample = {}
    for i in range(n):
        sample[cluster_ids[shuffled[i]]] = shuffled[i]
        if (len(sample) == train_len):
            break

    train_indices = Set(sample.values())

    for i in range(n):
        if (len(train_indices) < train_len):
            train_indices.add(shuffled[i])
        else:
            break

    train_indices = list(train_indices)

    train = feature_matrix[train_indices, :]
    train_target = target[train_indices]

    return train, train_target

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


def create_next_data(train, train_target, feature_matrix, target, y_pred, n):
    diff = np.absolute(y_pred - 0.5)
    sorted_ids = np.argsort(diff)

    certainty = (np.sum(diff) / len(diff)) * 2

    if certainty == 1.0:
        return train, train_target, 1.0

    #plt.hist(diff)
    #plt.show()

    trainl = []

    for i in range(n):
        trainl.append(feature_matrix[sorted_ids[i]])
        train = vstack((train, feature_matrix[sorted_ids[i]]))
        train_target = np.append(train_target, [target[sorted_ids[i]]])

    return train, train_target, certainty


def create_next_data_clustering(train, train_target, feature_matrix, target, y_pred, n):
    diff = np.absolute(y_pred - 0.5)
    sorted_ids = np.argsort(diff)

    plt.hist(diff)
    plt.show()

    uncertain_indices = np.where(diff < 0.5)[0]

    print uncertain_indices[0]

    cluster_data = feature_matrix[uncertain_indices]

    kmeans = MiniBatchKMeans(n_clusters=n, init='k-means++', random_state=0).fit(cluster_data)
    cluster_ids = kmeans.predict(cluster_data)

    index_dict = {}
    diff_dict = {}
    for i in range(len(cluster_ids)):
        if cluster_ids[i] in diff_dict:
            if diff[uncertain_indices[i]] < diff_dict[cluster_ids[i]]:
                diff_dict[cluster_ids[i]] = diff[uncertain_indices[i]]
                index_dict[cluster_ids[i]] = uncertain_indices[i]
        else:
            diff_dict[cluster_ids[i]] = diff[uncertain_indices[i]]
            index_dict[cluster_ids[i]] = uncertain_indices[i]

    for i in range(n):
        train = vstack((train, feature_matrix[index_dict[i]]))
        train_target = np.append(train_target, [target[sorted_ids[i]]])

    return train, train_target

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




def generate_column_feature(dirty, distinct_values, column):
    rows = len(dirty)
    cols = len(dirty.columns)

    current_value_fraction = np.array([0.0] * rows)
    current_value_length = np.array([0] * rows)
    current_row_avg_fraction = np.array([0.0] * rows)
    current_row_avg_length = np.array([0.0] * rows)

    i = 0
    for row in range(rows):

        row_sum_length = 0.0
        row_sum_fraction = 0.0
        for column in range(cols):
            row_sum_fraction += distinct_values.get(dirty.values[row][column], 0)
            row_sum_length += len(str(dirty.values[row][column]))
        row_avg_length = row_sum_length / cols
        row_avg_fraction = row_sum_fraction / cols


        # value based
        current_value_fraction[i] = distinct_values.get(dirty.values[row][column], 0)
        current_value_length[i] = len(str(dirty.values[row][column]))
        # row based
        current_row_avg_fraction[i] = row_avg_fraction
        current_row_avg_length[i] = row_avg_length

        i += 1

    schema = ["val_fraction", "val_length",
              "row_avg_fraction", "row_avg_length"]

    return [current_value_fraction, current_value_length, \
            current_row_avg_fraction, current_row_avg_length], schema

def compute_general_stats(dirty, column):
    distinct_values = {}
    N = len(dirty)

    for index, row in dirty.iterrows():
        value = row[dirty.columns[column]]
        distinct_values[value] = distinct_values.get(value, 0) + 1

    for column in range(len(dirty.columns)):
        for key, value in distinct_values.iteritems():
            distinct_values[key] = value / float(N)

    stats_feature_list,_ = generate_column_feature(dirty, distinct_values, column)

    return np.transpose(np.matrix(stats_feature_list))


def one_hot_encoding(dataset, column_id, matrix):
    X_int = np.matrix(LabelEncoder().fit_transform(dataset.dirty_pd[dataset.dirty_pd.columns[column_id]])).transpose()

    # transform to binary
    X_bin = OneHotEncoder().fit_transform(X_int)

    enhanced_matrix = hstack((matrix, X_bin)).tocsr()

    return enhanced_matrix


def print_stats(target, res):
    print "F-Score: " + str(f1_score(target, res))
    print "Precision: " + str(precision_score(target, res))
    print "Recall: " + str(recall_score(target, res))

def print_stats_whole(target, res):
    print "All F-Score: " + str(f1_score(target.flatten(), res.flatten()))
    print "All Precision: " + str(precision_score(target.flatten(), res.flatten()))
    print "All Recall: " + str(recall_score(target.flatten(), res.flatten()))

def go_to_next_column(column_id, min_certainties, round_robin):
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

pipeline = Pipeline([('vect', CountVectorizer(analyzer='char', lowercase=False)),
                     ('tfidf', TfidfTransformer())
                    ])

start_time = time.time()

#dataSet = BlackOakDataSet()
#dataSet = FlightLarysa()
from ml.datasets.flights import FlightHoloClean
dataSet = FlightHoloClean()
#dataSet = HospitalHoloClean()
#dataSet = IQContest()

print("read: %s seconds ---" % (time.time() - start_time))

start_time = time.time()


feature_name_list = []

feature_list = []
#create features
for column_id in range(dataSet.shape[1]):
    data_column = dataSet.dirty_pd.values[:, column_id].astype('U')

    print data_column

    # bag of characters
    clf = pipeline.fit(data_column)

    feature_matrix = pipeline.transform(data_column).astype(float)

    listed_tuples = sorted(pipeline.named_steps['vect'].vocabulary_.items(), key=operator.itemgetter(1))
    feature_name_list.extend([str(dataSet.clean_pd.columns[column_id]) + "_letter_" + tuple_dict_sorted[0] for tuple_dict_sorted in listed_tuples])


    # correlations
    svd = TruncatedSVD(n_components=(feature_matrix.shape[1] - 1), n_iter=10, random_state=42)
    svd.fit(feature_matrix)
    correlated_matrix = svd.transform(feature_matrix)

    feature_name_list.extend([dataSet.clean_pd.columns[column_id] + "_svd_" + str(svd_id) for svd_id in range(feature_matrix.shape[1] - 1)])

    feature_matrix = hstack((feature_matrix, correlated_matrix)).tocsr()

    #apply one hot encoding for categorical data
    if len(dataSet.dirty_pd[dataSet.dirty_pd.columns[column_id]].unique()) <= 100:
        length_old = feature_matrix.shape[1]

        feature_matrix = one_hot_encoding(dataSet, column_id, feature_matrix)

        feature_name_list.extend([dataSet.clean_pd.columns[column_id] + "_onehotencod_" + str(onehot_id) for onehot_id in
                                  range(feature_matrix.shape[1] - length_old)])

    feature_list.append(feature_matrix)


all_matrix_bag = hstack(feature_list).tocsr()

#all_matrix = hstack((read_compressed_deep_features(), all_matrix_bag)).tocsr()
all_matrix = all_matrix_bag

print all_matrix.shape
print len(feature_name_list)


print("features: %s seconds ---" % (time.time() - start_time))

checkN = 1


f = open('/home/felix/SequentialPatternErrorDetection/log_progress.csv', 'w+')
#f.write('column,labels,certainty,min_certainty,fscore\n')


print dataSet.shape[0] * dataSet.shape[1]

for check_this in range(checkN):

    data_result = []

    column_id = 0

    feature_matrix = all_matrix
    testdmat = xgb.DMatrix(feature_matrix)

    all_error_status = np.zeros(dataSet.dirty_pd.shape, dtype=bool)


    save_fscore = []
    save_labels = []
    save_certainty = []

    our_params = {}
    train = {}
    train_target = {}
    y_pred = {}
    certainty = {}
    min_certainty = {}
    final_gb = {}



    for round in range(21 * dataSet.shape[1]):
        print "round: " + str(round)

        #switch to column
        target = dataSet.matrix_is_error[:, column_id]


        if round < dataSet.shape[1]:
            start_time = time.time()

            num_errors = 2
            train[column_id], train_target[column_id] = create_user_start_data(feature_matrix, target, num_errors)
            if train[column_id] == None:
                certainty[column_id] = 1.0
                column_id = go_to_next_column(column_id, min_certainty, round < dataSet.shape[1])
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
                column_id = go_to_next_column(column_id, min_certainty, round < dataSet.shape[1])
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



            train[column_id], train_target[column_id], certainty[column_id] = create_next_data(train[column_id], train_target[column_id], feature_matrix, target, y_pred[column_id], 10)

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
        res = (y_pred[column_id] > 0.5)
        all_error_status[:, column_id] = res
        print("train & predict: %s seconds ---" % (time.time() - start_time))


        try:
            import os
            import webbrowser

            column_name = dataSet.clean_pd.columns[column_id]

            path = '/home/felix/SequentialPatternErrorDetection/html/Flights/' + column_name + '.html'
            url = 'file://' + path
            html = "<h1>" + column_name + "</h1>"
            html += "<h2>number of labels: " + str(train[column_id].shape[0]) + "</h2>"
            html += "<h2>F-Score: " + str(f1_score(target, res)) + "</h2>"
            html += show_weights(final_gb[column_id], feature_names=feature_name_list, importance_type="gain").data

            with open(path, 'w') as webf:
                webf.write(html)
            webf.close()
            #webbrowser.open(url)
        except jinja2.exceptions.UndefinedError:
            print format_as_text(explain_weights(final_gb[column_id], feature_names=feature_name_list))


        print "current train shape: " + str(train[column_id].shape)

        print "column: " + str(column_id)
        print_stats(target, res)
        print_stats_whole(dataSet.matrix_is_error, all_error_status)

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

        save_fscore.append(f1_score(dataSet.matrix_is_error.flatten(), all_error_status.flatten()))
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

            distinct_values_fraction = float(len(dataSet.dirty_pd[dataSet.dirty_pd.columns[column_id]].unique())) / float(dataSet.shape[0])


            f.write(str(distinct_values_fraction) + ',' + str(train[column_id].shape[0]) + ',' + str(certainty[column_id]) + ',' + str(np.min(np.absolute(y_pred[column_id] - 0.5))) + str_hist + ',' + str(f1_score(target, res)) + '\n')


        column_id = go_to_next_column(column_id, min_certainty, round < dataSet.shape[1])

    print save_fscore
    print save_labels
    print save_certainty