import math
import time
import warnings
from sets import Set

import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
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

    certainty = float(len(np.where(diff > 0.4)[0])) / len(diff)
    print "bigger certainty than 0.4: " + str(certainty)

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

print("read: %s seconds ---" % (time.time() - start_time))

start_time = time.time()

feature_list = []
#create features
for column_id in range(dataSet.shape[1]):
    data_column = dataSet.dirty_pd.values[:, column_id].astype('U')

    print data_column

    # bag of characters
    clf = pipeline.fit(data_column)

    feature_matrix = pipeline.transform(data_column).astype(float)

    # stats
    # stats = compute_general_stats(dataSet.dirty_pd, column_id)
    # feature_matrix = hstack((feature_matrix, stats)).tocsr()

    # correlations
    svd = TruncatedSVD(n_components=(feature_matrix.shape[1] - 1), n_iter=10, random_state=42)
    svd.fit(feature_matrix)
    correlated_matrix = svd.transform(feature_matrix)

    feature_matrix = hstack((feature_matrix, correlated_matrix)).tocsr()

    #apply one hot encoding for categorical data
    if len(dataSet.dirty_pd[dataSet.dirty_pd.columns[column_id]].unique()) <= 100:
        feature_matrix = one_hot_encoding(dataSet, column_id, feature_matrix)

    feature_list.append(feature_matrix)


all_matrix = hstack(feature_list).tocsr()

'''
svd = TruncatedSVD(n_components=200, n_iter=10, random_state=42)
svd.fit(all_matrix)
correlated_matrixa = svd.transform(all_matrix)

all_matrix = hstack((all_matrix, correlated_matrixa)).tocsr()
'''


print("features: %s seconds ---" % (time.time() - start_time))

checkN = 1
check_fscore = []
check_precision = []
check_recall = []
check_labeled = []

for check_this in range(checkN):

    data_result = []

    number_samples = 0

    for column_id in range(dataSet.shape[1]):

        #feature_matrix = feature_list[column_id]
        feature_matrix = all_matrix
        target = dataSet.matrix_is_error[:, column_id]

        # active learning

        f_scores = []
        precision_scores = []
        recall_scores = []

        models = dict()

        n = 1
        for train_size in [10]:
            f_current = 0.0
            precision_current = 0.0
            recall_current = 0.0

            for t in range(n):
                start_time = time.time()

                num_errors = 2
                train, train_target = create_user_start_data(feature_matrix, target, num_errors)
                if train == None:
                    res2 = np.zeros(feature_matrix.shape[0])
                    data_result.append(res2)
                    break

                #train, train_target = create_clustered_data(feature_matrix, target, 30)

                print "Number of errors in training: " + str(np.sum(train_target))

                print train.shape

                print("clustering: %s seconds ---" % (time.time() - start_time))

                start_time = time.time()
                #cross-validation
                our_params = run_cross_validation(train, train_target, num_errors)
                print("cv: %s seconds ---" % (time.time() - start_time))

                print our_params

                start_time = time.time()

                xgdmat = xgb.DMatrix(train, train_target)

                final_gb = xgb.train(our_params, xgdmat, num_boost_round=3000, verbose_eval=False)

                testdmat = xgb.DMatrix(feature_matrix)
                y_pred = final_gb.predict(testdmat)

                print("train & predict: %s seconds ---" % (time.time() - start_time))

                res = (y_pred > 0.5)

                print "1 - F-Score: " + str(f1_score(target, res))
                print "1 - Precision: " + str(precision_score(target, res))
                print "1 - Recall: " + str(recall_score(target, res))

                for r in range(10):
                    train, train_target, certainty = create_next_data(train, train_target, feature_matrix, target, y_pred, 10)

                    if certainty == 1.0:
                        break

                    #start_time = time.time()
                    # cross-validation
                    #our_params = run_cross_validation(train, train_target, 20)
                    #print("cv: %s seconds ---" % (time.time() - start_time))

                    xgdmat2 = xgb.DMatrix(train, train_target)
                    final_gb2 = xgb.train(our_params, xgdmat2, num_boost_round=3000, verbose_eval=False)

                    y_pred = final_gb2.predict(testdmat)

                    res2 = (y_pred > 0.5)

                    f1_now = f1_score(target, res2)

                    print train.shape
                    print "2 - F-Score: " + str(f1_now)
                    print "2 - Precision: " + str(precision_score(target, res2))
                    print "2 - Recall: " + str(recall_score(target, res2)) + "\n\n"

                    #if f1_now == 1.0 or f1_now == 0.0:
                    #    break

                data_result.append(res2)

                f_current += f1_score(target, res2)
                precision_current += precision_score(target, res2)
                recall_current += recall_score(target, res2)

            f_scores.append(f_current / n)
            precision_scores.append(precision_current / n)
            recall_scores.append(recall_current / n)

            if train != None:
                number_samples += train.shape[0]

        print "column: "+ str(column_id)
        print "F-Score: " + str(f_scores)
        print "Precision: " + str(precision_scores)
        print "Recall: " + str(recall_scores)

    my_result = np.transpose(np.array(data_result))


    nc= number_samples
    fc= f1_score(dataSet.matrix_is_error.flatten(), my_result.flatten())
    pc= precision_score(dataSet.matrix_is_error.flatten(), my_result.flatten())
    rc= recall_score(dataSet.matrix_is_error.flatten(), my_result.flatten())

    check_labeled.append(nc)
    check_fscore.append(fc)
    check_precision.append(pc)
    check_recall.append(rc)

    print "Current Fscore: " + str(fc)


print "Total number: " + str(check_labeled) + " -> avg: " + str(np.mean(check_labeled))
print "Fscore: " + str(check_fscore) + " -> avg: " + str(np.mean(check_fscore))
print "Precision: " + str(check_precision) + " -> avg: " + str(np.mean(check_precision))
print "Recall: " + str(check_recall) + " -> avg: " + str(np.mean(check_recall))