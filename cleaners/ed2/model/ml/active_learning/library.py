import operator
import os
import random
import warnings
#from sets import Set

import jinja2
import numpy as np
#from eli5 import explain_weights
#from eli5 import show_weights
from eli5.formatters import format_as_text
from scipy.sparse import hstack
from scipy.sparse import vstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.pipeline import Pipeline
import collections
import time
from ml.features.CompressedDeepFeatures import read_compressed_deep_features
from ml.configuration.Config import Config

from ml.features.MetaDataFeatures import MetaDataFeatures

warnings.filterwarnings('ignore')

def create_user_start_data(feature_matrix, target, num_errors=2, return_ids=False):
    error_ids = np.where(target == True)[0]
    correct_ids = np.where(target == False)[0]

    if (len(error_ids) < 2 or len(correct_ids) < 2):
        if return_ids:
            return None, None, None
        else:
            return None, None

    error_select_ids = list(range(len(error_ids)))
    #error_select_ids = np.random.shuffle(range(len(error_ids)))
    np.random.shuffle(error_select_ids)
    error_select_ids = error_select_ids[0:num_errors]

    correct_select_ids = list(range(len(correct_ids)))
    np.random.shuffle(correct_select_ids)
    correct_select_ids = correct_select_ids[0:num_errors]

    list_ids = []
    list_ids.extend(error_ids[error_select_ids])
    list_ids.extend(correct_ids[correct_select_ids])

    train = feature_matrix[list_ids, :]
    train_target = target[list_ids]
    print(train_target)
    print(list_ids)


    if return_ids:
        return train, train_target, list_ids
    else:
        return train, train_target



def add_data_next(trainx, trainy, id_list, x_next, y_next, id_next):
    for ii in range(len(x_next)):
        trainx = vstack((trainx, x_next[ii]))

    trainy = np.append(trainy, y_next)
    id_list.extend(id_next)

    return trainx, trainy, id_list


def create_next_part(feature_matrix, target, y_pred, n, data, column_id, user_error_probability = 0.0, id_list=[], shuffle_seed=42):
    diff = np.absolute(y_pred - 0.5)
    sorted_ids = np.argsort(diff)

    certainty = (np.sum(diff) / len(diff)) * 2

    #print diff

    if np.sum(diff) == 0.0:
        np.random.shuffle(sorted_ids)

    if False:#np.sum(diff) == 0.0: #fall back to outlier detection
        print("tets")
        #get value frequencies
        '''

        distinct_values_count = collections.OrderedDict()
        for tuple_id in range(data.shape[0]):
            val = data.dirty_pd.values[tuple_id, column_id]
            if val in distinct_values_count:
                distinct_values_count[val].append(tuple_id)
            else:
                distinct_values_count[val] = list([tuple_id])

        #print "distinct values: " + str(len(distinct_values_count))

        #get least n frequent values
        def top_k_hybrid(a, k):
            b = np.argpartition(a, k)[:k]
            return b[np.argsort(a[b])]

        count_array = []
        key_array = []
        for key, value in distinct_values_count.iteritems():
            count_array.append(len(value))
            key_array.append(key)

        sort_key_ids = np.argsort(count_array)

        #print "key array: " + str(len(key_array))

        #round robin

        next_target = []
        next_x = []
        current_set = Set()
        internal_id_list = []

        k_distinct = 0

        #kn = n
        #if len(key_array) < n:
        #    kn = len(key_array)
        kn = len(key_array)

        while len(internal_id_list) < n:
            #choose one id of kth value randomly
            cur_val_id = sort_key_ids[k_distinct]
            cur_value = key_array[cur_val_id]
            dif_id_list = set(distinct_values_count[cur_value]).difference(id_list).difference(internal_id_list)
            #print id_list
            if len(dif_id_list) == 0:
                k_distinct += 1
                if k_distinct >= kn:
                    k_distinct = 0
                continue

            dif_id_list = list(dif_id_list)
            random_id = dif_id_list[np.random.randint(len(dif_id_list))]
            print random_id


            if not random_id in id_list and not random_id in internal_id_list:
                k_distinct += 1
                if k_distinct >= kn:
                    k_distinct = 0

                print internal_id_list

                current_set.add(data.dirty_pd.values[random_id, column_id])
                internal_id_list.append(random_id)
                print(data.dirty_pd.values[random_id, column_id])

                next_x.append(feature_matrix[random_id])

                random01 = random.random()
                if random01 < user_error_probability:  # introduce user error
                    next_target.append([not target[random_id]])
                else:
                    next_target.append([target[random_id]])
        
        if len(id_list) == 0:
            return next_x, next_target, diff
        else:
            return next_x, next_target, diff, internal_id_list
        '''



    else:
        next_target = []

        next_x = []

        current_set = set()
        i = 0
        internal_id_list = []
        while len(current_set) < n and i < len(sorted_ids):
            if not data.dirty_pd.values[sorted_ids[i],column_id] in current_set:
                if not sorted_ids[i] in id_list and not sorted_ids[i] in internal_id_list:
                    current_set.add(data.dirty_pd.values[sorted_ids[i],column_id])
                    internal_id_list.append(sorted_ids[i])
                    print(data.dirty_pd.values[sorted_ids[i],column_id])
                    #if len(id_list) > 0:
                    #    id_list.append(sorted_ids[i])

                    next_x.append(feature_matrix[sorted_ids[i]])
                    #train = vstack((train, feature_matrix[sorted_ids[i]]))

                    random01 = random.random()
                    if random01 < user_error_probability: # introduce user error
                        next_target.append([not target[sorted_ids[i]]])
                    else:
                        next_target.append([target[sorted_ids[i]]])
            i += 1

        i = 0
        while len(internal_id_list) < n:
            if not sorted_ids[i] in internal_id_list:
                if not sorted_ids[i] in id_list:
                    print(data.dirty_pd.values[sorted_ids[i], column_id])
                    internal_id_list.append(sorted_ids[i])
                    #if len(id_list) > 0:
                    #    id_list.append(sorted_ids[i])

                    next_x.append(feature_matrix[sorted_ids[i]])
                    #train = vstack((train, feature_matrix[sorted_ids[i]]))

                    random01 = random.random()
                    if random01 < user_error_probability:  # introduce user error
                        next_target.append([not target[sorted_ids[i]]])
                    else:
                        next_target.append([target[sorted_ids[i]]])
            i += 1

        if len(id_list) == 0:
            return next_x, next_target, diff
        else:
            return next_x, next_target, diff, internal_id_list


def create_next_dumm(feature_matrix, target, y_pred, n, data, column_id, user_error_probability = 0.0, id_list=[]):
    diff = np.absolute(y_pred - 0.5)
    sorted_ids = np.argsort(diff)

    next_target = []

    next_x = []

    internal_id_list = []
    i = 0
    while len(internal_id_list) < n:
        print(data.dirty_pd.values[sorted_ids[i], column_id])
        internal_id_list.append(sorted_ids[i])
        if len(id_list) > 0:#check
            id_list.append(sorted_ids[i]) #check

        next_x.append(feature_matrix[sorted_ids[i]])
        #train = vstack((train, feature_matrix[sorted_ids[i]]))

        random01 = random.random()
        if random01 < user_error_probability:  # introduce user error
            next_target.append([not target[sorted_ids[i]]])
        else:
            next_target.append([target[sorted_ids[i]]])
        i += 1

    if len(id_list) == 0:
        return next_x, next_target, diff
    else:
        return next_x, next_target, diff, internal_id_list








def create_next_data(train, train_target, feature_matrix, target, y_pred, n, data, column_id, user_error_probability = 0.0, id_list=[]):
    diff = np.absolute(y_pred - 0.5)
    sorted_ids = np.argsort(diff)

    certainty = (np.sum(diff) / len(diff)) * 2

    if certainty == 1.0:
        return train, train_target, 1.0
        if len(id_list) == 0:
            return train, train_target, 1.0
        else:
            return train, train_target, 1.0, id_list

    #plt.hist(diff)
    #plt.show()

    trainl = []

    current_set = set()
    i = 0
    internal_id_list = []
    while len(current_set) < n and i < len(sorted_ids):
        if not data.dirty_pd.values[sorted_ids[i],column_id] in current_set:
            if not sorted_ids[i] in id_list:
                current_set.add(data.dirty_pd.values[sorted_ids[i],column_id])
                internal_id_list.append(sorted_ids[i])
                print(data.dirty_pd.values[sorted_ids[i],column_id])
                if len(id_list) > 0:
                    id_list.append(sorted_ids[i])

                trainl.append(feature_matrix[sorted_ids[i]])
                train = vstack((train, feature_matrix[sorted_ids[i]]))

                random01 = random.random()
                if random01 < user_error_probability: # introduce user error
                    train_target = np.append(train_target, [not target[sorted_ids[i]]])
                else:
                    train_target = np.append(train_target, [target[sorted_ids[i]]])
        i += 1

    i = 0
    while len(internal_id_list) < n:
        if not sorted_ids[i] in internal_id_list:
            if not sorted_ids[i] in id_list:
                print(data.dirty_pd.values[sorted_ids[i], column_id])
                internal_id_list.append(sorted_ids[i])
                if len(id_list) > 0:
                    id_list.append(sorted_ids[i])

                trainl.append(feature_matrix[sorted_ids[i]])
                train = vstack((train, feature_matrix[sorted_ids[i]]))

                random01 = random.random()
                if random01 < user_error_probability:  # introduce user error
                    train_target = np.append(train_target, [not target[sorted_ids[i]]])
                else:
                    train_target = np.append(train_target, [target[sorted_ids[i]]])
        i += 1

    if len(id_list) == 0:
        return train, train_target, certainty
    else:
        return train, train_target, certainty, id_list

'''
def run_cross_validation(train, train_target, folds):
    cv_params = {'min_child_weight': [1, 3, 5],
                 'subsample': [0.7, 0.8, 0.9],
                 'max_depth': [3, 5, 7]}
    ind_params = {#'min_child_weight': 1, # we could optimize this: 'min_child_weight': [1, 3, 5]
                  'learning_rate': 0.1, # we could optimize this: 'learning_rate': [0.1, 0.01]
                  #'max_depth': 3, # we could optimize this: 'max_depth': [3, 5, 7]
                  #'n_estimators': 1000, # we choose default 100
                  'colsample_bytree': 0.8,
                  'silent': 1,
                  'seed': 0,
                  'objective': 'binary:logistic'}

    optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params),
                                 cv_params,
                                 scoring='f1', cv=folds, n_jobs=1, verbose=0)

    print(train.shape)

    optimized_GBM.fit(train, train_target)

    #print "best scores: " + str(optimized_GBM.grid_scores_)

    our_params = ind_params.copy()
    our_params.update(optimized_GBM.best_params_)

    return our_params

def run_cross_validation_eval(train, train_target, folds, our_params):
    scores = cross_val_score(xgb.XGBClassifier(**our_params), train, train_target, cv = folds, scoring = 'f1')
    return scores
'''

def print_stats(target, res):
    print("F-Score: " + str(f1_score(target, res)))
    print("Precision: " + str(precision_score(target, res)))
    print("Recall: " + str(recall_score(target, res)))

def calc_my_fscore(target, res, dataSet):
    t = target.flatten()
    pred = res.flatten()

    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(t, pred).ravel()

    total_f = float((2 * tp)) / float(((2 * tp) + fn + fp))

    print("my total: " + str(total_f) + " sklearn: " + str(f1_score(t, pred)))

    error_indices = np.where(np.sum(dataSet.matrix_is_error, axis=0) != 0)[0]

    t_part = target[:, error_indices].flatten()
    pred_part = res[:, error_indices].flatten()

    tn, fp, fn, tp = confusion_matrix(t_part, pred_part).ravel()

    total_f = float((2 * tp)) / float(((2 * tp) + fn + fp))

    print("my part total: " + str(total_f) + " sklearn: " + str(f1_score(t_part, pred_part)))


def print_stats_whole(target, res, label):
    print(label + " F-Score: " + str(f1_score(target.flatten(), res.flatten())))
    print(label + " Precision: " + str(precision_score(target.flatten(), res.flatten())))
    print(label + " Recall: " + str(recall_score(target.flatten(), res.flatten())))

def go_to_next_column(column_id, min_certainties, dataSet):
    if True:#round_robin:
        column_id = column_id + 1
        if column_id == dataSet.shape[1]:
            column_id = 0
        return column_id
    else:
        min_certainty = 1.0
        min_certainty_index = -1

        print(min_certainties)

        for key, value in min_certainties.iteritems():
            if min_certainty > value:
                min_certainty = value
                min_certainty_index = key

        print(min_certainty_index)

        return min_certainty_index


def create_features(dataSet, train_indices, test_indices, ngrams=2, runSVD=False, is_word=False, use_tf_idf=True):
    feature_name_list = []
    feature_list_train = []
    feature_list_test = []

    analyzer = 'char'
    if is_word:
        analyzer = 'word'

    if use_tf_idf:
        pipeline = Pipeline([('vect', CountVectorizer(analyzer=analyzer, lowercase=False, ngram_range=(1, ngrams))),
                             ('tfidf', TfidfTransformer())
                             ])
    else:
        pipeline = Pipeline([('vect', CountVectorizer(analyzer=analyzer, lowercase=False, ngram_range=(1, ngrams)))])

    #create features
    for column_id in range(dataSet.shape[1]):
        data_column_train = dataSet.dirty_pd.values[train_indices, column_id]
        data_column_test = dataSet.dirty_pd.values[test_indices, column_id]

        # bag of characters
        try:
            pipeline.fit(data_column_train)

            feature_matrix_train = pipeline.transform(data_column_train).astype(float)
            if len(data_column_test) > 0:
                feature_matrix_test = pipeline.transform(data_column_test).astype(float)

            listed_tuples = sorted(pipeline.named_steps['vect'].vocabulary_.items(), key=operator.itemgetter(1))
            feature_name_list.extend([str(dataSet.clean_pd.columns[column_id]) + "_letter_" + tuple_dict_sorted[0] + "_" for tuple_dict_sorted in listed_tuples])

            # correlations
            if runSVD:
                svd = TruncatedSVD(n_components=(feature_matrix_train.shape[1] - 1), n_iter=10)
                svd.fit(feature_matrix_train)
                correlated_matrix_train = svd.transform(feature_matrix_train)
                #print(svd.explained_variance_ratio_)

                print(pipeline.named_steps['vect'].vocabulary_)
                print(svd.components_)

                from ml.VisualizeSVD import visualize_svd
                visualize_svd(svd.components_, dataSet, pipeline.named_steps['vect'].vocabulary_, column_id)


                feature_name_list.extend([str(dataSet.clean_pd.columns[column_id]) + "_svd_" + str(svd_id) for svd_id in range(feature_matrix_train.shape[1] - 1)])
                feature_matrix_train = hstack((feature_matrix_train, correlated_matrix_train)).tocsr()

                if len(data_column_test) > 0:
                    correlated_matrix_test = svd.transform(feature_matrix_test)
                    feature_matrix_test = hstack((feature_matrix_test, correlated_matrix_test)).tocsr()

        except ValueError:
            pass


        feature_list_train.append(feature_matrix_train)

        if len(data_column_test) > 0:
            feature_list_test.append(feature_matrix_test)


    all_matrix_train = hstack(feature_list_train).tocsr()
    if len(data_column_test) > 0:
        all_matrix_test = hstack(feature_list_test).tocsr()
    else:
        all_matrix_test = None

    return all_matrix_train, all_matrix_test , feature_name_list

def getTarget(dataSet, column_id, train_indices, test_indices):
    train = dataSet.matrix_is_error[train_indices, column_id]
    test = dataSet.matrix_is_error[test_indices, column_id]

    return train, test

def compare_change(old_prediction, new_prediction):
    size = float(len(old_prediction))

    no_change_1 = len((np.where(np.logical_and(old_prediction == True, new_prediction == True)))[0]) / size
    no_change_0 = len((np.where(np.logical_and(old_prediction == False, new_prediction == False)))[0]) / size
    change_1_to_0 = len((np.where(np.logical_and(old_prediction == True, new_prediction == False)))[0]) / size
    change_0_to_1 = len((np.where(np.logical_and(old_prediction == False, new_prediction == True)))[0]) / size

    return no_change_0, no_change_1, change_0_to_1, change_1_to_0

def add_metadata_features(data, train_indices, test_indices, all_features_train, all_features_test, feature_names, use_meta_only=False):
    data_train = data.dirty_pd.values[train_indices, :]
    data_test = data.dirty_pd.values[test_indices, :]

    metadatafeatures = MetaDataFeatures()

    metadatafeatures.fit(data_train)

    features_train, names = metadatafeatures.transform(data_train, data.clean_pd.columns)

    if use_meta_only:
        all_features_train_new = features_train
        feature_names = names
        all_features_test_new = all_features_test

    else:
        all_features_train_new = hstack((all_features_train, features_train)).tocsr()
        feature_names.extend(names)

        if data_test.shape[0] > 0:
            features_test, _ = metadatafeatures.transform(data_test, data.clean_pd.columns)
            all_features_test_new = hstack((all_features_test, features_test)).tocsr()
        else:
            all_features_test_new = all_features_test




    return all_features_train_new, all_features_test_new, feature_names


def visualize_model(dataSet, column_id, final_gb, feature_name_list, train, target_run, res):
    try:
        column_name = dataSet.clean_pd.columns[column_id]

        feature_name_list_err_corr = list(feature_name_list)
        print("missing features: " + str(len(final_gb[column_id].feature_names)- len(feature_name_list)))

        if len(final_gb[column_id].feature_names)- len(feature_name_list) > 0:
            for err_corr_id in range(dataSet.shape[1]):
                if dataSet.is_column_applicable(err_corr_id) and err_corr_id != column_id:
                    feature_name_list_err_corr.append("error_corr_" + str(dataSet.clean_pd.columns[err_corr_id]))

        directory = Config.get("logging.folder") + '/out/html/' + dataSet.name
        if not os.path.exists(directory):
            os.makedirs(directory)
        path = directory + '/' + str(column_name) + '_' + str(train[column_id].shape[0]) + '_' + str(time.time())  + '.html'

        table_content = show_weights(final_gb[column_id], feature_names=feature_name_list_err_corr, importance_type="gain").data

        # print table_content
        from ml.VisualizeSVD import replace_with_url

        table_content = replace_with_url(table_content, dataSet)

        url = 'file://' + path
        html = "<h1>" + str(column_name) + "</h1>"
        html += "<h2>number of labels: " + str(train[column_id].shape[0]) + "</h2>"
        html += "<h2>F-Score: " + str(f1_score(target_run, res[column_id])) + "</h2>"
        html += str(table_content)

        with open(path, 'w') as webf:
            webf.write(html)
        webf.close()
        # webbrowser.open(url)
    except jinja2.exceptions.UndefinedError:
        print(format_as_text(explain_weights(final_gb[column_id], feature_names=feature_name_list)))



def split_data_indices(dataSet, train_fraction, fold_number=0):


    number_splits = 10

    if train_fraction == 1.0:
        return range(dataSet.shape[0]), []


    from sklearn.model_selection import KFold
    kf = KFold(n_splits=number_splits, shuffle=False, random_state=42)

    from collections import deque
    circular_queue = deque([])


    for _, test_id in kf.split(range(dataSet.shape[0])):
        circular_queue.append(test_id)
        print(test_id)


    circular_queue.rotate(-1 * fold_number)

    train_indices = []
    test_indices = []

    train_N = int(train_fraction * number_splits)
    test_N = number_splits - train_N

    for train_parts in range(train_N):
        train_indices.extend(circular_queue.pop())
    for test_parts in range(test_N):
        test_indices.extend(circular_queue.pop())

    print("train: " + str(len(train_indices)))
    print("test: " + str(len(test_indices)))


    return train_indices, test_indices