from ml.active_learning.library import *

warnings.filterwarnings('ignore')

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference


def normalize(x):
    probabilities = x / np.sum(x)
    return probabilities

def go_to_next_column_certainty(column_id, certainty, dataSet):
    if len(certainty) < dataSet.get_number_dirty_columns():#round_robin:
        column_id = column_id + 1
        if column_id == dataSet.shape[1]:
            column_id = 0
        return column_id
    else:
        certainty_array = np.zeros(dataSet.get_number_dirty_columns())
        indexmap = {}
        index_i = 0
        for key, value in certainty.iteritems():
            certainty_array[index_i] = np.square(1.0 - value)
            indexmap[index_i] = key
            index_i += 1

        prob = normalize(certainty_array)

        print certainty_array
        print prob

        return indexmap[np.random.choice(len(prob), 1, p=prob)[0]]





#input



from ml.datasets.flights.FlightHoloClean import FlightHoloClean
dataSet = FlightHoloClean()
from ml.datasets.hospital.HospitalHoloClean import HospitalHoloClean
#dataSet = HospitalHoloClean()
from ml.datasets.blackOak.BlackOakDataSetUppercase import BlackOakDataSetUppercase
#dataSet = BlackOakDataSetUppercase()

#print("read: %s seconds ---" % (time.time() - start_time))

start_time = time.time()

train_fraction = 1.0
ngrams = 1
runSVD = False
use_metadata = True
user_error_probability = 0.0
step_size = 10
cross_validation_rounds = 1 #1

checkN = 5 #total runs #5
label_iterations = 6 #6


f = open(Config.get("active.learning.logfile"), 'w+')

for check_this in range(checkN):

    start_time = time.time()
    total_start_time = time.time()

    all_matrix_train, all_matrix_test, feature_name_list = create_features(dataSet, train_fraction, ngrams, runSVD)

    '''
    #blackoak
    #all_matrix_train = read_compressed_deep_features()
    #all_matrix_test = None
    #feature_name_list = ['deep ' + str(dfeature) for dfeature in range(all_matrix_train.shape[1])]
    
    '''
    '''
    # tax
    all_matrix_train_deep = read_compressed_deep_features("/home/felix/SequentialPatternErrorDetection/deepfeatures/Tax/full_row/last_state/")
    '''

    '''
    # flights
    all_matrix_train_deep = read_compressed_deep_features(
        "/home/felix/SequentialPatternErrorDetection/deepfeatures/Flights/last/")

    '''
    #BlackOakUppercase
    #all_matrix_train_deep = read_compressed_deep_features(
    #    "/home/felix/SequentialPatternErrorDetection/deepfeatures/BlackOakUppercase/last/")

    '''
    #Hospital
    all_matrix_train_deep = read_compressed_deep_features(
        "/home/felix/SequentialPatternErrorDetection/deepfeatures/HospitalHoloClean/last/")
    '''
    '''
    all_matrix_test = None
    feature_name_list_deep = ['deep ' + str(dfeature) for dfeature in range(all_matrix_train_deep.shape[1])]
    all_matrix_train = all_matrix_train_deep
    #all_matrix_train = hstack((all_matrix_train, all_matrix_train_deep)).tocsr()
    feature_name_list = feature_name_list_deep
    #feature_name_list.extend(feature_name_list_deep)
    '''


    if use_metadata:
        all_matrix_train, all_matrix_test, feature_name_list = add_metadata_features(dataSet, train_fraction, all_matrix_train, all_matrix_test, feature_name_list)



    #loss_zip = np.load("/home/felix/SequentialPatternErrorDetection/mlp_features/loss_zip.npy")
    #all_matrix_train = hstack((all_matrix_train, loss_zip)).tocsr()
    #feature_name_list.append('ZIP_loss')


    #print type(all_matrix_train.todense())
    #np.save("names", feature_name_list)
    #np.save("data", all_matrix_train.todense())


    split_id = int(dataSet.shape[0] * train_fraction)

    print("features: %s seconds ---" % (time.time() - start_time))

    data_result = []

    column_id = 0

    feature_matrix = all_matrix_train

    print "type: " + str(type(all_matrix_train))
    print "isnan: " + str(np.isnan(all_matrix_train.data).any())
    print "finite: " + str(np.isfinite(all_matrix_train.data).all())

    from ml.active_learning.classifier.XGBoostClassifier import XGBoostClassifier
    classifier = XGBoostClassifier(all_matrix_train, all_matrix_test)
    # classifier = LinearSVMClassifier(all_matrix_train, all_matrix_test)
    # classifier = LogisticRegressionClassifier(all_matrix_train, all_matrix_test)
    #from ml.classifier.NaiveBayesClassifier import NaiveBayesClassifier
    #classifier = NaiveBayesClassifier(all_matrix_train, all_matrix_test)


    all_error_status = np.zeros((all_matrix_train.shape[0], dataSet.shape[1]), dtype=bool)
    if all_matrix_test != None:
        all_error_status_test = np.zeros((all_matrix_test.shape[0], dataSet.shape[1]), dtype=bool)


    save_fscore = []
    save_precision = []
    save_recall = []
    save_labels = []
    save_certainty = []
    save_fscore_general = []
    save_time = []

    our_params = {}
    train = {}
    train_target = {}
    y_pred = {}
    certainty = {}
    certainty_new = {}
    min_certainty = {}
    final_gb = {}
    res = {}
    prediction_change = {}

    zero_change_count = {}

    for round in range(label_iterations * dataSet.shape[1]):
        print "round: " + str(round)

        '''
        #check if column is in queue
        if column_id in zero_change_count and zero_change_count[column_id] >= 1:
            column_id = go_to_next_column(column_id, min_certainty)
            continue
        '''



        #switch to column
        target_run, target_test = getTarget(dataSet, column_id, train_fraction)


        if round < dataSet.shape[1]:
            start_time = time.time()

            num_errors = 2
            train[column_id], train_target[column_id] = create_user_start_data(feature_matrix, target_run, num_errors)
            if train[column_id] == None:
                certainty[column_id] = 1.0
                column_id = go_to_next_column(column_id, min_certainty, dataSet)
                continue

            print "Number of errors in training: " + str(np.sum(train_target[column_id]))
            print("clustering: %s seconds ---" % (time.time() - start_time))

            #cross-validation
            start_time = time.time()
            classifier.run_cross_validation(train[column_id], train_target[column_id], num_errors, column_id)
            print("cv: %s seconds ---" % (time.time() - start_time))

            min_certainty[column_id] = 0.0

        else:
            if train[column_id] == None:
                column_id = go_to_next_column(column_id, min_certainty, dataSet)
                continue

            if column_id in certainty:
                min_certainty[column_id] = np.min(np.absolute(y_pred[column_id] - 0.5))
            else:
                min_certainty[column_id] = 0.0


            # important stop criterion
            diff = np.absolute(y_pred[column_id] - 0.5)
            print "min certainty: " + str(np.min(diff))
            '''
            if np.min(diff) >= 0.4: #and certainty[column_id] > 0.9:
                column_id = go_to_next_column(column_id, min_certainty)
                continue
            '''


            train[column_id], train_target[column_id], certainty[column_id] = create_next_data(train[column_id],
                                                                                               train_target[column_id],
                                                                                               feature_matrix,
                                                                                               target_run,
                                                                                               y_pred[column_id],
                                                                                               step_size,
                                                                                               dataSet,
                                                                                               column_id,
                                                                                               user_error_probability)

            certainty_new[column_id] = certainty[column_id]

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
            if round < dataSet.shape[1] * cross_validation_rounds:
                our_params[column_id] = classifier.run_cross_validation(train[column_id], train_target[column_id], num_errors)
            #print("cv: %s seconds ---" % (time.time() - start_time))

            eval_scores = classifier.run_cross_validation_eval(train[column_id], train_target[column_id], 7, column_id)

        start_time = time.time()
        # train
        # predict
        y_pred[column_id], res_new = classifier.train_predict(train[column_id], train_target[column_id], column_id)

        if column_id in res:
            no_change_0, no_change_1, change_0_to_1, change_1_to_0 = compare_change(res[column_id], res_new)

            print "no change 0: " + str(no_change_0) + " no change 1: " + str(no_change_1) + " sum no change: " + str(no_change_0 + no_change_1)
            print "change 0 ->1: " + str(change_0_to_1) + " change 1->0: " + str(change_1_to_0) + " sum change: " + str(change_0_to_1 + change_1_to_0)

        res[column_id] = res_new
        all_error_status[:, column_id] = res[column_id]
        print("train & predict: %s seconds ---" % (time.time() - start_time))

        if all_matrix_test != None:
            y_pred_test, res_gen = classifier.predict(column_id)
            all_error_status_test[:, column_id] = res_gen

        #visualize_model(dataSet, column_id, final_gb, feature_name_list, train, target_run, res)


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
        save_precision.append(precision_score(dataSet.matrix_is_error[0:split_id,:].flatten(), all_error_status.flatten()))
        save_recall.append(recall_score(dataSet.matrix_is_error[0:split_id, :].flatten(), all_error_status.flatten()))

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


            print "cross-val-score: " + str(np.mean(eval_scores))

            str_hist += "," + str(np.mean(eval_scores))

            str_hist += "," + str(no_change_0)
            str_hist += "," + str(no_change_1)
            str_hist += "," + str(change_0_to_1)
            str_hist += "," + str(change_1_to_0)

            prediction_change[column_id] = change_0_to_1 + change_1_to_0

            if (change_0_to_1 + change_1_to_0) < 0.001 and certainty[column_id] > 0.4:
                if column_id in zero_change_count:
                    zero_change_count[column_id] = zero_change_count[column_id] + 1
                else:
                    zero_change_count[column_id] = 1
            else:
                zero_change_count[column_id] = 0

            print "current zero count: " + str(zero_change_count[column_id])


            distinct_values_fraction = float(len(dataSet.dirty_pd[dataSet.dirty_pd.columns[column_id]].unique())) / float(dataSet.shape[0])


            f.write(str(distinct_values_fraction) + ',' + str(train[column_id].shape[0]) + ',' + str(certainty[column_id]) + ',' + str(np.min(np.absolute(y_pred[column_id] - 0.5))) + str_hist + ',' + str(f1_score(target_run, res[column_id])) + '\n')


        column_id = go_to_next_column_certainty(column_id, certainty_new, dataSet)

        current_runtime = (time.time() - total_start_time)
        print("iteration end: %s seconds ---" % current_runtime)
        save_time.append(current_runtime)

    print "fscore_train: " + str(save_fscore)
    print "precision_train: " + str(save_precision)
    print "recall_train: " + str(save_recall)
    print "fscore_test: " + str(save_fscore_general)
    print "number labelled cells: " + str(save_labels)
    print "certainty: " + str(save_certainty)
    print "time: " + str(save_time)
