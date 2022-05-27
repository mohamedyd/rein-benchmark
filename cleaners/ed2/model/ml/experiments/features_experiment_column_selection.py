from ml.classes.active_learning_total_uncertainty_error_correlation_lib import run_multi
from ml.classes.active_learning_total_uncertainty_error_correlation_lib import run
import multiprocessing as mp

from ml.datasets.flights.FlightHoloClean import FlightHoloClean
from ml.datasets.blackOak.BlackOakDataSetUppercase import BlackOakDataSetUppercase
from ml.datasets.hospital.HospitalHoloClean import HospitalHoloClean
from ml.datasets.MoviesMohammad.Movies import Movies
from ml.datasets.RestaurantMohammad.Restaurant import Restaurant
from ml.datasets.BeersMohammad.Beers import Beers
from ml.datasets.Citations.Citation import Citation
from ml.datasets.salary_data.Salary import Salary

from ml.active_learning.classifier.XGBoostClassifier import XGBoostClassifier
from ml.active_learning.classifier.LinearSVMClassifier import LinearSVMClassifier
from ml.active_learning.classifier.NaiveBayesClassifier import NaiveBayesClassifier

import numpy as np

from ml.configuration.Config import Config
import os
import time


path_folder = Config.get("logging.folder") + "/out/column_selection_beers"
if not os.path.exists(path_folder):
    os.makedirs(path_folder)


#data_list = [FlightHoloClean, BlackOakDataSetUppercase, HospitalHoloClean, Movies, Restaurant, Citation, Beers, Salary]
data_list = [Restaurant]

parameters = []
#parameters.append({'use_metadata': False, 'correlationFeatures': False}) #char unigrams
#parameters.append({'use_metadata': False, 'correlationFeatures': False, 'is_word': True}) #word unigrams
#parameters.append({'use_metadata_only': True, 'correlationFeatures': False}) #metadata
#parameters.append({'use_metadata': False, 'ngrams': 2, 'correlationFeatures': False}) #char unigrams + bigrams
#parameters.append({'correlationFeatures': False}) #char unigrams + meta data
#parameters.append({}) #char unigrams + meta data + correlation
#parameters.append({'use_word2vec': True, 'use_word2vec_only': False, 'w2v_size': 100}) #char unigrams + meta data + correlation + word2vec
#parameters.append({'use_metadata_only': False, 'correlationFeatures': False, 'use_metadata': False, 'use_word2vec': True, 'use_word2vec_only': True, 'w2v_size': 100}) #word2vec
#parameters.append({'use_metadata_only': False, 'correlationFeatures': False, 'use_metadata': False, 'use_active_clean': True, 'use_activeclean_only': True}) #active clean
#parameters.append({'use_metadata_only': False, 'correlationFeatures': False, 'use_metadata': False, 'use_word2vec': True, 'use_word2vec_only': True, 'w2v_size': 100, 'use_boostclean_metadata': True}) #boostclean


parameters.append({'use_word2vec': True, 'use_word2vec_only': False, 'w2v_size': 100, 'number_of_round_robin_rounds': 10000, 'use_min_certainty_column_selection': False, 'label_iterations': 12}) #round robin
#parameters.append({'use_word2vec': True, 'use_word2vec_only': False, 'w2v_size': 100, 'use_random_column_selection': True, 'use_min_certainty_column_selection': False}) #random
#parameters.append({'use_word2vec': True, 'use_word2vec_only': False, 'w2v_size': 100, 'use_max_pred_change_column_selection': True, 'use_min_certainty_column_selection': False}) #prediction change
#parameters.append({'use_word2vec': True, 'use_word2vec_only': False, 'w2v_size': 100, 'use_max_error_column_selection': True, 'use_min_certainty_column_selection': False}) #max error


#LSTM

feature_names = ['round_robin',
                 #'random',
                 #'prediction_change',
                 #'max_error'
                 ]

#classifiers = [XGBoostClassifier, LinearSVMClassifier, NaiveBayesClassifier]
classifiers = [XGBoostClassifier]

fnames = []
my_array = []
for dataset in data_list:
    data = dataset()

    for param_i in range(len(parameters)):
        for classifier in classifiers:
            my_dict = parameters[param_i].copy()
            my_dict['dataSet'] = data
            my_dict['classifier_model'] = classifier
            my_dict['checkN'] = 10
            fnames.append(feature_names[param_i])

            my_array.append(my_dict)

pool = mp.Pool(processes=13)
results = pool.map(run_multi, my_array)


for r_i in range(len(results)):
    r = results[r_i]
    data = my_array[r_i]['dataSet']

    fscore_lists = r['fscore']

    ts = time.time()
    my_file = open(
        path_folder + '/labels_experiment_data_' + str(data.name) + '_' + str(fnames[r_i]) + '_' + my_array[r_i]['classifier_model'].name + '_time_' + str(ts) + '.csv', 'w+')

    if len(fscore_lists) > 0:
        label = r['labels']
        all_precision = r['precision']
        all_recall = r['recall']
        all_time = r['time']

        f_matrix = np.matrix(fscore_lists)

        average = list(np.mean(f_matrix, axis=0).A1)

        latex = ""

        lower_quartile = np.percentile(f_matrix, 25, axis=0)
        median = np.percentile(f_matrix, 50, axis=0)
        upper_quartile = np.percentile(f_matrix, 75, axis=0)
        minimum = np.min(f_matrix, axis=0).A1
        maximum = np.max(f_matrix, axis=0).A1

        latex = ""

        for i in range(len(lower_quartile)):
            latex += "\\boxplotlabels{" + str(label[i]) + "}{" + str(median[i]) + "}{" + str(
                lower_quartile[i]) + "}{" + str(upper_quartile[i]) + "}{" + str(minimum[i]) + "}{" + str(
                maximum[i]) + "}\n"

        latex += "\n\n\n"

        latex += "\\addplot+[mark=none] coordinates{"

        for c in range(len(average)):
            latex += "(" + str(label[c]) + "," + str(average[c]) + ")"
        latex += "};\n"


        my_file.write(latex)

        my_file.write("\n\n")

        avg_prec = list(np.mean(np.matrix(all_precision), axis=0).A1)
        avg_rec = list(np.mean(np.matrix(all_recall), axis=0).A1)
        avg_f = list(np.mean(np.matrix(fscore_lists), axis=0).A1)
        avg_time = list(np.mean(np.matrix(all_time), axis=0).A1)

        for i in range(len(label)):
            my_file.write(
                str(label[i]) + "," + str(avg_time[i]) + "," + str(avg_prec[i]) + "," + str(avg_rec[i]) + "," + str(
                    avg_f[i]) + "\n")

        my_file.write("\n\n\nAVG Precision: " + str(avg_prec))
        my_file.write("\nAll Precision: " + str(all_precision))
        my_file.write("\n\nAVG Recall: " + str(avg_rec))
        my_file.write("\nAll Recall: " + str(all_recall))
        my_file.write("\n\nLabels: " + str(label))


    else:
        my_file.write("error" + str(r['error']))

    my_file.close()



