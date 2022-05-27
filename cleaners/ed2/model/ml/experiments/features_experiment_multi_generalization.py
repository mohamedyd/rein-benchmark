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


path_folder = Config.get("logging.folder") + "/out/generalization"
if not os.path.exists(path_folder):
    os.makedirs(path_folder)


#data_list = [FlightHoloClean, BlackOakDataSetUppercase, HospitalHoloClean, Movies, Restaurant, Citation, Beers, Salary]
data_list = [FlightHoloClean, BlackOakDataSetUppercase, Movies, Beers]


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

#parameters.append({'use_metadata_only': False, 'use_metadata': False, 'correlationFeatures': False, 'use_lstm_only': True, 'use_lstm': True}) #LSTM


#store ed2
#parameters.append({'use_word2vec': True, 'use_word2vec_only': False, 'w2v_size': 100, 'output_detection_result': 148}) #char unigrams + meta data + correlation + word2vec
#store activeClean
#parameters.append({'use_metadata_only': False, 'correlationFeatures': False, 'use_metadata': False, 'use_active_clean': True, 'use_activeclean_only': True, 'output_detection_result': 148}) #active clean
#store boostclean
#parameters.append({'use_metadata_only': False, 'correlationFeatures': False, 'use_metadata': False, 'use_word2vec': True, 'use_word2vec_only': True, 'w2v_size': 100, 'use_boostclean_metadata': True, 'output_detection_result': 148}) #boostclean

#ed test 0.1
parameters.append({'use_word2vec': True, 'use_word2vec_only': False, 'w2v_size': 100, 'train_fraction': 0.2})
parameters.append({'use_word2vec': True, 'use_word2vec_only': False, 'w2v_size': 100, 'train_fraction': 0.4})
parameters.append({'use_word2vec': True, 'use_word2vec_only': False, 'w2v_size': 100, 'train_fraction': 0.6})
parameters.append({'use_word2vec': True, 'use_word2vec_only': False, 'w2v_size': 100, 'train_fraction': 0.8})


#LSTM

feature_names = ['20 static data',
'40 static data',
'60 static data',
'80 static data']

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


#results = [run(**my_array[0])]
pool = mp.Pool(processes=20)
results = pool.map(run_multi, my_array)



def process_fscore(fscore_lists, label):
    latex = ""
    if len(fscore_lists) > 0:
        f_matrix = np.matrix(fscore_lists)
        average = list(np.mean(f_matrix, axis=0).A1)

        lower_quartile = np.percentile(f_matrix, 25, axis=0)
        median = np.percentile(f_matrix, 50, axis=0)
        upper_quartile = np.percentile(f_matrix, 75, axis=0)
        minimum = np.min(f_matrix, axis=0).A1
        maximum = np.max(f_matrix, axis=0).A1

        for i in range(len(lower_quartile)):
            latex += "\\boxplotlabels{" + str(label[i]) + "}{" + str(median[i]) + "}{" + str(
                lower_quartile[i]) + "}{" + str(upper_quartile[i]) + "}{" + str(minimum[i]) + "}{" + str(
                maximum[i]) + "}\n"

        latex += "\n\n\n"

        latex += "\\addplot+[mark=none] coordinates{"

        for c in range(len(average)):
            latex += "(" + str(label[c]) + "," + str(average[c]) + ")"
        latex += "};\n"

    return latex


for r_i in range(len(results)):
    r = results[r_i]
    data = my_array[r_i]['dataSet']

    ts = time.time()
    my_file = open(
        path_folder + '/labels_experiment_data_' + str(data.name) + '_' + str(fnames[r_i]) + '_' + my_array[r_i]['classifier_model'].name + '_time_' + str(ts) + '.csv', 'w+')

    fscore_lists = r['fscore']
    if len(fscore_lists) > 0:
        label = r['labels']
        all_precision = r['precision']
        all_recall = r['recall']
        all_time = r['time']

        my_file.write("NORMAL:\n\n")
        my_file.write(process_fscore(fscore_lists, label))

        my_file.write("TEST:\n\n")
        my_file.write(process_fscore(r['fscore_test'], label))

        my_file.write("\n\n")

        avg_prec = list(np.mean(np.matrix(all_precision), axis=0).A1)
        avg_rec = list(np.mean(np.matrix(all_recall), axis=0).A1)
        avg_f = list(np.mean(np.matrix(fscore_lists), axis=0).A1)

        std_prec = list(np.std(np.matrix(all_precision), axis=0).A1)
        std_rec = list(np.std(np.matrix(all_recall), axis=0).A1)
        std_f = list(np.std(np.matrix(fscore_lists), axis=0).A1)


        avg_time = list(np.mean(np.matrix(all_time), axis=0).A1)

        for i in range(len(label)):
            my_file.write(
                str(label[i]) + ", time: " + str(avg_time[i])
                + ", AVG_Precision: " + str(avg_prec[i]) + ", STD_Precision: " + str(std_prec[i])
                + ", AVG Recall: " + str(avg_rec[i]) + ", STD_Recall: " + str(std_rec[i])
                + ", AVG Fscore: " + str(avg_f[i]) + ", STD Fscore: " + str(std_f[i])
                + "\n")

        my_file.write("\n\n\nAVG Precision: " + str(avg_prec))
        my_file.write("\n\n\nSTD Precision: " + str(std_prec))
        my_file.write("\nAll Precision: " + str(all_precision))

        my_file.write("\n\nAVG Recall: " + str(avg_rec))
        my_file.write("\n\nSTD Recall: " + str(std_rec))
        my_file.write("\nAll Recall: " + str(all_recall))


        my_file.write("\n\nLabels: " + str(label))


    else:
        my_file.write("error" + str(r['error']))

    my_file.close()



