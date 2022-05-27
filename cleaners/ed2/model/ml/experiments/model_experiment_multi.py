from ml.classes.active_learning_total_uncertainty_error_correlation_lib import run_multi


from ml.datasets.flights.FlightHoloClean import FlightHoloClean
from ml.datasets.blackOak.BlackOakDataSetUppercase import BlackOakDataSetUppercase
from ml.datasets.hospital.HospitalHoloClean import HospitalHoloClean
from ml.datasets.MoviesMohammad.Movies import Movies
from ml.datasets.RestaurantMohammad.Restaurant import Restaurant
from ml.datasets.BeersMohammad.Beers import Beers
from ml.datasets.Citations.Citation import Citation

from ml.active_learning.classifier.XGBoostClassifier import XGBoostClassifier
from ml.active_learning.classifier.LinearSVMClassifier import LinearSVMClassifier
from ml.active_learning.classifier.NaiveBayesClassifier import NaiveBayesClassifier

from ml.datasets.salary_data.Salary import Salary
import numpy as np

from ml.configuration.Config import Config
import os
import time


path_folder = Config.get("logging.folder") + "/out/model"
if not os.path.exists(path_folder):
    os.makedirs(path_folder)


data_list = [FlightHoloClean, BlackOakDataSetUppercase, HospitalHoloClean, Movies, Restaurant, Citation, Beers, Salary]
classifiers = [XGBoostClassifier,LinearSVMClassifier, NaiveBayesClassifier]

parameters = []




my_array = []
for dataset in data_list:
    data = dataset()
    for classifier in classifiers:
        for param_i in range(len(parameters)):
            my_dict = parameters[param_i].copy()
            my_dict['dataSet'] = data
            my_dict['classifier_model'] = classifier
            my_dict['checkN'] = 10

            my_array.append(my_dict)

import multiprocessing as mp
pool = mp.Pool(processes=10)

results = pool.map(run_multi, my_array)

for r_i in range(len(results)):
    r = results[r_i]
    data = my_array[r_i]['dataSet']

    fscore_lists = r['fscore']
    label = r['labels']
    all_precision = r['precision']
    all_recall = r['recall']
    all_time = r['time']

    f_matrix = np.matrix(fscore_lists)

    average = list(np.mean(f_matrix, axis=0).A1)

    latex = ""
    latex += "\\addplot+[mark=none] coordinates{"

    for c in range(len(average)):
        latex += "(" + str(label[c]) + "," + str(average[c]) + ")"
    latex += "};\n"


    ts = time.time()

    my_file = open( path_folder + '/labels_experiment_data_' + str(data.name) + "_" + classifier.name  + "_time_" + str(ts) + '.csv', 'w+')
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

    my_file.close()


