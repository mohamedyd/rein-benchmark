from ml.classes.active_learning_total_uncertainty_error_correlation_lib import run_multi
import time

from ml.datasets.flights.FlightHoloClean import FlightHoloClean
from ml.datasets.blackOak.BlackOakDataSetUppercase import BlackOakDataSetUppercase
from ml.datasets.hospital.HospitalHoloClean import HospitalHoloClean
from ml.datasets.MoviesMohammad.Movies import Movies
from ml.datasets.RestaurantMohammad.Restaurant import Restaurant
from ml.datasets.BeersMohammad.Beers import Beers
from ml.datasets.Citations.Citation import Citation
from ml.datasets.salary_data.Salary import Salary

from ml.active_learning.classifier.XGBoostClassifier import XGBoostClassifier
import numpy as np

from ml.configuration.Config import Config
import os


path_folder = Config.get("logging.folder") + "/out/labels"
if not os.path.exists(path_folder):
    os.makedirs(path_folder)


data_list = [FlightHoloClean, BlackOakDataSetUppercase, HospitalHoloClean, Movies, Restaurant, Citation, Beers, Salary]

params = {'use_word2vec': True,
          'use_word2vec_only': False,
          'w2v_size': 20}


classifier = XGBoostClassifier

my_array = []
for dataset in data_list:
    data = dataset()
    my_dict = params.copy()
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

    lower_quartile = np.percentile(f_matrix, 25, axis=0)
    median = np.percentile(f_matrix, 50, axis=0)
    upper_quartile = np.percentile(f_matrix, 75, axis=0)
    minimum = np.min(f_matrix, axis=0).A1
    maximum = np.max(f_matrix, axis=0).A1

    latex = ""

    for i in range(len(lower_quartile)):
        latex += "\\boxplotlabels{"+ str(label[i]) +"}{"+ str(median[i]) +"}{"+ str(lower_quartile[i]) +"}{"+ str(upper_quartile[i]) +"}{"+ str(minimum[i]) +"}{"+ str(maximum[i]) +"}\n"


    ts = time.time()

    my_file = open(path_folder + '/labels_experiment_data_' + str(data.name) + "_time_" + str(ts) + '.csv', 'w+')
    my_file.write(latex)
    my_file.write("\n\n")

    avg_prec = list(np.mean(np.matrix(all_precision), axis=0).A1)
    avg_rec = list(np.mean(np.matrix(all_recall), axis=0).A1)
    avg_f = list(np.mean(np.matrix(fscore_lists), axis=0).A1)
    avg_time = list(np.mean(np.matrix(all_time), axis=0).A1)

    for i in range(len(label)):
        my_file.write(str(label[i]) + "," + str(avg_time[i]) + "," + str(avg_prec[i]) + "," + str(avg_rec[i]) + "," + str(avg_f[i]) + "\n" )

    my_file.write("\n\n\nAVG Precision: " + str(avg_prec))
    my_file.write("\nAll Precision: " + str(all_precision))
    my_file.write("\n\nAVG Recall: " + str(avg_rec))
    my_file.write("\nAll Recall: " + str(all_recall))
    my_file.write("\n\nLabels: " + str(label))

    my_file.close()



