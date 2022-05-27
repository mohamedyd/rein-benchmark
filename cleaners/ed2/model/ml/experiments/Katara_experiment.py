from ml.datasets.flights.FlightHoloClean import FlightHoloClean
from ml.datasets.blackOak.BlackOakDataSetUppercase import BlackOakDataSetUppercase
from ml.datasets.hospital.HospitalHoloClean import HospitalHoloClean
from ml.datasets.MoviesMohammad.Movies import Movies
from ml.datasets.RestaurantMohammad.Restaurant import Restaurant
from ml.datasets.BeersMohammad.Beers import Beers
from ml.datasets.salary_data.Salary import Salary
from ml.datasets.Citations.Citation import Citation
from ml.datasets.salary_data.Salary import Salary
import random
import time
import csv
import multiprocessing as mp
from ml.configuration.Config import Config
import numpy as np
import os
from ml.tools.katara_new.Katara import Katara


path_folder = Config.get("logging.folder") + "/out/katara"

if not os.path.exists(path_folder):
    os.makedirs(path_folder)

path_folder_tmp = Config.get("logging.folder") + "/out/katara_tmp"
if not os.path.exists(path_folder_tmp):
    os.makedirs(path_folder_tmp)

data_list = [FlightHoloClean]



def run_katara(data):
    ts = time.time()
    tmp_katara_out = path_folder_tmp + "/katara_time_" + str(ts) + "_" + str(random.randint(1,100000)) + "_KATARA_" + ".txt"

    dirty_dataset = path_folder_tmp + '/dirty_dataset_' + str(ts) + '_' + str(random.randint(1, 100000)) + '.csv'
    dirty_df = data.dirty_pd.copy()

    for column_i in range(dirty_df.shape[1]):
        dirty_df[dirty_df.columns[column_i]] = dirty_df[dirty_df.columns[column_i]].apply(lambda x: x.upper())

    dirty_df.to_csv(dirty_dataset, index=False, encoding='utf-8')

    start_time = time.time()

    command = "cd " + Config.get("abstractionlayer.folder") + "/\n" + "python2 cleaning_api.py " + dirty_dataset + " " + tmp_katara_out
    print command
    os.system(command)

    return_dict= {}
    return_dict['output'] = tmp_katara_out
    return_dict['time'] = time.time() - start_time

    return return_dict


my_array = []
for dataset in data_list:
    data = dataset()
    my_array.append(data)

pool = mp.Pool(processes=1)

results = pool.map(run_katara, my_array)

for r_i in range(len(results)):
    ts = time.time()
    log_file = path_folder + "/" + str(my_array[r_i].name) + "_time_" + str(ts) + "_KATARA_" + ".txt"
    data = my_array[r_i]

    my_file = open(log_file, 'w+')
    if os.stat(results[r_i]['output']).st_size == 0:
        my_file.write("KATARA did not work\n")
    else:
        tool = Katara(results[r_i]['output'], data)
        my_file.write("Fscore: " + str(tool.calculate_total_fscore()) + "\n")
        my_file.write("Precision: " + str(tool.calculate_total_precision()) + "\n")
        my_file.write("Recall: " + str(tool.calculate_total_recall()) + "\n\n")

        for c in range(data.shape[1]):
            my_file.write(str(data.clean_pd.columns[c]) + ": " + str(tool.calculate_recall_by_column(c)) + '\n')
            my_file.write(str(data.clean_pd.columns[c]) + ": " + str(tool.calculate_precision_by_column(c)) + '\n')

    my_file.write("\n\nRuntime: " + str(results[r_i]['time']) + "\n")
    my_file.close()