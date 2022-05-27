from ml.datasets.csv_dataset.CsvDataset import CsvDataset

import time
from ml.tools.dboost.TestDBoost import test_multiple_sizes_hist
from ml.tools.dboost.TestDBoost import test_multiple_sizes_gaussian
from ml.tools.dboost.TestDBoost import test_multiple_sizes_mixture

from ml.configuration.Config import Config
import os
from os import listdir
from os.path import isfile, join


path_folder = Config.get("logging.folder") + "/out/dboost_bench"

if not os.path.exists(path_folder):
    os.makedirs(path_folder)

available_datasets_dir = Config.get("bench.folder") + "/data"

datasets_folders = os.listdir(available_datasets_dir)

print datasets_folders

for dataset_dir in datasets_folders:
    #get all files in folder
    mypath = available_datasets_dir + "/" + dataset_dir
    dirtyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    groundtruth_dirs = [x[0] for x in os.walk(mypath)]
    groundtruth_dir = ""
    for file_i in groundtruth_dirs:
        if '_GT' in file_i:
            groundtruth_dir = file_i

    cleanfile = [f for f in listdir(groundtruth_dir) if isfile(join(groundtruth_dir, f))][0]
    
    #create datasets
    data_list = []
    for dirty_file in dirtyfiles:
        data_list.append(CsvDataset(groundtruth_dir + "/" + cleanfile, mypath + "/" + dirty_file, dirty_file.split('.')[0]))
    
    print data_list
    
    steps = 100
    N = 1
    
    
    dBoost_methods = [test_multiple_sizes_hist, test_multiple_sizes_gaussian, test_multiple_sizes_mixture]
    
    for dataset in data_list:
        data = dataset
        rows_number= data.shape[0]
    
        for dBoost in dBoost_methods:
            ts = time.time()
            log_file = path_folder + "/" + str(data.name) + "_time_" + str(ts) + "_dBoost_" + dBoost.func_name  + ".txt"
            dBoost(data, steps, N, [rows_number], log_file)

        
        


'''
str(best_params) + ", " +
str(runtime) + ", " +
str(precision) + ", " +
str(recall) + ", " +
str(best_fscore_all)
'''
