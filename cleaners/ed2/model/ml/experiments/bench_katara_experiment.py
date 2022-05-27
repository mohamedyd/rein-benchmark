import multiprocessing as mp
import time
from ml.configuration.Config import Config
import os
from ml.tools.katara_new.Katara import Katara
from os import listdir
from os.path import isfile, join
from ml.datasets.csv_dataset.CsvDataset import CsvDataset
from ml.experiments.Katara_experiment import run_katara


path_folder = Config.get("logging.folder") + "/out/bench_katara"
if not os.path.exists(path_folder):
    os.makedirs(path_folder)

available_datasets_dir = Config.get("bench.folder") + "/data"

datasets_folders = os.listdir(available_datasets_dir)

print datasets_folders

for dataset_dir in datasets_folders:
    # get all files in folder
    mypath = available_datasets_dir + "/" + dataset_dir
    dirtyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    groundtruth_dirs = [x[0] for x in os.walk(mypath)]
    groundtruth_dir = ""
    for file_i in groundtruth_dirs:
        if '_GT' in file_i:
            groundtruth_dir = file_i

    cleanfile = [f for f in listdir(groundtruth_dir) if isfile(join(groundtruth_dir, f))][0]


    # create datasets
    data_list = []
    for dirty_file in dirtyfiles:
        print dirty_file
        data_list.append(
            CsvDataset(groundtruth_dir + "/" + cleanfile, mypath + "/" + dirty_file, dirty_file.split('.')[0]))

    print data_list


my_array = []
for dataset in data_list:
    data = dataset()
    my_array.append(data)

pool = mp.Pool(processes=26)

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