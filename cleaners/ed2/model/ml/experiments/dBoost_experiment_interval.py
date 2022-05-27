from ml.datasets.flights.FlightHoloClean import FlightHoloClean
from ml.datasets.blackOak.BlackOakDataSetUppercase import BlackOakDataSetUppercase
from ml.datasets.hospital.HospitalHoloClean import HospitalHoloClean
from ml.datasets.MoviesMohammad.Movies import Movies
from ml.datasets.RestaurantMohammad.Restaurant import Restaurant
from ml.datasets.BeersMohammad.Beers import Beers
from ml.datasets.Citations.Citation import Citation
from ml.datasets.salary_data.Salary import Salary

import time
from ml.tools.dboost.TestDBoost import test_multiple_sizes_hist
from ml.tools.dboost.TestDBoost import test_multiple_sizes_gaussian
from ml.tools.dboost.TestDBoost import test_multiple_sizes_mixture

from ml.configuration.Config import Config
import numpy as np
import os
import numpy as np

from ml.datasets.flights.FlightHoloClean import FlightHoloClean
from ml.tools.dboost.TestDBoost import test_multiple_sizes_gaussian
from ml.tools.dboost.TestDBoost import toLatex
from ml.configuration.Config import Config
import os
import time
import multiprocessing as mp
import sys


#data_list = [FlightHoloClean, BlackOakDataSetUppercase, HospitalHoloClean, Restaurant, Movies, Beers, Citation]
data_list = [HospitalHoloClean, Restaurant, Movies, Beers]



steps = 100
N = 10

dBoost_methods = [test_multiple_sizes_gaussian, test_multiple_sizes_mixture, test_multiple_sizes_hist]

defined_range_labeled_cells = {}
defined_range_labeled_cells[FlightHoloClean.name] = range(10, 121, 10)
defined_range_labeled_cells[BlackOakDataSetUppercase.name] = range(10, 151, 10)
defined_range_labeled_cells[Beers.name] = range(10, 341, 10)
defined_range_labeled_cells[HospitalHoloClean.name] = range(10, 501, 10)
defined_range_labeled_cells[Movies.name] = range(10, 301, 10)
defined_range_labeled_cells[Restaurant.name] = range(10, 501, 10)


def run_dboost(dBoost, data, defined_range_labeled_cells, steps, N):
    ts = time.time()

    path_folder = Config.get("logging.folder") + "/out/dboost_interval"

    if not os.path.exists(path_folder):
        os.makedirs(path_folder)

    log_file = path_folder + "/" + str(data.name) + "_time_" + str(ts) + "_dBoost_" + dBoost.func_name + ".txt"

    sizes = np.array(defined_range_labeled_cells, dtype=float)  # in cells

    dirty_column_fraction = data.get_number_dirty_columns() / float(data.shape[1])
    sizes /= dirty_column_fraction  # cells converted
    sizes /= float(data.shape[1])  # cells to rows
    row_sizes = np.array(sizes, dtype=int)  # in rows

    avg_times, avg_fscores, avg_precision, avg_recall, std_fscores, std_precision, std_recall = dBoost(
        data, steps, N, row_sizes, log_file)

    toLatex(defined_range_labeled_cells, avg_times, avg_fscores, avg_precision, avg_recall, std_fscores,
            std_precision, std_recall, log_file)


def run_multi(params):
    try:
        return run_dboost(**params)
    except:
        print("Unexpected error:" + str(sys.exc_info()[0]))


my_array = []
for dataset in data_list:
    data = dataset()

    for dBoost in dBoost_methods:
        my_dict = {}
        my_dict['data'] = data
        my_dict['defined_range_labeled_cells'] = defined_range_labeled_cells[data.name]
        my_dict['steps'] = steps
        my_dict['N'] = N
        my_dict['dBoost'] = dBoost

        my_array.append(my_dict)


pool = mp.Pool(processes=12)
results = pool.map(run_multi, my_array)






