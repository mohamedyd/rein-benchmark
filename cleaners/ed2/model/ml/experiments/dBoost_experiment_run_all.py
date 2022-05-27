from ml.datasets.flights.FlightHoloClean import FlightHoloClean
from ml.datasets.blackOak.BlackOakDataSetUppercase import BlackOakDataSetUppercase
from ml.datasets.hospital.HospitalHoloClean import HospitalHoloClean
from ml.datasets.MoviesMohammad.Movies import Movies
from ml.datasets.RestaurantMohammad.Restaurant import Restaurant
from ml.datasets.BeersMohammad.Beers import Beers
from ml.datasets.Citations.Citation import Citation
from ml.datasets.salary_data.Salary import Salary

import numpy as np

from ml.datasets.flights.FlightHoloClean import FlightHoloClean
from ml.tools.dboost.TestDBoost import run_histogram_stat
from ml.tools.dboost.TestDBoost import run_gaussian_stat
from ml.tools.dboost.TestDBoost import run_mixture_stat
from ml.configuration.Config import Config
import os
import time
import multiprocessing as mp
import itertools
from ml.tools.dboost.DBoostMe import DBoostMe

#data_list = [FlightHoloClean, BlackOakDataSetUppercase, HospitalHoloClean, Restaurant, Movies, Beers, Citation]
data_list = [FlightHoloClean, BlackOakDataSetUppercase, Movies, Beers]

models = [run_gaussian_stat, run_histogram_stat, run_mixture_stat]

processes = 20



def create_grid(model):
    paramater_dict = {}

    if model == run_gaussian_stat:
        steps = 10
        paramater_dict['gaussian'] = [((4.0 - 0.0) / steps) * step for step in range(steps)]
        paramater_dict['statistical'] = [((4.0 - 0.0) / steps) * step for step in range(steps)]
    if model ==run_mixture_stat:
        paramater_dict['n_subpops'] = [1, 2, 3]
        paramater_dict['threshold'] = [0.01, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
        paramater_dict['statistical'] = [0.5]
    if model ==run_histogram_stat:
        paramater_dict['peak'] = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
        paramater_dict['outlier'] = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        paramater_dict['statistical'] = [0.5]

    return paramater_dict


def generate_dBoost_result_file_name(model, data, parameter_grid_dict, keys):
    path = Config.get("logging.folder") + "/out/dboost_results"

    if not os.path.exists(path):
        os.makedirs(path)

    dBoost_result = path + "/dboost_" + str(model.__name__) + "_" + str(data.name)
    for p_i in range(len(keys)):
        dBoost_result += '_' + str(keys[p_i]) + '_' + str(parameter_grid_dict[keys[p_i]])
    dBoost_result += '.npy'

    return dBoost_result


def create_runs(datasets, models=[run_gaussian_stat, run_histogram_stat, run_mixture_stat]):
    my_array = []
    result_counter = 0

    for dataset in datasets:
        d = dataset()
        sample_file = "/tmp/data_sample_" + str(d.name)  + ".csv"
        d.dirty_pd.to_csv(sample_file, index=False, encoding='utf8')

        for model in models:
            paramater_dict = create_grid(model)
            id_list = []
            key_id = []
            for key in paramater_dict.keys():
                id_list.append(range(len(paramater_dict[key])))
                key_id.append(key)

            permutations = list(itertools.product(*id_list))

            #print permutations
            #print len(permutations)

            for p in permutations:
                my_dict = {}
                my_dict['model'] = model
                my_dict['data'] = d

                parameter_grid_dict = {}
                for p_i in range(len(p)):
                    parameter_grid_dict[key_id[p_i]] = paramater_dict[key_id[p_i]][p[p_i]]
                my_dict['keys'] = key_id
                parameter_grid_dict['sample_file'] = sample_file
                parameter_grid_dict['result_file'] = "/tmp/dboostres_" + str(time.time()) + "_" + str(d.name) + "_" + str(result_counter) + ".csv"
                result_counter += 1

                my_dict['parameter_grid_dict'] = parameter_grid_dict

                final_file = generate_dBoost_result_file_name(model, d, parameter_grid_dict, key_id)
                if not os.path.isfile(final_file):
                    my_array.append(my_dict)
    return my_array





def run_and_evaluate_dBoost(params):
    model = params['model']

    final_file = generate_dBoost_result_file_name(model, params['data'], params['parameter_grid_dict'], params['keys'])

    if not os.path.isfile(final_file):
        model = params['model']
        model(**params['parameter_grid_dict'])

        result_file = str(params['parameter_grid_dict']['result_file'])
        del params['parameter_grid_dict']['sample_file']
        del params['parameter_grid_dict']['result_file']

        run = DBoostMe(params['data'], result_file)
        run.write_detected_matrix(final_file)


def clean_up(datasets):
    for dataset in datasets:
        d = dataset()
        sample_file = "/tmp/data_sample_" + str(d.name)  + ".csv"
        os.remove(sample_file)


my_array = create_runs(data_list, models)

pool = mp.Pool(processes=processes)
results = pool.map(run_and_evaluate_dBoost, my_array)

clean_up(data_list)