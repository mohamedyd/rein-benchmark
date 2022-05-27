####################################################
# Benchmark: script for repairing various errors in the datasets
# Authors: Mohamed Abdelaal
# Date: March 2022
# Software AG
# All Rights Reserved
####################################################

# Create a path to the Profiler directory
import argparse as argparse

from rein.auxiliaries.configurations import *
from rein.auxiliaries.datasets_dictionary import datasets_dictionary
from rein.benchmark import Benchmark

####################################################


if __name__ == '__main__':

    # Initialize an argument parser
    parser = argparse.ArgumentParser()
    # Add the parser's options
    parser.add_argument('--dataset_name', default=None, required=True)
    parser.add_argument('--n_iterations', type=int, default=10)
    parser.add_argument('--repair_method', type=RepairMethod, choices=list(RepairMethod), nargs='+', default=None)
    parser.add_argument('--store_postgres', action='store_true')
    args = parser.parse_args()

    # Retrieve the input arguments
    dataset_name = args.dataset_name
    iterations = args.n_iterations
    repair_list = args.repair_method

    # Check if the dataset has a dictionary
    if dataset_name not in datasets_dictionary:
        raise ValueError(f"Dataset {dataset_name} is not known.")

    # Instantiate a benchmark object
    app = Benchmark(args.store_postgres)

    # ======================= Define Experimental settings ==============================

    # Define a list of datasets dataset to run experiments with
    datasets_list = [dataset_name]

    # All repair methods are executed, iff empty list is passed to which_repair
    app.run_cleaners(exp_id=1, dataset_list=datasets_list, specified_cleaners=repair_list,
                     iterations=iterations)
