####################################################
# Benchmark: script for detecting errors in the datasets
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
    parser.add_argument('--hyperopt', type=bool, default=False)
    parser.add_argument('--early_termination', type=bool, default=False)
    parser.add_argument('--ml_models', type=MLModel, choices=list(MLModel), nargs='+', default=None)
    parser.add_argument('--store_postgres', action='store_true')
    args = parser.parse_args()

    # Retrieve the input arguments
    dataset_name = args.dataset_name
    iterations = args.n_iterations
    models_list = args.ml_models

    # Check if the dataset has a dictionary
    if dataset_name not in datasets_dictionary:
        raise ValueError(f"Dataset {dataset_name} is not known.")

    # Instantiate a benchmark object
    app = Benchmark(args.store_postgres)

    # ======================= Define Experimental settings ==============================

    # Define a list of datasets dataset to run experiments with
    datasets_list = [dataset_name]

    # All detectors are executed, iff empty list is passed to which_detectors
    if not models_list:
        app.run_all_models(dataset_list=datasets_list, iterations=iterations, hyperopt=args.hyperopt,
                           fastTermination=args.early_termination)
    else:
        for model in models_list:
            app.run_models(datasets_list=datasets_list, ml_model=model, iterations=iterations, hyperopt=args.hyperopt,
                           fastTermination=args.early_termination)
