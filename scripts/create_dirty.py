####################################################
# Benchmark: script for injecting errors into the datasets
# Authors: Mohamed Abdelaal
# Date: March 2022
# Software AG
# All Rights Reserved
####################################################

import argparse

from rein.auxiliaries.configurations import *
from rein.auxiliaries.datasets_dictionary import datasets_dictionary
from rein.datasets import Datasets

####################################################


if __name__ == '__main__':

    # Initialize an argument parser
    parser = argparse.ArgumentParser()
    # Add the parser's options
    parser.add_argument('--dataset_name', default=None, required=True)
    parser.add_argument('--muted_attribs', nargs='+', default=None)
    parser.add_argument('--error_rate', type=float, default=0.1)
    parser.add_argument('--outlier_degree', type=int, default=3)
    parser.add_argument('--store_postgres', action='store_true')
    parser.add_argument('--error_type', nargs='+', type=ErrorType, choices=list(ErrorType), default=None, required=True)

    args = parser.parse_args()

    # Retrieve the input arguments
    dataset_name = args.dataset_name
    muted_attribs = args.muted_attribs
    error_rate = args.error_rate
    outlier_degree = args.outlier_degree

    # Check if the dataset has a dictionary
    if dataset_name not in datasets_dictionary:
        raise ValueError(f"Dataset {dataset_name} is not known.")

    if not muted_attribs:
        muted_attribs = []
        # Mute the labels, if no attributes specified in the command
        ml_tasks = datasets_dictionary[dataset_name]['ml_tasks']
        if 'regression' in ml_tasks:
            muted_attribs.extend(datasets_dictionary[dataset_name]['labels_reg'])
        if 'classification' in ml_tasks:
            muted_attribs.extend(datasets_dictionary[dataset_name]['labels_clf'])

    # Initialize a dataset object using its dictionary
    app = Datasets(datasets_dictionary[dataset_name])

    configurations = [error_rate, outlier_degree]
    # Define the error types to be injected, e.g., missing values, outliers, [implicit_mv, swapping]
    error_list = []
    error_functions = []
    for error in args.error_type:
        if str(error) == 'outliers':
            error_list.append(outliers)
        else:
            error_functions.append(error.func)
    # Append all error types
    error_list.append(error_functions)
    # Inject the errors & save the dirty version
    app.inject_errors(error_list, configurations, muted_attribs, args.store_postgres)