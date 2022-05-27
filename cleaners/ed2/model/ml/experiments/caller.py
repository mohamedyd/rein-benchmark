import os
import sys

path = os.path.join(os.path.dirname(__file__), os.pardir,os.pardir)
sys.path.append(os.path.abspath(path))

from ml.classes.active_learning_total_uncertainty_error_correlation_lib import run_multi
from ml.classes.active_learning_total_uncertainty_error_correlation_lib import run
import multiprocessing as mp
import pandas as pd
from ml.datasets.specificDataset import SpecificDataset

from ml.active_learning.classifier.XGBoostClassifier import XGBoostClassifier

import numpy as np
import time


def run_ed2(clean_df, dirty_df, name, label_cutoff):

    #ed default settings
    parameters={'use_word2vec': True, 'use_word2vec_only': False, 'w2v_size': 100} #char unigrams + meta data + correlation + word2vec

    feature_name = 'ed2'

    #classifiers = [XGBoostClassifier, LinearSVMClassifier, NaiveBayesClassifier]
    classifier = XGBoostClassifier

    data = SpecificDataset(name, dirty_df, clean_df)
    my_array = []
    my_dict = parameters.copy()
    my_dict['dataSet'] = data
    my_dict['classifier_model'] = classifier
    my_dict['checkN'] = 1
    my_dict['label_threshold'] = label_cutoff
   
    my_array.append(my_dict)
    # runs experiment checkN rounds
    results = [run(**my_array[0])]

    # get dataframe which cell is true if cell is an detected error and else false
    all_error_statusDF = pd.DataFrame(results[0][1])
    labels = results[0][0]["labels"].pop()
    return all_error_statusDF, labels



