####################################################
# Benchmark: configuration
# Authors: Mohamed Abdelaal, Christian Hammacher
# Date: April 2021
# Software AG
# All Rights Reserved
####################################################


import sys
import os.path
import pandas as pd
import os
import csv
import tqdm
import numpy as np
import pickle
import time
import json
from datetime import datetime
import itertools
from scipy.stats import pearsonr, wilcoxon
import logging
import traceback
from enum import Enum

# Error generation
from error_generator import Explicit_Missing_Value
from error_generator import Implicit_Missing_Value
from error_generator import White_Noise
from error_generator import Gaussian_Noise
from error_generator import Random_Active_Domain
from error_generator import Similar_Based_Active_Domain
from error_generator import Typo_Keyboard
from error_generator import Typo_Butterfingers
from error_generator import Word2vec_Nearest_Neighbor

# Solve the bug: No module named 'sklearn.neighbors.base', caused by missingpy
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base

import warnings       # `do not disturbe` mode
warnings.filterwarnings("ignore", category=FutureWarning)

# Avoid warning due to hpsklearn
os.environ['OMP_NUM_THREADS'] = '1'

# =============================================================================================
#                                   Logging Configurations
#==============================================================================================

# Two logging configs methods are needed to avoid generating two log files in each run. Knowing that we execute this
# method three times to properly confugre the logging process
def logging_configs_console():
    """This methos defines the configurations for logging on the console"""
    logging.basicConfig(filename=None, format="%(asctime)s - [%(levelname)5s] - %(message)s", datefmt='%H:%M:%S')
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    return root_logger

def logging_configs_file():
    """This methods defines the configurations for logging in a file whose name includes the currrent date and time"""
    logging.basicConfig(filename='logs_{}.txt'.format(time.strftime("%Y%m%d-%H%M%S")),
                            filemode='w', format="%(asctime)s - [%(levelname)5s] - %(message)s", datefmt='%H:%M:%S')
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    return root_logger


# Configure the logger
logging_configs_console()


# =============================================================================================
#                                   Error Injection Configurations
#==============================================================================================

# Error types injected using the error generator module
class ErrorType(Enum):
    swapping = 'cell_swapping'
    implicit_mv = 'implicit_mv'
    explicit_mv = 'explicit_mv'
    noise = 'gaussian_noise'
    typos = 'typo_keyboard'
    outliers = 'outliers'

    @property
    def func(self):
        func_dict = {
        ErrorType.swapping: Random_Active_Domain(),
        ErrorType.implicit_mv: Implicit_Missing_Value(),
        ErrorType.explicit_mv: Explicit_Missing_Value(),
        ErrorType.noise: Gaussian_Noise(),
        ErrorType.typos: Typo_Keyboard()
        }
        return func_dict[self]

    def __str__(self):
        return self.value

# Error Types injected using the BART tools
missing_values = "missing_values"
pattern_violation = "pattern_violation"
rule_violation = "rule_violation"
outliers = "outliers"
mislabels = "mislabels"
duplicates = "duplicates"
typos = "typos"
inconsistency = 'inconsistency'

# =============================================================================================
#                                   Dataset Configurations
#==============================================================================================

smartfactory = 'smartfactory'
nasa = 'nasa'
citation = 'citation'
mercedes = 'mercedes'
print3d = 'print3d'
adult = 'adult'
beers = 'beers'
bike = 'bike'
soilmoisture = 'soilmoisture'
har = 'har'
water = 'water'
breast_cancer = 'breast_cancer'
nursery = 'nursery'
power = 'power'
soccer = 'soccer'
customers = 'customers'
flights = 'flights'
hospital = 'hospital'
tax = 'tax'
airbnb = 'airbnb'

# =============================================================================================
#                                   Error Detection Configurations
#==============================================================================================

class DetectMethod(Enum):
    # Detectors
    raha = "raha"
    mvdetector = "mvdetector"
    fahes = "fahes"
    nadeef = "nadeef"
    holoclean = "holoclean"
    dboost = "dboost"
    outlierdetector = "outlierdetector"
    duplicatesdetector = "duplicatesdetector"
    mislabeldetector = "mislabeldetector"
    katara = "katara"
    activeclean = "activeclean"
    metadata_driven = "metadata_driven"
    openrefine = "openrefine"
    max_entropy = "max_entropy"
    min_k = "min_k"
    zeroer = 'zeroer'
    cleanlab = 'cleanlab'
    picket = 'picket'
    ed2 = 'ed2'

    def __str__(self):
        return self.value

""" * These methods rely on the results of other detectors. 
    * Therefore, they should be executed, after running all other detectors """
ensemble_detectors = [DetectMethod.max_entropy, DetectMethod.min_k]

# =============================================================================================
#                                   Error Repair Configurations
#==============================================================================================

class RepairMethod(Enum):
    # Repair Methods
    cleanWithGroundTruth = "cleanWithGroundTruth"
    standardImputer = "standardImputer"
    duplicatesCleaner = "duplicatesCleaner"
    dcHoloCleaner = "dcHoloCleaner"
    baran = "baran"
    mlImputer = 'mlImputer'
    cleanlab = 'cleanlab'
    openrefine = 'openrefine'
    boostClean = 'boostClean'
    cpClean = 'CPClean'
    activecleanCleaner = "activecleanCleaner"

    def __str__(self):
        return self.value

# =============================================================================================
#                                   ML Modeling Configurations
#==============================================================================================

# ======== ML Models
classification = 'classification'
regression = 'regression'
clustering = 'clustering'
classes_list = ['binary', 'multi'] # binary or multi-class classification

class MLModel(Enum):
    rf_clf = 'forest_clf'
    logit_clf = "logit_clf"
    svc_clf = "svc_clf"
    sgd_svc_clf = "sgd_svc_clf"
    knn_clf = "knn_clf"
    dt_clf = "tree_clf"
    adaboost_clf = "adaboost_clf"
    gaussian_nb_clf = "gaussian_nb_clf"
    xgboost_clf = "xgboost_clf"
    ridge_clf = "ridge_clf"
    mlp_clf = "mlp_clf"
    multinomial_nb_clf = 'mn_clf'
    # Regression methods
    linear_regression_reg = "lin_reg"
    dt_reg = "tree_reg"
    rf_reg = "forest_reg"
    adaboost_reg = "adaboost_reg"
    svm_reg = "svm_reg"
    bayes_ridge_reg = "bayes_ridge_reg"
    xgboost_reg = "xgboost_reg"
    mlp_reg = "mlp_reg"
    ridge_reg = "ridge_reg"
    knn_reg = "knn_reg"
    ransac_reg = 'ransac_reg'
    # Clustering methods
    gaussian_mixture_cls = "gm_cls"
    kmeans_cls = "kmeans_cls"
    affinity_propagation_cls = "affinity_cls"
    hierarchical_cls = "hierarchical_cls"
    optics_cls = "optics_cls"
    birch_cls = "birch_cls"

    def __str__(self):
        return self.value

# =============================================================================================
#                                   Plotting Configurations
#==============================================================================================

# CSV files of the results
iou_file = 'iou.csv'
detection_file = 'detection_results.csv'
repair_file = 'cleaning_results.csv'
ml_file = 'model_results.csv'
abtest_file = 'abtesting.csv'
cleaners_models_file = 'cleaner_model_results.csv'
# Mapping dictionaries
detectors_mapper = {'min_k': ['Min', 'M'],
                    'metadata_driven': ['Meta', 'T'],
                    'ed2': ['ED2', 'E'],
                    'raha': ['RAHA', 'R'],
                    'katara': ['Katara', 'K'],
                    'mvdetector': ['MVD', 'V'],
                    'max_entropy': ['Max', 'X'],
                    'holoclean': ['Holo', 'H'],
                    'nadeef': ['NADEEF', 'N'],
                    'openrefine': ['OpnR', 'O'],
                    'picket': ['Picket', 'P'],
                    'fahes_ALL': ['FAHES', 'F'],
                    'fahes': ['FAHES', 'F'],
                    'duplicatesdetector': ['DuplD', 'D'],
                    'dboost': ['dBoost', 'B'],
                    'outlierdetector_IF': ['IF', 'I'],
                    'outlierdetector': ['SD', 'S'],
                    'outlierdetector_IQR': ['IQR', 'Q'],
                    'zeroer': ['ZeroER', 'Z'],
                    'cleanlab': ['Cleanlab', 'C'],
                    'cleanlab-forest_clf': ['Cleanlab', 'C'],
                    'outlierdetector_SD': ['SD', 'S'],
                    'dirty': ['', 'R']}
cleaners_mapper = {'cleanWithGroundTruth': ['GT', '1'],
                   'dirty': ['Dirty', ''],
                   'baran': ['BARAN', '15'],
                   'dcHoloCleaner-without_init': ['Holo', '13'],
                   'duplicatesCleaner': ['DuplC', '2'],
                   'mlImputer-seperate-missForest-missForest': ['MISS-Sep', '8'],
                   'mlImputer-seperate-missForest-datawig': ['MISS-DataWig', '9'],
                   'mlImputer-seperate-decisionTree-missForest': ['DT-MISS', '10'],
                   'mlImputer-seperate-bayesianRidge-missForest': ['Bayes-MISS', '11'],
                   'mlImputer-seperate-knn-missForest': ['KNN-MISS', '12'],
                   'standardImputer-impute-mean-mode': ['Impute', '3'],
                   'standardImputer-impute-mean-dummy': ['Impute', '3'],
                   'standardImputer-impute-median-mode': ['Impute', '4'],
                   'standardImputer-impute-median-dummy': ['Impute', '4'],
                   'standardImputer-impute-mode-mode': ['Impute', '5'],
                   'standardImputer-impute-mode-dummy': ['Impute', '5'],
                   'standardImputer-delete': ['Delete', '2'],
                   'standardImputerdelete': ['Delete', '2'],
                   'mlImputer-mix-missForest': ['MISS-Mix', '6'],
                    'mlImputer-mix-datawig': ['DataWig-Mix', '7'],
                   'cleanlab-forest_clf': ['Cleanlab', '16'],
                   'activecleanCleaner-0.2': ['ActiveClean', '17'],
                   'CPClean': ['CPClean', '19'],
                   'boostClean': ['BoostClean', '18'],
                   'openrefine': ['OpnR', '14']
                   }
models_mapper = {
                 # Classification
                 'forest_clf': ['RF'],
                 'logit_clf': ['Logit'],
                 'svc_clf': ['SVC'],
                 'sgd_svc_clf': ['SGD'],
                 'knn_clf': ['KNN'],
                 'tree_clf': ['DT'],
                 'adaboost_clf': ['AdaB'],
                 'gaussian_nb_clf': ['GNB'],
                 'xgboost_clf': ['XGB'],
                 'ridge_clf': ['Ridge'],
                 'mlp_clf': ['MLP'],
                 'autosklearn_clf': ['AutoSK'],
                 'tpot_clf': ['TPOT'],
                 # Regression
                 'forest_reg': ['RF'],
                 'lin_reg': ['Linear'],
                 'tree_reg': ['DT'],
                 'adaboost_reg': ['AdaB'],
                 'svm_reg': ['SVM'],
                 'bayes_ridge_reg': ['BRidge'],
                 'xgboost_reg': ['XGB'],
                 'mlp_reg': ['MLP'],
                 'ridge_reg': ['Ridge'],
                 'knn_reg': ['KNN'],
                 'ransac_reg': ['RANSAC'],
                 # Clustering
                 'gm_cls': ['GMM'],
                 'kmeans_cls': ['KMeans'],
                 'affinity_cls': ['AP'],
                 'hierarchical_cls': ['HC'],
                 'optics_cls': ['Optics'],
                 'birch_cls': ['Birch']
}
cleaners_list = cleaners_mapper.keys()
detectors_list = detectors_mapper.keys()
models_list = models_mapper.keys()