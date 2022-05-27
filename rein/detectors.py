####################################################
# Benchmark: A collection of errors detection methods
# Authors: Christian Hammacher, Mohamed Abdelaal
# Date: February 2021
# Software AG
# All Rights Reserved
###################################################


###################################################
# import benchmark
import re
import shutil
import random
import hashlib
import tempfile
import copy
import operator
import torch
import sklearn.linear_model
import sklearn.feature_extraction
import sklearn.ensemble
from sklearn.ensemble import IsolationForest
from rein.auxiliaries.configurations import *
from rein.auxiliaries.datasets_dictionary import datasets_dictionary
from rein.auxiliaries.models_dictionary import models_dictionary
from cleanlab.pruning import get_noise_indices
from rein.models import Models, models
from sklearn.model_selection import StratifiedKFold     
from sklearn.linear_model import LogisticRegression
from cleanlab.latent_estimation import estimate_latent, estimate_confident_joint_and_cv_pred_proba

# Create a path to the cleaners directory
#cleaners_path = os.path.join(os.path.dirname(__file__), os.pardir, "cleaners")
# Add the cleaners directory to the system path
sys.path.append(os.path.abspath("cleaners"))

# Import cleaners-related methods
import holoclean.holoclean
from holoclean.repair.featurize import *
from holoclean.dataset.table import Table, Source
from holoclean.detect import NullDetector, ViolationDetector
from raha.detection import Detection as Raha_Detection
from FAHES.fahes_caller import *
from dBoost.dboost.imported_dboost import *
from katara.katara import *
from zeroer.zeroer import *
from Picket.picket.prepare.dataPrepare import TrainTestSplit
from picket.transformer.PicketNet import PicketNetModel
from picket.filter.filtersTrain import Attribute
from ed2.model.ml.experiments.caller import run_ed2

###################################################


###################################################
class Detectors:
    """
    Class encapsulates all errors detection methods
    """

    def __init__(self, dataset_name, actual_errors):
        """
        The constructor.
        """
        self.actual_errors = actual_errors
        self.__dataset_name = dataset_name
        self.__DATASET_CONSTRAINTS = {
            adult: {
                'functions': [
                    #fds resulting from FD_generator
                    ["education", "capital_gain"],
                    ["capital_gain", "educational_num"],
                    
                    # other fds
                    ['marital_status', 'relationship'],
                    ['education', 'educational_num']
                ],
                "patterns": []
                },
            bike:{
                'functions': [
                    #fds resulting from FD_generator
                    ["yr", "casual"],
                    ["mnth", "season"],
                    ["workingday", "holiday"],
                ],
                "patterns": [
                    ["season", "^[\d]{1}$", "ONM"],
                    ["yr", "^(0|1)$", "ONM"],
                    ["mnth", "^[\d]+$", "ONM"],
                    ["hr", "^[\d]+$", "ONM"],
                    ["weekday", "^[\d]{1}$", "ONM"],
                    ["workingday", "^(0|1)$", "ONM"],
                    ["weathersit", "^(1|2|3|4)$", "ONM"],
                    ["casual", "^[\d]+$", "ONM"],
                    ["registered", "^[\d]+$", "ONM"],
                    ["cnt", "^[\d]+$", "ONM"],
                    ["temp", "^([\d]+\.[\d]+|1)$", "ONM"],
                    ["atemp", "^([\d]+\.[\d]+|0|1)$", "ONM"],
                    ["hum", "^([\d]+\.[\d]+|0|1)$", "ONM"],
                    ["windspeed", "^([\d]+\.[\d]+|0)$", "ONM"],
                ],
                },
            customers: {
                'functions' : [
                    #fds resulting from FD_generator
                    ["Fresh", "Frozen"],
                    ["Milk", "Detergents_Paper"],
                    ["Channel", "index"],
                    ["index", "Region"],
                ],"patterns": [],},
            flights: {
                'functions' : [
                    #fds resulting from FD_generator
                    ["act_arr_time_sin", "act_dep_time_sin"],
                    ["tuple_id", "flight"],
                ], "patterns": [],},
            har: {
                'functions' : [
                    #fds resulting from FD_generator
                    [],
                ], "patterns": [
                    ["gt", "^(stand|sit)$", "ONM"],
                    ["Index", "^[\d]+$", "ONM"],
                    ["z", "^[\d]+\.[\d]+$", "ONM"],
                    ["x", "^[-]?[\d]+\.[\d]+$", "ONM"],
                    ["y", "^[-]?[\d]+\.[\d]+$", "ONM"],
                ]},
            hospital: {
                "functions": [
                    #fds resulting from FD_generator
                    ["HospitalName", "ZipCode"],
                    ["HospitalName", "PhoneNumber"],
                    ["MeasureCode", "MeasureName"],
                    ["MeasureCode", "Stateavg"],
                    ["ProviderNumber", "HospitalName"],
                    ["MeasureCode", "Condition"],
                    ["HospitalName", "Address1"],
                    ["HospitalName", "HospitalOwner"],
                    ["HospitalName", "ProviderNumber"],
                    ["HospitalName", "PhoneNumber"],
                    ["City", "CountyName"],
                    ["ZipCode", "EmergencyService"],
                    ["HospitalName", "City"],
                    ["MeasureName", "MeasureCode"],
                    
                    # other fds
                    ["City", "ZipCode"],
                    ["ZipCode", "City"],
                    ["ZipCode", "State"],
                    ["ZipCode", "CountyName"],
                    ["CountyName", "State"],
                ],
                "patterns": [
                    #["Index", "^[\d]+$", "ONM"],
                    ["ProviderNumber", "^[\d]+$", "ONM"],
                    ["ZipCode", "^[\d]{5}$", "ONM"],
                    ["State", "^[a-z]{2}$", "ONM"],
                    ["PhoneNumber", "^[\d]+$", "ONM"],
                ],
            },
            mercedes: {'functions' : [], "patterns": [],},
            nasa: {'functions' : [
                    #fds resulting from FD_generator
                    ["angle", "chord_length"],
                ], "patterns": [],},
            print3d: {'functions' : [], "patterns": [],},
            smartfactory : {'functions' : [
                    #fds resulting from FD_generator
                    ["o_w_blo_voltage", "i_w_blo_weg"],
                    ["i_w_blo_weg", "i_w_bru_weg"],
                ], "patterns": [],},
            soilmoisture: {'functions' : [], "patterns": [],},
            power: {'functions': [], "patterns": [
                ["X0", "^[\d]+\.?[\d]*$", "ONM"],
                ["X1", "^[\d]+\.?[\d]*$", "ONM"],
                ["X2", "^[\d]+\.?[\d]*$", "ONM"],
                ["X3", "^[\d]+\.?[\d]*$", "ONM"],
                ["X4", "^[\d]+\.?[\d]*$", "ONM"],
                ["X5", "^[\d]+\.?[\d]*$", "ONM"],
                ["X6", "^[\d]+\.?[\d]*$", "ONM"],
                ["X7", "^[\d]+\.?[\d]*$", "ONM"],
                ["X8", "^[\d]+\.?[\d]*$", "ONM"],
                ["X9", "^[\d]+\.?[\d]*$", "ONM"],
                ["X10", "^[\d]+\.?[\d]*$", "ONM"],
                ["X11", "^[\d]+\.?[\d]*$", "ONM"],
                ["X12", "^[\d]+\.?[\d]*$", "ONM"],
                ["X13", "^[\d]+\.?[\d]*$", "ONM"],
                ["X14", "^[\d]+\.?[\d]*$", "ONM"],
                ["X15", "^[\d]+\.?[\d]*$", "ONM"],
                ["X16", "^[\d]+\.?[\d]*$", "ONM"],
                ["X17", "^[\d]+\.?[\d]*$", "ONM"],
                ["X18", "^[\d]+\.?[\d]*$", "ONM"],
                ["X19", "^[\d]+\.?[\d]*$", "ONM"],
                ["X20", "^[\d]+\.?[\d]*$", "ONM"],
                ["X21", "^[\d]+\.?[\d]*$", "ONM"],
                ["X22", "^[\d]+\.?[\d]*$", "ONM"],
                ["X23", "^[\d]+\.?[\d]*$", "ONM"],
            ], },
            soccer: {
                'functions' : [
                ['player_name', 'birthday'],
                ['id_x', 'player_name'],
            ], 'patterns' : [
                ["overall_rating", "^[\d]+$", "ONM"],
                ["potential", "^[\d]+$", "ONM"],
                ["crossing", "^[\d]+$", "ONM"],
                ["finishing", "^[\d]+$", "ONM"],
                ["heading_accuracy", "^[\d]+$", "ONM"],
                ["short_passing", "^[\d]+$", "ONM"],
                ["volleys", "^[\d]+$", "ONM"],
                ["dribbling", "^[\d]+$", "ONM"],
                ["curve", "^[\d]+$", "ONM"],
                ["free_kick_accuracy", "^[\d]+$", "ONM"],
                ["long_passing", "^[\d]+$", "ONM"],
                ["ball_control", "^[\d]+$", "ONM"],
                ["acceleration", "^[\d]+$", "ONM"],
                ["sprint_speed", "^[\d]+$", "ONM"],
                ["agility", "^[\d]+$", "ONM"],
                ["reactions", "^[\d]+$", "ONM"],
                ["balance", "^[\d]+$", "ONM"],
                ["shot_power", "^[\d]+$", "ONM"],
                ["jumping", "^[\d]+$", "ONM"],
                ["stamina", "^[\d]+$", "ONM"],
                ["strength", "^[\d]+$", "ONM"],
                ["long_shots", "^[\d]+$", "ONM"],
                ["aggression", "^[\d]+$", "ONM"],
                ["interceptions", "^[\d]+$", "ONM"],
                ["positioning", "^[\d]+$", "ONM"],
                ["vision", "^[\d]+$", "ONM"],
                ["penalties", "^[\d]+$", "ONM"],
                ["marking", "^[\d]+$", "ONM"],
                ["standing_tackle", "^[\d]+$", "ONM"],
                ["sliding_tackle", "^[\d]+$", "ONM"],
                ["gk_diving", "^[\d]+$", "ONM"],
                ["gk_handling", "^[\d]+$", "ONM"],
                ["gk_kicking", "^[\d]+$", "ONM"],
                ["gk_positioning", "^[\d]+$", "ONM"],
                ["gk_reflexes", "^[\d]+$", "ONM"],
                ["weight", "^[\d]+$", "ONM"],
                ["id_x", "^[\d]+$", "ONM"],
                [ "birthday", "^[0-9]+[/][0-9]+[/][0-9]{4}$", "ONM"],
                ["date", "^[0-9]+[/][0-9]+[/][0-9]{4}$", "ONM"],
                ["preferred_foot", "^(left|right)$", "ONM"],
            ]},
            beers: {
                "functions": [
                    #fds resulting from FD_generator
                    ["ounces", "id"],
                    
                    # other fds
                    ["brewery_id", "brewery_name"],
                    ["brewery_id", "city"],
                    ["brewery_id", "state"]],
                "patterns": [["state", "^[A-Z]{2}$", "ONM"], ["brewery_id", "^[\d]+$", "ONM"]]
            },
            water: {'functions' : [
                    #fds resulting from FD_generator
                    ["x7", "x13"],
                    ["x10", "x16"],
                    ["x10", "x3"],
                    ["x21", "x32"],
                    ["x34", "x36"],
                    ["x22", "x9"],
                    ["x22", "x29"],
                    ["x24", "x33"],
                    ["x24", "x26"],
                    ["x38", "x8"],
                    ["x8", "x12"]
                ], "patterns": [],},
            tax: {
                "functions": [
                    # other fds
                    ["zip", "city"],
                    ["zip", "state"],
                    ["f_name", "gender"],
                    ["area_code", "state"],
                ],
                "patterns": [
                    ["gender", "^(M|F)$", "ONM"],
                    ["area_code", "^[\d]{3}$", "ONM"],
                    ["phone", "^[0-9]{3}[-][0-9]{4}$", "ONM"],
                    ["state", "^[a-z]{2}$", "ONM"],
                    ["zip", "^[1-9][0-9]*$", "ONM"],
                    ["marital_status", "^(M|S)$", "ONM"],
                    ["has_child", "^(N|Y)$", "ONM"],
                    ["salary", "^[\d]+$", "ONM"],
                ],
            },
            airbnb: {
                "functions": [
                    #fds resulting from FD_generator
                    ["NumGuests", "Beds"],
                    ["NumGuests", "Bedrooms"],
                    ["LocationName", "Rating"],
                    ["Count of Abnb", "Density of Abnb"],
                    ["median taxes with_mortgage", "median house value"],
                    ["median monthly owner costs no_mortgage", "prop taxes paid 2016"],
                    ["latitude", "land_area sqmi"],
                    ["longitude", "Average Number of Beds by_zipcode"],
                    ["Average Number of Beds by_zipcode", "Average Number of Bedrooms by_zipcode"],
                    ["Average Number of Bedrooms by_zipcode", "Average Number of Bathrooms by_zipcode"],
                    ["cost_living_index US avg_100", "pop_density people_per_mile"],
                    ["zipcode", "Average Rating by_zipcode"],
                    ["median houshold income", "median gross rent"],
                    ["median house value", "median asking price for vacant for sale home_condo"],
                    ["median taxes with_mortgage", "Average Abnb Price by_zipcode"],
                    ["pop2010", "pop2000"],
                    
                    # other fds
                    ["zipcode", "land_area sqmi"],
                    ["zipcode", "cost_living_index US avg_100"],
                    ["zipcode", "water_area sqmi"],
                ],
                 "patterns": [],
                #     ["Bathrooms", "^[\d]+$", "ONM"],
                #     ["Bedrooms", "^[\d]+$", "ONM"],
                #     ["Beds", "^[\d]+$", "ONM"],
                #     ["NumGuests", "^[\d]+$", "ONM"],
                #     ["NumReviews", "^[\d]+$", "ONM"],
                #     ["Price", "^[\d]+$", "ONM"],
                #     ["Rating", "^(Y|N)$", "ONM"],
                #     ["latitude", "(^\-?[\d]+$) |(^\-?[\d]+[\.][\d]+$)", "ONM"],
                #     ["longitude", "(^\-?[\d]+$) |(^\-?[\d]+[\.][\d]+$)", "ONM"],
                #     ["zipcode", "^[\d]+$", "ONM"],
                #     ["pop2016", "^[\d]+$", "ONM"],
                #     ["pop2010", "^[\d]+$", "ONM"],
                #     ["pop2000", "^[\d]+$", "ONM"],
                #     [
                #         "cost_living_index (US avg. = 100)",
                #         "(^\-?[\d]+$) |(^\-?[\d]+[\.][\d]+$)",
                #         "ONM",
                #     ],
                #     [
                #         "land_area (sq.mi.)",
                #         "(^\-?[\d]+$) |(^\-?[\d]+[\.][\d]+$)",
                #         "ONM",
                #     ],
                #     [
                #         "water_area (sq.mi.)",
                #         "(^\-?[\d]+$) |(^\-?[\d]+[\.][\d]+$)",
                #         "ONM",
                #     ],
                #     ["pop_density (people per mile)", "^[\d]+$", "ONM"],
                #     ["number of males", "^[\d]+$", "ONM"],
                #     ["number of females", "^[\d]+$", "ONM"],
                #     [
                #         "prop taxes paid 2016",
                #         "(^\-?[\d]+$)|(^\-?[\d]+[\.][\d]+$)",
                #         "ONM",
                #     ],
                #     ["median taxes (with mortgage", "^[\d]+$", "ONM"],
                #     ["median taxes (no mortgage)", "^[\d]+$", "ONM"],
                #     ["median house value", "^[\d]+$", "ONM"],
                #     ["median houshold income", "^[\d]+$", "ONM"],
                #     ["median monthly owner costs (with mortgage)", "^[\d]+$", "ONM"],
                #     ["median monthly owner costs (no mortgage)", "^[\d]+$", "ONM"],
                #     ["median gross rent", "^[\d]+$", "ONM"],
                #     [
                #         "median asking price for vacant for-sale home/condo",
                #         "^[\d]+$",
                #         "ONM",
                #     ],
                #     ["unemployment (%)", "(^\d{1,2}$)|(^\d{1,2}\.\d+$)", "ONM"],
                #     ["Number of Homes", "(^[\d]+$) |(^[\d]+[\.][\d]+$)", "ONM"],
                #     ["Count of Abnb", "^[\d]+$", "ONM"],
                #     ["Density of Abnb (%)", "(^\d{1,2}$)|(^\d{1,2}\.\d+$)", "ONM"],
                #     [
                #         "Average Abnb Price (by zipcode)",
                #         "(^[\d]+$) |(^[\d]+[\.][\d]+$)",
                #         "ONM",
                #     ],
                #     [
                #         "Average NumReviews (by zipcode)",
                #         "(^[\d]+$) |(^[\d]+[\.][\d]+$)",
                #         "ONM",
                #     ],
                #     [
                #         "Average Rating (by zipcode)",
                #         "(^[\d]+$) |(^[\d]+[\.][\d]+$)",
                #         "ONM",
                #     ],
                #     [
                #         "Average Number of Bathrooms (by zipcode)",
                #         "(^[\d]+$) |(^[\d]+[\.][\d]+$)",
                #         "ONM",
                #     ],
                #     [
                #         "Average Number of Bedrooms (by zipcode)",
                #         "(^[\d]+$) |(^[\d]+[\.][\d]+$)",
                #         "ONM",
                #     ],
                #     [
                #         "Average Number of Beds (by zipcode)",
                #         "(^[\d]+$) |(^[\d]+[\.][\d]+$)",
                #         "ONM",
                #     ],
                #     [
                #         "Average Number of Guests (by zipcode)",
                #         "(^[\d]+$) |(^[\d]+[\.][\d]+$)",
                #         "ONM",
                #     ],
                # ],
            },
        }

        # List all implemented detection methods
        self.detectors_list = [
             self.nadeef,
             self.outlierdetector,
             self.mvdetector,
             self.duplicatesdetector,
             self.raha,
             self.mislabeldetector,
             self.holoclean,
             self.fahes,
             self.dboost,
            self.katara,
            self.activeclean,
            self.metadata_driven,
            self.max_entropy,
            self.openrefine,
            self.min_k,
            self.zeroer,
            self.cleanlab,
            self.picket,
            self.ed2
        ]

    def __get_detector_directory(self, detector_name):
        """
        Arguments:
        detector_name (String) -- name of the detector that is applied

        Returns:
        detector_directory (String) -- path to the detector folder for a dataset
        """
        path = datasets_dictionary[self.__dataset_name]["dataset_path"]

        detector_directory = os.path.join(path, detector_name)
        if not os.path.exists(detector_directory):
            # creating a new directory if it does not exit
            os.mkdir(detector_directory)

        return detector_directory

    def __store_detections(self, detections, detector_path):
        """
        This method stores a given detection dictionary as a csv in the detector folder for a dataset

        Arguments:
        detections (dict) -- dictionary where each entry with keys (i,j) is one detected error in row i and column j
        detector_path (String) -- path to the folder where detections should be stored
        """
        try:
            outfile = os.path.join(detector_path, "detections.csv")
            with open(outfile, 'w') as f_object:
                # Create a file object and prepare it for writing the results
                writefile = csv.writer(f_object)
                for key,value in detections.items():
                    row = [key, value]
                    # Write the values after flattening the row list obtained in the above line
                    writefile.writerow(np.hstack(row))
            f_object.close()
        except IOError:
            logging.info("I/O error")

    def __evaluate(self, detections):
        """
        This method evaluates the detection performance

        Arguments:
        detections (dict) -- keys represent i,j of dirty cells & values are constant string "JUST A DUUMY VALUE" 

        Returns:
        precision, recall, f1
        """
        tp = 0.0
        output_size = 0.0

        #for key, value in self.actual_errors.items():
        #    logging.info("actual_errors ", key, value)

        for cell in detections:
            output_size = output_size + 1
            if cell in self.actual_errors:
                tp = tp + 1

        precision = 0.0 if output_size == 0 else tp / output_size
        recall = 0.0 if len(self.actual_errors) == 0 else tp / (len(self.actual_errors))
        f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)

        return precision, recall, f1

    def nadeef(self, dirtyDF, dataset, configs):
        """
        This method runs NADEEF.
        It will return an empty dictionary and result dictionary if there are not constraints defined for the given dataset.

        Arguments:
        dirtyDF -- dataframe of shape n_R (# of records) x n_A (# of attributes) - containing a dirty version of a dataset

        Returns:
        detection_dictionary -- dictionary - keys represent i,j of dirty cells & values are constant string "JUST A DUUMY VALUE"
        """
        start_time = time.time()

        # define a dictionary to store the indices of the detected dirty cells
        detection_dictionary = {}

        pattern_violation_count = 0
        fd_violation_count = 0
        
        # return empty detection and results dict if dataset has no nadeef constraints
        if self.__dataset_name not in self.__DATASET_CONSTRAINTS.keys():
            return {},{}

        # adds (index, left_value) and (index, right_value) to dictionary for every functional dependency
        for fd in self.__DATASET_CONSTRAINTS[self.__dataset_name]["functions"]:

            # get attribute of interest
            l_attribute, r_attribute = fd

            # get values of each attribute
            l_j = dirtyDF.columns.get_loc(l_attribute)
            r_j = dirtyDF.columns.get_loc(r_attribute)

            value_dictionary = {}

            # fills value dictionary with {value_left_i : {value_right_i : 1} value_left_i : {value_right_i : 1} ... }
            #
            for i, row in dirtyDF.iterrows():
                if row[l_attribute]:
                    if row[l_attribute] not in value_dictionary:
                        value_dictionary[row[l_attribute]] = {}
                    if row[r_attribute]:
                        value_dictionary[row[l_attribute]][row[r_attribute]] = 1

            # adds violation to detection_dictionary for every left attribute value that has more than 1 right attribute value
            # violations example: {value_left_i : {value_right_i : 1} value_left_i : {value_right_i : 1, value_right_j : 2} ... }
            for i, row in dirtyDF.iterrows():
                if (
                        row[l_attribute] in value_dictionary
                        and len(value_dictionary[row[l_attribute]]) > 1
                ):
                    detection_dictionary[(i, l_j)] = "JUST A DUUMY VALUE"
                    detection_dictionary[(i, r_j)] = "JUST A DUUMY VALUE"
                    # increment fd violation by two for each pair of row, attribute_left and row attribute_right
                    fd_violation_count = fd_violation_count + 2

        for attribute, pattern, opcode in self.__DATASET_CONSTRAINTS[
            self.__dataset_name
        ]["patterns"]:
            j = dirtyDF.columns.get_loc(attribute)
            for i, value in dirtyDF[attribute].iteritems():
                if opcode == "OM":
                    if len(re.findall(pattern, value, re.UNICODE)) > 0:
                        detection_dictionary[(i, j)] = "JUST A DUUMY VALUE"
                        # increase pattern violation count for every pattern violation
                        pattern_violation_count = pattern_violation_count + 1
                else:
                    if len(re.findall(pattern, value, re.UNICODE)) == 0:
                        detection_dictionary[(i, j)] = "JUST A DUUMY VALUE"
                        # increase pattern violation count for every pattern violation
                        pattern_violation_count = pattern_violation_count + 1

        error_detect_runtime = time.time() - start_time

        # get detector path
        detector_path = self.__get_detector_directory("nadeef")

        # store detections in detector directory
        self.__store_detections(detection_dictionary, detector_path)

        precision, recall, f1 = self.__evaluate(detection_dictionary)

        evaluation_dict = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "detection_runtime": error_detect_runtime,
            "#detections": (pattern_violation_count + fd_violation_count),
            "#pattern_violations": pattern_violation_count,
            "#fd_violations": fd_violation_count,
            "#detected_duplicates": None,
            "detected_error_rate": len(detection_dictionary) / dirtyDF.size,
        }

        return detection_dictionary, evaluation_dict

    def raha(self, dirtydf, dataset, configs):
        start_time = time.time()
        dataset_name = dataset

        # dict to process raha steps internally
        internal_dataset_dict = {
            "name": dataset_name,
            "path": os.path.abspath(
                os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "dirty.csv")),
            "clean_path": os.path.abspath(
                os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "clean.csv"))
        }

        # detect errors and get detection dictionary
        app = Raha_Detection()
        detection_dictionary = app.run(internal_dataset_dict)
        logging.info("------ received detection dictionary -------")
        # get runtime
        error_detect_runtime = time.time() - start_time

        # get detector path
        detector_path = self.__get_detector_directory(str(DetectMethod.raha))

        # store detections in detector directory
        self.__store_detections(detection_dictionary, detector_path)

        # length of detection_dictionary represents number of detected errors
        precision, recall, f1 = self.__evaluate(detection_dictionary)

        evaluation_dict = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "detection_runtime": error_detect_runtime,
            "#detections": len(detection_dictionary),
            "#pattern_violations": None,
            "#fd_violations": None,
            "#detected_duplicates": None,
            "detected_error_rate": len(detection_dictionary) / dirtydf.size,
        }

        return detection_dictionary, evaluation_dict

    def mvdetector(self, dirtydf, dataset, configs):
        """
        This method detects explicit missing values.

        As the data is loaded with keep_default_na = False, empty cells are interpreted as 
        empty string. Thus every cell with an empty string is counted as a explicit missing value.

        Arguments:
        dirtyDF -- dataframe of shape n_R (# of records) x n_A (# of attributes) - containing a dirty version of a dataset

        Returns:
        detection_dictionary -- dictionary - keys represent i,j of dirty cells & values are constant string "JUST A DUUMY VALUE"
        """
        start_time = time.time()

        detection_dictionary = {}
        
        # save every (row,col) where entry is true to detection dictionary
        for col in dirtydf.columns:

            col_j = dirtydf.columns.get_loc(col)
            
            for i, row in dirtydf.iterrows():
                
                # dataset is read with keep_default_na = False, so empty cells
                # are represented by empty strings
                if dirtydf.iat[i, col_j]=='':
                    detection_dictionary[(i, col_j)] = "JUST A DUMMY VALUE"
        
        # get runtime
        error_detect_runtime = time.time() - start_time

        # get detector path
        detector_path = self.__get_detector_directory(str(DetectMethod.mvdetector))

        # store detections in detector directory
        self.__store_detections(detection_dictionary, detector_path)

        precision, recall, f1 = self.__evaluate(detection_dictionary)

        evaluation_dict = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "detection_runtime": error_detect_runtime,
            "#detections": len(detection_dictionary),
            "#pattern_violations": None,
            "#fd_violations": None,
            "#detected_duplicates": None,
            "detected_error_rate": len(detection_dictionary) / dirtydf.size,
        }

        return detection_dictionary, evaluation_dict

    def outlierdetector(self, dirtydf, dataset, configs):
        """
        This method finds outliers for numerical columns. 

        Configuration options: set nstd for method SD, set k for method IQR, set contamination for Method IF

        Arguments:
        dirtyDF -- dataframe of shape n_R (# of records) x n_A (# of attributes) - containing a dirty version of a dataset
        detect_method (String) -- can be SD, IQR, IF

        Returns:
        detection_dictionary -- dictionary - keys represent i,j of dirty cells & values are constant string "JUST A DUUMY VALUE"

        """

        def SD(x, nstd=3.0):
            # Standard Deviaiton Method (Univariate)
            mean, std = np.mean(x), np.std(x)
            cut_off = std * nstd
            lower, upper = mean - cut_off, mean + cut_off
            return lambda y: (y > upper) | (y < lower)

        def IQR(x, k=1.5):
            # Interquartile Range (Univariate)
            q25, q75 = np.percentile(x, 25), np.percentile(x, 75)
            iqr = q75 - q25
            cut_off = iqr * k
            lower, upper = q25 - cut_off, q75 + cut_off
            return lambda y: (y > upper) | (y < lower)

        def IF(x, contamination=0.01):
            # Isolation Forest (Univariate)
            #IF = IsolationForest(contamination='auto')
            IF = IsolationForest(contamination=contamination)
            IF.fit(x.reshape(-1, 1))
            return lambda y: (IF.predict(y.reshape(-1, 1)) == -1)

        start_time = time.time()

        # Extract the detection method
        detect_method = configs["detect_method"]

        # can be set
        detect_fn_dict = {'SD':SD, 'IQR':IQR, "IF":IF}
        detect_fn = detect_fn_dict[detect_method]

        # transform dirtdf which has dtype string to numeric type if possible
        dirtydf = dirtydf.apply(pd.to_numeric, errors="ignore")

        num_df = dirtydf.select_dtypes(include='number')
        cat_df = dirtydf.select_dtypes(exclude='number')
        X = num_df.values
        m = X.shape[1]

        # calculate for each row the detector in form of a lambda expression
        detectors = []
        for i in range(m):
            x = X[:, i]
            detector = detect_fn(x)
            detectors.append(detector)

        ind_num = np.zeros_like(num_df).astype('bool')
        ind_cat = np.zeros_like(cat_df).astype('bool')

        # check for each column if respective lambda expression is true
        # if there is a outlier
        for i in range(m):
            x = X[:, i]
            detector = detectors[i]
            is_outlier = detector(x)
            ind_num[:, i] = is_outlier

        ind_num = pd.DataFrame(ind_num, columns=num_df.columns)
        ind_cat = pd.DataFrame(ind_cat, columns=cat_df.columns)
        ind = pd.concat([ind_num, ind_cat], axis=1).reindex(columns=dirtydf.columns)       

        # create detection dict
        detection_dictionary = {}
        for col in ind.columns:

            col_j = ind.columns.get_loc(col)
            
            for i, row in ind.iterrows():
                
                # if cell is true it is outlier
                if ind.iat[i, col_j]:
                    detection_dictionary[(i, col_j)] = "JUST A DUMMY VALUE"

        # get runtime
        error_detect_runtime = time.time() - start_time

        # get detector path
        detector_path = self.__get_detector_directory("{}_{}".format(str(DetectMethod.outlierdetector), detect_method))

        # store detections in detector directory
        self.__store_detections(detection_dictionary, detector_path)

        precision, recall, f1 = self.__evaluate(detection_dictionary)

        evaluation_dict = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "detection_runtime": error_detect_runtime,
            "#detections": len(detection_dictionary),
            "#pattern_violations": None,
            "#fd_violations": None,
            "#detected_duplicates": None,
            "detected_error_rate": len(detection_dictionary) / dirtydf.size,
        }

        return detection_dictionary, evaluation_dict

    def duplicatesdetector(self, dirtydf, dataset, configs):
        """
        This method finds duplicated records.

        A record is a duplicate if it has the same values for the key columns (keys). This 
        method adds all cells of the duplicated row to the detection_dictionary. In the case of duplicates
        the first record is not markt as an duplicate, but all following.

        Arguments:
        dirtyDF -- dataframe of shape n_R (# of records) x n_A (# of attributes) - containing a dirty version of a dataset
        keys (List (String)) -- list of column names that define the key columns for the given dataset

        Returns:
        detection_dictionary -- dictionary - keys represent i,j of dirty cells & values are constant string "JUST A DUUMY VALUE"

        """
        start_time = time.time()

        # Extract the keys parameter
        keys = configs["keys"]

        key_col = pd.DataFrame(dirtydf, columns=keys)

        # if two records are duplicates, last one is set true in is_dup
        is_dup = key_col.duplicated(keep='first')
        is_dup = pd.DataFrame(is_dup, columns=['is_dup'])

        # adds every cell of row that is duplicate to detection dictionary
        detection_dictionary = {}

        for i, row in is_dup.iterrows():
            if row["is_dup"]==True:
                for col_j, col in enumerate(dirtydf.columns):
                    detection_dictionary[(i, col_j)] = "JUST A DUMMY VALUE"

        # get runtime
        error_detect_runtime = time.time() - start_time

        # get detector path
        detector_path = self.__get_detector_directory(str(DetectMethod.duplicatesdetector))

        # store detections in detector directory
        self.__store_detections(detection_dictionary, detector_path)

        precision, recall, f1 = self.__evaluate(detection_dictionary)

        evaluation_dict = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "detection_runtime": error_detect_runtime,
            "#detections": len(detection_dictionary),
            "#pattern_violations": None,
            "#fd_violations": None,
            "#detected_duplicates": len(is_dup[is_dup["is_dup"] == True]),
            "detected_error_rate": len(detection_dictionary) / dirtydf.size,
        }

        return detection_dictionary, evaluation_dict

    def mislabeldetector(self, dirtydf, dataset, configs):
        """
        This method finds mislabeles in a dataset given the correct labels.

        Compares all labels of dirtydf with the correct labels (correct_labels) and if not equal add
        detected (row,col) to detection dictionary

        Arguments:
        dirtyDF -- dataframe of shape n_R (# of records) x n_A (# of attributes) - containing a dirty version of a dataset
        correct_labels -- dataframe with shape n_R (# of records) x n_L (# of labels) - contains correct labels for dataset

        Returns:
        detection_dictionary -- dictionary - keys represent i,j of dirty cells & values are constant string "JUST A DUUMY VALUE"

        """

        start_time = time.time()

        # Extract the correct lables from configs
        correct_labels = configs["correct_labels"]

        # adds for every mislabeld cell the cells position (row, col) to detection dictionary
        detection_dictionary = {}

        for col_j, col in enumerate(correct_labels.columns):
            or_index = dirtydf.columns.get_loc(col)
            for i in range(0, len(correct_labels[col])):
                if dirtydf.iat[i,or_index] != correct_labels.iat[i, col_j]:
                    detection_dictionary[(i, or_index)] = "JUST A DUMMY VALUE"

        # get runtime
        error_detect_runtime = time.time() - start_time

        # get detector path
        detector_path = self.__get_detector_directory(str(DetectMethod.mislabeldetector))

        # store detections in detector directory
        self.__store_detections(detection_dictionary, detector_path)

        precision, recall, f1 = self.__evaluate(detection_dictionary)

        evaluation_dict = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "detection_runtime": error_detect_runtime,
            "#detections": len(detection_dictionary),
            "#pattern_violations": None,
            "#fd_violations": None,
            "#detected_duplicates": None,
            "detected_error_rate": len(detection_dictionary) / dirtydf.size,
        }

        return detection_dictionary, evaluation_dict

    def holoclean(self, dirtydf, dataset, configs):
        """
        This method finds attributes in the dataframe that violate the defined denial constraints.

        Arguments:
        dirtyDF -- dataframe of shape n_R (# of records) x n_A (# of attributes) - containing a dirty version of a dataset
        dataset (String) -- name of the dataset

        Returns:
        detection_dictionary -- dictionary - keys represent i,j of dirty cells & values are constant string "JUST A DUUMY VALUE"
        """

        start_time = time.time()

        # 1. Setup a HoloClean session.
        hc = holoclean.HoloClean(
            db_name='holo',
            domain_thresh_1=0,
            domain_thresh_2=0,
            weak_label_thresh=0.99,
            max_domain=10000,
            cor_strength=0.6,
            nb_cor_strength=0.8,
            epochs=10,
            weight_decay=0.01,
            learning_rate=0.001,
            threads=1,
            batch_size=1,
            verbose=False,
            timeout=3*60000,
            feature_norm=False,
            weight_norm=False,
            print_fw=True
        ).session

        
        # Concatinate all txt files in dataset/constraints into _all_constraints.txt
        dataset_dir = self.__get_detector_directory(dataset)
        try:
            dir = os.path.join(datasets_dictionary[dataset]["dataset_path"], "constraints")
            all_constraints_file  = open(os.path.join(dir, "_all_constraints.txt"), 'w+')
            all_constraints_file.truncate()
            for filename in os.listdir(dir):
                if filename.endswith(".txt") and not filename.startswith("_"):
                    with open(os.path.join(dir, filename), "r") as infile:
                        for line in infile:
                            all_constraints_file.write(line)
            all_constraints_file.close()
        except:
            logging.info("No constraints exist for the {} dataset".format(dataset))

        # 2. Load training data and denial constraints. Pass copy of dirtydf as load_data alters the parameter df
        copy_dirtydf = dirtydf.copy()
        hc.load_data(dataset, '',df=copy_dirtydf)
        hc.load_dcs(os.path.join(dir,"_all_constraints.txt"))
        hc.ds.set_constraints(hc.get_dcs())

        # detect errors with violation detector
        detectors = [ViolationDetector()]
        errors_df = hc.detect_errors(detectors, return_errors=True)

        # transform detected errors from dataframe to detection dictionary
        detection_dictionary = {}
        for index, row in errors_df.iterrows():
            detection_dictionary[(row['_tid_'], dirtydf.columns.get_loc(row['attribute']))] = "JUST A DUMMY VALUE"
        
        # get runtime
        error_detect_runtime = time.time() - start_time

        # get detector path
        detector_path = self.__get_detector_directory("holoclean")

        # store detections in detector directory
        self.__store_detections(detection_dictionary, detector_path)

        precision, recall, f1 = self.__evaluate(detection_dictionary)

        evaluation_dict = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "detection_runtime": error_detect_runtime,
            "#detections": len(detection_dictionary),
            "#pattern_violations": None,
            "#fd_violations": None,
            "#detected_duplicates": None,
            "detected_error_rate": len(detection_dictionary) / dirtydf.size,
        }
        
        return detection_dictionary, evaluation_dict

    def fahes(self, dirtydf, dataset, configs):
        """
        This method detects disguised missing values.

        In order to run this method, fahes has to be compiled. Therefore go to cleaners/FAHES/src run "make clean" and then "make all".

        Arguments:
        dirtyDF -- dataframe of shape n_R (# of records) x n_A (# of attributes) - containing a dirty version of a dataset
        dataset (String) -- name of the dataset
        path_to_dirtydf (String) -- path to the dirty dataframe, will be used by fahes.
        tool (String) -- which fahes component is to be run. SYN-OD = check syntactic outliers only;
                     RAND = detect DMVs that replace MAR values; NUM-OD = detect DMVs that are numerical outliers only;
                     ALL = check all DMVs

        Returns:
        detection_dictionary -- dictionary - keys represent i,j of dirty cells & values are constant string "JUST A DUUMY VALUE"
        evaluation_dict -- dictionary - hold evaluation information.
        """
        
        start_time = time.time()

        # Extract the path_to_dirtydf parameter
        path_to_dirtydf = configs["path_dirty"]
        detect_method = configs["detect_method"]

        # dict to map modules that fahes should run and its respective identifier
        module_dict={
            "SYN-OD" : 1,
            "RAND" : 2,
            "NUM-OD" : 3,
            "ALL" : 4
        }
        # run fahes and get path to results .csv
        path_fahes_res = executeFahes(path_to_dirtydf, module_dict[detect_method])
        
        # load results .csv as dataframe
        fahes_res_df = pd.read_csv(
            path_fahes_res,
            dtype=str,
            header="infer",
            encoding="utf-8",
            keep_default_na=False,
            low_memory=False,
        )

        detection_dictionary = {}

        # for each entry in fahes results go through the respective
        # column in dirtydf and mark every cell as detected that
        # has the DMV value defined in the fahes results entry
        for i_fahes, row_fahes in fahes_res_df.iterrows():
            for j_dirty, row_dirty in dirtydf.iterrows():
                # get index of respective column in dirtdf
                col_index = dirtydf.columns.get_loc(row_fahes["Attribute Name"])
                if row_dirty[row_fahes["Attribute Name"]] == row_fahes["DMV"]:
                    detection_dictionary[(j_dirty, col_index)] = "JUST A DUMMY VALUE"

        # get runtime
        error_detect_runtime = time.time() - start_time

        # get detector path
        detector_path = self.__get_detector_directory("{}_{}".format(str(DetectMethod.fahes), detect_method))

        # store detections in detector directory
        self.__store_detections(detection_dictionary, detector_path)

        precision, recall, f1 = self.__evaluate(detection_dictionary)

        evaluation_dict = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "detection_runtime": error_detect_runtime,
            "#detections": len(detection_dictionary),
            "#pattern_violations": None,
            "#fd_violations": None,
            "#detected_duplicates": None,
            "detected_error_rate": len(detection_dictionary) / dirtydf.size,
        }
        
        return detection_dictionary, evaluation_dict

    def dboost(self, dirtydf, dataset, configs):
        """
        This method detects outliers with dboost.

        Arguments:
        dirtyDF -- dataframe of shape n_R (# of records) x n_A (# of attributes) - containing a dirty version of a dataset
        dataset (String) -- name of the dataset

        Returns:
        detection_dictionary -- dictionary - keys represent i,j of dirty cells & values are constant string "JUST A DUUMY VALUE"
        evaluation_dict -- dictionary - hold evaluation information.
        """
        start_time = time.time()
        algorithm_and_configurations = []
        algorithm="OD"

        # Define the detector path
        detector_path = self.__get_detector_directory(str(DetectMethod.dboost))
        # Define the configuations JSON file
        configs_path = os.path.join(detector_path, 'dboost_configs.json')

        # Use the predefined configs, if defined in the dataset dictionary or in the configurations JSON file
        if 'dboost_configs' in datasets_dictionary[dataset]:
            configuration_list = [datasets_dictionary[dataset]['dboost_configs']]
        elif os.path.exists(configs_path):
            with open(configs_path, 'r') as fp:
                configs = json.load(fp)
            configuration_list = [configs]
        else:
            # list of configurations to run dboost mit
            configuration_list = [
                list(a) for a in
                list(itertools.product(["histogram"], ["0.1", "0.3", "0.5", "0.7", "0.9"], ["0.1", "0.3", "0.5", "0.7", "0.9"])) +
                list(itertools.product(["gaussian"], ["1.0", "1.3", "1.5", "1.7", "2.0", "2.3", "2.5", "2.7", "3.0"]))]
            random.shuffle(configuration_list)

        # run each strategy and save the evaluation results and detection dictionary
        # of the strategy with the highest f1 score
        best_f1 = -1.0
        best_strategy_profile={}
        for config in configuration_list:
            outputted_cells={}
            strategy_name = json.dumps([algorithm, config])
            strategy_name_hash = str(int(hashlib.sha1(strategy_name.encode("utf-8")).hexdigest(), 16))

            # create a tmp directory and write the dirty dataframe to it
            dataset_path = os.path.join(tempfile.gettempdir(), dataset + "-" + strategy_name_hash + ".csv")
            dirtydf.to_csv(dataset_path, sep=",", header=True, index=False, encoding="utf-8")

            # run dboost with respective parameters of configuration
            params = ["-F", ",", "--statistical", "0.5"] + ["--" + config[0]] + config[1:] + [dataset_path]
            run_dboost(params)

            # get results from dboost and create detection dictionar (outputted cells) of it
            # the remove the tmp directory
            algorithm_results_path = dataset_path + '-dboost_output.csv' 
            if os.path.exists(algorithm_results_path):
                ocdf = pd.read_csv(algorithm_results_path, sep=",", header=None, encoding="utf-8", dtype=str,
                                       keep_default_na=False, low_memory=False).apply(lambda x: x.str.strip())
                for i, j in ocdf.values.tolist():
                    if int(i) > 0:
                        outputted_cells[(int(i) - 1, int(j))] = "JUST A DUMMY VALUE"
                os.remove(algorithm_results_path)
            os.remove(dataset_path)

            # evaluate strategy save as best strategy it f1 score is so far the highest
            precision, recall, f1 = self.__evaluate(outputted_cells)
            # TODO to be removed
            logging.info("strategy:{}, f1:{}".format(strategy_name, f1))

            if f1>best_f1:
                best_strategy_profile={"name" : strategy_name, "detection_dict" : outputted_cells, "precision" : precision, "recall" : recall, "f1" : f1}
                best_config = config
                best_f1 = f1

        error_detect_runtime = time.time()-start_time
        detection_dictionary = best_strategy_profile["detection_dict"]

        # store detections in detector directory
        self.__store_detections(detection_dictionary, detector_path)

        # Store the best configurations, to be resued in subsequent executions
        with open(configs_path, 'w') as fp:
            json.dump(best_config, fp)
        
        evaluation_dict = {
            "precision": best_strategy_profile["precision"],
            "recall": best_strategy_profile["recall"],
            "f1": best_strategy_profile["f1"],
            "detection_runtime": error_detect_runtime,
            "#detections": len(detection_dictionary),
            "#pattern_violations": None,
            "#fd_violations": None,
            "#detected_duplicates": None,
            "detected_error_rate": len(detection_dictionary) / dirtydf.size,
        }
        
        return detection_dictionary, evaluation_dict

    def katara(self, dirtydf, dataset, configs):
        """
        This method detects cells that violate the knowledge base in cleaner/KATARA/knowledge-base.

        Arguments:
        dirtyDF -- dataframe of shape n_R (# of records) x n_A (# of attributes) - containing a dirty version of a dataset
        dataset (String) -- name of the dataset

        Returns:
        detection_dictionary -- dictionary - keys represent i,j of dirty cells & values are constant string "JUST A DUUMY VALUE"
        evaluation_dict -- dictionary - hold evaluation information.
        """

        start_time = time.time()

        # get list of Relations from knowledge base to run KATARA mit
        path_to_knowledge = os.path.join("cleaners","katara","knowledge-base")
        configuration_list = [os.path.join(path_to_knowledge, pat) for pat in os.listdir(path_to_knowledge)]
        random.shuffle(configuration_list)
        
        detection_dictionary = {}

        # fill detection_dictionary with detections based on different relations of knowledge-base
        for config in configuration_list:
            outputted_cells = run_KATARA(dirtydf, config)
            detection_dictionary.update({cell: "JUST A DUMMY VALUE" for cell in outputted_cells})
        
        error_detect_runtime = time.time()-start_time

        # get detector path
        detector_path = self.__get_detector_directory(str(DetectMethod.katara))

        # store detections in detector directory
        self.__store_detections(detection_dictionary, detector_path)
        
        precision, recall, f1 = self.__evaluate(detection_dictionary)

        evaluation_dict = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "detection_runtime": error_detect_runtime,
            "#detections": len(detection_dictionary),
            "#pattern_violations": None,
            "#fd_violations": None,
            "#detected_duplicates": None,
            "detected_error_rate": len(detection_dictionary) / dirtydf.size,
        }
        
        return detection_dictionary, evaluation_dict
        
    def activeclean(self, dirtydf, dataset, configs):
        """
        Detects cells by learning from pre-labeled cells (#sampling_budget).

        Arguments:
        dirtyDF -- dataframe of shape n_R (# of records) x n_A (# of attributes) - containing a dirty version of a dataset
        dataset (String) -- name of the dataset
        configs (dict) -- has to contain "sampling_budget" (int), which is the number of labeled tuples (from actual_errors) that are beeing used for learning

        Returns:
        detection_dictionary -- dictionary - keys represent i,j of dirty cells & values are constant string "JUST A DUUMY VALUE"
        evaluation_dict -- dictionary - hold evaluation information.
        """
        
        start_time = time.time()
        sampling_budget = configs["sampling_budget"]

        actual_errors_dictionary = self.actual_errors
        vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(min_df=1, stop_words="english")
        text = [" ".join(row) for row in dirtydf.values.tolist()]
        acfv = vectorizer.fit_transform(text).toarray()
        labeled_tuples = {}
        adaptive_detector_output = []
        detection_dictionary = {}
        while len(labeled_tuples) < sampling_budget:
            if len(adaptive_detector_output) < 1:
                adaptive_detector_output = [i for i in range(dirtydf.shape[0]) if i not in labeled_tuples]
            labeled_tuples.update({i: 1 for i in np.random.choice(adaptive_detector_output, 1, replace=False)})
            x_train = []
            y_train = []
            for i in labeled_tuples:
                x_train.append(acfv[i, :])
                y_train.append(int(sum([(i, j) in actual_errors_dictionary for j in range(dirtydf.shape[1])]) > 0))
            adaptive_detector_output = []
            x_test = [acfv[i, :] for i in range(dirtydf.shape[0]) if i not in labeled_tuples]
            test_rows = [i for i in range(dirtydf.shape[0]) if i not in labeled_tuples]
            if sum(y_train) == len(y_train):
                predicted_labels = len(test_rows) * [1]
            elif sum(y_train) == 0 or len(x_train[0]) == 0:
                predicted_labels = len(test_rows) * [0]
            else:
                model = sklearn.linear_model.SGDClassifier(loss="log", alpha=1e-6, max_iter=200, fit_intercept=True)
                model.fit(x_train, y_train)
                predicted_labels = model.predict(x_test)
            detection_dictionary = {}
            for index, pl in enumerate(predicted_labels):
                i = test_rows[index]
                if pl:
                    adaptive_detector_output.append(i)
                    for j in range(dirtydf.shape[1]):
                        detection_dictionary[(i, j)] = "JUST A DUMMY VALUE"
            for i in labeled_tuples:
                for j in range(dirtydf.shape[1]):
                    detection_dictionary[(i, j)] = "JUST A DUMMY VALUE"
        
        error_detect_runtime = time.time()-start_time

        # get detector path
        detector_path = self.__get_detector_directory(str(DetectMethod.activeclean))

        # store detections in detector directory
        self.__store_detections(detection_dictionary, detector_path)
        
        precision, recall, f1 = self.__evaluate(detection_dictionary)

        evaluation_dict = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "detection_runtime": error_detect_runtime,
            "#detections": len(detection_dictionary),
            "#pattern_violations": None,
            "#fd_violations": None,
            "#detected_duplicates": None,
            "detected_error_rate": len(detection_dictionary) / dirtydf.size,
        }
        
        return detection_dictionary, evaluation_dict

    def metadata_driven(self, dirtydf, dataset, configs):
        """
        Mechine Learning Aggregator that combines detection results of dboost, nadeef and katara.
        Nadeef returns an empty detection dictionary if there are not constraints defined for the dataset.

        Arguments:
        dirtyDF -- dataframe of shape n_R (# of records) x n_A (# of attributes) - containing a dirty version of a dataset
        dataset (String) -- name of the dataset
        configs (dict) -- has to contain "sampling_budget" (int), which is the number of labeled tuples (from actual_errors) that are beeing used for learning

        Returns:
        detection_dictionary -- dictionary - keys represent i,j of dirty cells & values are constant string "JUST A DUUMY VALUE"
        evaluation_dict -- dictionary - hold evaluation information.
        """

        start_time = time.time()

        sampling_budget = configs["sampling_budget"]
        actual_errors_dictionary = self.actual_errors

        # Define the required configurations
        configs = {}

        # run stand-alone detection algorithms
        dboost_output = self.dboost(dirtydf, dataset, configs)
        nadeef_output = self.nadeef(dirtydf, dataset, configs)
        katara_output = self.katara(dirtydf, dataset, configs)
        
        lfv = {}
        columns_frequent_values = {}

        # get top 10 values (in terms of frequency) for each column
        for j, attribute in enumerate(dirtydf.columns.tolist()):
            fd = {}
            for value in dirtydf[attribute].tolist():
                if value not in fd:
                    fd[value] = 0
                fd[value] += 1
            sorted_fd = sorted(fd.items(), key=operator.itemgetter(1), reverse=True)[:int(dirtydf.shape[0] / 10.0)]
            
            # store value + frequency the top 10 values of column j
            columns_frequent_values[j] = {v: f for v, f in sorted_fd}
        
        # cell_list is 1 Dimensional list with cartesian product of row x columns as iterator (row, col)
        cells_list = list(itertools.product(range(dirtydf.shape[0]), range(dirtydf.shape[1])))
        
        # create feature for every cell: The feature is a single integer
        # which gets incremented by one for every condition that applies to the 
        # corrosponding cell
        for cell in cells_list:
            lfv[cell] = []
            lfv[cell] += [1 if cell in dboost_output else 0]
            lfv[cell] += [1 if cell in nadeef_output else 0]
            lfv[cell] += [1 if cell in katara_output else 0]
            value = dirtydf.iloc[cell[0], cell[1]]
            lfv[cell] += [1 if value in columns_frequent_values[cell[1]] else 0]
            lfv[cell] += [1 if re.findall(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", value) else 0]
            lfv[cell] += [1 if re.findall("https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+", value) else 0]
            lfv[cell] += [1 if re.findall("^[\d]+$", value) else 0]
            lfv[cell] += [1 if re.findall(r"[\w.-]+@[\w.-]+", value) else 0]
            lfv[cell] += [1 if re.findall("^[\d]{16}$", value) else 0]
            lfv[cell] += [1 if value.lower() in ["m", "f"] else 0]
            lfv[cell] += [1 if re.findall("^[\d]{4,6}$", value) else 0]
            lfv[cell] += [1 if not value else 0]
            for la, ra in self.__DATASET_CONSTRAINTS[dataset]["functions"]:
                lfv[cell] += [1 if dirtydf.columns.tolist()[cell[1]] in [la, ra] else 0]
        random_tuples_list = [i for i in random.sample(range(dirtydf.shape[0]), dirtydf.shape[0])]
        labeled_tuples = {i: 1 for i in random_tuples_list[:sampling_budget]}

        x_train = []
        y_train = []

        # create training set
        for cell in cells_list:
            # if row of cell is in tuples that should be used in training set 
            # add feature (lfv[cell]) to train set with label = true if cell is 
            # an error (based on actual_errors_dictionary)
            if cell[0] in labeled_tuples:
                x_train.append(lfv[cell])
                y_train.append(int(cell in actual_errors_dictionary))
        detection_dictionary = {}
        if sum(y_train) != 0:

            # create x_test with all features lfv[cell] for every cell
            x_test = [lfv[cell] for cell in cells_list]
            test_cells = [cell for cell in cells_list]

            # train model and apply to x_train
            if sum(y_train) != len(y_train):
                model = sklearn.ensemble.AdaBoostClassifier(n_estimators=6)
                model.fit(x_train, y_train)
                predicted_labels = model.predict(x_test)
            else:
                predicted_labels = len(test_cells) * [1]
            
            detection_dictionary = {}

            # fill detection_dictionary with labels for each cell
            for index, pl in enumerate(predicted_labels):
                # get position of cell (row, col)
                cell = test_cells[index]

                # if cell was in training_set we already now the correct label
                # based on actual_error_dict
                if cell[0] in labeled_tuples:
                    if cell in actual_errors_dictionary:
                        detection_dictionary[cell] = "JUST A DUMMY VALUE"
                elif pl:
                    detection_dictionary[cell] = "JUST A DUMMY VALUE"

        error_detect_runtime = time.time()-start_time

        # get detector path
        detector_path = self.__get_detector_directory(str(DetectMethod.metadata_driven))

        # store detections in detector directory
        self.__store_detections(detection_dictionary, detector_path)
        
        precision, recall, f1 = self.__evaluate(detection_dictionary)

        evaluation_dict = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "detection_runtime": error_detect_runtime,
            "#detections": len(detection_dictionary),
            "#pattern_violations": None,
            "#fd_violations": None,
            "#detected_duplicates": None,
            "detected_error_rate": len(detection_dictionary) / dirtydf.size,
        }
        
        return detection_dictionary, evaluation_dict
        
    def openrefine(self, dirtydf, dataset, configs):
        """
        uses clusters generated by openrefine to create detection dictionary. Clusters have to be
        exported from openrefine as json and stored in dataset/clusters. For each 
        choise in the cluster those cells that do not match the proposed value are added to the detection_dict
        
        For generating the clusters in OpenRefine "key collision" is used as Method and "fingerprint" as keying function
        """
        
        start_time = time.time()
        dataset_dir = self.__get_detector_directory(dataset)
        
        # check if path to clusters folder exist
        try:
            dir = os.path.join(datasets_dictionary[dataset]["dataset_path"], "clusters")
        except:
            logging.info("No clusters exist for the {} dataset".format(dataset))
            sys.exit(1)
        
        detection_dictionary = {}
        
        # iterate over each json in /clusters and extract row, col for each value
        # that doe not equal the proposed_value by openrefine
        for filename in os.listdir(dir):
            if filename.endswith(".json"):
                with open(os.path.join(dir, filename)) as file:
                    clusters_dict = json.load(file)
        
                col_name = clusters_dict["columnName"]
                if col_name in dirtydf.columns:
                    col = dirtydf.columns.get_loc(col_name)
                    for cluster in clusters_dict["clusters"]:
                        correct_value = cluster["value"]
                        for choise in cluster["choices"]:
                            if choise["v"] != correct_value:
                                row_list = dirtydf.index[dirtydf[col_name]== choise["v"]]
                                for row in row_list:
                                    detection_dictionary[(row, col)] = "JUST A DUMMY"

        error_detect_runtime = time.time()-start_time

        # get detector path
        detector_path = self.__get_detector_directory(str(DetectMethod.openrefine))

        # store detections in detector directory
        self.__store_detections(detection_dictionary, detector_path)
        
        precision, recall, f1 = self.__evaluate(detection_dictionary)

        evaluation_dict = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "detection_runtime": error_detect_runtime,
            "#detections": len(detection_dictionary),
            "#pattern_violations": None,
            "#fd_violations": None,
            "#detected_duplicates": None,
            "detected_error_rate": len(detection_dictionary) / dirtydf.size,
        }
        
        return detection_dictionary, evaluation_dict
    
    def max_entropy(self, dirtydf, dataset, configs):
        """
        It should be executed after running other detectors.
        Run at the very end, after all other non-ensemble detectors ran.
        Ensemble method that tries to minimize the human involvement (here simulated by the actual_error_dict). 
        Each detection_dict (of stand_alone detectors) is treated as the result of a detection tool. Firstly all detection_dicts get loaded,
        secondly the precision of each detection_dict (each detection tool) is predicted, thirdly the tool with 
        the highest precision is chosen and added to the overall_detection_dict (that is beeing returned). Then the correctly
        detected errors by the chose tool get removed from all other detection_dicts. Then start over with the second step.

        Arguments:
        dirtyDF -- dataframe of shape n_R (# of records) x n_A (# of attributes) - containing a dirty version of a dataset
        dataset (String) -- name of the dataset
        configs (dict) -- has to contain "precision_threshold" (int) and "sampling_budget" (int) where the first defines the stopping
        condition for the algorithm (i.e. if there is no tool with an precision that is higer that the threshold the algorithm terminates). The
        sampling budget defines the number of samples based on which the precision of each tool gets estimates

        Returns:
        detection_dictionary -- dictionary - keys represent i,j of dirty cells & values are constant string "JUST A DUUMY VALUE"
        evaluation_dict -- dictionary - hold evaluation information.
        """
        
        start_time = time.time()
        precision_threshold = configs["precision_threshold"]
        # number of samples on which precision is estimated
        sampling_budget = configs["sampling_budget"]
        
        # List of all detections dictionaries for dataset. Loads all detections.csv files in
        # dataset directory into detection dictionaries and adds them to list (detection_dicts)
        detection_dicts = []
        for root, dirs, files in os.walk(datasets_dictionary[self.__dataset_name]["dataset_path"]):
            if "detections.csv" in files:
                try:
                    if os.path.basename(root) not in ensemble_detectors:
                        reader = pd.read_csv(os.path.join(root, "detections.csv"), names=['i','j','dummy'])
                        detection_dicts.append(reader.groupby(['i','j'])['dummy'].apply(list).to_dict())
                except:
                    pass
        
        overall_detection_dictionary={}
        while len(detection_dicts)>0:
            best_precision = -1
            best_detections_index = -1
            # estimate precision for each detections dict
            for i, detections in enumerate(detection_dicts):
                tp = 0.0

                # get samples of detections dictionary
                if len(detections)<=sampling_budget:
                    samples = list(detections.keys())
                else: 
                    samples = random.sample(list(detections.keys()), sampling_budget)

                for cell in samples:
                    if cell in self.actual_errors:
                        tp += 1
                
                # get precision for detections and check if it has the best precision so far
                precision = 0.0 if len(detections)==0 else tp/len(detections)
                #logging.info("%d-te detection_dict, precision %f" %(i, precision))
                if precision > best_precision:
                    best_precision=precision
                    best_detections_index=i
            
            # if best precision is smaller then the required threshold
            # break out while loop
            if best_precision<precision_threshold:
                break
            
            # add detections of best detections dict to overall detections
            overall_detection_dictionary.update(detection_dicts[best_detections_index])
            
            # remove already correctly detected cells from other detection dict, so that 
            # they the precision gets estimated based on the deteted errors (that actually are errors), 
            # that were not yet detected
            for i, detections in enumerate(detection_dicts):
                if i != best_detections_index:
                    for key in detection_dicts[best_detections_index].keys():
                        if key in detections and key in self.actual_errors:    
                            detections.pop(key)
                
            # remove detections dict with best precision from list of detection_dicts
            detection_dicts.pop(best_detections_index)
        
        detection_dictionary = overall_detection_dictionary
        error_detect_runtime = time.time()-start_time

        # get detector path
        detector_path = self.__get_detector_directory(str(DetectMethod.max_entropy))

        # store detections in detector directory
        self.__store_detections(detection_dictionary, detector_path)
        
        precision, recall, f1 = self.__evaluate(detection_dictionary)

        evaluation_dict = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "detection_runtime": error_detect_runtime,
            "#detections": len(detection_dictionary),
            "#pattern_violations": None,
            "#fd_violations": None,
            "#detected_duplicates": None,
            "detected_error_rate": len(detection_dictionary) / dirtydf.size,
        }
        return detection_dictionary, evaluation_dict
        
    def min_k(self, dirtydf, dataset, configs):
        """
        Run at the very end, after all other non-ensembledetectors ran.
        Considers those errors that are detected by at least configs["threshold"] percent of detectors (only stand alone detectors).
        
        Arguments:
        dirtyDF -- dataframe of shape n_R (# of records) x n_A (# of attributes) - containing a dirty version of a dataset
        dataset (String) -- name of the dataset
        configs (dict) -- has to contain "threshold" which specifies the minimum percentage of how
        many detectors need to detect an error in order to be included
        
        Returns:
        detection_dictionary -- dictionary - keys represent i,j of dirty cells & values are constant string "JUST A DUUMY VALUE"
        evaluation_dict -- dictionary - hold evaluation information.
        """

        start_time = time.time()
        threshold = configs["threshold"]

        # List of all detections dictionaries for dataset. Loads all detections.csv files in
        # dataset directory into detection dictionaries and adds them to list (detection_dicts)
        detection_dicts = []
        for root, dirs, files in os.walk(datasets_dictionary[self.__dataset_name]["dataset_path"]):
            if "detections.csv" in files:
                try:
                    # only consider detections generated by stand alone detectors and not ensemble detectors
                    if os.path.basename(root) not in ensemble_detectors:
                        reader = pd.read_csv(os.path.join(root, "detections.csv"), names=['i','j','dummy'])
                        detection_dicts.append(reader.groupby(['i','j'])['dummy'].apply(list).to_dict())
                except:
                    pass

        # for each detected cell count the number of times it was detected over all detection_dicts
        cells_counter = {}
        for i, detections in enumerate(detection_dicts):
            for cell in detections.keys():
                if cell not in cells_counter:
                    cells_counter[cell] = 0.0
                cells_counter[cell] += 1.0

        # for each detected error get percentage of detectors that detected the error
        for cell in cells_counter:
            cells_counter[cell] /= len(detection_dicts)

        # fill detection_dictionary with detections that have been detected
        # by a minimum of threshold-percent of the detectors
        detection_dictionary = {}
        for cell in cells_counter:
            if cells_counter[cell] >= threshold:
                detection_dictionary[cell] = "JUST A DUMMY VALUE"

        error_detect_runtime = time.time()-start_time

        # get detector path
        detector_path = self.__get_detector_directory(str(DetectMethod.min_k))

        # store detections in detector directory
        self.__store_detections(detection_dictionary, detector_path)

        precision, recall, f1 = self.__evaluate(detection_dictionary)

        evaluation_dict = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "detection_runtime": error_detect_runtime,
            "#detections": len(detection_dictionary),
            "#pattern_violations": None,
            "#fd_violations": None,
            "#detected_duplicates": None,
            "detected_error_rate": len(detection_dictionary) / dirtydf.size,
        }
        return detection_dictionary, evaluation_dict
    
    def zeroer(self, dirtydf, dataset, configs):
        """
        Finds tuples/rows that represent the same real world entity through zeroer.
        
        Need to provide a blocking function in /cleaners/zeroer/blocking_functions.py for each dataset
        and add it to the mapping of dataset and function in the bottom of blocking_functions.py. Dataset column names must not contain
        "(", ")", "-", ".", "=", "%"
    
        Arguments:
        dirtyDF -- dataframe of shape n_R (# of records) x n_A (# of attributes) - containing a dirty version of a dataset
        dataset (String) -- name of the dataset
        configs (dict) -- 
        
        Returns:
        detection_dictionary -- dictionary - keys represent i,j of dirty cells & values are constant string "JUST A DUUMY VALUE"
        evaluation_dict -- dictionary - hold evaluation information.
        """
        
        start_time = time.time()
        # create a tmp directory and write the dirty dataframe to it
        dataset_path = os.path.join(tempfile.gettempdir(), dataset + ".csv")
        metadata_path = os.path.join(os.path.dirname(dataset_path), "metadata.txt")

        # Delete temp folder if still exists, otherwise previously calculated features will be loaded
        #try:
        #    shutil.rmtree(os.path.dirname(dataset_path), ignore_errors=False)
        #except:
        #    pass

        # if existent delete id column and replace it with another id colum
        # important to assure that ids are from 0 to #rows-1
        if "id" in dirtydf.columns:
            dirtydf.drop('id', inplace=True, axis=1)
        dirtydf["id"] = [id for id in range(0, dirtydf.shape[0])]
        dirtydf.to_csv(dataset_path, sep=",", header=True, index=False, encoding="utf-8")
        
        # create metadata file in tmp folder and set same .csv as left and right relation
        f = open(metadata_path, 'w')
        f.write(dataset+".csv"+"\n")
        f.write(dataset+".csv"+"\n")
        f.close()  
        
        # get entity resolution from zeroer
        pred_df = get_zeroer(dataset, os.path.dirname(dataset_path))
    
        # transform predictions into dictionary with (id1,id2) as key where
        # id1 is always smaller then id2
        pred_dict = {}
        for row_i, row in pred_df.iterrows():
            if int(row["ltable_id"])<int(row["rtable_id"]):
                pred_dict[(row["ltable_id"],row["rtable_id"])] = row["pred"]
            else:
                pred_dict[(row["rtable_id"],row["ltable_id"])] = row["pred"]

        # transform tuples that are predicted to refer to the same entity into
        # detection_dict. Every cell of tuple/row is treated as one error
        detection_dictionary = {}
        for (ltable_id, rtable_id), prediction in pred_dict.items():
            if prediction == 1  and not ltable_id == rtable_id:
                for col in dirtydf.columns:
                    col_i = dirtydf.columns.get_loc(col)
                    detection_dictionary[(int(rtable_id), col_i)] = "JUST A DUMMY VALUE"
                
        # delete temporary dict
        shutil.rmtree(os.path.dirname(dataset_path), ignore_errors=True)
        
        error_detect_runtime = time.time()-start_time

        # get detector path
        detector_path = self.__get_detector_directory(str(DetectMethod.zeroer))

        # store detections in detector directory
        self.__store_detections(detection_dictionary, detector_path)
        
        precision, recall, f1 = self.__evaluate(detection_dictionary)

        evaluation_dict = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "detection_runtime": error_detect_runtime,
            "#detections": len(detection_dictionary),
            "#pattern_violations": None,
            "#fd_violations": None,
            "#detected_duplicates": None,
            "detected_error_rate": len(detection_dictionary) / dirtydf.size,
        }
        return detection_dictionary, evaluation_dict
    
    def cleanlab(self, dirtydf, dataset, configs):
        """
        This method uses cleanlab library in order to find errors in noisy labels.
        
        Arguments:
        dirtyDF -- dataframe of shape n_R (# of records) x n_A (# of attributes) - containing a dirty version of a dataset
        dataset (String) -- name of the dataset
        configs (dict) -- "model_name" has to be provided and be contained in /auxilaries/models_dictionary
        
        Returns:
        detection_dictionary -- dictionary - keys represent i,j of dirty cells & values are constant string "JUST A DUUMY VALUE"
        evaluation_dict -- dictionary - hold evaluation information.
        """
        
        start_time = time.time()
        # set model that is to be used
        model_name = configs["model_name"]
        
        # get daataset specific ml_task and model
        for model_type in models:
            if model_name in models_dictionary[model_type]:
                ml_task = model_type
                model = models_dictionary[model_type][model_name]

        # return empty if model_task is not classification
        if ml_task != classification:
            return {},{}

        # instantiate  model class
        app = Models(dataset)

        x_train, y_train, x_test, y_test = app.preprocess(dirtydf, ml_task)

        if not isinstance(x_train, np.ndarray):
            x_train = x_train.toarray()

        # Use the ML model with default parameters (i.e., not optimized)
        estimator = model["fn"](**model["fixed_params"])
        
        # for svc classifier
        if "probability" in estimator.__dict__.keys():
            setattr(estimator, 'probability', True)
        
        # if possible call cleanlab function that creates psx with StratifiedKFolt
        # else just use one fold
        (unique, class_counts) = np.unique(np.concatenate((y_train, y_test), axis=0), return_counts=True)
        if min(class_counts) == 1:
            estimator.fit(np.concatenate((x_train, x_test), axis=0), np.concatenate((y_train, y_test), axis=0))
            psx = estimator.predict_proba(np.concatenate((x_train, x_test), axis=0))
        else:
            _, psx = estimate_confident_joint_and_cv_pred_proba(
                X=np.concatenate((x_train, x_test), axis=0), 
                s=np.concatenate((y_train, y_test), axis=0),
                clf=estimator,
                cv_n_folds = 5 if min(class_counts)>=5 else min(class_counts),
                seed=1
            )
        
        # Label errors are ordered by likelihood of being an error.
        ordered_label_errors = get_noise_indices(
            s=np.concatenate((y_train, y_test), axis = 0),
            psx=psx,
            sorted_index_method='normalized_margin', # Orders label errors
        )
        
        # create detection dict from list of errors
        detection_dictionary={}
        col_i = dirtydf.columns.get_loc(datasets_dictionary[dataset]["labels_clf"][0])
        for x in ordered_label_errors:
            detection_dictionary[(x,col_i)] = "JUST A DUMMY VARIABLE"
        
        error_detect_runtime = time.time()-start_time

        # get detector path"outlierdetector_{}".format(detect_method)
        detector_path = self.__get_detector_directory("{}-{}".format(str(DetectMethod.cleanlab), model_name))

        # store detections in detector directory
        self.__store_detections(detection_dictionary, detector_path)
        
        precision, recall, f1 = self.__evaluate(detection_dictionary)

        evaluation_dict = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "detection_runtime": error_detect_runtime,
            "#detections": len(detection_dictionary),
            "#pattern_violations": None,
            "#fd_violations": None,
            "#detected_duplicates": None,
            "detected_error_rate": len(detection_dictionary) / dirtydf.size,
        }
        return detection_dictionary, evaluation_dict

    def picket(self, dirtydf, dataset, configs):
        """
        This method the PicketNetModel. It detects corrupted rows. We mark every cell of the detected row
        as an error. Only possible for datasets with classification task. The PicketNetModel parameters can be altered below
        as well as the embed_dim when calling TrainTestSplit.
        
        Arguments:
        dirtyDF -- dataframe of shape n_R (# of records) x n_A (# of attributes) - containing a dirty version of a dataset
        dataset (String) -- name of the dataset
        configs (dict) -- 
        
        Returns:
        detection_dictionary -- dictionary - keys represent i,j of dirty cells & values are constant string "JUST A DUUMY VALUE"
        evaluation_dict -- dictionary - hold evaluation information.
        """

        # column tzpe is text if more than 9 unique values in column
        def infer_column_type(c, data):
            if np.issubdtype(c, np.number):
                return "numeric"
            if data.unique().shape[0] >= 10:
                return "text"
            return "categorical"
        
        start_time = time.time()
        
        # transform dirtdf which has dtype string to numeric type if possible
        dirtydf = dirtydf.apply(pd.to_numeric, errors="ignore")
        
        feature_dtypes = []
        # infere dtypes for features (NOT LABEL). possible options are "numeric", "categorical", "text"
        for col_i, t in enumerate(dirtydf.dtypes):
            
            # only for features, not for label
            if not(datasets_dictionary[dataset]["labels_clf"][0] == dirtydf.columns[col_i]):
                feature_dtypes.append(infer_column_type(t, dirtydf.iloc[:, col_i]))
            
        # set train size to 1 because we want to get detections for the entire data
        ds = TrainTestSplit(dataset, dirtydf, datasets_dictionary[dataset]["labels_clf"][0], train_size=1.0, dtypes=feature_dtypes, embed_dim=64, resample=False, save=False)
        
        X_train = ds["X_train"]
        idx_train = ds["idx_train"]
        
        # Train PicketNet and get the loss in the early stage
        param = {
            'model_dim': 64,
            'input_dim': X_train.shape[2],
            'attribute_num': X_train.shape[1],
            'transformer_layer': 6,
            'head_num': 2,
            'hidden_dim': 64,
            'dropout': 0.1,
            'numerical_ids': [i for i, x in enumerate(feature_dtypes) if x == "numeric"],
            'categorical_ids': [i for i, x in enumerate(feature_dtypes) if x == 'categorical'],
            # Use a learnable lookup for categorical attributes
            'useEncoding': True,
            'batch_size': 500,
            'epochs': 0, # can be set to 0 we only need the warmup and trim epochs
            'loss_warm_up_epochs': 4, # originally 50
            'loss_trim_epochs': 2, # originally 20
            # The proportion to remove after early stage training
            'loss_trim_p': 0.2
        }
        
        PicketN = PicketNetModel(param)
        attribute_info = [Attribute(ds['vec'][i]) for i in range(len(ds['vec']))]
        PicketN.loadData(torch.Tensor(X_train).double(), None, attribute_info, 
                    tuple_idx = torch.Tensor(idx_train))
        
        # calculates a loss for each row indexes and then rows with outlier losses
        # are saved in PicketN.indices_to_remove
        PicketN.loss_based_train()
        detected_rows = PicketN.indices_to_remove
        
        detection_dictionary={}
        for row_index in detected_rows:
            for col_j, col in enumerate(dirtydf.columns):
                detection_dictionary[(row_index, col_j)] = "JUST A DUMMY VALUE"
        
        error_detect_runtime = time.time()-start_time

        # get detector path"outlierdetector_{}".format(detect_method)
        detector_path = self.__get_detector_directory(str(DetectMethod.picket))

        # store detections in detector directory
        self.__store_detections(detection_dictionary, detector_path)
        
        precision, recall, f1 = self.__evaluate(detection_dictionary)

        evaluation_dict = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "detection_runtime": error_detect_runtime,
            "#detections": len(detection_dictionary),
            "#pattern_violations": None,
            "#fd_violations": None,
            "#detected_duplicates": None,
            "detected_error_rate": len(detection_dictionary) / dirtydf.size,
        }

        return detection_dictionary, evaluation_dict
    
    def ed2(self, dirtydf, dataset, configs):
        """
        This method calls the ed2 algorithm to detect incorrect cells. Ed2 uses a active learning approach, thus the
        groundtruth dataset is needed (to simulate the human).
        
        Arguments:
        dirtyDF -- dataframe of shape n_R (# of records) x n_A (# of attributes) - containing a dirty version of a dataset
        dataset (String) -- name of the dataset
        configs (dict) -- "label_cutoff" specifies the budget of labels in the active learning process. The algoithm stop
            as soon as the number of labels exceeds the label_cutoff (in worst case it is 9 higher than the label_cutoff)
        
        Returns:
        detection_dictionary -- dictionary - keys represent i,j of dirty cells & values are constant string "JUST A DUUMY VALUE"
        evaluation_dict -- dictionary - hold evaluation information.
        """
        
        start_time = time.time()
        
        # groundtruth is needed to simulate human in the lopp during active learning
        cleanDF = pd.read_csv(
                os.path.abspath(
                os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset, "clean.csv")),
                dtype=str,
                header="infer",
                encoding="utf-8",
                keep_default_na=False,
                low_memory=False,
            )

        label_cutoff = configs["label_cutoff"]
        
        # dataframe of same size as dirtydf that contains flase/true for every cell
        # true meaning, that the cell is an error. labels is the number of labels used in active learning
        all_error_statusDF, labels = run_ed2(cleanDF, dirtydf, dataset, label_cutoff)

        detection_dictionary = {}
        for row_i in range(dirtydf.shape[0]):
            for col_i in range(dirtydf.shape[1]):
                if all_error_statusDF.iat[row_i, col_i] == True:
                    detection_dictionary[(row_i, col_i)] = "JUST A DUMMY VALUE"
        
        error_detect_runtime = time.time()-start_time

        # get detector path"outlierdetector_{}".format(detect_method)
        detector_path = self.__get_detector_directory(str(DetectMethod.ed2))

        # store detections in detector directory
        self.__store_detections(detection_dictionary, detector_path)
        
        precision, recall, f1 = self.__evaluate(detection_dictionary)

        evaluation_dict = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "detection_runtime": error_detect_runtime,
            "#detections": len(detection_dictionary),
            "#pattern_violations": None,
            "#fd_violations": None,
            "#detected_duplicates": None,
            "detected_error_rate": len(detection_dictionary) / dirtydf.size,
        }

        return detection_dictionary, evaluation_dict

###################################################


###################################################
if __name__ == "__main__":
    detector = Detectors("name")
    detection_dictionary, evaluation_dict = detector.ed2({}, "flights", {})

###################################################
