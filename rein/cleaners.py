################################################
# Benchmark: A collection of data repair methods
# Authors: Christian Hammacher, Mohamed Abdelaal
# Date: February 2021
# Software AG
# All Rights Reserved
################################################

################################################
from rein.auxiliaries.configurations import *
from rein.auxiliaries.datasets_dictionary import datasets_dictionary
from rein.auxiliaries.models_dictionary import models_dictionary
from rein.models import Models, models
from rein.datasets import Datasets, Database

import math
import numbers
import time
from sklearn import preprocessing
import datawig
import os
import pandas as pd
from missingpy import MissForest
from impyute.imputation.cs import em as impEM
from sklearn.preprocessing import OrdinalEncoder
from cleanlab.classification import LearningWithNoisyLabels
from sklearn.impute import KNNImputer as sklearnKNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer as sklearnIterativeImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, accuracy_score

# missingpy calles sklearn.neighbors.base (older scikit-learn version)
# but got renamed to sklearn.neighbors._base
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base


# Create a path to the cleaners directory
#cleaners_path = os.path.join(os.path.dirname(__file__), os.pardir, "cleaners")
# Add the cleaners directory to the system path
sys.path.append(os.path.abspath("cleaners"))

import holoclean.holoclean
from holoclean.detect import NullDetector, ViolationDetector
from holoclean.repair.featurize import *
from holoclean.dataset.table import Table, Source

from raha.correction import Correction as Raha_Correction
from raha.dataset import Dataset as Raha_Dataset
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support

from CPClean.code.repair.repair import repair
from CPClean.code.cleaner.boost_clean import boost_clean, modified_boost_clean
from CPClean.code.training.knn import KNN
from CPClean.code.training.preprocess import preprocess
from CPClean.code.cleaner.CPClean.clean import CPClean

################################################


class Cleaners:
    """
    A class containing all data reapir methods 
    """

    def __init__(self, dataset_name, detector_name, groundTruthDF, actual_errors_dict, store_postgres=None):
        """
        The constructor

        Arguments:
        dataset_name (String) -- name of the examined dataset
        detector_name (String) -- name of the detector for which results the respective cleaner is applied
        groundTruthDF (dataframe) -- ground truth of the dataset
        """
        self.__dataset_name = dataset_name
        self.__detector_name = detector_name
        self.groundTruthDF = groundTruthDF
        self.actual_errors_dict = actual_errors_dict
        self.store_postgres = store_postgres
        self.detect_clean_dict = {
            "duplicatesdetector": [self.duplicatesCleaner],
            "mvdetector": [self.standardImputer],
            "nadeef": [self.cleanWithGroundTruth],
            "outlierdetector_IF": [self.standardImputer],
            "outlierdetector_IQR": [self.standardImputer],
            "outlierdetector_SD": [self.standardImputer],
            "raha": [self.cleanWithGroundTruth, self.duplicatesCleaner, self.standardImputer],
        }
        # List all implemented cleaning methods
        self.cleaners_list = [
            self.standardImputer,
            self.mlImputer,
            self.baran,
            self.cleanWithGroundTruth,
            self.duplicatesCleaner,
            self.dcHoloCleaner,
            self.openrefine,
            self.lop,
        ]
        self.model_oriented_cleaners_list = [
            self.boostClean,
            self.CPClean,
            self.cleanlab,
            self.activecleanCleaner,
        ]

    def __get_cleaner_directory(self, cleaner_name):
        """
        This method creates a cleaner directory, if not exist, and return a path to this directory

        :Arguments:
        cleaner_name --String denoting the name of a cleaner

        Returns:
        cleaner_directory -- String denoting the path to the cleaner directory
        """
        path = datasets_dictionary[self.__dataset_name]["dataset_path"]

        cleaner_directory = os.path.join(path, self.__detector_name, cleaner_name)
        if not os.path.exists(cleaner_directory) and not self.store_postgres:
            # creating a new directory if it does not exit
            os.mkdir(cleaner_directory)

        return cleaner_directory

    def __store_cleaned_data(self, cleanedDF, cleaner_path):
        """
        stores given dataframe as .csv in cleaner_path or as postgres relations in REIN database

        Arguments:
        cleanedDF (dataframe) -- dataframe that was cleaned
        cleaner_path (String) -- path to the folder in which cleaned dataframe should be stored
        """
        # Store the repaired data in PostgreSQL
        # PostgreSQL restricts the table's name to 63 characters
        db_object = Database()
        if db_object.db_exists_postgresql() and self.store_postgres:
            db_object.write_df_postgresql(cleanedDF, cleaner_path[-63:-1])
        else:
            cleanedDF.to_csv(cleaner_path, index=False, encoding="utf-8")


    def __evaluate(self, detections, dirtyDF, cleanedDF):
        """
        Runs 3 Experiments for evaluating the cleaned dataset.
        1. Experiment: interprete all columns as categorical and calculate precision, recall, f1 relative to all error (actual_errors_dict)
        2. Experiment: get numerical columns from groundtruthDF and calulate RMSE for dirty dataset and for the repaired dataset (considers cells
                        where groundtruth, dirty and repaired dataset have numerical values). Uses StandardScaler
        3. Expriment: get categorical columns from groundtruthDF and calculate precision, recall, f1 relativ to all errors in the respective columns

        Arguments:
        detections -- dictionary - keys represent i,j of dirty cells & values are constant string "JUST A DUUMY VALUE"
        dirtyDF (dataframe) -- dirty dataframe
        cleanedDF (dataframe) -- dataframe that was repaired
        """

        def convert_to_float_or_nan(matrix):
            for (x,y), _ in np.ndenumerate(matrix):
                try:
                    matrix[x,y] = float(matrix[x,y])
                except ValueError:
                    matrix[x,y] = np.nan
            return matrix

        # Initialize a dictionary to pack the results
        evaluation_dict = {}

        # Get numerical and categorical columns of the ground truth
        groundTruthDF = self.groundTruthDF.apply(pd.to_numeric, errors="ignore")
        gt_num_columns = groundTruthDF.select_dtypes(include="number").columns
        gt_cat_columns = groundTruthDF.select_dtypes(exclude="number").columns
        print(gt_num_columns)
        
        # ===================================================================================
        # Extract metadata of groundtruth, repaired and dirty dataset
        # ===================================================================================

        # Return metrics not available if shapes dont equal, e.g., in case of deleting dirty tuples
        if dirtyDF.shape != cleanedDF.shape != self.groundTruthDF.shape:

            evaluation_dict =  { "gt_#cat_col": len(gt_cat_columns),
                                 'gt_#num_col': len(gt_num_columns),
                                 "repaired_#cat_col" : len(cleanedDF.select_dtypes(exclude="number").columns),
                                 'repaired_#num_col': len(cleanedDF.select_dtypes(include="number").columns),
                                 "dirty_#cat_col" : len(dirtyDF.select_dtypes(exclude="number").columns),
                                 'dirty_#num_col': len(dirtyDF.select_dtypes(include="number").columns),
                                 'allCat_total_repairs': None,
                                 'allCat_tp': None,
                                 'allCat_actual_#errors': None,
                                 'allCat_p': None,
                                 'allCat_r': None,
                                 'allCat_f': None,
                                 'onlyNum_rmse_repaired': None,
                                 'onlyNum_rmse_dirty': None,
                                 'onlyCat_total_repairs': None,
                                 'onlyCat_tp': None,
                                 'onlyCat_actual_#errors': None,
                                 'onlyCat_p': None,
                                 'onlyCat_r': None,
                                 'onlyCat_f': None,
                                 'models': None
            }

        else:

            # =================================================================================
            # Experiment 2: Only Numerical
            # only numerical columns of groundtruth, removes values from consideration for RMSE
            #     in dirtydf and cleaneddf that are not numerical. Looks at overlap of cells where groundtruth,
            #     repaired and dirty cells are numerical.
            # =================================================================================

            if len(gt_num_columns) != 0:
                y_groundtruth = groundTruthDF[gt_num_columns].to_numpy(dtype=float)
                y_cleaned = cleanedDF[gt_num_columns].to_numpy()
                y_dirty = dirtyDF[gt_num_columns].to_numpy()
                
                # convert each element in y_cleaned, y_dirty, and y_groundtruth to a float, and 
                # if it fails (due to the element not being a number), sets that element to NaN.
                y_cleaned = convert_to_float_or_nan(y_cleaned)
                y_dirty = convert_to_float_or_nan(y_dirty)
                y_groundtruth = convert_to_float_or_nan(y_groundtruth)

                scaler = StandardScaler()
                """ y_groundtruth, y_cleaned, y_dirty have nan at the same positions
                    thus nan values can be simply removed and the resulting arrays still fit """

                # scale, remove nan values
                y_true = scaler.fit_transform(y_groundtruth).flatten().astype(float)
                #y_true = y_true[np.logical_not(np.isnan(y_true))] # remove nan
                y_true = np.nan_to_num(y_true) # replace nan with zero
                
                # scale, remove nan values and calculate rmse for repaired dataset
                y_pred = scaler.transform(y_cleaned).flatten().astype(float)
                #y_pred = y_pred[np.logical_not(np.isnan(y_pred))] # remove nan
                y_pred = np.nan_to_num(y_pred)
                rmse_repaired = mean_squared_error(y_true, y_pred, squared=False)

                # scale, remove nan values and calculate rmse for dirty dataset
                y_pred2 = scaler.transform(y_dirty).flatten().astype(float)
                #y_pred = y_pred[np.logical_not(np.isnan(y_pred))] # remove nan
                y_pred2 = np.nan_to_num(y_pred2)
                rmse_dirty = mean_squared_error(y_true, y_pred2, squared=False)

            else:
                rmse_repaired, rmse_dirty = 0.0, 0.0

            exp2_evaluation = {
                                   "rmse_repaired": rmse_repaired,
                                   "rmse_dirty": rmse_dirty,
                                      }

            #evaluation_dict = {
            #    "gt_#cat_col": len(gt_cat_columns),
            #    'gt_#num_col': len(gt_num_columns),
            #    "repaired_#cat_col": len(cleanedDF.select_dtypes(exclude="number").columns),
            #    'repaired_#num_col': len(cleanedDF.select_dtypes(include="number").columns),
            #    "dirty_#cat_col": len(dirtyDF.select_dtypes(exclude="number").columns),
            #    'dirty_#num_col': len(dirtyDF.select_dtypes(include="number").columns),

            #    'onlyNum_rmse_repaired': exp2_evaluation["rmse_repaired"],
            #    'onlyNum_rmse_dirty': exp2_evaluation["rmse_dirty"],

            #}
            
            # =============================================================================
            #  Experiment 3: Only Categorical
            #  calculate f1, precision, recall for only categorical columns of groundtruth
            # =============================================================================

            tp_cat = 0.0
            cat_repairsCounter = 0
            actual_errors_cat_dict = {(row, col): value for (row, col), value in self.actual_errors_dict.items() if self.groundTruthDF.columns[col] in gt_cat_columns}
            for (row_i, col_i), dummy in detections.items():

                if row_i >= dirtyDF.shape[0]:
                    continue 

                # if detection is in a categorical column of groundtruth
                if cleanedDF.columns[col_i] in gt_cat_columns:
                    errors_in_cat=+1
                    # check if repair has happend
                    if cleanedDF.iat[row_i, col_i] != dirtyDF.iat[row_i, col_i]:
                        # counter all repairs
                        cat_repairsCounter = cat_repairsCounter + 1

                        # check if detected error was corretly repaired
                        if cleanedDF.iat[row_i, col_i] == self.groundTruthDF.iat[row_i, col_i]:
                            tp_cat = tp_cat + 1

            precision_cat = 0.0 if cat_repairsCounter == 0 else tp_cat / cat_repairsCounter
            recall_cat = 0.0 if len(actual_errors_cat_dict) == 0 else tp_cat / len(actual_errors_cat_dict)
            f1_cat = 0.0 if (precision_cat + recall_cat) == 0 else (2 * precision_cat * recall_cat) / (precision_cat + recall_cat)

            exp3_evaluation = {
                                   "#total_repairs" : cat_repairsCounter,
                                   "#correct_repairs(tp)" : tp_cat,
                                   "#actual_errors" : len(actual_errors_cat_dict),
                                   "precision": precision_cat,
                                   "recall": recall_cat,
                                   "f1": f1_cat,
                                      }

            evaluation_dict = {
                "gt_#cat_col": len(gt_cat_columns),
                'gt_#num_col': len(gt_num_columns),
                "repaired_#cat_col": len(cleanedDF.select_dtypes(exclude="number").columns),
                'repaired_#num_col': len(cleanedDF.select_dtypes(include="number").columns),
                "dirty_#cat_col": len(dirtyDF.select_dtypes(exclude="number").columns),
                'dirty_#num_col': len(dirtyDF.select_dtypes(include="number").columns),

                'onlyNum_rmse_repaired': exp2_evaluation["rmse_repaired"],
                'onlyNum_rmse_dirty': exp2_evaluation["rmse_dirty"],

                'onlyCat_total_repairs': exp3_evaluation['#total_repairs'],
                'onlyCat_tp': exp3_evaluation['#correct_repairs(tp)'],
                'onlyCat_actual_#errors': exp3_evaluation['#actual_errors'],
                'onlyCat_p': exp3_evaluation['precision'],
                'onlyCat_r': exp3_evaluation['recall'],
                'onlyCat_f': exp3_evaluation['f1'],

                'model': None
            }
            
            

        return evaluation_dict

    def cleanWithGroundTruth(self, dirtyDF, detection_dictionary, configs):
        """
        A workaround for data cleaning

        * Replacing the detected dirty cells with their clean values from the ground truth
        * In this case, data cleaning is excluded from the evaluation, i.e. ec_p, ec_r, or ec_f are not computed
        
        Arguments:
        detection_dictionary -- dictionary - keys represent i,j of dirty cells & values are constant string "JUST A DUUMY VALUE"
        dirtyDF -- dataframe of shape n_R (# of records) x n_A (# of attributes) - containing a dirty version of a dataset
        
        Returns:
        cleanedDF -- dataframe of shape n_R x n_A - containing a cleaned version of the dirty dataset 
        results (dict) -- dictionary with results from evaluation
        """
        if len(detection_dictionary) == 0:
            return dirtyDF, {"Problem": "No Errors to be cleaned"}

        start_time = time.time()

        # Extract the dataset name
        #dataset_name = configs["dataset_name"]
        dataset_name = self.__dataset_name

        dataset_dict = datasets_dictionary[self.__dataset_name]
        results = {}

        # initialize cleaned data with dirty data
        cleanedDF = dirtyDF.copy()

        # iterate through detection_dictionary and set correct values at detected error cells
        for (row_i, col_i), dummy in detection_dictionary.items():
            # replace a dirty cell with its clean value
            cleanedDF.iat[row_i, col_i] = self.groundTruthDF.iat[row_i, col_i]

        cleaning_runtime = time.time() - start_time
        cleaner_directory = self.__get_cleaner_directory("cleanWithGroundTruth")
        self.__store_cleaned_data(
            cleanedDF, os.path.join(cleaner_directory, "repaired.csv")
        )

        results = self.__evaluate(detection_dictionary, dirtyDF, cleanedDF)
        results["runtime"] = cleaning_runtime

        return cleanedDF, results

    def standardImputer(self, dirtyDF, detections, configs, **kwargs):
        """
        This method cleans missing values with either deletion or imputation.

        This method can delete or impute (with different strategies) missing values for a given dirty dataset.
        Non-numerical columns are assumed to be categorical.
        
        Arguments:
        detection_dictionary -- dictionary - keys represent i,j of dirty cells & values are constant string "JUST A DUUMY VALUE"
        dirtyDF -- dataframe of shape n_R (# of records) x n_A (# of attributes) - containing a dirty version of a dataset
        method (String) -- is "delete" per default but can be set to "impute"
        num (String) -- only needed for method="impute". Possible values: "mean", "median", "mode". Defines how to impute missing vales
                        in numerical columns
        cat (String) -- only needed for method="impute". Possible values: "mode", "dummy". Defines how to impute missing values 
                        in non-numerical columns, which are assumed to be categorical
        
        Returns:
        repairedDF -- dataframe of shape n_R x n_A - containing a cleaned version of the dirty dataset 
        results (dict) -- dictionary with results from evaluation
        """

        if len(detections) == 0:
            return dirtyDF, {"Problem": "No Errors to be cleaned"}

        start_time = time.time()

        # Extract the method
        method = configs["method"]

        # if method is delete, drop all rows from detections keys (row, col)
        if method == "delete":
            drop_rows = []
            for (row_i, col_i), dummy in detections.items():
                drop_rows.append(row_i)

            repairedDF = dirtyDF.drop(drop_rows)

            cleaner_directory = self.__get_cleaner_directory(str(RepairMethod.standardImputer)+"delete")

        # if method is impute, impute detected cells with respective strategy provided in **kwargs
        elif method == "impute":
            if "num" not in kwargs or "cat" not in kwargs:
                logging.info("Must give imputation method for numerical and categorical data")
                sys.exit(1)

            num_method = kwargs["num"]
            cat_method = kwargs["cat"]

            # transform dirtdf which has dtype string to numeric type if possible
            dirtydf = dirtyDF.apply(pd.to_numeric, errors="ignore")

            num_df = dirtydf.select_dtypes(include="number")
            cat_df = dirtydf.select_dtypes(exclude="number")

            if num_method == "mean":
                num_imp = num_df.mean()
            if num_method == "median":
                num_imp = num_df.median()
            if num_method == "mode":
                num_imp = num_df.mode().iloc[0]

            if cat_method == "mode":
                cat_imp = cat_df.mode().iloc[0]
            if cat_method == "dummy":
                cat_imp = ["missing"] * len(cat_df.columns)
                cat_imp = pd.Series(cat_imp, index=cat_df.columns)

            impute = pd.concat([num_imp, cat_imp], axis=0)

            repairedDF = dirtyDF.copy()

            # for every entry in detections impute the value in repairedDF with
            # the impute value for the respective column
            for (row_i, col_i), dummy in detections.items():
                repairedDF.iat[row_i, col_i] = impute[repairedDF.columns[col_i]]

            cleaner_directory = self.__get_cleaner_directory(
                str(RepairMethod.standardImputer)+"-impute-{}-{}".format(num_method, cat_method)
            )

        else:
            logging.info("incorrect parameters for method")
            sys.exit(1)

        cleaning_runtime = time.time() - start_time
        self.__store_cleaned_data(
            repairedDF, os.path.join(cleaner_directory, "repaired.csv")
        )

        results = self.__evaluate(detections, dirtyDF, repairedDF)
        results["cleaning_runtime"] = cleaning_runtime

        return repairedDF, results

    def mlImputer(self, dirtyDF, detections, configs, **kwargs):
        """
        Imputes cells in detections with ml based imputation methods. 

        Runs either seperate imputation methods for categorical and numerical columns or one mixed imputation method
        for both. For mixed methods, a mix_method has to be provieded. For seperate methods, num and cat parameters have to be provided.
        To be installed: sklearn.impute, missingpy, datawig
        
        Arguments:
        detection_dictionary -- dictionary - keys represent i,j of dirty cells & values are constant string "JUST A DUUMY VALUE"
        dirtyDF -- dataframe of shape n_R (# of records) x n_A (# of attributes) - containing a dirty version of a dataset
        configs: method (String) -- "mix" oder "seperate". "mix" means that a method is applied to all columns, "seperate" means that
                                seperate methods get applied to categorical (cat) and numerical (num) columns
        mix_method (String) -- method to be applied if configs["method"] == "mix". Available: "missForest", "datawig"
        num (String) -- method to be applied for numerical columns if configs["method"] == "seperate". Available: "missForest", "knn", "em",
                        "decisionTree", "bayesianRidge", "extraTrees", "datawig"
        cat (String) -- method to be applied for categorical columns if configs["method"] == "seperate". Available: "missForest", "datawig"
        
        Returns:
        repairedDF -- dataframe of shape n_R x n_A - containing a cleaned version of the dirty dataset 
        results (dict) -- dictionary with results from evaluation
        """

        def encode_cat(X_c):
            data = X_c.copy()
            nonulls = data.dropna().values
            impute_reshape = nonulls.reshape(-1,1)
            encoder = OrdinalEncoder()
            impute_ordinal = encoder.fit_transform(impute_reshape)
            data.loc[data.notnull()] = np.squeeze(impute_ordinal)
            return data, encoder

        def decode_cat(X_c, encoder):
            data = X_c.copy()
            nonulls = data.dropna().values.reshape(-1,1)
            n_cat = len(encoder.categories_[0])
            nonulls = np.round(nonulls).clip(0, n_cat-1)
            nonulls = encoder.inverse_transform(nonulls)
            data.loc[data.notnull()] = np.squeeze(nonulls)
            return data

        start_time = time.time()

        if configs["method"] == "mix":
            num_method = None
            cat_method = None
            mix_method = kwargs["mix_method"]
            save_extension = "mix-{}".format(mix_method)
        elif configs["method"] == "seperate":
            mix_method = None
            num_method = kwargs["num"]
            cat_method = kwargs["cat"]
            save_extension = "seperate-{}-{}".format(num_method, cat_method)
        else:
            logging.info("incorrect parameters for method")
            sys.exit(1)


        dirtyDF_nan = dirtyDF.copy()

        # change all occurances detections to np.nan
        # imputers identify cells to impute by checking if they are np.nan
        for (row_i, col_i), dummy in detections.items():
            dirtyDF_nan.iat[row_i, col_i] = np.nan


        # transform dirtdf which has dtype string to numeric type if possible.
        # np.nan is float and thus is numeric
        dirtyDF_nan = dirtyDF_nan.apply(pd.to_numeric, errors="ignore")

        num_df_orig = dirtyDF_nan.select_dtypes(include="number")
        cat_df = dirtyDF_nan.select_dtypes(exclude="number")

        # for numerical columns save columns that are all nan and create
        # a new dataframe that excludes those columns
        num_all_nan_cols = []
        for col in num_df_orig.columns:
            if num_df_orig[col].isnull().sum() == num_df_orig.shape[0]:
                num_all_nan_cols.append(col)
        num_df = num_df_orig.drop(columns=num_all_nan_cols)

        if num_method == "knn":
            # assigns mean of n_neighbors closests values

            imputer = sklearnKNNImputer(missing_values=np.nan, n_neighbors=5)
            num_repaired = imputer.fit_transform(num_df)

            # repaired version of numerical columns that are not all nan
            num_repaired = pd.DataFrame(num_repaired, columns=num_df.columns)

        if num_method == "missForest":
            # runs missingpy's MissForest on numerical columns. 
            # Impute value are based on other values in row (numerical columnns)

            imputer = MissForest(missing_values=np.nan)
            num_repaired = imputer.fit_transform(num_df)

            # repaired version of numerical columns that are not all nan
            num_repaired = pd.DataFrame(num_repaired, columns=num_df.columns)

        if num_method == "em":
            # runs expected maximization imputer from impyute library

            # impEM only works if there are any np.nan's in the dataset
            num_repaired = impEM(num_df.to_numpy().astype(np.float)) if num_df.isnull().values.any() else num_df
            # repaired version of numerical columns that are not all nan
            num_repaired = pd.DataFrame(num_repaired, columns=num_df.columns)

        if num_method == "decisionTree" or num_method == "bayesianRidge" or num_method == "extraTrees":

            # instantiate estimator
            if num_method == "decisionTree":
                estimator = DecisionTreeRegressor(max_features='sqrt')
            elif num_method == "bayesianRidge":
                estimator = BayesianRidge()
            elif num_method == "extraTrees":
                estimator = ExtraTreesRegressor(n_estimators=10)

            imputer = sklearnIterativeImputer(estimator=estimator, missing_values=np.nan)
            num_repaired = imputer.fit_transform(num_df)

            # repaired version of numerical columns that are not all nan
            num_repaired = pd.DataFrame(num_repaired, columns=num_df.columns)

        if num_method == "datawig":

            num_repaired = datawig.SimpleImputer.complete(num_df)

        if cat_method == "missForest":
            # decodes categorical variables and runs missingpy's MissForest on 
            # categorical columns. Impute value are based on other values in row (only categorical columns)

            # encode categorical columns
            cat_encoders = {}
            cat_X_enc = []
            for c in cat_df.columns:
                X_c_enc, encoder = encode_cat(cat_df[c])
                cat_X_enc.append(X_c_enc)
                cat_encoders[c] = encoder
            cat_X_enc = pd.concat(cat_X_enc, axis=1)
            cat_columns = cat_df.columns
            cat_indices = [i for i, c in enumerate(cat_X_enc.columns) if c in cat_columns]

            # impute np.nan values           
            imputer = MissForest(missing_values=np.nan)
            cat_repaired_enc = imputer.fit_transform(cat_X_enc.values.astype(float), cat_vars=cat_indices)
            cat_repaired_enc = pd.DataFrame(cat_repaired_enc, columns=cat_X_enc.columns)

            #decode encoded representation
            cat_X_imp = cat_repaired_enc
            cat_X_dec = []
            for c in cat_df.columns:
                X_c_dec = decode_cat(cat_X_imp[c], cat_encoders[c])
                cat_X_dec.append(X_c_dec)
            cat_X_dec = pd.concat(cat_X_dec, axis=1)

            # repaired version of categorical columns that are not all nan
            cat_repaired = cat_X_dec

        if cat_method == "datawig":

            cat_repaired = datawig.SimpleImputer.complete(cat_df)

        if mix_method == "missForest":
            # decodes categorical variables and runs missingpy's MissForest on 
            # all columns (numerical = cateogircal). Impute value are based on other values in row (numerical + cateogrical columnns)

            # only if there are any categorical columns
            if cat_df.shape[1]>0:
                cat_encoders = {}
                cat_X_enc = []
                for c in cat_df.columns:
                    X_c_enc, encoder = encode_cat(cat_df[c])
                    cat_X_enc.append(X_c_enc)
                    cat_encoders[c] = encoder
                cat_X_enc = pd.concat(cat_X_enc, axis=1)
                X_enc = pd.concat([num_df, cat_X_enc], axis=1) # because mix_method
                cat_columns = cat_df.columns
                cat_indices = [i for i, c in enumerate(X_enc.columns) if c in cat_columns]
            else:
                X_enc = num_df # because mix method
                cat_indices = None

            # impute np.nan values           
            imputer = MissForest(missing_values=np.nan)
            repaired_enc = imputer.fit_transform(X_enc.values.astype(float), cat_vars=cat_indices)
            repaired_enc = pd.DataFrame(repaired_enc, columns=X_enc.columns)

            if cat_df.shape[1]>0:
                #decode encoded representation
                num_X_imp = repaired_enc[num_df.columns] # new
                cat_X_imp = repaired_enc[cat_df.columns] # new
                cat_X_dec = []
                for c in cat_df.columns:
                    X_c_dec = decode_cat(cat_X_imp[c], cat_encoders[c])
                    cat_X_dec.append(X_c_dec)
                cat_X_dec = pd.concat(cat_X_dec, axis=1)
                X_dec = pd.concat([num_X_imp, cat_X_dec], axis=1)

            # repaired version of categorical columns that are not all nan
            repaired = X_dec

        if mix_method == "datawig":

            X  = pd.concat([num_df, cat_df], axis=1)
            repaired = datawig.SimpleImputer.complete(X)

        repairedDF = dirtyDF.copy()
        if  configs["method"] == "mix":
            outoforder_concat = pd.concat([repaired, dirtyDF[num_all_nan_cols]], axis=1)
        else:
            outoforder_concat = pd.concat([num_repaired, cat_repaired, dirtyDF[num_all_nan_cols]], axis=1)
        for col in outoforder_concat.columns:
            repairedDF[col] = outoforder_concat[col]

        cleaner_directory = self.__get_cleaner_directory(
                str(RepairMethod.mlImputer)+"-{}".format(save_extension)
            )
        cleaning_runtime = time.time() - start_time
        self.__store_cleaned_data(
            repairedDF, os.path.join(cleaner_directory, "repaired.csv")
        )

        results = self.__evaluate(detections, dirtyDF, repairedDF)
        results["cleaning_runtime"] = cleaning_runtime

        return repairedDF, results

    def duplicatesCleaner(self, dirtyDF, detections, configs):
        """
        This method cleanes duplicates by deleting the entire entry.

        For each given key in detections (key, col), this method deletes the row from the dataset.

        Arguments:
        detection_dictionary -- dictionary - keys represent i,j of dirty cells & values are constant string "JUST A DUUMY VALUE"
        dirtyDF -- dataframe of shape n_R (# of records) x n_A (# of attributes) - containing a dirty version of a dataset
        
        Returns:
        repairedDF -- dataframe of shape n_R x n_A - containing a cleaned version of the dirty dataset 
        results (dict) -- dictionary with results from evaluation
        """
        if len(detections) == 0:
            return dirtyDF, {"Problem": "No Errors to be cleaned"}

        start_time = time.time()

        # Extract the dataset_name
        #dataset_name = configs["dataset_name"]
        dataset_name = self.__dataset_name

        # get row of each entry
        drop_rows = []
        for (row_i, col_i), dummy in detections.items():
            drop_rows.append(row_i)

        # delete all rows which are duplicates
        repairedDF = dirtyDF.drop(drop_rows)
        cleaning_runtime = time.time() - start_time

        cleaner_directory = self.__get_cleaner_directory("duplicatesCleaner")
        self.__store_cleaned_data(
            repairedDF, os.path.join(cleaner_directory, "repaired.csv")
        )

        results = self.__evaluate(detections, dirtyDF, repairedDF)
        results["cleaning_runtime"] = cleaning_runtime

        return repairedDF, results

    def dcHoloCleaner(self, dirtyDF, detections, configs):
        """
        This method repairs errors detected with denial constraints,

        Arguments:
        detections -- dictionary - keys represent i,j of dirty cells & values are constant string "JUST A DUUMY VALUE"
        dirtyDF -- dataframe of shape n_R (# of records) x n_A (# of attributes) - containing a dirty version of a dataset
        with_init -- boolean - if true InitAttributeFeaturizer is used as feature
        path_to_constraints -- String - Path to the txt-file, containing the constraints with which the errors where detected

        Returns:
        repairedDF -- dataframe of shape n_R x n_A - containing a cleaned version of the dirty dataset 
        results (dict) -- dictionary with results from evaluation
        """

        if len(detections) == 0:
            return dirtyDF, {"Problem": "No Errors to be cleaned"}

        start_time = time.time()

        # Extract the necessary parameters
        #with_init = configs["with_init"]
        with_init = True if configs["method"] == "with_init" else False
        path_to_constraints = os.path.abspath(os.path.join(datasets_dictionary[self.__dataset_name]["dataset_path"], "constraints"))
        #path_to_constraints = configs["path_to_constraints"]

        # 1. Setup a HoloClean session.
        hc = holoclean.HoloClean(
            db_name="holo",
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
            timeout=3 * 60000,
            feature_norm=False,
            weight_norm=False,
            print_fw=True,
        ).session

        copy_dirtyDF = dirtyDF.copy()

        # holoclean can not work with dataframes that contain a column named index
        index_col_pos = -1
        if "index" in copy_dirtyDF.columns:
            index_col_pos = copy_dirtyDF.columns.get_loc("index")
            index_col_values = copy_dirtyDF["index"]
            copy_dirtyDF = copy_dirtyDF.drop(columns=["index"])

        # load the dirty data in holoclean
        hc.load_data(self.__dataset_name, "", df=copy_dirtyDF)

        # load the constraints from dataset constraint directory
        hc.load_dcs(os.path.join(path_to_constraints, "_all_constraints.txt"))

        # set the constraints in holoclean
        hc.ds.set_constraints(hc.get_dcs())
        # transform detections {row_i: col_i} into new dataframe with columns _tid_ and attribute
        # _tid_ = row_i attribute = Column name of col_i
        holoclean_error_df = pd.DataFrame(columns=["_tid_", "attribute"])
        for (row_i, col_i), dummy in detections.items():
            if dirtyDF.columns[col_i] != "index":
                holoclean_error_df = holoclean_error_df.append(
                    {"_tid_": row_i, "attribute": dirtyDF.columns[col_i]}, ignore_index=True
                )
        holoclean_error_df.drop_duplicates()
        holoclean_error_df["_cid_"] = holoclean_error_df.apply(
            lambda x: hc.ds.get_cell_id(x["_tid_"], x["attribute"]), axis=1
        )

        # Store errors to db fur further processing in holoclean
        hc.detect_engine.store_detected_errors(holoclean_error_df)

        # 4. Repair errors utilizing the defined features.
        hc.setup_domain()
        if with_init == True:
            featurizers = [
                InitAttrFeaturizer(),
                OccurAttrFeaturizer(),
                FreqFeaturizer(),
                ConstraintFeaturizer(),
            ]
        else:
            featurizers = [
                OccurAttrFeaturizer(),
                FreqFeaturizer(),
                ConstraintFeaturizer(),
            ]

        # repair errors and get repaired dataframe
        _, repairedDF = hc.repair_errors(featurizers)

        cleanedDF = repairedDF.drop("_tid_", 1)

        # if dataset contained index column, insert it again into df
        if index_col_pos != -1:
            cleanedDF.insert(index_col_pos, "index", index_col_values)

        cleaning_runtime = time.time() - start_time

        cleaner_directory = self.__get_cleaner_directory(
            "dcHoloCleaner-{specification}".format(
                specification="with_init" if with_init == True else "without_init"
            )
        )
        self.__store_cleaned_data(
            cleanedDF, os.path.join(cleaner_directory, "repaired.csv")
        )

        results = self.__evaluate(detections, dirtyDF, cleanedDF)
        results["cleaning_runtime"] = cleaning_runtime

        return cleanedDF, results

    def baran(self, dirtyDF, detections, configs):
        """
        This method repairs detected errors with baran.

        Arguments:
        dirtyDF -- dataframe of shape n_R (# of records) x n_A (# of attributes) - containing a dirty version of a dataset
        detections -- dictionary - keys represent i,j of dirty cells & values are constant string "JUST A DUUMY VALUE"

        Returns:
        repairedDF -- dataframe of shape n_R x n_A - containing a cleaned version of the dirty dataset 
        results (dict) -- dictionary with results from evaluation
        """
        start_time = time.time()

        #dataset_name = configs["dataset_name"]
        dataset_name = self.__dataset_name

        # dict to process raha steps internally
        internal_dataset_dict = {
            "name": dataset_name,
            "path": os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    os.pardir,
                    "datasets",
                    dataset_name,
                    "dirty.csv",
                )
            ),
            "clean_path": os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    os.pardir,
                    "datasets",
                    dataset_name,
                    "clean.csv",
                )
            ),
        }

        app = Raha_Correction()
        app.LABELING_BUDGET = 20
        app.VERBOSE = True
        app.SAVE_RESULTS = False

        # simulate detector initialization
        d = Raha_Dataset(internal_dataset_dict)
        d.dictionary = internal_dataset_dict
        d.results_folder = os.path.join(
            os.path.dirname(internal_dataset_dict["path"]),
            "raha-baran-results-" + d.name,
        )
        d.labeled_tuples = {} if not hasattr(d, "labeled_tuples") else d.labeled_tuples
        d.labeled_cells = {} if not hasattr(d, "labeled_cells") else d.labeled_cells
        d.labels_per_cluster = (
            {} if not hasattr(d, "labels_per_cluster") else d.labels_per_cluster
        )
        d.detected_cells = {} if not hasattr(d, "detected_cells") else d.detected_cells

        # if self.SAVE_RESULTS and not os.path.exists(d.results_folder):
        #    os.mkdir(d.results_folder)

        # set detections for dataset instance
        d.detected_cells = detections
        d.clean_dataframe = self.groundTruthDF
        d.has_ground_truth = True

        # initialize dataset for correcting with simulated dataset instance of detecting
        d = app.initialize_dataset(d)

        # initialize the Error Corrector Models
        app.initialize_models(d)

        # label tuples with ground truth, update models, generate features, predict correction
        while len(d.labeled_tuples) < app.LABELING_BUDGET:
            app.sample_tuple(d)
            if d.has_ground_truth:
                app.label_with_ground_truth(d)
            else:
                import IPython.display

                logging.info("Label the dirty cells in the following sampled tuple.")
                sampled_tuple = pd.DataFrame(
                    data=[d.dataframe.iloc[d.sampled_tuple, :]],
                    columns=d.dataframe.columns,
                )
                IPython.display.display(sampled_tuple)
                for j in range(d.dataframe.shape[1]):
                    cell = (d.sampled_tuple, j)
                    value = d.dataframe.iloc[cell]
                    d.labeled_cells[cell] = input(
                        "What is the correction for value '{}'?\n".format(value)
                    )
                d.labeled_tuples[d.sampled_tuple] = 1
            app.update_models(d)
            app.generate_features(d)
            app.predict_corrections(d)

        repairedDF = dirtyDF.copy()

        # repair dirty dataset
        for (row_i, col_i), correction in d.corrected_cells.items():
            repairedDF.iat[row_i, col_i] = correction

        cleaning_runtime = time.time() - start_time

        cleaner_directory = self.__get_cleaner_directory("baran")
        self.__store_cleaned_data(
            repairedDF, os.path.join(cleaner_directory, "repaired.csv")
        )

        results = self.__evaluate(detections, dirtyDF, repairedDF)
        results["cleaning_runtime"] = cleaning_runtime

        return repairedDF, results

    def openrefine(self, dirtyDF, detections, configs):
        """
        uses the clusters generated by openrefine as json in order to clean the respective
        cells with the proposed correct value (proposed by OpenRefine).
        
        For generating the clusters in OpenRefine "key collision" is used as Method and "fingerprint" as keying function
        """

        start_time = time.time()
        try:
            dir = os.path.join(datasets_dictionary[self.__dataset_name]["dataset_path"], "clusters")
        except:
            logging.info("No clusters exist for the {} dataset".format(dataset))
            sys.exit(1)

        # extracts row and col from json file and edits the cells with 
        # the proposed value 
        repairedDF = dirtyDF.copy()
        for filename in os.listdir(dir):
            if filename.endswith(".json"):
                with open(os.path.join(dir, filename)) as file:
                    clusters_dict = json.load(file)

                col_name = clusters_dict["columnName"]
                col = dirtyDF.columns.get_loc(col_name)
                for cluster in clusters_dict["clusters"]:
                    correct_value = cluster["value"]
                    for choise in cluster["choices"]:
                        if choise["v"] != correct_value:
                            row_list = dirtyDF.index[dirtyDF[col_name]== choise["v"]]
                            for row in row_list:
                                repairedDF.iat[row, col] = correct_value

        cleaning_runtime = time.time() - start_time

        cleaner_directory = self.__get_cleaner_directory(str(RepairMethod.openrefine))
        self.__store_cleaned_data(
            repairedDF, os.path.join(cleaner_directory, "repaired.csv")
        )

        results = self.__evaluate(detections, dirtyDF, repairedDF)
        results["cleaning_runtime"] = cleaning_runtime

        return repairedDF, results

    def lop(self, dirtyDF, detection_dictionary, configs):
        """_summary_

        Args:
            dirtyDF (DataFrame): dirty version of a dataset
            detection_dictionary (Dictionary): indexes of detected dirty cells
            configs (Dictionary): configurations to run the method
        
        Returns:
            _type_: _description_
        """
        
        # Extract the dataset name
        dataset_name = self.__dataset_name
        
        dataset_dict = datasets_dictionary[self.__dataset_name]
        results = {}
        
        # Load the repaired data
        repaired_path = self.__get_cleaner_directory("lop")
        cleanedDF = pd.read_csv(os.path.join(repaired_path, 'repaired.csv'), encoding="utf-8", header="infer", 
                                keep_default_na=False, low_memory=False)
        cleanedDF.reset_index(drop=True, inplace=True) 
        results = self.__evaluate(detection_dictionary, dirtyDF, cleanedDF)
        print(results)
        
        return cleanedDF, results

    ##########################################################################
    # The following cleaners dont create a repaired version of the dataset
    # but repairing and training a model is interconnected. Insted of creating
    # a repaired datasets these cleaners store their precision, recall and f1

    def activecleanCleaner(self, dirtyDF, detections, configs):
        '''
        Runs activeclean, which does not clean the data but uses a specific sampling strategy to samples entries that should
        be cleaned by an human annotator. The human annotator is simulated by the ground truth dataset. This method creates the test results for scenarios 
        S1, S4 of the created model.
        
        The method uses gradient descent strategy to sample the optimal entries. It therefore needs a maximum number of samples which are examined.
        Sampling_budget * number_of_entries sets this number.
        
        
        Arguments:
        detection_dictionary -- dictionary - keys represent i,j of dirty cells & values are constant string "JUST A DUUMY VALUE"
        dirtyDF -- dataframe of shape n_R (# of records) x n_A (# of attributes) - containing a dirty version of a dataset
        configs: sampling_budget (float) -- sampling budget in percent of the total number of entries. e.g. sampling budget = 0.2 means that
                                            20% of all data are used.
        
        Returns:
        results (dict) -- dictionary with p, r, f1 for the applicable scenarios
        app -- instance of model class that can be used to store results
        '''

        def activeclean_orig(dirty_data, clean_data, test_data, full_data, indextuple, batchsize=50, total=1000):
            #makes sure the initialization uses the training data
            X = dirty_data[0][translate_indices(indextuple[0],indextuple[1]),:]
            y = dirty_data[1][translate_indices(indextuple[0],indextuple[1])]

            X_clean = clean_data[0]
            y_clean = clean_data[1]

            X_test = test_data[0]
            y_test = test_data[1]

            logging.info("[ActiveClean Real] Initialization")

            lset = set(indextuple[2])

            dirtyex = [i for i in indextuple[0]]
            cleanex = []

            # alternative for initializing dirtex with dirty entries in training set
            # and cleanex with clean entries in training set
            #dirtyex = [i for i in translate_indices(indextuple[0],indextuple[1])] #dirty training indices
            #cleanex = [i for i in translate_indices(indextuple[0],indextuple[2])] #clean training indices

            #dirtyex = indextuple[1] #dirty training indices
            #cleanex = indextuple[2]

            total_labels = []

            if (len(indextuple[1])==0):
                raise ValueError("no dirty tuples to be cleaned")

            import time
            timeout = time.time()+5*60 #now + 2 minutes

            while len(np.unique(y_clean[cleanex]))<len(np.unique(y_clean)):

                ##Not in the paper but this initialization seems to work better, do a smarter initialization than
                ##just random sampling (use random initialization)
                topbatch = np.random.choice(range(0,len(dirtyex)), batchsize, replace=False)
                examples_real = [dirtyex[j] for j in topbatch]
                examples_map = translate_indices(examples_real, indextuple[2])
                cleanex.extend(examples_map)
                if (time.time()>timeout):
                    raise TimeoutError("couldnt find warm up configuration with clean data")

            for j in examples_real:
                dirtyex.remove(j)


            clf = SGDClassifier(loss="hinge", alpha=0.000001, max_iter=200, fit_intercept=True, warm_start=True)
            clf.fit(X_clean[cleanex,:],y_clean[cleanex])

            for i in range(20, total, batchsize):

                logging.info("[ActiveClean Real] Number Cleaned So Far ", len(cleanex))
                ypred = clf.predict(X_test)
                logging.info("[ActiveClean Real] Prediction Freqs",np.sum(ypred), np.shape(ypred))
                logging.info(classification_report(y_test, ypred))

                #Sample a new batch of data
                examples_real = np.random.choice(dirtyex, batchsize)
                examples_map = translate_indices(examples_real, indextuple[2])

                total_labels.extend([(r, (r in lset)) for r in examples_real])

                #on prev. cleaned data train error classifier
                ec = error_classifier(total_labels, full_data)

                for j in examples_real:
                    try:
                        dirtyex.remove(j)
                    except ValueError:
                        pass

                dirtyex = ec_filter(dirtyex, full_data, ec)

                #Add Clean Data to The Dataset
                cleanex.extend(examples_map)

                #uses partial fit (not in the paper--not exactly SGD)
                clf.partial_fit(X_clean[cleanex,:],y_clean[cleanex], classes=np.unique(y_clean))


                logging.info("[ActiveClean Real] Accuracy ", i ,accuracy_score(y_test, ypred))

                if len(dirtyex) < batchsize:
                    logging.info("[ActiveClean Real] No More Dirty Data Detected")
                    break

            average = "micro" if len(np.unique(y_test)) > 2 else "binary"

            p, r, f1 , _ = precision_recall_fscore_support(y_test, ypred, average=average)
            return p, r, f1


        def translate_indices(globali, imap):
            lset = set(globali)
            return [s for s,t in enumerate(imap) if t in lset]

        def error_classifier(total_labels, full_data):
            indices = [i[0] for i in total_labels]
            labels = [int(i[1]) for i in total_labels]
            if np.sum(labels) < len(labels):
                clf = SGDClassifier(loss="log", alpha=1e-6, max_iter=200, fit_intercept=True)
                clf.fit(full_data[indices,:],labels)

                return clf
            else:
                return None

        def ec_filter(dirtyex, full_data, clf, t=0.90):
            if clf != None:
                pred = clf.predict_proba(full_data[dirtyex,:])

                return [j for i,j in enumerate(dirtyex) if pred[i][0] < t]

            logging.info("CLF none")

            return dirtyex

        dataset = self.__dataset_name
        sampling_budget = configs["sampling_budget"]
        # instantiate  model class
        app = Models(dataset)

        # Use the ML model with default parameters (i.e., not optimized)
        #estimator = model["fn"](**model["fixed_params"])

        # preprocess data, especially featurization
        X_train_dirty, y_train_dirty, X_test_dirty, y_test_dirty = app.preprocess(dirtyDF, classification)
        X_train_gt, y_train_gt, X_test_gt, y_test_gt = app.preprocess(self.groundTruthDF, classification)

        # concatinate to get same format as activeclean
        X_dirty = np.concatenate((X_train_dirty, X_test_dirty), axis=0)
        y_dirty = np.concatenate((y_train_dirty, y_test_dirty), axis=0)

        X_gt = np.concatenate((X_train_gt, X_test_gt), axis=0)
        y_gt = np.concatenate((y_train_gt, y_test_gt), axis=0)

        # get indices of dirty records/rows of training data
        indices_dirty = np.unique([row_i for (row_i, col_i), dummy in detections.items()])
        indices_clean = [i for i in range(0, X_train_dirty.shape[0]) if i not in indices_dirty]
        #indices_clean = [i for i in range(0, X_dirty.shape[0]) if i not in indices_dirty]

        # get indicies of splitted data
        N = X_dirty.shape[0]
        np.random.seed(1)
        idx = np.random.permutation(N)

        test_size = int(0.2*N)
        train_size = N-test_size

        # get indices of train, test, val
        test_indices = idx[:test_size]
        train_indices = idx[test_size: N]
        clean_test_indices = translate_indices(test_indices,indices_clean)


        # initialize results dict
        results = {
            "model" : str(RepairMethod.activecleanCleaner),
            "S1": [],
            "S2" :[0.0, 0.0, 0.0],
            "S3" :[0.0, 0.0, 0.0],
            "S4" :[],
            "S5" :[0.0, 0.0, 0.0],
        }

        # S4: training: repaired/activeclean, test: gt
        p,r,f1 = activeclean_orig((X_dirty, y_dirty),
                        (X_gt, y_gt),
                        (X_gt[test_indices,:], y_gt[test_indices]), # test on gt
                        #(X_gt[clean_test_indices,:], y_gt[clean_test_indices]),
                        X_dirty, # X_full
                        (train_indices,indices_dirty,indices_clean), total=int(sampling_budget*dirtyDF.shape[0]))
        results["S4"] = [p,r,f1]

        # S1: training: dirty, test: dirty
        clf = SGDClassifier(loss="hinge", alpha=0.000001, max_iter=200, fit_intercept=True, warm_start=True)
        clf.fit(X_dirty[train_indices,:], y_dirty[train_indices])

        average = "micro" if len(np.unique(y_dirty[test_indices])) > 2 else "binary"
        y_pred = clf.predict(X_dirty[test_indices,:])
        p, r, f1 , _ = precision_recall_fscore_support(y_gt[test_indices], y_pred, average=average)
        results["S1"] = [p,r,f1]


        return app, results

    def boostClean(self, dirtyDF, detections, configs):
        '''
        Runs boostClean, as implemented in CPClean. Only works on datasets with a binary classification task (throws error otherwise). This Cleaner does not
        return a cleaned version of the dataset as cleaning and model training is interconnected. This method generates the test results for scenarios 
        S1, S4, S5 of the created model.
        
        BoostCleans runs different Imputation methods on the detected cells resulting in multiple repaired versions of the dataset.
        On each version a classifier (KNN, is the only one that is suppoerted) is trained that predicts the label of the dataset. 
        Then boosting is applied to combine the classifiers. Algorithm calculates accuracy on training and validation set in the end.
        Val: 20%, Train: 60%, Test:20%
        
        Arguments:
        detection_dictionary -- dictionary - keys represent i,j of dirty cells & values are constant string "JUST A DUUMY VALUE"
        dirtyDF -- dataframe of shape n_R (# of records) x n_A (# of attributes) - containing a dirty version of a dataset
        configs: method (String) -- empty
        
        Returns:
        results (dict) -- dictionary with p, r, f1 for the applicable scenarios
        app -- instance of model class that can be used to store results
        '''

        ##### 1. impute np.nan into detected cells in dirtyDF_nan

        dirtyDF_nan = dirtyDF.copy()#.apply(pd.to_numeric, errors="ignore")

        # change all occurances detections to np.nan
        # imputers identify cells to impute by checking if they are np.nan
        for (row_i, col_i), dummy in detections.items():
            dirtyDF_nan.iat[row_i, col_i] = np.nan

        dirtyDF_nan = dirtyDF_nan.apply(pd.to_numeric, errors="ignore")
        dirtyDF = dirtyDF.apply(pd.to_numeric, errors="ignore")
        groundTruthDF = self.groundTruthDF.apply(pd.to_numeric, errors="ignore")
        app = Models(self.__dataset_name)

        ##### 2. Split data into train, validation, test set
        # train is used to train the classifiers per repairer, validation for 
        # adaboost to choose best classifier and test set to test the resulting
        # boosting classifier in the end

        np.random.seed(1)
        N = dirtyDF_nan.shape[0]
        idx = np.random.permutation(N)

        test_size = int(0.2*N)
        val_size = int(0.2*N)
        train_size = N-test_size-val_size

        # get indices of train, test, val
        idx_test = idx[:test_size]
        idx_val = idx[test_size: test_size+val_size]
        idx_train = idx[test_size+val_size: N]

        scenarios = {
            "S1" : {"df" : dirtyDF, "repair" : False}, # dirty train set and dirty test set (equivelent to knn on dirty data)
            "S4" : {"df" : groundTruthDF, "repair" : False}, # clean train set and clean test set (equivelent to knn on clean data)
            "S5" : {"df" : dirtyDF_nan, "repair" : True} # repaired train set, dirty test set
        }

        # initialize results dict
        results = {
            "model" : str(RepairMethod.boostClean),
            "S1": [],
            "S2" :[0.0, 0.0, 0.0],
            "S3" :[0.0, 0.0, 0.0],
            "S4" :[],
            "S5" :[],
        }

        # iterate over scenarios
        for s, configuration in scenarios.items():
            df = configuration["df"]

            # seperate dirty features and dirty/clean labels,  
            y_dirty = df[app.labels_list_clf]
            y_clean = groundTruthDF[app.labels_list_clf]
            if "features_clf" in datasets_dictionary[app.dataset_name]:  # list of features are given
                logging.info("Loading given features ..")
                X = df[app.features_clf]
            else:
                X = df.drop(app.labels_list_clf, axis=1)  # no features are given

            # exit if more than two classes
            if len(np.unique(y_clean)) > 2 or len(np.unique(y_dirty)) > 2:
                print('unique values in ground truth', np.unique(y_clean))
                print('unique values in dirty dataset', np.unique(y_dirty))
                logging.info("Invalid dataset: more than two classes")
                break

            # X_train, y_train. in S1: both dirty (not modified), in S4: both groundtruth,
            # in S5: both dirty but with imputed np.nan where error was detected (detection_dict)
            X_train = X.iloc[idx_train].reset_index(drop=True)
            y_train = y_dirty.iloc[idx_train].reset_index(drop=True)

            # validation. y_val always groundtruth. S1: X_val dirty (not modifiyed), S4: X_val groundtruth,
            # S5: X_val dirty with imputed np.nan where error was detected (detection_dict)
            X_val = X.iloc[idx_val].reset_index(drop=True)
            y_val = y_clean.iloc[idx_val].reset_index(drop=True)

            # test. y_test always clean X. S1: X_test dirty (not modifiyed), S4: X_test groundtruth,
            # S5: X_test dirty with imputed np.nan where error was detected (detection_dict)
            X_test = X.iloc[idx_test].reset_index(drop=True)
            y_test = y_clean.iloc[idx_test].reset_index(drop=True)

            # For S5: repair() returns a dictionary with the repaired version
            # of the dataset for each repair method within CPClean
            if configuration["repair"]:
                X_train_repairs = repair(X_train)
            else:
                # if no detection are provided, training happens on the not repaired data
                # for S1: X_train = dirty, S4: X_train = groundTruth
                X_train_repairs = {}
                X_train_repairs["default"] = X_train

            # set data to the format needed by CPClean/boostclean
            data = {}

            ##
            # need to be set, for CPClean/boostclean to run
            data["X_full"] = X
            data["y_full"] = y_clean
            data["X_train_dirty"] = X_train
            data["X_train_clean"] = X_train
            data["indicator"] = df.isnull()
            ##

            data["X_train_repairs"] = X_train_repairs
            data["X_test"] = X_test
            data["X_val"] = X_val

            data["y_train"] = y_train # y_train can be dirty
            data["y_test"] = y_test # needs to be clean
            data["y_val"] =y_val # needs to be clean

            ##### 3. preprocess the data
            data = preprocess(data)


            ##### 4. set model used for boosting, implementation by CPClean only supports KNN
            model = {
                "fn": KNN,
                "params": {"n_neighbors":3}
            }

            ##### 5. call boostcleean and get accuracy on test and validationt set plus precision, recall f1 on test data
            test_acc, val_acc, p, r, f1 =  modified_boost_clean(model, list(data["X_train_repairs"].values()), data["y_train"], data["X_val"], data["y_val"], data["X_test"], data["y_test"], T=5)
            results[s] = [p, r, f1]

        return app, results

    def CPClean(self, dirtyDF, detections, configs):
        '''
        Runs CPClean and saves the models test results for Scenarios S1, S4 and S5 in the model .csv file. Can only be applied
        on binary classification tasks (throws error otherwise).
        
        Val: 20%, Train: 60%, Test:20%
        Parameter k of KNN can be modified.
        
        Arguments:
        detection_dictionary -- dictionary - keys represent i,j of dirty cells & values are constant string "JUST A DUUMY VALUE"
        dirtyDF -- dataframe of shape n_R (# of records) x n_A (# of attributes) - containing a dirty version of a dataset
        configs: method (String) -- empty
        
        Returns:
        results (dict) -- dictionary with p, r, f1 for the applicable scenarios
        app -- instance of model class that can be used to store results
        '''

        ##### 1. impute np.nan into detected cells in dirtyDF_nan

        dirtyDF_nan = dirtyDF.copy()#.apply(pd.to_numeric, errors="ignore")

        # change all occurances detections to np.nan
        # imputers identify cells to impute by checking if they are np.nan
        for (row_i, col_i), dummy in detections.items():
            dirtyDF_nan.iat[row_i, col_i] = np.nan

        dirtyDF_nan = dirtyDF_nan.apply(pd.to_numeric, errors="ignore")
        dirtyDF = dirtyDF.apply(pd.to_numeric, errors="ignore")
        groundTruthDF = self.groundTruthDF.apply(pd.to_numeric, errors="ignore")
        app = Models(self.__dataset_name)

        ##### 2. Split data into train, validation, test set

        np.random.seed(1)
        N = dirtyDF_nan.shape[0]
        idx = np.random.permutation(N)

        test_size = int(0.2*N)
        val_size = int(0.2*N)
        train_size = N-test_size-val_size

        # get indices of train, test, val
        idx_test = idx[:test_size]
        idx_val = idx[test_size: test_size+val_size]
        idx_train = idx[test_size+val_size: N]

        scenarios = {
            "S1" : {"df" : dirtyDF, "repair" : False}, # dirty train set and dirty test set (equivelent to knn on dirty data)
            "S4" : {"df" : groundTruthDF, "repair" : False}, # clean train set and clean test set (equivelent to knn on clean data)
            "S5" : {"df" : dirtyDF_nan, "repair" : True} # repaired train set, dirty test set
        }

        # initialize results dict
        results = {
            "model" : str(RepairMethod.cpClean),
            "S1": [],
            "S2" :[0.0, 0.0, 0.0],
            "S3" :[0.0, 0.0, 0.0],
            "S4" :[],
            "S5" :[],
        }

        # iterate over scenarios
        for s, configuration in scenarios.items():
            df = configuration["df"]

            # seperate dirty features and dirty/clean labels,  
            y_dirty = df[app.labels_list_clf]
            y_clean = groundTruthDF[app.labels_list_clf]
            if "features_clf" in datasets_dictionary[app.dataset_name]:  # list of features are given
                logging.info("Loading given features ..")
                X = df[app.features_clf]
            else:
                X = df.drop(app.labels_list_clf, axis=1)  # no features are given

            # exit if more than two classes
            if len(np.unique(y_clean)) > 2 or len(np.unique(y_dirty)) > 2:
                print('unique values in ground truth', np.unique(y_clean))
                print('unique values in dirty dataset', np.unique(y_dirty))
                logging.info("Invalid dataset: more than two classes")
                break

            # X_train, y_train. in S1: both dirty (not modified), in S4: both groundtruth,
            # in S5: both dirty but with imputed np.nan where error was detected (detection_dict)
            X_train = X.iloc[idx_train].reset_index(drop=True)
            y_train = y_dirty.iloc[idx_train].reset_index(drop=True)

            # validation. y_val always groundtruth. S1: X_val dirty (not modifiyed), S4: X_val groundtruth,
            # S5: X_val dirty with imputed np.nan where error was detected (detection_dict)
            X_val = X.iloc[idx_val].reset_index(drop=True)
            y_val = y_clean.iloc[idx_val].reset_index(drop=True)

            # test. y_test always clean X. S1: X_test dirty (not modifiyed), S4: X_test groundtruth,
            # S5: X_test dirty with imputed np.nan where error was detected (detection_dict)
            X_test = X.iloc[idx_test].reset_index(drop=True)
            y_test = y_clean.iloc[idx_test].reset_index(drop=True)


            # For S5: repair() returns a dictionary with the repaired version
            # of the dataset for each repair method within CPClean
            data = {}
            if configuration["repair"]:
                X_train_repairs = repair(X_train)
                data["X_train_repairs"] = X_train_repairs
            else:
                # for S1: X_train = dirty, S4: X_train = groundTruth
                X_train_repairs = {}
                data["X_train_repairs"] = X_train_repairs
                #X_train_repairs["default"] = X_train
                data["X_train_repairs"]["mean"] = X_train

            ##
            # need to be set, in order for CPClean to run
            data["X_full"] = X
            data["y_full"] = y_clean
            data["X_train_dirty"] = X_train
            data["X_train_clean"] = X_train
            data["indicator"] = df.isnull()
            ##

            data["X_test"] = X_test
            data["X_val"] = X_val

            data["y_train"] = y_train # y_train can be dirty
            data["y_test"] = y_test # needs to be clean
            data["y_val"] =y_val # needs to be clean

            ##### 3. preprocess the data
            data = preprocess(data)

            print(data)

            ##### 4. set model used for boosting, implementation by CPClean only supports KNN
            model = {
                "fn": KNN,
                "params": {"n_neighbors":3}
            }

            X_train_repairs = np.array([data["X_train_repairs"][m] for m in data["repair_methods"]])


            ##### 5. call CPClean and get accuracy on test and validationt set plus precision, recall f1 on test data
            cleaner = CPClean(K=model["params"]["n_neighbors"], n_jobs=4)
            cleaner.fit(X_train_repairs, data["y_train"], data["X_val"], data["y_val"],
                        gt=data["X_train_gt"], X_train_mean=data["X_train_repairs"]["mean"],
                        restore=False, method="sgd_cpclean", sample_size=64)

            p, r, f1 = cleaner.prf(data["X_test"], data["y_test"])

            results[s] = [p, r, f1]

        return app, results

    def cleanlab(self, dirtyDF, detections, configs):
        """
        This method runs cleanlabs cleanwithNoisyData function, which is a wrapper for a classifier as e.g. randomForest. This
        Method does not cleaning but only creates a Model and stores precision, recall and f1 on test data. Therefore the results are 
        independent from the detections, the detection are not used. Can be used on multi-classes classification tasks

        Arguments:
        dirtyDF -- dataframe of shape n_R (# of records) x n_A (# of attributes) - containing a dirty version of a dataset
        detections -- dictionary - keys represent i,j of dirty cells & values are constant string "JUST A DUUMY VALUE"
        configs (dict) -- "model" has to be provided and be contained in /auxilaries/models_dictionary. Model is used for LearningWithNoisyLabels function

        Returns:
        results (dict) -- dictionary with p, r, f1 for the applicable scenarios
        app -- instance of model class that can be used to store results
        """
        dataset = self.__dataset_name

        # set model that is to be used
        model_name = configs["model"]

        # get dataset specific ml_task and model
        for model_type in models:
            if model_name in models_dictionary[model_type]:
                ml_task = model_type
                model = models_dictionary[model_type][model_name]

        # exit if model_task is not classification
        if ml_task != classification:
            logging.info("Invalid model_name: only classification as ml_task allowed")
            sys.exit(1)

        # instantiate  model class
        app = Models(dataset)

        # Use the ML model with default parameters (i.e., not optimized)
        estimator = model["fn"](**model["fixed_params"])

        # split dirty data
        X_train_dirty, y_train_dirty, X_test_dirty, y_test_dirty = app.preprocess(dirtyDF, ml_task)

        # split groundtruth data. Model class instantaied seed, thus split is the same
        X_train_gt, y_train_gt, X_test_gt, y_test_gt = app.preprocess(self.groundTruthDF, ml_task)

        average = "micro" if len(np.unique(y_test_gt)) > 2 else "binary"

        # scenarios to run
        scenarios = {
            "S1" : {"X_train" : X_train_dirty, "y_train": y_train_dirty, "X_test": X_test_dirty}, # dirty train set and dirty test set
            "S4" : {"X_train" : X_train_gt, "y_train": y_train_gt, "X_test": X_test_gt}, # clean train set and clean test set
        }

        # initialize results dict
        results = {
            "model" : str(RepairMethod.cleanlab),
            "S1": [],
            "S2" :[0.0, 0.0, 0.0],
            "S3" :[0.0, 0.0, 0.0],
            "S4" :[],
            "S5" :[0.0, 0.0, 0.0],
        }

        # iterate through scenarios
        for s, configuration in scenarios.items():

            # cleanlab uses stratifiedKfold of size 5. If condition is true
            # each fold cannot include all possible classes which leads to a missmatch
            if (configuration["X_train"].shape[0] / 5 < len(np.unique(configuration["y_train"]))):
                results[s] = ["too many classes"]
                continue

            # train classifier
            lnl = LearningWithNoisyLabels(clf=estimator)
            lnl.fit(configuration["X_train"], configuration["y_train"])

            y_pred = lnl.predict(configuration["X_test"])

            p, r, f1, _ = precision_recall_fscore_support(y_test_gt, y_pred, average=average)

            results[s] = [p, r, f1]

        return app, results
        
        
################################################

################################################
if __name__ == "__main__":
    pass

################################################

