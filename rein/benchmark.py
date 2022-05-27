####################################################
# Benchmark: main script for running the experiments
# Authors: Christian Hammacher, Mohamed Abdelaal
# Date: February 2021
# Software AG
# All Rights Reserved
####################################################


####################################################

from rein.auxiliaries.datasets_dictionary import datasets_dictionary
from rein.auxiliaries.detectors_dictionary import detectors_dictionary
from rein.auxiliaries.cleaners_configurations import cleaners_configurations
from rein.auxiliaries.models_dictionary import *
from rein.datasets import Datasets, Database
from rein.detectors import Detectors
from rein.cleaners import Cleaners
from rein.models import Models, models
from functools import reduce
import shutil

####################################################
class Benchmark:
    """
    Class encapsulates all required methods to run the different experiments
    """

    def __init__(self, store_postgres=False, no_ground_truth=False):
        """
        Constructor defining default variables
        """
        self.__logging_setup(logging_configs_console())
        self.store_postgres = store_postgres
        self.no_ground_truth = no_ground_truth

    def __logging_setup(self, root_logger):
        """This method setup necessary configurations for logging"""
        # Remove possible already-existent handlers
        while root_logger.handlers:
              root_logger.handlers.clear()
        # Define the new configurations
        logging_configs_file()

    def __get_dataset_dictionary(self, dataset_name):
        """
        returns dictionary for a given dataset_name

        Arguments:
        dataset_name (String) -- representing the name of the relative dataset
        """
        return datasets_dictionary[dataset_name]

    def __store_results(self, results, results_path):
        """
        stores results dict as a json file in results_path

        Arguments:
        results (dict) -- dictionary with the results
        results_path (String) -- path to the folder where results should be stores
        """
        with open(os.path.join(results_path, "results.json"), "w") as handle:
            json.dump(results, handle)

    def __store_detection_results(self, results, dataset_name, detector_name, exp_id):
        """
          This method stores the cleaning results in an CSV file located at the results directory

          :param
            results-- a dictionary containing the quality scores
            detector_name -- string denoting the name of the detector
            cleaner_name -- string denoting the name of the repairing method
          """

        # Create a list of keys in the dictionary
        key_list = list(results.keys())

        # ============================ Setting up the directories and pathes ===============
        # Create the path of the results directory
        results_dir = os.path.join(datasets_dictionary[dataset_name]["dataset_path"], "results")
        if not os.path.exists(results_dir):
            # Create a new directory if it does not exit
            os.mkdir(results_dir)
        results_path = os.path.join(results_dir, "detection_results.csv")

        #===================== Stroing results ============================
        write_header = False
        # Check if the file already exists
        if not os.path.exists(results_path):
            write_header = True

        # Open an CSV file in append mode. Create a file object for this file
        with open(results_path, 'a') as f_object:
            # Create a file object and prepare it for writing the results
            writefile = csv.writer(f_object)
            if write_header:
                writefile.writerow(["exp_id", "time", "detector"] + key_list)  # write the header
            # Prepare the row which is to be written to the file
            row = [exp_id, datetime.now(), detector_name, [results[index] for index in key_list]]
            # Write the values after flattening the row list obtained in the above line
            writefile.writerow(np.hstack(row))

        # Close the file object
        f_object.close()

    def __store_cleaning_results(self, exp_id, results, dataset_name, detector_name, cleaner_name):
        """
          This method stores the cleaning results in an CSV file located at the results directory

          :param
            results-- a dictionary containing the quality scores
            detector_name -- string denoting the name of the detector
            cleaner_name -- string denoting the name of the repairing method
          """

        # Create a list of keys in the dictionary
        key_list = list(results.keys())

        # ============================ Setting up the directories and pathes ===============

        # Create the path of the results directory
        results_dir = os.path.join(datasets_dictionary[dataset_name]["dataset_path"], "results")
        if not os.path.exists(results_dir):
            # Create a new directory if it does not exit
            os.mkdir(results_dir)
        results_path = os.path.join(results_dir, "cleaning_results.csv")

        #===================== Stroing results ============================
        write_header = False
        # Check if the file already exists
        if not os.path.exists(results_path):
            write_header = True

        # Open an CSV file in append mode. Create a file object for this file
        with open(results_path, 'a') as f_object:
            # Create a file object and prepare it for writing the results
            writefile = csv.writer(f_object)
            # Check if the file already exists
            if write_header:
                writefile.writerow(["exp_id", "time", "detector", "cleaner"] + key_list)  # write the header
            # Prepare the row which is to be written to the file
            row = [exp_id, datetime.now(), detector_name, cleaner_name, [results[index] for index in key_list]]
            # Write the values after flattening the row list obtained in the above line
            writefile.writerow(np.hstack(row))

        # Close the file object
        f_object.close()

    def __find_files(self, path, name):
        """
            This method searches for all detections.csv files in the dataset directory
        :param
            path -- string, directory of the dataset
        :return:
            result -- list of paths (strings) to the required files
        """

        result = []
        for root, dirs, files in os.walk(path):
            if name in files:
                result.append(os.path.join(root, name))
        return result

    def __find_detections_files(self, path, name="detections.csv"):
        return self.__find_files(path, name)

    def __find_repaired_datasets(self, path, name="repaired.csv"):
        return self.__find_files(path, name)

    def __constraints_exist(self, dataset_name):
        """This method checks if the dataset has FD constrains"""
        dir_path = os.path.join(datasets_dictionary[dataset_name]["dataset_path"], "constraints")
        return os.path.exists(dir_path)

    def __extract_from_path(self, dir, loc):
        """
        This method extracts a component from the path
        :param dir -- string, path to be split
        :return -- string denoting the name of a method
        """
        # Split the path to its individual components
        dir_split = os.path.normpath(dir).split(os.path.sep)
        # Extract the name of the current method from the path
        return dir_split[-loc]

    def __get_detections_dictionary(self, path):
        """
        df = pd.read_csv(os.path.join(path,"detections.csv"), header=None, names=["left", "right","value"])
        df["left"] = [x[1:] for x in df["left"]]
        df["right"] = [x[:-1] for x in df["right"]]
        df["key"] = df[["left","right"]].apply(tuple, axis=1)
        df[["left","right"]]=df[["left", "right"]].apply(pd.to_numeric)
        detection_dict = df.set_index("key")["value"].to_dict()
        """
        try:
            reader = pd.read_csv(os.path.join(path,"detections.csv"), names=['i','j','dummy'])
            detection_dict = reader.groupby(['i','j'])['dummy'].apply(list).to_dict()
        except:
            logging.info("No detections found ..")
            return {}

        # if detections found, return them as a dictionary
        return detection_dict

    def __get_detectors_list(self, dataset):
        """
        This method retieves a list of detectors to detect the errors exist in this dataset
        :param dataset: string of the dataset
        :param return_detectors: predicate to identify whether to return detectors or cleaners
        :return:
        """
        # Retrieve a list of error types in the dataset
        errors = datasets_dictionary[dataset]["error_types"]
        detect_list = []
        for error_type in errors:
            detect_list.extend(detectors_dictionary[error_type]["detectors_list"])
        detectors_list = list(set(detect_list))

        return detectors_list, errors

    def __get_true_detections(self, dataset, detections):
        """
        This method extracts the truly-detected errors from a detections.csv file
        :param detections: dictionary of the detections
        :return:
        """

        # Retrieve the dataset path
        dataset_path = self.__get_dataset_dictionary(dataset)["dataset_path"]

        # === Load the actual errors ===
        actual_errors_path = os.path.join(dataset_path, "actual_errors.csv")
        if os.path.exists(actual_errors_path):
            # Load the actual_errors dictionary
            actual_errors = pd.read_csv(actual_errors_path, names=['rows', 'attribs', 'dummy'])
            #actual_errors_dictionary = reader.groupby(['i', 'j'])['dummy'].apply(list).to_dict()
        else:
            logging.info("Actual errors have not been estimated.")

        # === Finding the common entries in both the detections and actual errors
        common_detections = pd.merge(detections, actual_errors, how='inner', on=['rows', 'attribs'])
        true_detections = common_detections.drop_duplicates()

        return true_detections

    def get_iou(self, dataset):
        """
        This method computes the IoU metric for detectors run on a certain dataset
        :param
            dataset -- string, denoting the name of a dataset
        :return:
        """

        logging.info("==========================================================================")
        logging.info("=========================  IoU Testing Phase =============================")
        logging.info("==========================================================================\n")

        # Obtain the path of the directories containing the detections.csv files
        dataset_path = datasets_dictionary[dataset]["dataset_path"]
        directories = self.__find_detections_files(dataset_path)

        # Get a list of the name of the detection methods
        methods = []
        for dir in directories:
            method = self.__extract_from_path(dir, loc=2)
            # Add the name of the current method to the methods list
            methods.append(method)

        # ======= Prepare a 2D matrix for storing the IoU values =========
        iou_matrix = np.zeros((len(methods), len(methods)))
        # Get a list of indices for the methods
        indices_matrix = range(len(iou_matrix))
        # Create a dictionary to map from a method to an index
        mapping = {key:value for key, value in zip(methods, indices_matrix)}


        # Find all combinations to compute the IoU between every two methods
        # comb is a list of tuples, where each tuple contains directories of two methods
        comb_path = [ ' '.join([directories[i], directories[j]]) for i, j in
                 itertools.combinations(range(len(directories)), 2)]
        comb_method = [ ' '.join([methods[i], methods[j]]) for i, j in
                      itertools.combinations(range(len(methods)), 2)]

        colmuns = ['rows', 'attribs', 'dummy']
        for dirs, methods_pair in zip(comb_path, comb_method):
            try:
                # Split the string into two paths
                dirs_list = dirs.split(sep=' ')
                # Check if one of the files (or both) is empty, i.e., file_size = 0
                if os.path.getsize(dirs_list[0]) != 0 and os.path.getsize(dirs_list[1]) != 0:
                    #=== Load the detections into two dataframes ===
                    detections_1 = pd.read_csv(dirs_list[0], names=colmuns)
                    detections_2 = pd.read_csv(dirs_list[1], names=colmuns)
                    # Obtain the true positives
                    tp_1 = self.__get_true_detections(dataset, detections_1)
                    tp_2 = self.__get_true_detections(dataset, detections_2)
                    #logging.info("#True Detections by {}: {}, {}".format(methods, len(tp_1), len(tp_2)))
                    # Obtain common rows between two detections
                    common_detections = pd.merge(tp_1, tp_2, on=['rows','attribs'], how='inner')
                    intersection = len(common_detections.drop_duplicates())
                    # Estimate the union using the inclusion-exclusion principle
                    union = len(tp_2) + len(tp_1) - intersection
                    logging.info("Union & intersection: {}, {}".format(union, intersection))
                    iou = intersection / union
                    logging.info("IoU (based on True Positives) for {}: {}".format(methods_pair, iou))

                    # ====== Store the obtained IoU in the correct location in the 2D matrix =====
                    methods_list = methods_pair.split(sep= ' ')
                    iou_matrix[mapping[methods_list[0]]][mapping[methods_list[1]]] = iou
                    iou_matrix[mapping[methods_list[1]]][mapping[methods_list[0]]] = iou
                else:
                    logging.info("One of the methods (or both) did not detect errors ..")
            except Exception as e:
                logging.info("Exception: {}".format(e.args[0]))

        # ====== Save the IoU matrix in an CSV file ==========
        results_dir = os.path.join(datasets_dictionary[dataset]["dataset_path"], "results")
        if not os.path.exists(results_dir):
            # Create a new directory if it does not exit
            os.mkdir(results_dir)
        iou_path = os.path.join(results_dir, 'iou.csv')
        with open(iou_path, "w+") as my_csv:
            csvWriter = csv.writer(my_csv, delimiter=',')
            # Write the header
            csvWriter.writerow(methods)
            # Write the IoU matrix
            csvWriter.writerows(iou_matrix)

    def run_detectors(self, exp_id, datasets_list, specified_detectors, iterations):
        """
        This method runs several detectors on a list of datasets

        :param
            datasets_list -- list of strings denoting the datasets to be considered
            exp_id -- integer denoting the ID of the experiment, used for scalability analysis
        """

        for dataset in datasets_list:

            # Retrieve a dataset dictionary
            dataset_dict = self.__get_dataset_dictionary(dataset)

            # Instantiate a database Object
            db_object = Database()

            # Handle the case of missing ground truth
            # Copy the dirty data, rename to ground truth to run methods which rely on ground truth
            if self.no_ground_truth:
                assert not os.path.exists(dataset_dict["groundTruth_path"]), 'CSV file of GT already exists'
                assert not db_object.db_table_exists_postgresql('ground_truth_{}'.format(dataset)), 'GT already exists in the rein database'
                logging.info("Creating an artificial ground truth file")
                shutil.copy2(dataset_dict["dirty_path"], dataset_dict["groundTruth_path"])


            # Initiate a dataset object
            curr_dataset = Datasets(dataset_dict)

            # Load dirty & ground truth datasets
            if db_object.db_table_exists_postgresql('dirty_{}'.format(dataset)):
                logging.info("Loading the **{}** dataset from the rein database".format(dataset))
                dirtyDF = db_object.load_df_postgresql('dirty_{}'.format(dataset))
                groundtruthDF = db_object.load_df_postgresql('ground_truth_{}'.format(dataset))
            else:
                logging.info("Loading the **{}** dataset from the CSV files".format(dataset))
                dirtyDF = curr_dataset.load_data(dataset_dict["dirty_path"])
                groundtruthDF = curr_dataset.load_data(dataset_dict["groundTruth_path"])
                # Store the dataset in the REIN database
                if not self.no_ground_truth:
                    try:
                        db_object.write_df_postgresql(dirtyDF, 'dirty_{}'.format(dataset))
                        db_object.write_df_postgresql(groundtruthDF, 'ground_truth_{}'.format(dataset))
                    except:
                        logging.info("Unable to insert the datasets into the REIN database")

            # Estimate the error rate and actual_error_dictionary
            logging.info("Getting error rate and actual_error_dict")
            actual_errors_path = os.path.join(curr_dataset.dataset_path, "actual_errors.csv")
            error_rate_path = os.path.join(curr_dataset.dataset_path, "error_rate.json")
            if os.path.exists(actual_errors_path) and os.path.exists(error_rate_path):
                # Load the actual_errors dictionary
                reader = pd.read_csv(actual_errors_path, names=['i','j','dummy'])
                #actual_errors_dictionary = reader.set_index(['i','j'], drop=True, inplace=True)
                #actual_errors_dictionary = reader.T.to_dict('records')
                actual_errors_dictionary = reader.groupby(['i','j'])['dummy'].apply(list).to_dict()
                # Load the JSON file
                with open(error_rate_path) as json_file:
                    error_rate = json.load(json_file)
            else:
                actual_errors_dictionary, error_rate = curr_dataset.get_actual_errors(
                    dirtyDF, groundtruthDF)

            # Instantiate a detector instance
            detector = Detectors(dataset, actual_errors_dictionary)

            # Loading the key cloumns for duplicates detection
            keys = [] if "keys" not in datasets_dictionary[dataset] else \
                datasets_dictionary[dataset]["keys"]

            # Define correct labels, which is necessary for mislable detector
            correct_labels = []

            # Retrieve detectors list
            relevant_detectors, _ = self.__get_detectors_list(dataset)
            # Limit the detectors to only user-sepecified detectors
            relevant_detectors = specified_detectors if specified_detectors else relevant_detectors

            # Re-order the list by adding ensemble methods at the end
            all_detectors = list(relevant_detectors).copy()
            # Convert the detectors from DetectMethod to string
            for item in all_detectors:
                if item in ensemble_detectors:
                    relevant_detectors.append(item)
                    relevant_detectors.remove(item)

            logging.info("Relevant detectors: {}".format(relevant_detectors))

            logging.info("==========================================================================")
            logging.info("========================  Error Detection Phase ==========================")
            logging.info("==========================================================================\n")

            # Map the detectors from strings onto functions
            # We can only perform mapping after instantiating a Detectors instance
            functions_list = []
            for item in relevant_detectors:
                for function in detector.detectors_list:
                    if str(item) == function.__name__:
                        functions_list.append(function)

            try:
                # Detect dirty cells using all available detectors
                for function in tqdm.tqdm(functions_list):

                    # Extract the name of the detection method
                    method = function.__name__

                    # Create a list of outlier detection methods
                    detect_methods = ["SD", "IQR", "IF"] if  method == "outlierdetector" else ["ALL"]

                    # Iterate over each duplicates detection method
                    # For other detectors, these loop will be executed once, only "SD" in the list
                    for detect_method in detect_methods:

                        # Prepare the (necessary) configurations  for each detector
                        configs = {}
                        configs["detect_method"] = detect_method
                        configs["keys"] = keys
                        configs["path_dirty"] = dataset_dict["dirty_path"]
                        configs["correct_labels"] = correct_labels
                        # min-k
                        configs['threshold'] = 0.4
                        # max entropy
                        configs["precision_threshold"] = 0.01
                        configs["sampling_budget"] = 200
                        # ed2: labeling budget
                        configs["label_cutoff"] = 20 * groundtruthDF.shape[1]

                        # cleanlab
                        configs['model_name'] = "forest_clf"

                        # Setting the directory name, where the detections will be stored
                        dir_name = '_'.join([method, detect_method]) if method in ["outlierdetector", "fahes"] else method

                        logging.info("Detecting errors using ------------> ***********{}***********".format(dir_name))
                        # Find the dirty cells and generate a detections.csv file
                        for index in range(iterations):
                            try:
                                detection_dictionary, detection_results_dict = function(dirtyDF, dataset, configs)
                                # Add the error rate to the results dictionary
                                detection_results_dict["error_rate"] = error_rate
                                # Store results from error detection
                                logging.info("Iteration {}: Storing detection results".format(index))
                                for key, value in detection_results_dict.items():
                                    logging.info('{}: {}'.format(key, value))
                                self.__store_detection_results(detection_results_dict, dataset, dir_name, exp_id)
                            except Exception as e:
                                logging.info("Exception: {}".format(e.args[0]))
                                logging.info("Exception: {}".format(sys.exc_info()))
                                break
            except Exception as e:
                logging.info("Exception: {}".format(e.args[0]))
                logging.info("Exception: {}".format(sys.exc_info()))
                continue

            # Remove intermediate files and data
            if self.no_ground_truth:
                os.remove(dataset_dict["groundTruth_path"])
                os.remove(error_rate_path)
                os.remove(actual_errors_path)

    def run_models(self, datasets_list, ml_model, iterations, hyperopt=False, fastTermination=False):
        """
        This method runs various ML tasks on a set of clean, dirty, and repaired datasets

        Steps:
            * run a model on the dirty dataset
            * run a model on all repaired datasets

        :param 
            dataset_lists -- list of strings denoting the names of datasets
            model -- name of model from models_dictionary to be used
            iterations -- integer defines the numer of times the ML training process will be repeated
            hyperopt -- Bolean defines whether to run hyperparameters optimization
            fastTermination -- Bolean defines whether to limit the number of iterations if the difference between the average of n iterations and the current F1 is negligable
        :return:
        """

        logging.info("==========================================================================")
        logging.info("========================  ML Testing Phase ===============================")
        logging.info("==========================================================================\n")

        # Convert model name to string
        model_name = str(ml_model)

        # Identify the ML task, i.e., "classification", "regression", or "clustering"
        for model_type in models:
            if model_name in models_dictionary[model_type]:
                ml_task = model_type

        for dataset in datasets_list:

            if self.store_postgres:
                db_object = Database()
                dirty_repaired_paths  = db_object.get_table_names_postgresql()
                # Exclude ground truth from the list
                if 'ground_truth_{}'.format(dataset) in dirty_repaired_paths:
                    dirty_repaired_paths.remove("ground_truth_{}".format(dataset))
                elif not dirty_repaired_paths:
                    # Handle the case if no data in the REIN database
                    logging.info("No data exist in the database")
            else:
                # ================ Prepare the data paths ====================
                # Obtain the path to the dataset
                path = datasets_dictionary[dataset]["dataset_path"]
                # Obtain the path to the dirty dataset
                dirty_path = datasets_dictionary[dataset]["dirty_path"]
                # Obtain a list of paths to the repaired datasets
                repaired_paths= self.__find_repaired_datasets(path=datasets_dictionary[dataset]["dataset_path"])
                # Insert all paths to one list
                dirty_repaired_paths = []
                dirty_repaired_paths = [dirty_path] + repaired_paths

            logging.info("Number of repaired/dirty datasets to be trained: **{}**".format(len(dirty_repaired_paths) * iterations))

            # ================= Looping over dirty and all repaired datasets =====
            for data_path in tqdm.tqdm(dirty_repaired_paths):
                if self.store_postgres:
                    logging.info("Current data path: {}".format(data_path))
                    if data_path == 'dirty_{}'.format(dataset):
                        data_name = data_path.split('_')[0]
                        data = db_object.load_df_postgresql(data_path)
                    else:
                        cleaner_name = self.__extract_from_path(data_path, loc=2)
                        detector_name = self.__extract_from_path(data_path, loc=3)
                        data_name = detector_name + "_" + cleaner_name
                        data = db_object.load_df_postgresql(data_path)

                else:
                    # Obtain the name of the data
                    logging.info("Current data path: {}".format(data_path))
                    cleaner_name = self.__extract_from_path(data_path, loc=2)
                    detector_name = self.__extract_from_path(data_path, loc=3)
                    data_name = "dirty" if os.path.split(data_path)[1] == "dirty.csv" else \
                        detector_name + "_" + cleaner_name
                    # Load a dataset (dirty, or repaired) to test against several ML models
                    data = pd.read_csv(data_path, low_memory=False)

                logging.info("Training & evaluating **{}** on the **{}** dataset .. ".format(model_name, data_name))

                # Check if the data file is empty or has unsufficient samples
                if len(data) < 50:
                    logging.info("Skipping the dataset {}: No sufficient records.".format(data_name))
                    # skip this version of the dataset
                    continue

                # Append the results to run the statistical tests
                results_all_runs = {}
                results_all_runs['S1'] = []
                results_all_runs['S4'] = []

                # Repeat the process x times,
                for index in tqdm.tqdm(range(iterations)):

                    # Instantiate a model object
                    model = Models(dataset)

                    # Initialize the results dictionary
                    results = {}
                    data_prepared = {}
                    average = 0

                    # Prepare the input dataset
                    """
                    The input dataset is preprocessed here (not in train_and_test) to avoid being preprocessed, 
                    each time a model is to be trained using the same dataset
                    """
                    try:
                        data_prepared = model.prepare_datasets(data, ml_task)
                    except Exception as e:
                        logging.info(e)
                        logging.info(sys.exc_info())
                        logging.info('Data preparation failed')
                        break

                    # Break the loop if it has only one label class
                    if (len(np.unique(data_prepared["dataset"][1])) == 1):
                        logging.info("Dataset ignored: The dirty/reapired dataset has only one label class ..")
                        break

                    try:
                        # Train and evaluate the ML model using the input dataset and the ground truth dataset
                        results = model.train_and_test(data_name, model_name, data_prepared, optimization=hyperopt)
                    except Exception as e:
                        logging.info(e)
                        logging.info(sys.exc_info())
                        logging.info('Failed operation: Training the {} model'.format(model_name))
                        break


                    # Store the results of each iteration
                    results_all_runs['S1'].append(results['S1'][0])
                    results_all_runs['S4'].append(results['S4'][0])

                    # Terminate the execution, if the average accuracy over several runs is lower than a threshold
                    average = reduce(lambda a, b: a+b, results_all_runs['S4']) / len(results_all_runs['S4'])
                    logging.info("Iteration {}: the accuracy of S4: {}".format(index, results['S4'][0]))
                    logging.info("Average accuracy of S4: {}".format(average))
                    if index > iterations/2 and fastTermination:
                        if abs(average - results['S4'][0]) < 0.5:
                            break

                    # ==================== Storing the results ========================

                    # Append the name of the model to the results
                    results["model"] = model_name + "_" + data_name
                    if hyperopt:
                        results["model"] += "_opt"

                    # Store the results in an CSV file located in the models folder
                    model.store_results(results, ml_task)
                    logging.info("Results have been successfully saved to the results directory")

                # Statistical tests
                try:
                    t, p_value = wilcoxon(results_all_runs['S1'], results_all_runs['S4'], zero_method='wilcox', correction=True)
                    effect_size, _ = pearsonr(results_all_runs['S1'], results_all_runs['S4'])
                    logging.info("Wilcoxon test: The p-value: {}".format(p_value))
                    logging.info("Pearson’s correlation coefficient: The effect size: {}".format(effect_size))
                    model.store_abtesting(model_name, data_name, p_value, effect_size)
                except:
                    logging.info("Cannot estimate the Wilcoxon metric.")
                    continue

    def run_all_models(self, dataset_list, iterations, hyperopt, fastTermination):
        """
        This method runs run_models for all possible models for all datasets in dataset_list.

        Steps:
            * run a model on the dirty dataset
            * run a model on all repaired datasets

        :param 
            dataset_lists -- list of strings denoting the names of datasets
        :return:
        """

        dataset_dict = {}
        for i, dataset_name in enumerate(dataset_list):
            dataset_dict = self.__get_dataset_dictionary(dataset_name)
            ml_tasks = dataset_dict["ml_tasks"]
            logging.info("Running ml tasks {} for dataset {}\n".format(ml_tasks, dataset_name))
            
            for ml_task in ml_tasks:
                logging.info("Running {} model".format(ml_task))
                for model in models_dictionary[ml_task]:
                    self.run_models([dataset_name], model, iterations, hyperopt=hyperopt, fastTermination=fastTermination)
            logging.info("Finished running all models for {}".format(dataset_name))
        
    def run_cleaners(self, exp_id, dataset_list, specified_cleaners, iterations):

        for dataset in dataset_list:

            # Retrieve a dataset dictionary
            dataset_dict = self.__get_dataset_dictionary(dataset)

            # Instantiate datasets and database Object
            curr_dataset = Datasets(dataset_dict)
            db_object = Database()

            logging.info("==========================================================================")
            logging.info("========================  Data Repair Phase ==============================")
            logging.info("==========================================================================\n")

            # Load dirty & ground truth datasets
            logging.info("Loading **{}** dataset and its ground truth".format(str(dataset)))
            if db_object.db_table_exists_postgresql('dirty_{}'.format(dataset)):
                dirtyDF = db_object.load_df_postgresql('dirty_{}'.format(dataset))
                groundtruthDF = db_object.load_df_postgresql('ground_truth_{}'.format(dataset))
            else:
                dirtyDF = curr_dataset.load_data(dataset_dict["dirty_path"])
                groundtruthDF = curr_dataset.load_data(dataset_dict["groundTruth_path"])
                # Store the dataset in the REIN database
                try:
                    db_object.write_df_postgresql(dirtyDF, 'dirty_{}'.format(dataset))
                    db_object.write_df_postgresql(groundtruthDF, 'ground_truth_{}'.format(dataset))
                except:
                    logging.info("Unable to insert the datasets into the 'rein' database")

            # Estimate the error rate and actual_error_dictionary
            logging.info("Getting error rate and actual_error_dict")
            actual_errors_path = os.path.join(curr_dataset.dataset_path, "actual_errors.csv")
            error_rate_path = os.path.join(curr_dataset.dataset_path, "error_rate.json")
            if os.path.exists(actual_errors_path) and os.path.exists(error_rate_path):
                # Load the actual_errors dictionary
                reader = pd.read_csv(actual_errors_path, names=['i', 'j', 'dummy'])
                actual_errors_dictionary = reader.groupby(['i', 'j'])['dummy'].apply(list).to_dict()
                # Load the JSON file
                with open(error_rate_path) as json_file:
                    error_rate = json.load(json_file)
            else:
                actual_errors_dictionary, error_rate = curr_dataset.get_actual_errors(
                    dirtyDF, groundtruthDF)

            # Retrieve detectors list and error types
            dataset_dets, error_types = self.__get_detectors_list(dataset)
            logging.info("Error types in this dataset: {}".format(error_types))
            # Convert detectors into strings
            dataset_detectors = [str(x) for x in dataset_dets]

            # Adjust the detector name, if the detector has several methods, e.g., outlierdetector_IF, FAHES_ALL
            relevant_detectors = list(dataset_detectors.copy())
            for item in dataset_detectors:
                if item == str(DetectMethod.outlierdetector):
                    relevant_detectors.remove(str(DetectMethod.outlierdetector))
                    relevant_detectors.extend(["outlierdetector_IF", "outlierdetector_IQR","outlierdetector_SD"])
                elif item == str(DetectMethod.fahes):
                    relevant_detectors.remove(str(DetectMethod.fahes))
                    relevant_detectors.append("fahes_ALL")

            logging.info("Relevant detectors for this dataset: {}".format(relevant_detectors))

            # ============= Traverse the detectors to repair the detected errors ================
            for detector_name in tqdm.tqdm(relevant_detectors):
                logging.info("Applying cleaning methods for detector ----------> ****{}****".format(detector_name))
                detector_path= os.path.join(dataset_dict["dataset_path"],detector_name)
                # Load the detections
                detection_dict = self.__get_detections_dictionary(detector_path)
                # Ignore the detector, if there exists no detections.csv file
                if not detection_dict:
                    continue

                # Instantiate a cleaners instance for each detector
                cleaner = Cleaners(dataset, detector_name, groundtruthDF, actual_errors_dictionary, self.store_postgres)

                # ================================== Finding relevant cleaners for each detector ======
                # Initialize an empty list to add relevant cleaners
                cleaners_list = []
                # Obtain the first part of the detector name, if the name includes an underscore
                #detect_name = detector_name.split('_')[0] if detector_name.find('_') != -1 else detector_name
                # 2 - append the cleaners existing in each error type which has the detector name
                for error in error_types:
                    #if detect_name in detectors_dictionary[error]["detectors_list"]:
                    cleaners_list.extend(detectors_dictionary[error]["cleaners_list"])
                # Remove duplicated methods from the list
                relevant_cleaners = list(set(cleaners_list))

                # Limit the cleaners to the unser-specified cleaners list
                clean_methods = specified_cleaners if specified_cleaners else relevant_cleaners
                # Convert cleaners into strings
                relevant_cleaners = [str(x) for x in clean_methods]

                # Find DC constraints for HoloClean
                if self.__constraints_exist(dataset):
                    constraints_path = os.path.join(datasets_dictionary[dataset]["dataset_path"], "constraints")
                elif str(RepairMethod.dcHoloCleaner) in relevant_cleaners:
                    # Remove HoloCleaner reapir methods, if no DCs exist
                    relevant_cleaners.remove(str(RepairMethod.dcHoloCleaner))
                    constraints_path = ""
                else:
                    constraints_path = ""

                # Rearrange list to add cleanWithGroundTruth at the beginning, useful for the cleaning_results.csv
                if str(RepairMethod.cleanWithGroundTruth) in relevant_cleaners:
                    relevant_cleaners.remove(str(RepairMethod.cleanWithGroundTruth))
                    relevant_cleaners.insert(0, str(RepairMethod.cleanWithGroundTruth))

                # Ignore CPClean & boostClean if the ML task of the dataset is not binary classification
                if classification not in dataset_dict['ml_tasks'] or dataset_dict['classes'] != classes_list[0]:
                    logging.info("{} not suitable for boostClean/CPClean: ML task is not binary-classification".format(
                        dataset_dict['name']))
                    if str(RepairMethod.cpClean) in relevant_cleaners:
                        relevant_cleaners.remove(str(RepairMethod.cpClean))
                    if  str(RepairMethod.boostClean) in relevant_cleaners:
                        relevant_cleaners.remove(str(RepairMethod.boostClean))

                logging.info("Relevant repair methods: {}".format(relevant_cleaners))

                # Map the detectors from strings onto functions
                # We can only perform mapping after instantiating a Detectors instance
                functions_list = []
                for item in relevant_cleaners:
                    for function in cleaner.cleaners_list + cleaner.model_oriented_cleaners_list:
                        if item == function.__name__:
                            functions_list.append(function)

                # Obtain a list of outlier detectors
                outlier_detectors = detectors_dictionary[outliers]["detectors_list"]

                # ============== Loop over all relevant cleaners for this dataset =============
                for function in tqdm.tqdm(functions_list):

                    # ============== run cleaner with all configurations =============
                    for config in cleaners_configurations[RepairMethod(function.__name__)]:
                        save_extension = "-" + "-".join([str(x) for x in list(config["configs"].values())] + list(config["kwargs"].values())) \
                            if len(list(config["configs"].values()) + list(config["kwargs"].values()))>0 else ""
                        logging.info("Running Cleaner ----------> ***{}***".format(function.__name__ + save_extension))

                        try:
                            for index in range(iterations):
                                # if cleaner is model-oriented
                                if function in cleaner.model_oriented_cleaners_list:
                                    app, results = function(dirtyDF, detection_dict, config["configs"], **config["kwargs"])
                                    results["model"] = detector_name + "-" + results["model"] + save_extension
                                    app.store_results(results)
                                #elif config['configs']['method'] != 'delete':
                                else:
                                    _, results = function(dirtyDF, detection_dict, config["configs"], **config["kwargs"])
                                    self.__store_cleaning_results(exp_id, results, dataset, detector_name, function.__name__ + save_extension)
                        except Exception as e:
                            logging.info("Exception: {}".format(e.args[0]))
                            logging.info("Exception: {}".format(sys.exc_info()))
                            continue

    def run_experiment(self, datasets, settings):
        """
        This method runs error-detection & cleaners, before testing several ML models
        :param dataset_list: python list contains the dataset to be included in the experiment
        :return:

        :Requirements
            * activate postgresql
            * make sure that DCs, patterns, FDs are defined for all datasets
        """
        # Unpacking all the settings
        det_settings, cleaners_settings, ml_settings = settings
        exp_id = 1  # ID used only in scalability analysis, it is value has no meaning here

        # ========= Error Detection Phase ==========
        if det_settings["run_detectors"]:
            self.run_detectors(exp_id, datasets, det_settings['which_detectors'], iterations=det_settings['iterations'])
            # Obtain the IoU metric between the various detectors for each dataset
            for dataset in datasets:
                self.get_iou(dataset)
        else:
            logging.info("Error detection phase skipped!")

        # ========= Data Repair Phase ==========
        if cleaners_settings['run_cleaners']:
            self.run_cleaners(exp_id, datasets, cleaners_settings['which_cleaners'], iterations=cleaners_settings['iterations'])
        else:
            logging.info("Data repair phase skipped!")

        # ========= ML Modeling Phase ==========
        if ml_settings['run_models']:
            if not ml_settings['which_models']:
                self.run_all_models(datasets, iterations=ml_settings["iterations"], hyperopt=ml_settings['hyperopt'], fastTermination=ml_settings['fastTermination'])
            else:
                for model in ml_settings['which_models']:
                    self.run_models(datasets, model, iterations=ml_settings["iterations"], hyperopt=ml_settings['hyperopt'], fastTermination=ml_settings['fastTermination'])
        else:
            logging.info("ML Modeling phase skipped!")

    def run_scalability_exp(self, datasets, detectors_list, iterations):
        """
        The method uses stratified sampling to generate #of splits of the GT and dirty data before running
         the scalability analysis for the error detection and repair methods
        datasets -- list of strings denoting the examined datasets
        settings -- dictionary of all configurations
        :return:
        """

        for dataset in datasets:
            # Retrieve a dataset dictionary
            dataset_dict = self.__get_dataset_dictionary(dataset)
            db_object = Database()

            # Define paths to the error_rate and actual errors files
            error_rate_path = os.path.abspath(os.path.join(dataset_dict['dataset_path'], 'error_rate.json'))
            actual_errors_path = os.path.abspath(os.path.join(dataset_dict['dataset_path'], 'actual_errors.csv'))

            # Instantiate datasets Object
            curr_dataset = Datasets(dataset_dict)

            # Load dirty & ground truth datasets
            logging.info("Loading **{}** dataset and its ground truth".format(str(dataset)))
            if db_object.db_table_exists_postgresql('dirty_{}'.format(dataset)):
                dirtyDF = db_object.load_df_postgresql('dirty_{}'.format(dataset))
                groundtruthDF = db_object.load_df_postgresql('ground_truth_{}'.format(dataset))
            else:
                dirtyDF = curr_dataset.load_data(dataset_dict["dirty_path"])
                groundtruthDF = curr_dataset.load_data(dataset_dict["groundTruth_path"])
                # Store the dataset in the REIN database
                try:
                    db_object.write_df_postgresql(dirtyDF, 'dirty_{}'.format(dataset))
                    db_object.write_df_postgresql(groundtruthDF, 'ground_truth_{}'.format(dataset))
                except:
                    logging.info("Unable to insert the datasets into the 'rein' database")

            if self.store_postgres:
               db_object.rename_table_postgrsql("dirty_{}".format(dataset), "dirty_{}_temp".format(dataset))
               db_object.rename_table_postgrsql("ground_truth_{}".format(dataset), "ground_truth_{}_temp".format(dataset))
            else:
                # Rename the clean and dirty CSV files to generate new splits
                new_clean_path = os.path.abspath(os.path.join(dataset_dict['dataset_path'],'clean_temp.csv'))
                new_dirty_path = os.path.abspath(os.path.join(dataset_dict['dataset_path'], 'dirty_temp.csv'))
                os.rename(dataset_dict['groundTruth_path'], new_clean_path)
                os.rename(dataset_dict['dirty_path'], new_dirty_path)

            # Get the labels
            labels = []
            # Mute the labels, if no attributes specified in the command
            ml_tasks = dataset_dict['ml_tasks']
            if 'regression' in ml_tasks:
                labels.extend(dataset_dict['labels_reg'])
            if 'classification' in ml_tasks:
                labels.extend(dataset_dict['labels_clf'])

            for percent in np.arange(0.1, 1.1, 0.1):

                # Handle the case of percent = 1
                if percent == 1:
                    # Restore the original datasets
                    if self.store_postgres:
                        db_object.rename_table_postgrsql("dirty_{}_temp".format(dataset), "dirty_{}".format(dataset))
                        db_object.rename_table_postgrsql("ground_truth_{}_temp".format(dataset),
                                                         "ground_truth_{}".format(dataset))
                    else:
                        os.rename(new_clean_path, dataset_dict['groundTruth_path'])
                        os.rename(new_dirty_path, dataset_dict['dirty_path'])
                    self.run_detectors(100, datasets, detectors_list, iterations)
                    exit()
                # Initalize a stratified sampler
                split = StratifiedShuffleSplit(n_splits=1, test_size=percent, random_state=42)

                # ==== Generate data splits ===
                # Sample the GT and dirty data according to the generated indices
                for train_index, test_index in split.split(groundtruthDF, groundtruthDF[labels]):
                    test_gt_set = groundtruthDF.loc[test_index]
                    test_dirty_set = dirtyDF.loc[test_index]

                # Set the path to save the splits
                logging.info('Generating new data splits with percent: {}'.format(percent))
                if self.store_postgres:
                    db_object.write_df_postgresql(test_gt_set, "ground_truth_{}".format(dataset))
                    db_object.write_df_postgresql(test_dirty_set, "dirty_{}".format(dataset))
                else:
                    out_gt_path = os.path.abspath(os.path.join(dataset_dict['dataset_path'], 'clean.csv'))
                    out_dirty_path = os.path.abspath(os.path.join(dataset_dict['dataset_path'] , 'dirty.csv'))
                    test_gt_set.to_csv(out_gt_path, sep=",", index=False, encoding="utf-8")
                    test_dirty_set.to_csv(out_dirty_path, sep=",", index=False, encoding="utf-8")

                try:
                    for detect_method in detectors_list:
                        self.run_detectors(np.abs(percent*100), datasets, [detect_method], iterations=iterations)
                except Exception as e:
                    logging.info("Exception: {}".format(e.args[0]))
                    logging.info("Exception: {}".format(sys.exc_info()))
                    continue

                # Remove the current split and its relevant files
                logging.info('Removing current data splits and their relevant files')
                if self.store_postgres:
                    db_object.remove_table_postgres("dirty_{}".format(dataset))
                    db_object.remove_table_postgres("ground_truth_{}".format(dataset))
                else:
                    os.remove(out_gt_path)
                    os.remove(out_dirty_path)
                os.remove(actual_errors_path)
                os.remove(error_rate_path)
                # Remove detection files
                raha_files_path = os.path.abspath(os.path.join(dataset_dict['dataset_path'], 'raha-baran-results-{}'.format(dataset)))
                shutil.rmtree(raha_files_path, ignore_errors=True)

            # Restore the original datasets
            if self.store_postgres:
                db_object.rename_table_postgrsql("dirty_{}_temp".format(dataset), "dirty_{}".format(dataset))
                db_object.rename_table_postgrsql("ground_truth_{}_temp".format(dataset), "ground_truth_{}".format(dataset))
            else:
                os.rename(new_clean_path, dataset_dict['groundTruth_path'])
                os.rename(new_dirty_path, dataset_dict['dirty_path'])

    def run_robustness_exp(self, datasets, detectors_list, iterations, error_rate_bool):
        """
        The method examines performance of error detection and repair for different error rates
        datasets -- list of strings denoting the examined datasets
        settings -- dictionary of all configurations
        :return:
        """

        for dataset in datasets:
            # Retrieve a dataset dictionary
            dataset_dict = self.__get_dataset_dictionary(dataset)

            # Instantiate datasets Object
            curr_dataset = Datasets(dataset_dict)

            # Define the muted attributes, which should remain clean
            muted_attribs = []
            # Mute the labels, if no attributes specified in the command
            ml_tasks = dataset_dict['ml_tasks']
            if 'regression' in ml_tasks:
                muted_attribs.extend(dataset_dict['labels_reg'])
            if 'classification' in ml_tasks:
                muted_attribs.extend(dataset_dict['labels_clf'])

            # Define paths to the error_rate and actual errors files
            error_rate_path = os.path.abspath(os.path.join(dataset_dict['dataset_path'], 'error_rate.json'))
            actual_errors_path = os.path.abspath(os.path.join(dataset_dict['dataset_path'], 'actual_errors.csv'))

            # Initilize a dataset object
            data_object = Datasets(dataset_dict)
            # Define the range of error rates or outlier degrees
            iter_range = np.arange(0.1,1.1,0.1) if error_rate_bool else np.arange(1,11,1)

            for percent in iter_range:
                # ==== Generate dirty version ===
                # Set the configurations
                configurations = [percent, 3] if error_rate_bool else [0.3, percent]
                # Inject outliers
                logging.info("Generating a dirty version with percent: {}".format(percent))
                if not error_rate_bool:
                    data_object.inject_errors([outliers], configurations, muted_attribs, self.store_postgres)
                else:
                    data_object.inject_errors([outliers, [ErrorType.explicit_mv.func]], configurations, muted_attribs, self.store_postgres)

                # Run the detectors on the generated split
                exp_id = np.abs(percent*100) if error_rate_bool else np.abs(percent*10)
                self.run_detectors(exp_id, datasets, detectors_list, iterations=iterations)

                # Remove the current split and its relevant files
                logging.info('Removing irrelevant files')
                os.remove(actual_errors_path)
                os.remove(error_rate_path)
                # Remove detection files
                raha_files_path = os.path.abspath(os.path.join(dataset_dict['dataset_path'], 'raha-baran-results-{}'.format(dataset)))
                shutil.rmtree(raha_files_path, ignore_errors=True)
                logging.info("=================================================================")

####################################################


####################################################
if __name__ == "__main__":

    # Instantiate a benchmark object
    app = Benchmark()

    # ======================= Define Experimental settings ==============================

    # Define a list of datasets dataset to run experiments with
    datasets_list = [nursery]

    # Select the type of operations to be executed on the datasets defined above
    detection_settings = dict(run_detectors=False, which_detectors=[DetectMethod.min_k], iterations=1)  # run all detectors if empty list is passed
    repair_settings = dict(run_cleaners=False, which_cleaners=[RepairMethod.cleanWithGroundTruth], iterations=1)  # run all cleaners if empty list
    ml_settings = dict(run_models=True, which_models=[MLModel.logit_clf], iterations=1, hyperopt=False, fastTermination=True)  # run all ML models if empty list

    # Group all settings in one list
    all_settings = [detection_settings, repair_settings, ml_settings]

    # ==== Run an experiment =====
    app.run_experiment(datasets_list, all_settings)

    # ==== Scalability experiment =====
    # app.run_scalability_exp(datasets_list, all_settings)

    # Generate beep at the end of the execution
    print('\007')
