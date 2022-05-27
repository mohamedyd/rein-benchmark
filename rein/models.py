####################################################
# Benchmark: A collection of machine learning methods
# Authors: Mohamed Abdelaal
# Date: February 2021
# Software AG
# All Rights Reserved
###################################################

# Import necessary libraries and modules
from rein.auxiliaries.configurations import *
from rein.auxiliaries.models_dictionary import *
from rein.auxiliaries.datasets_dictionary import *
#from auxiliaries.hyperopt_functions import *

from sklearn.pipeline import Pipeline
from mlxtend.evaluate import paired_ttest_5x2cv
from mlxtend.evaluate import permutation_test, cochrans_q, mcnemar_table, mcnemar, combined_ftest_5x2cv
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, RepeatedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error, silhouette_score, r2_score, \
    classification_report, homogeneity_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.inspection import permutation_importance
import imblearn
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.compose import ColumnTransformer


###################################################
class Models:
    """
      Class encapsulates all ML-related methods
      """

    def __init__(self, dataset_name):
        """
          The constructor performs the following tasks:
            - extract the necessary params from datasets dictionary
            - creates a directory to store the models and their results
            - prepare the ground truth dataset to use to for estimating the results in 4 scenarios (S1, S2, S3, S4)
          """

        # Define the seed used for splitting train & test data
        np.random.seed()
        self.split_seed = np.random.randint(1000)

        # Load the name of the dataset
        self.dataset_name = dataset_name

        # Load the list of labels for each ML task
        if "labels_reg" in datasets_dictionary[dataset_name]:
            self.labels_list = datasets_dictionary[dataset_name]["labels_reg"]
        if "labels_clf" in datasets_dictionary[dataset_name]:
            self.labels_list_clf = datasets_dictionary[dataset_name]["labels_clf"]
        if "labels_cls" in datasets_dictionary[dataset_name]:
            self.labels_list_cls = datasets_dictionary[dataset_name]["labels_cls"]
        if "features_clf" in datasets_dictionary[dataset_name]:
            self.features_clf = datasets_dictionary[dataset_name]["features_clf"]
        if 'excluded_attribs' in datasets_dictionary[dataset_name]:
            self.excluded_attribs = datasets_dictionary[dataset_name]['excluded_attribs']

        # Create a directory to cache all the trained models
        self.models_path = self.__get_results_directory()

        # Extract the number of clustere for clustering tasks
        if "n_clusters" in datasets_dictionary[dataset_name]:
            self.n_clusters = datasets_dictionary[dataset_name]["n_clusters"]

    def __get_quality_reg(self, ds_model, gt_model, prepared_data):
        """
          This method evaluates a regressor in four scenarios

          :param
            ds_model -- regression model trained on a dirty/repaired dataset
            gt_model -- regression model trained on ground truth
            prepared_data -- dictionary storing the prepared ground truth and dirty/repaired data
          :return:
            score -- dictionary, RMSE, MAE, R2 values in each scenario
          """

        # Unpack the prepared data
        _, _, gt_test_features, gt_test_labels = prepared_data["ground_truth"]
        _, _, ds_test_features, ds_test_labels = prepared_data["dataset"]
        ignore_s2_s3 = prepared_data["ignore_s2_s3"]

        # prepare the input data, i.e., convert it to numpy ndarray, if necessary
        ds_test_features = ds_test_features.toarray() if not isinstance(ds_test_features, np.ndarray) else ds_test_features
        gt_test_features = gt_test_features.toarray() if not isinstance(gt_test_features, np.ndarray) else gt_test_features

        # Use the trained model to predict the labels in each scenario
        y_predicted = {}
        y_predicted["S1"] = ds_model.predict(ds_test_features)  # dirty-model, test-dirty
        y_predicted["S2"] = ds_model.predict(gt_test_features) if not ignore_s2_s3 else [0]  # dirty-model, test-clean
        y_predicted["S3"] = gt_model.predict(ds_test_features) if not ignore_s2_s3 else [0]  # clean-model, test-dirty
        y_predicted["S4"] = gt_model.predict(gt_test_features)  # clean-model, test-clean

        # Initialize a results dictionary
        score = {}
        model_rmse, model_mae, model_r2 = 0, 0, 0
        for key in y_predicted.keys():
            # Define the prediction
            predictions = y_predicted[key].copy()
            # Define the actual labels
            actual_labels = ds_test_labels.copy() if key in ['S1', 'S3'] else gt_test_labels.copy()

            if len(predictions) == len(actual_labels):
                model_rmse = np.sqrt(mean_squared_error(actual_labels, predictions))
                model_r2 = r2_score(actual_labels, predictions)  # multioutput='uniform_average'
                model_mae = mean_absolute_error(actual_labels, predictions)
                score[key] = [model_rmse, model_mae, model_r2]
            else:
                score[key] = [0, 0, 0]

        return score

    def __get_quality_clf(self, ds_model, gt_model, prepared_data):
        """
          This method evaluates a classifier in four scenarios

          :param
            ds_model -- regression model trained on a dirty/repaired dataset
            gt_model -- regression model trained on ground truth
            prepared_data -- dictionary storing the prepared ground truth and dirty/repaired data
          :return:
            score -- dictionary of the precision, recall, and F1 measure in each scenario
          """

        # Unpack the prepared data
        _, _, gt_test_features, gt_test_labels = prepared_data["ground_truth"]
        _, _, ds_test_features, ds_test_labels = prepared_data["dataset"]
        ignore_s2_s3 = prepared_data["ignore_s2_s3"]

        # prepare the input data, i.e., convert it to numpy ndarray, if necessary
        ds_test_features = ds_test_features.toarray() if not isinstance(ds_test_features, np.ndarray) else ds_test_features
        gt_test_features = gt_test_features.toarray() if not isinstance(gt_test_features, np.ndarray) else gt_test_features

        # Use the trained model to predict the labels in each scenario
        y_predicted = {}
        y_predicted["S1"] = ds_model.predict(ds_test_features)  # dirty-model, test-dirty
        y_predicted["S2"] = ds_model.predict(gt_test_features) if not ignore_s2_s3 else [0]  # dirty-model, test-clean
        y_predicted["S3"] = gt_model.predict(ds_test_features) if not ignore_s2_s3 else [0]  # clean-model, test-dirty
        y_predicted["S4"] = gt_model.predict(gt_test_features)  # clean-model, test-clean

        # Initialize a results dictionary
        score = {}
        model_precision, model_recall, model_f1 = 0, 0, 0
        # For multiclass problems, use micro averaging, otherwise, use binary (default)
        average = "micro" if len(np.unique(ds_test_labels)) > 2 else "binary"
        # ========= estimate the quality metrics in each scenario ============
        for key in y_predicted.keys():
            # Define the prediction
            predictions = y_predicted[key].copy()
            # Define the actual labels
            actual_labels = ds_test_labels.copy() if key in ['S1', 'S3'] else gt_test_labels.copy()
            if len(predictions) == len(actual_labels):
                model_precision = precision_score(actual_labels, predictions, average=average)
                model_recall = recall_score(actual_labels, predictions, average=average)
                model_f1 = f1_score(actual_labels, predictions, average=average)
                score[key] = [model_f1, model_precision, model_recall]
            else:
                score[key] = [0, 0, 0]

        return score

    def __get_quality_cls(self, ds_model, gt_model, prepared_data):
        """
          This method evaluates a clustering method in four scenarios
          :param
            ds_model -- regression model trained on a dirty/repaired dataset
            gt_model -- regression model trained on ground truth
            prepared_data -- dictionary storing the prepared ground truth and dirty/repaired data
          :return:
            score -- dictionary, Silhouette index in each scenario
          """

        # Unpack the prepared data
        gt_features, gt_labels, _, _ = prepared_data["ground_truth"]
        ds_features, ds_labels, _, _ = prepared_data["dataset"]
        ignore_s2_s3 = prepared_data["ignore_s2_s3"]

        # prepare the input data, i.e., convert it to numpy ndarray, if necessary
        ds_features = ds_features.toarray() if not isinstance(ds_features, np.ndarray) else ds_features
        gt_features = gt_features.toarray() if not isinstance(gt_features, np.ndarray) else gt_features

        # Use the trained model to predict the labels in each scenario
        y_predicted = {}
        y_predicted["S1"] = ds_model.fit_predict(ds_features)  # dirty-model, test-dirty
        y_predicted["S2"] = ds_model.fit_predict(gt_features) if not ignore_s2_s3 else [0]  # dirty-model, test-clean
        y_predicted["S3"] = gt_model.fit_predict(ds_features) if not ignore_s2_s3 else [0]  # clean-model, test-dirty
        y_predicted["S4"] = gt_model.fit_predict(gt_features)  # clean-model, test-clean

        # Initialize a dictionary to cache the obtained quality metrics
        score = {}
        model_silhouette, model_db_index, model_ch_index = 0, 0, 0
        # ========= estimate the quality metrics in each scenario ============
        for key in y_predicted.keys():
            # Define the prediction
            predictions = y_predicted[key].copy()
            # Define the actual labels
            features = ds_features.copy() if key in ['S1', 'S3'] else gt_features.copy()
            if len(predictions) == len(features):
                model_silhouette = silhouette_score(features, predictions)
                model_db_index = davies_bouldin_score(features, predictions)
                model_ch_index = calinski_harabasz_score(features, predictions)
                score[key] = [model_silhouette, model_db_index, model_ch_index]
            else:
                score[key] = [0, 0, 0]

        return score

    def __get_results_directory(self):
        """
          This method creates a new directory to cache all models and their results

          :return
            model_directory -- string, path of the "models" directory
        """

        # Define a path to the "models" directory
        models_directory = os.path.abspath(
            # os.path.join(os.path.dirname(__file__), os.pardir, "datasets", self.dataset_name, "results"))
            os.path.join("datasets", self.dataset_name, "results"))

        if not os.path.exists(models_directory):
            # Create a new directory if it does not exit
            os.mkdir(models_directory)

        return models_directory

    def __evaluate(self, estimator_dataset, estimator_gt, prepared_data, ml_task):
        """
        This method tests the trained models

        :param
            estimator_dataset -- ML model trained using input dataset
            estimator_gt -- ML model trained using ground truth
            x_test, y_test -- dataframe, test data
            ml_task -- string, denoting the ML task, e.g., regression, clustering, or classification
            scoring -- boolean, whether to evalute based on the estimator scoring function

        :return
            score -- dictionary of quality metrics
        """

        if ml_task == regression:
            return self.__get_quality_reg(estimator_dataset, estimator_gt, prepared_data)
        elif ml_task == classification:
            return self.__get_quality_clf(estimator_dataset, estimator_gt, prepared_data)
        elif ml_task == clustering:
            return self.__get_quality_cls(estimator_dataset, estimator_gt, prepared_data)

    def __find_best_model(self, model):
        """
        This method performs hyperparameters optimization using the ground truth dataset

        :param
            model -- dictionary, different params of an ML task
        :return
            best_params_ -- dictionary, optimal hyperparameters
            best_estimator_ -- ML model trained with the ground truth dataset & the optimal hyperparameters
        """

        # ================== Extract relevant iformation from the models dictionary ==========

        # Extract the fixed parameters of the ML model
        fixed_params = model["fixed_params"]
        # Extract the hyperparameters and their range of values
        hyperparams = model["hyperparams"]
        # Extract the sklearn function of the ML model
        estimator = model["fn"](**fixed_params)

        # Extract the cleaned train data
        for model_type in models:
            if model["name"] in models_dictionary[model_type]:
                x_train = self.groundtruth[model_type][0]
                y_train = self.groundtruth[model_type][1]

        # ================ Traverse the search space to tune the hyperparameters ========

        # Use grid search to find the optimal hyperparameters
        search = GridSearchCV(estimator=estimator, param_grid=hyperparams,
                              cv=5, scoring="balanced_accuracy", n_jobs=-1, verbose=1)

        # Use the obtained hyperparameters to train a model using the cleaned train data
        if not isinstance(x_train, np.ndarray):
            x_train = x_train.toarray()
        search.fit(x_train, y_train.values.flatten())

        # Print the model parameters
        # pprint(search.best_params_)

        return search.best_params_, search.best_estimator_

    def __separate_features(self, dataset, ml_task):
        """
        This method seperates features and labels
        :param dataset: dataframe contains a dataset
        :param ml_task: string denotes the name of the ML task
        :return:
        """
        # Use the ML task to identify the right features and labels
        if ml_task == regression:
            labels = dataset[self.labels_list].copy()
            features = dataset.drop(self.labels_list, axis=1)
        elif ml_task == classification:
            labels = dataset[self.labels_list_clf].copy()
            if "features_clf" in datasets_dictionary[self.dataset_name]:  # list of features are given
                print("Loading given features ..")
                features = dataset[self.features_clf]
            else:
                features = dataset.drop(self.labels_list_clf, axis=1)  # no features are given
        else: # Clustering
            labels = pd.DataFrame()
            features = dataset.copy()

        return features, labels

    def __features_zero_var(self, df):
        """ A function to remove the features of train dataset with zero variance """
        df_original_var = pd.DataFrame(df.var(axis=0), columns=['Variance'])
        return ((df_original_var[df_original_var.Variance == 0]))

    def __split_train_test(self, data_df, ml_task, rand_seed):
        try:
            return train_test_split(data_df, test_size=0.2, random_state=rand_seed)
        except:
            logging.info("Not enough data samples to split")

    def __separate_categorical(self, features):
        """
        This method separates numerical and categorical attributes
        :param features: dataframe contains features of a dataset
        :return:
        """
        # Select the columns whose data type is object
        features_cat = features.select_dtypes(include=['object']).copy()

        # Extract the numerical features
        if not features_cat.empty:
            features_num = features.select_dtypes(exclude=['object']).copy()
        else:
            features_num = features.copy()  # there are no categorical features
        return features_cat, features_num

    def __smote_oversampling(self, data):
        """This method employes the SMOTE method to resolve the class imbalance problem"""

        # Unpack the input data
        x_train_gt, y_train_gt, x_test_gt, y_test_gt = data["ground_truth"]
        x_train, y_train, x_test, y_test = data["dataset"]

        try:
            # SMOTE: Oversampling minority classes
            smote = SMOTE(k_neighbors=2)
            # Fit predictor and target variable
            x_train_gt, y_train_gt = smote.fit_resample(x_train_gt, y_train_gt)
            x_test_gt, y_test_gt = smote.fit_resample(x_test_gt, y_test_gt)
            x_train, y_train = smote.fit_resample(x_train, y_train)
            x_test, y_test = smote.fit_resample(x_test, y_test)
        except Exception:
            logging.info("SMOTE failed to execute due to sparse classes!")

        # Pack the balanced data
        balanced_data = {}
        balanced_data["ground_truth"] = [x_train_gt, y_train_gt, x_test_gt, y_test_gt]
        balanced_data["dataset"] = [x_train, y_train, x_test, y_test]
        balanced_data["ignore_s2_s3"] = data["ignore_s2_s3"]

        return balanced_data

    def prepare_without_splitting(self, dataset, task):
        """This method prepare a dataset for the purpose of carrying out statistical tests"""
        # Initialize dictionary to pack the prepared data
        prepared_data = {}

        # Extract features and labels from the ground truth and the repaired/dirty data
        features_dataset, labels_dataset = self.__separate_features(dataset, task)

        # Extract numerical and categorical features from the ground truth and the dirty/repaired data
        features_cat, features_num = self.__separate_categorical(features_dataset)

        # ==== Preparing the numerical and categorical attributes
        # Prepare the numerical pipeline
        num_pipeline = Pipeline([('std_scaler', StandardScaler())])

        num_attribs = list(features_num)
        cat_attribs = list(features_cat)

        full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(handle_unknown='ignore'), cat_attribs), ])

        features_prepared = full_pipeline.fit_transform(features_dataset)

        # Prepare the labels, if necessary
        encoder = LabelEncoder()
        if not labels_dataset.empty:
            labels = encoder.fit_transform(labels_dataset.values.flatten())
            labels_dataset = labels.copy()

        return features_prepared, labels_dataset, 0, 0

    def permutation_test(self, dataset1, dataset2):
        """
        This method performs the permutation test to check if two datasets come from the same distribution
        It can be used as a evaluation metric for data repair
        """
        dataset1_flatten = []
        for row in dataset1[0]:
            dataset1_flatten.extend(row)
        dataset2_flatten = []
        for row in dataset2[0]:
            dataset2_flatten.extend(row)
        p_value = permutation_test(dataset1_flatten[0:10000], dataset2_flatten[0:10000], method='approximate', num_rounds=10000, seed=0)
        logging.info("Estimating the p-value: {}".format(p_value))

    def prepare_datasets(self, dataset, ml_task):
        """
        This method prepares a dirty/repaired dataset along with their ground truth
        :param dataset: dataframe contains features and labels of a dirty/repaired dataset
        :param ml_task: string denotes the name of the ML task, e.g., classification, regression, or clustering
        :return:
        """
        # Initialize dictionary to pack the prepared data
        prepared_data = {}
        # Set a predicate to ignore S2 and S3, if the data types in ground truth and the dataset are different
        prepared_data["ignore_s2_s3"] = False

        # Load the ground truth
        ground_truth = pd.read_csv(datasets_dictionary[self.dataset_name]["groundTruth_path"]).dropna()
        # Drop missing values
        dataset = dataset.dropna()

        # Exclude non-useful attributes, if necessary
        try:
            if self.excluded_attribs:
                logging.info("Excluding non-useful attributes: {}".format(self.excluded_attribs))
                gt_only_relevant_attrs = ground_truth.drop(self.excluded_attribs, axis=1)
                dataset_only_relevant_attrs = dataset.drop(self.excluded_attribs, axis=1)
                ground_truth = gt_only_relevant_attrs.copy()
                dataset = dataset_only_relevant_attrs.copy()
        except Exception:
            logging.info("No attributes selected for exclusion!")

        if ml_task == clustering:
            prepared_data['ground_truth'] = self.prepare_without_splitting(ground_truth, ml_task)
            prepared_data['dataset'] = self.prepare_without_splitting(dataset, ml_task)
            return prepared_data

        # Split the dataset horizontally into train (80%) and test data (20%)
        """Test and train sets are separated before encoding categorical attributes to avoid data leakage"""
        split_seed = self.split_seed
        gt_train_set, gt_test_set = self.__split_train_test(ground_truth, ml_task, split_seed)
        ds_train_set, ds_test_set = self.__split_train_test(dataset, ml_task, split_seed)

        # ==== Split train set vertically: Extract features and labels from the ground truth and the repaired/dirty data
        # Train sets
        gt_train_features, gt_train_labels = self.__separate_features(gt_train_set, ml_task)
        ds_train_features, ds_train_labels = self.__separate_features(ds_train_set, ml_task)
        # Test sets
        gt_test_features, gt_test_labels = self.__separate_features(gt_test_set, ml_task)
        ds_test_features, ds_test_labels = self.__separate_features(ds_test_set, ml_task)

        # ==== Extract numerical and categorical features from the ground truth and the dirty/repaired data
        gt_train_cat, gt_train_num = self.__separate_categorical(gt_train_features)
        ds_train_cat, ds_train_num = self.__separate_categorical(ds_train_features)

        # =============== Preparing the numerical and categorical attributes
        # Prepare the numerical pipeline
        num_pipeline = Pipeline([('std_scaler', StandardScaler())])

        # Generate lists of the numerical and categorical attributes
        gt_num_attribs = list(gt_train_num)
        gt_cat_attribs = list(gt_train_cat)
        ds_num_attribs = list(ds_train_num)
        ds_cat_attribs = list(ds_train_cat)

        # Create a processing pipeline to prepare all columns in once step
        # Two pipelines are necessary since # of categorical data may differ in the GT and dirty/repaired versions
        # Pipeline for the ground truth dataset
        gt_full_pipeline = ColumnTransformer([
            ("num", num_pipeline, gt_num_attribs),
            ("cat", OneHotEncoder(handle_unknown='ignore'), gt_cat_attribs),])
        # Pipeline for the dirty/repaired dataset
        ds_full_pipeline = ColumnTransformer([
            ("num", num_pipeline, ds_num_attribs),
            ("cat", OneHotEncoder(handle_unknown='ignore'), ds_cat_attribs),])

        # Transform the train sets
        gt_features_prepared = gt_full_pipeline.fit_transform(gt_train_features)
        ds_features_prepared = ds_full_pipeline.fit_transform(ds_train_features)
        # Transform the test sets
        gt_test_prepared = gt_full_pipeline.transform(gt_test_features)
        ds_test_prepared = ds_full_pipeline.transform(ds_test_features)

        # Ignore scenarios S2 and S3 if the generated features in both datasets differ
        # Note that both prepared data are sparse matrices
        if gt_features_prepared.shape[1] != ds_features_prepared.shape[1]:
            prepared_data['ignore_s2_s3'] = True
            logging.info("S2 and S3 will be ignored: the dataset and its ground truth have different data types")

        try:
            # Prepare the labels, if necessary
            if not gt_train_labels.empty and ml_task == classification:
                encoder = LabelEncoder()
                gt_labels = encoder.fit_transform(gt_train_labels.values.flatten())
                ds_labels = encoder.fit_transform(ds_train_labels.values.flatten())
                gt_t_labels = encoder.fit_transform(gt_test_labels.values.flatten())
                ds_t_labels = encoder.fit_transform(ds_test_labels.values.flatten())
                # Rename lables to conform with other ML tasks
                gt_train_labels = gt_labels.copy()  # Train labels
                ds_train_labels = ds_labels.copy()
                gt_test_labels = gt_t_labels.copy() # Test labels
                ds_test_labels = ds_t_labels.copy()

            # Encapsulate all return parameters in a dictionary
            prepared_data["ground_truth"] = [gt_features_prepared, gt_train_labels, gt_test_prepared, gt_test_labels]
            prepared_data["dataset"] = [ds_features_prepared, ds_train_labels, ds_test_prepared, ds_test_labels]

        except Exception as ex:
            print("Exception: {}".format(ex.args[0]))
            prepared_data["ignore_s2_s3"] = True
            prepared_data["ground_truth"] = self.preprocess(ground_truth, ml_task)
            prepared_data["dataset"] = self.preprocess(dataset, ml_task)

        # For classification and regression, apply SMOTE or Oversampling
        #if ml_task != clustering:
            # Apply SMOTE or Oversampling to resolve the calss imbalance problem
            #prepared_data = self.__smote_oversampling(prepared_data)

        return prepared_data

    def preprocess(self, dataset, ml_task):
        """
          This method prepares a dataset via data separation, scaling, encoding

          :param
            dataset -- the dataset to be preprocessed
            ml_task -- string, denote the ML task, e.g., regression, classification, or clustering
          :return:
            X_train, Y_train -- training data
            X_test, Y_test --  test data
          """
        # Remove NaN rows
        dataset = dataset.dropna()

        # Split the dataset into train (80%) and test data (20%)
        train_set, test_set = self.__split_train_test(dataset, ml_task, self.split_seed)

        # Extract features and labels from the train set
        train_features, train_labels = self.__separate_features(train_set, ml_task)
        test_features, test_labels = self.__separate_features(test_set, ml_task)

        # Drop features with zero varience
        #features = features.drop(columns=self.__features_zero_var(features).index)

        # Split numerical and categorical attributes of the train set
        train_cat, train_num = self.__separate_categorical(train_features)

        # ==== Preparing the numerical and categorical attributes
        # Prepare the numerical pipeline
        num_pipeline = Pipeline([('std_scaler', StandardScaler())])

        num_attribs = list(train_num)
        cat_attribs = list(train_cat)

        full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(handle_unknown='ignore'), cat_attribs),])

        train_features_prepared = full_pipeline.fit_transform(train_features)
        test_features_prepared = full_pipeline.transform(test_features)

         # Prepare the labels, if necessary
        encoder = LabelEncoder()
        if not train_labels.empty and ml_task == classification:
            tr_labels = encoder.fit_transform(train_labels.values.flatten())
            ts_labels = encoder.fit_transform(test_labels.values.flatten())
            train_labels = tr_labels.copy()
            test_labels = ts_labels.copy()

        return train_features_prepared, train_labels, test_features_prepared, test_labels

    def store_results(self, scoring_dictionary, ml_task):
        """
          This method stores the obtained results in an CSV file located at the models directory

          :param
            scoring_dictionary -- a dictionary containing the quality scores
          """

        # Extract the name of the model
        model_name = scoring_dictionary["model"]
        # Delete the key "model" to write only the quality results
        del scoring_dictionary["model"]

        # Create a list of keys in the dictionary
        key_list = list(scoring_dictionary.keys())

        # Get the length of the list in each key, e.g. Precision, Recall, F1
        limit = 0 if not key_list else len(scoring_dictionary[key_list[0]])

        # Create the path of the output file
        file_path = os.path.join(self.models_path, "model_results.csv")

        # ========== Create the header ===============
        metrics_list = []
        header = ['time', 'model']
        # Select the right titles according to the ML task
        if ml_task == 'classification':
            metrics_list = ['f1', 'p', 'r']
        elif ml_task == 'regression':
            metrics_list = ['rmse', 'r2', 'mae']
        else:
            metrics_list = ['silhouette', 'db', 'ch']

        for metric in metrics_list:
            for key in key_list:
                header.append(key + '_' + metric)


        # Check whether the file already exists: useful for writing the header only once
        file_exists = os.path.isfile(file_path)

        # Open an CSV file in append mode. Create a file object for this file
        with open(file_path, 'a') as f_object:
            # Create a file object and prepare it for writing the results
            writefile = csv.writer(f_object)

            if not file_exists:
                writefile.writerow(header)  # file doesn't exist yet, write a header

            # Prepare the row which is to be written to the file
            row = [datetime.now(), model_name, [scoring_dictionary[index][x] for x in range(limit) for index in key_list]]
            # Write the values after flattening the row list obtained in the above line
            writefile.writerow(np.hstack(row))

        # Close the file object
        f_object.close()

    def store_abtesting(self, model_name, data_name, p_value, effect_size):
        """
          This method stores the obtained results in an CSV file located at the models directory

          :param
            scoring_dictionary -- a dictionary containing the quality scores
          """

        # Create a list of keys in the dictionary
        header = ['model', 'data', 'p_value', 'effect_size']

        # Create the path of the output file
        file_path = os.path.join(self.models_path, "abtesting.csv")

        # Check whether the file already exists: useful for writing the header only once
        file_exists = os.path.isfile(file_path)

        # Open an CSV file in append mode. Create a file object for this file
        with open(file_path, 'a') as f_object:
            # Create a file object and prepare it for writing the results
            writefile = csv.writer(f_object)

            if not file_exists:
                writefile.writerow(header)  # file doesn't exist yet, write a header

            # Prepare the row which is to be written to the file
            row = [model_name, data_name, p_value, effect_size]
            # Write the values after flattening the row list obtained in the above line
            writefile.writerow(row)

        # Close the file object
        f_object.close()

    def __5x2cv_paired_t_test(self, estimator1, estimator2, features, labels):
        """This method performs A/B testing between two different ML models (classification or regression)"""

        t, p = paired_ttest_5x2cv(estimator1=clf1, estimator2=clf2, X=features, y=labels, random_seed=1)

        if p < 0.05:
            print("The null hypothesis is rejected: The models are significantly different ..")
        else:
            print("The null hypothesis Cannot be rejected: The models are significantly similar ..")

    def feature_importance(self, prepared_data):
        """This method estimates the feature importance through permutations

        Arguments:
            prepared_data -- dictionary of preprocesed data, including x_train, and y_train
            importance -- list contains the importance scores of all attributes in the datasets
        """
        # Extract the data
        x_train, y_train,_,_ = prepared_data["ground_truth"]
        # Initiate output dictionary
        importance = {}
        # Identify the associated ML tasks
        ml_tasks = datasets_dictionary[self.dataset_name]["ml_tasks"]
        # For each task, employ a corresponding model
        for task in ml_tasks:
            if task == "regression":
                # define the model
                model = Ridge()
                # fit the model
                model.fit(x_train, y_train)
                # perform permutation importance
                results = permutation_importance(model, x_train, y_train,
                                        scoring='neg_mean_squared_error', n_jobs=5, n_repeats=3, random_state=42)
                # get importance
                importance[task] = results.importances_mean
            elif task == "classification":
                # define the model
                model = LogisticRegression()
                # fit the model
                model.fit(x_train, y_train)
                # perform permutation importance
                results = permutation_importance(model, X, y, scoring='accuracy',
                                                 n_jobs=5, n_repeats=3, random_state=42)
                # get importance
                importance[task] = results.importances_mean
            else:
                importance = {}

        return importance

    # evaluate a model
    def evaluate_model(self, X, y, model):
        # define model evaluation method
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        # evaluate model
        scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
        # force scores to be positive
        results = np.absolute(scores)
        print('Mean MAE: %.3f (%.3f)' % (np.mean(results), np.std(results)))

    def train_and_test(self, dataset_name, model_name, prepared_data, optimization=True):
        """
        This method trains a model with/without optimized hyperparameters

        Steps
            - find optimal hyperparameters using ground truth
            - fit the model with optimal params using dirty/repaired data
            - test the obtained model
            - return test results in four scenarios

        :param
            dataset_name -- string, name of the input dataset, e.g., "airbnb"
            model_name -- string, name of the ML model, e.g., "lin_reg", "tree_clf", etc.
            x_train, y_train -- Dataframes, training data
            x_test, y_test -- Dataframes, test data
            optimization -- boolean, whether to tune the hyperparameters

        :return
            results -- dictionary, quality metrics in four scenarios
        """

        # Unpack the prepared data
        x_train_gt, y_train_gt, x_test_gt, y_test_gt = prepared_data["ground_truth"]
        x_train, y_train, _, _ = prepared_data["dataset"]

        # Change the shape of the labels to (n_samples,). Without this line a warning appears with random forest
        y_train_gt = np.ravel(y_train_gt)
        y_train = np.ravel(y_train)


        # Locate the model dictionary and the cleaned train data
        model = {}
        ml_task = ""
        for model_type in models:
            if model_name in models_dictionary[model_type]:
                # Obtain the model dictionary
                ml_task = model_type
                model = models_dictionary[model_type][model_name]

        # ===== Train the models ==========
        # use the ground truth dataset to find the optimal hyperparameters.
        # use those parameters to train a model using the input dataset (dirty/repaired)
        if optimization and models_dictionary[ml_task][model_name]['hyperparams']:
            obj_function = Hyperoptimization(prepared_data["ground_truth"], model_name)
            study = optuna.create_study(direction="maximize")
            study.optimize(obj_function, n_trials=5)

            trial = study.best_trial
            print('Accuracy: {}'.format(trial.value))
            print('Best Hyperparameters: {}'.format(trial.params))

            estimator_gt = model["fn"](**trial.params)
            estimator_gt.fit(x_train_gt, y_train_gt)
            estimator_dataset = model["fn"](**trial.params)
            estimator_dataset.fit(x_train, y_train)

            # Find optimal hyperparams using the ground truth dataset
            #best_hyperparams, estimator_gt = self.__find_best_model(model)

            # Train the model using a dataset (dirty/repaired)
            #estimator_dataset = model["fn"](**best_hyperparams)
            #estimator_dataset.fit(x_train, y_train)
        else:
            logging.info("Training two {} models using the {} and its ground truth".format(model_name, dataset_name))
            # Set the number of clusters, for clustering tasks
            if ml_task == "clustering":
                if model_name not in {"optics_cls", "affinity_cls", "gm_cls"}:
                    model["fixed_params"]["n_clusters"] = self.n_clusters
                elif model_name == "optics_cls":
                    model["fixed_params"]["min_samples"] = 5  # 2 * (x_train.shape[1] - 1)

            # Use the ML model with default parameters (i.e., not optimized)
            estimator_dataset = model["fn"](**model["fixed_params"])
            estimator_gt = model["fn"](**model["fixed_params"])

            # Use the above models to train the input and ground truth data
            estimator_gt.fit(x_train_gt, y_train_gt)  # .values.flatten()
            estimator_dataset.fit(x_train, y_train)

            if ml_task in [regression, classification]:
                # define model evaluation method
                cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
                # evaluate model
                results_gt = cross_val_score(estimator_gt, x_train_gt, y_train_gt, scoring='neg_mean_absolute_error', cv=cv)
                results_ds = cross_val_score(estimator_dataset, x_train, y_train, scoring='neg_mean_absolute_error', cv=cv)
                logging.info('Cross validation for GT: {}'.format(np.mean(np.absolute(results_gt))))
                logging.info('Cross validation for dataset: {}'.format(np.mean(np.absolute(results_ds))))

        # Return the quality metrics in each scenario
        return self.__evaluate(estimator_dataset, estimator_gt, prepared_data, ml_task)

    def run_autoML(self, prepared_data):
        "This method trains autoML meothds"

        # Unpack the prepared data
        x_train_gt, y_train_gt, x_test_gt, y_test_gt = prepared_data["ground_truth"]
        x_train, y_train, x_test, y_test = prepared_data["dataset"]

        # define search
        ml_task = 'classification'
        logging.info("Instantiating autosklearn classifiers")
        model_gt = AutoSklearnClassifier(time_left_for_this_task=180)
        model_dataset = AutoSklearnClassifier(time_left_for_this_task=180)
        # perform the search
        logging.info("Fitting the two autosklearn classifiers using GT and dirty datasets")
        model_gt.fit(x_train_gt, y_train_gt)
        model_dataset.fit(x_train, y_train)
        # summarize
        print(model_gt.sprint_statistics())
        # evaluate best model
        #y_hat = model.predict(x_test_gt)
        #acc = accuracy_score(y_test_gt, y_hat)
        #print("Accuracy: %.3f" % acc)
        return self.__evaluate(model_dataset, model_gt, prepared_data, ml_task)

if __name__ == "__main__":

    # ==================== Setting up the environment ==================

    # Set the name of the dataset to be used for training and testing the ML models
    dataset_name = print3d

    # Select an ML model
    model_name = RF_reg

    # Select whether to tune the hyperparameters
    optimized = False

    # Identify the ML task, i.e., "classification", "regression", or "clustering"
    for model_type in models:
        if model_name in models_dictionary[model_type]:
            ml_task = model_type

    # Instantiate a models object
    app = Models(dataset_name)

    # Load a dataset (dirty, or repaired) to test against several ML models
    dataset = pd.read_csv(datasets_dictionary[dataset_name]["dirty_path"])
    print(datasets_dictionary[dataset_name]["dirty_path"])

    # Prepare the dirty/repaired dataset and its ground truth
    prepared_data = app.prepare_datasets(dataset, ml_task)

    # Train and evaluate the ML model using the input dataset and the ground truth dataset
    results = app.train_and_test("dirty", model_name, prepared_data, optimization=optimized)
    #results = app.run_autoML(prepared_data=prepared_data)

    # Append the name of the model to the results
    results["model"] = model_name + "_opt" if optimized else model_name

    # Store the results in an CSV file located in the models folder
    app.store_results(results, ml_task)
    logging.info("Results have been successfully saved to the models directory")

    # Generate beep at the end of the execution
    print('\007')
