####################################################
# Benchmark: A collection of datasets-related methods
# Authors: Christian Hammacher, Mohamed Abdelaal
# Date: February 2021
# Software AG
# All Rights Reserved
###################################################

##############################
# Importing Necessary Packages
##############################

from rein.auxiliaries.datasets_dictionary import *
from rein.models import Models

import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from functools import reduce
from copy import deepcopy
from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
from math import isnan
import tqdm
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from error_generator import Value_Selector
from error_generator import List_selected
from error_generator import Error_Generator
from error_generator import Read_Write
from sqlalchemy import create_engine
from sqlalchemy_utils.functions import database_exists
import psycopg2
# Create a path to the cleaners directory
#profiler_path = os.path.abspath(reduce(os.path.join,[os.path.dirname(__file__), os.pardir, "tools", "Profiler"]))
# Add the cleaners directory to the system path
sys.path.append(os.path.abspath(os.path.join("tools", "Profiler")))
# Profiler for generating FD rules
from profiler.core import *


##############################
# Matplotlib Settings
##############################
size = 7
params = {
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size*0.75,
          'ytick.labelsize': size*0.75,
          }
plt.rcParams.update(params)


##############################
# Datasets Class Definition
##############################
class Datasets:
    """
    Class encapsulates all datasets-related methods
    """

    def __init__(self, dataset_dictionary):
        """
        The constructor creates a dataset.
        """
        self.name = dataset_dictionary["name"]
        self.dirty_path = dataset_dictionary["dirty_path"]

        # create a path to the dataset directory
        #self.dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "datasets", self.name))
        self.dataset_path = os.path.abspath(os.path.join("datasets", self.name))

        if "groundTruth_path" in dataset_dictionary:
            self.groundTruth_path = dataset_dictionary["groundTruth_path"]
            self.groundTruthDF = pd.read_csv(
                dataset_dictionary["groundTruth_path"], sep=",", encoding="utf-8")
        if "repaired_path" in dataset_dictionary:
            self.has_been_repaired = True
            self.repaired_path = dataset_dictionary["repaired_path"]
            self.repaired_dataframe = pd.read_csv(
                dataset_dictionary["repaired_path"]
            )

    def __store_actual_errors(self, actual_errors, error_rate):
        """
        This method stores the actual errors into a csv file and the error rate into a json file

        :param actual_errors -- dictionay, indices of actual errors in a dataset
        :param error_rate -- float, the number of dirty cells relative to the total number of cells in a dataset
        :return:
        """

        # Create the path of the output files
        csv_path = os.path.join(self.dataset_path, "actual_errors.csv")
        json_path = os.path.join(self.dataset_path, "error_rate.json")

        with open(csv_path, 'w') as f_object:
            # Create a file object and prepare it for writing the results
            writefile = csv.writer(f_object)
            # Prepare the row which is to be written to the file
            for key, value in actual_errors.items():
                row = [key, value]
                # Write the values after flattening the row list obtained in the above line
                writefile.writerow(np.hstack(row))

        # Close the file object
        f_object.close()

        with open(json_path, "w") as handle:
            json.dump(error_rate, handle)

    def get_clus_tendency(self):
        """
        This method estimates the Hopkins statistic to quantify the clustering tendency.

        - Hopkins test tells how much percentage different is our data from random scatter data.
        - we can do Hopkins test before scaling or after the scaling as scaling does not affect the spread of points

        """

        logging.info("Estimating the clustering tendency of the {} dataset via the Hopkins statistic".format(self.name))

        # Exclude categorical data
        #logging.info("Excluding the categorical data")
        dataset = self.groundTruthDF
        app = Models(self.name)
        # Prepare the dirty/repaired dataset and its ground truth
        prepared_data = app.prepare_datasets(dataset, 'clustering')
        dataset, _, _, _ = prepared_data['ground_truth']

        logging.info("Shape of the dataset after exclusing categorical attributes: {}".format(dataset.shape))

        # Identify the dimensions of the dataset
        d = dataset.shape[1]
        n = len(dataset)  # rows
        m = int(0.1 * n)

        # Instantiate a nearst neighbor learner
        nbrs = NearestNeighbors(n_neighbors=1).fit(dataset)

        # Generate a simulated dataset drawn from random uniform distribution
        rand_X = sample(range(0, n, 1), m)

        ujd = []
        wjd = []
        for j in range(0, m):
            # Randomly sample m points from the input dataset, before computing the distance
            # from each point in the sampled data to each nearest neighbors
            u_dist, _ = nbrs.kneighbors(uniform(np.amin(dataset, axis=0), np.amax(dataset, axis=0), d).reshape(1, -1),
                                        2, return_distance=True)
            ujd.append(u_dist[0][1])

            # Calculate the distance from each simulated point to the nearest real (input) sampled data point
            w_dist, _ = nbrs.kneighbors(dataset[rand_X[j]].reshape(1, -1), 2, return_distance=True)
            wjd.append(w_dist[0][1])

        # Estimate the Hopkins metric
        Hopkins = sum(ujd) / (sum(ujd) + sum(wjd))
        if isnan(Hopkins):
            logging.debug("The nearest neighbors distances: {}, {}".format(ujd, wjd))
            return 0
        else: return Hopkins

    def get_n_clusters(self):
        """This method returns the number of clusters for kmeans"""

        # Extract numberical values
        dataset = self.groundTruthDF

        # Instantiate a models object
        app = Models(dataset_name)

        # Prepare the dirty/repaired dataset and its ground truth
        prepared_data = app.prepare_datasets(dataset, 'clustering')
        dataset, _, _, _ = prepared_data['ground_truth']
        # Scaling the numerical attributes
        # scaler = StandardScaler()
        # dataset_scaled = pd.DataFrame(scaler.fit_transform(dataset), columns=dataset.columns)

        logging.info('Finding the optimal value of K using the Silhouette graph')

        sse_ = []
        ssd = []
        centroids_from_kmeans = []
        iterations = []

        for k in tqdm.tqdm(range(2, 15)):
            kmeans = KMeans(n_clusters=k, init='random', n_init=100, max_iter=500, tol=1e-10).fit(dataset)
            sse_.append([k, silhouette_score(dataset, kmeans.labels_)])
            ssd.append([k, kmeans.inertia_])
            centroids_from_kmeans.append(kmeans.cluster_centers_)
            iterations.append([k, kmeans.n_iter_])

        plot = True
        if plot:
            logging.info('Plotting the Silhouette graph')
            plt.plot(pd.DataFrame(sse_)[0], pd.DataFrame(sse_)[1], label="Silhouette Score")
            plt.title("Silhouette Scores for Varying Number of Clusters")
            plt.xlabel("# of clusters")
            plt.ylabel("Silhouette Score")
            plt.show()

    def get_gmm_components(self):
        """This method returns the number of clusters for kmeans"""

        # Extract numberical values
        dataset = self.groundTruthDF

        # Instantiate a models object
        app = Models(self.name)

        # Prepare the dirty/repaired dataset and its ground truth
        prepared_data = app.prepare_datasets(dataset, 'clustering')
        dataset, _, _, _ = prepared_data['ground_truth']

        logging.info('Finding the optimal number of components using the Silhouette graph')

        sse_ = []

        for k in tqdm.tqdm(range(2, 15)):
            kmeans = KMeans(n_clusters=k, init='random', n_init=100, max_iter=500, tol=1e-10).fit(dataset)
            gmm = GaussianMixture(n_components=k).fit_predict(dataset)
            sse_.append([k, silhouette_score(dataset, gmm)])

        plot = True
        if plot:
            logging.info('Plotting the Silhouette graph')
            plt.plot(pd.DataFrame(sse_)[0], pd.DataFrame(sse_)[1], label="Silhouette Score")
            plt.title("Silhouette Scores for Varying Number of Components")
            plt.xlabel("# of components")
            plt.ylabel("Silhouette Score")
            plt.show()

    def load_data(self, dataset_path):
        """
        This method reads a dataset from a csv file path.

        Arguments:
        dataset_path -- string denoting the path of the dataset
        """

        # load data
        dataDF = pd.read_csv(
            dataset_path,
            dtype=str,
            header="infer",
            encoding="utf-8",
            keep_default_na=False,
            low_memory=False,
        )

        return dataDF

    def visualize_dataset(self, data_num):
        """
        This method uses PCA to visualuze the entire dataset.

         It reduces the dimensions of the dataset to only two dimensions, before drawing a scatter plot.
        :argument:
        data_num -- dataframe of the numerical attributes
        :return: None
        """

        # Build the model
        pca = PCA(n_components=2)

        # Reduce the data, output is ndarray
        reduced_data = pca.fit_transform(data_num)

        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=self.groundTruthDF["Price"], cmap='viridis')
        plt.show()

    def plot_versus_label(self, label, feature):
        """
        This method draws a scatter plot for the label versus a feature
        :param label -- string, denoting the label data
        :param feature -- string, denotingt a certain feature
        :return: None
        """
        self.groundTruthDF.plot(kind="scatter", x=label, y=feature, alpha=0.1)
        plt.show()

    def find_correlation(self, label, plot):
        """
        This method estimates the correlation matrix.

        It plots the heatmap and the scatter matrix for the highly-correlated attributes with the label

        :param label -- string, denoting the label data
        :param plot -- boolean, whether to plot the figures
        :return: None
        """

        dataset = self.groundTruthDF.copy()
        # check if the label is categorical and has only two values
        if dataset[label].dtypes == "object":
            dataset[label] = LabelEncoder().fit_transform(dataset[label])

        # compute the correlation matrix
        corrmat = dataset.corr()
        print(corrmat.shape)
        logging.info("Printing the correlation matrix sorted according to the label..")
        print(corrmat[str(label)].sort_values(ascending=False))
        print("-----------------------------------------------\n")
        if plot:
            f, ax = plt.subplots(figsize=(12,9))
            sns.heatmap(corrmat, vmax=.8, square=True, cmap=sns.diverging_palette(220, 10, as_cmap=True))
            corr_fig = os.path.join(self.dataset_path, "correlation.png")
            #plt.savefig(corr_fig, format = "png", bbox_inches='tight',
            #            pad_inches=0.5, orientation='portrait' )

            # plot pair relationships if the number of attributes is relatively reasonable, e.g., 10
            if len(dataset.columns) < 10:
                sns.pairplot(dataset.loc[:, dataset.dtypes == 'float64'])

            # plot the highly-correlated attributes to the label, e.g. threshold = 0.5
            # initialize a list to add the highly-correlated attributes with the label
            attributes = []
            # loop over all attributes to pick up the one which has high correlation with the label
            for att in corrmat:
                if corrmat[label][att] > 0.1:
                    attributes.append(att)
            # draw a scatter matrix plot
            scatter_matrix(dataset[attributes], figsize=(12, 8))
            plt.show()

    def explore_dataset(self, plot):
        """
        This method provides info about the dataset, i.e. stats, description of the columns, and categorical data

        The correlation figure is saved at the dataset directory
        :argument
        plot -- boolean, defines whether to plot figures or not
        :return: None
        """

        # Get the labels
        labels_exist = False
        if 'labels_clf' in datasets_dictionary[self.name].keys():
            labels = self.groundTruthDF[datasets_dictionary[self.name]["labels_clf"][0]]
            labels_exist = True
        elif 'labels_reg' in datasets_dictionary[self.name]:
            labels = self.groundTruthDF[datasets_dictionary[self.name]["labels_reg"][0]]
            labels_exist = True

        # Get information about the dataset
        print(self.groundTruthDF.info())
        # get a description of the dataset
        logging.info("Getting a description of the dataset..")
        description = self.groundTruthDF.describe()
        # storing the dataset description in an CSV file
        file_path = os.path.join(self.dataset_path, "data_description.csv")
        outfile = open(file_path, 'w')
        description.to_csv(outfile, index = True, header = True, sep = ',', encoding = 'utf-8')

        # Print value counts of the labels
        if labels_exist:
            labels.value_counts().to_csv(outfile, index = True, header = True, sep = ',', encoding = 'utf-8')

        outfile.close()

        # printing the dataset description to the console
        print(description)
        print("-----------------------------------------------\n")
        logging.info("The description of the dataset has been successfully stored in an CSV file.")

        # get stats of the dataset
        # look at "Non-Null Count" to detect whether there are (explicit) missing values
        logging.info("Getting more information about the dataset")
        print(self.groundTruthDF.info())
        print("-----------------------------------------------\n")

        # =========== Separate the features into categorical and numerical data =====

        # select the columns whose data type is object
        data_cat = self.groundTruthDF.select_dtypes("object")

        # extract the numerical features
        if not data_cat.empty:
            data_num = self.groundTruthDF.select_dtypes("number")
        else:
            data_num = self.groundTruthDF.copy()  # there are no categorical features


        # list the different options in each categorical data
        for cat in data_cat:
            print("The attribute \"{}\" has the following options: {}".format(cat, self.groundTruthDF[cat].unique()))
            print("-----------------------------------------------\n")

        # draw the histogram of each attribute in the dataset
        if plot:
            self.groundTruthDF.hist(bins=50, figsize=(20,15))
            plt.show()

    def get_dataframes_difference(self, dataframe_1, dataframe_2):
        """
        This method compares two dataframes and returns the different cells.
        """
        if dataframe_1.shape != dataframe_2.shape:
            sys.stderr.write("Two compared datasets do not have equal sizes!\n")
        difference_dictionary = {}
        difference_dataframe = dataframe_1.where(dataframe_1.values != dataframe_2.values).notna()
        for j in range(dataframe_1.shape[1]):
            for i in difference_dataframe.index[difference_dataframe.iloc[:, j]].tolist():
                difference_dictionary[(i, j)] = dataframe_2.iloc[i, j]
        return difference_dictionary

    def get_actual_errors(self, dirtyDF, groundTruthDF):
        """
        This method estimates the actual errors in a dataset and the error rate

        Arguments:
        dirtyDF (dataframe) -- dirty dataset
        groundTruthDF (dataframe) -- ground truth of the dataset

        Returns:
        actual_errors_dictionary (dictionary) -- keys represent i,j of dirty cells & values are constant string "DUUMY VALUE"
        error_reate -- error rate in dirtDF compared to groundtruthDF
        """

        # Create dictionary for the output
        actual_errors_dictionary = {}

        for col in dirtyDF.columns:
            # Get the location of the next column
           col_j = dirtyDF.columns.get_loc(col)

           for i, row in dirtyDF.iterrows():

                try:
                    if int(float(dirtyDF.iat[i, col_j])) != int(float(groundTruthDF.iat[i, col_j])):
                        actual_errors_dictionary[(i, col_j)] = "DUMMY VALUE"
                except ValueError:
                    if dirtyDF.iat[i, col_j] != groundTruthDF.iat[i, col_j]:
                        actual_errors_dictionary[(i, col_j)] = "DUMMY VALUE"

        #actual_errors_dictionary = self.get_dataframes_difference(dirtyDF, groundTruthDF)
        error_rate = len(actual_errors_dictionary) / groundTruthDF.size

        # Store the error rate and actual errors for later use
        self.__store_actual_errors(actual_errors_dictionary, error_rate)

        return actual_errors_dictionary, error_rate

    def drop_duplicates(self):
        """This method drops duplicates from a dataset """
        dataset = pd.read_csv(self.dirty_path)
        dataset.drop_duplicates(keep=False, inplace=True)
        path = os.path.join(self.dataset_path, "deduplicated_dataset.csv")
        dataset.to_csv(path)

        return

    def generate_fd_rules(self):
        """This method uses the Profiler package to generate a set of functional dependancy (FD) rules"""

        # Instantiate an engine
        """
        * workers : number of processes
        * tol : tolerance for differences when creating training data (set to 0 if data is completely clean)
        * eps : error bound for inverse covariance estimation (since we use conservative calculation when determining 
                minimum sample size, we recommend to set eps <= 0.01)
        * embedtxt: if set to true, differentiate b/w textual data and categorical data, and use word embedding 
                for the former 
        """
        pf = Profiler(workers=2, tol=0, eps=0.05, embedtxt=True)

        # Load data
        pf.session.load_data(name='soccer', src=DF, df=self.groundTruthDF, check_param=True, na_values='empty')

        # Change data types of attributes
        """ 
        * required input: a list of attributes, a list of data types (must match the order of the attributes; 
                can be CATEGORICAL, NUMERIC, TEXT, DATE)
        * optional input: a list of regular expression extractor
        """
        # pf.session.change_dtypes(
        #    ['ProviderNumber', 'ZipCode', 'PhoneNumber', 'State', 'EmergencyService', 'Score', 'Sample', 'HospitalType',
        #     'HospitalOwner', 'Condition'],
        #    [CATEGORICAL, NUMERIC, CATEGORICAL, TEXT, TEXT, NUMERIC, NUMERIC, TEXT, TEXT, TEXT],
        #    [None, None, None, None, None, r'(\d+)%', r'(\d+)\spatients', None, None, None])

        # Load/Train Embeddings for TEXT
        """
        * path: path to saved/to-save embedding folder
        * load: set to true -- load saved vec from 'path'; set to false -- train locally
        * save: (only for load = False) save trained vectors to 'path'
        """
        embeddings_path = os.path.abspath(os.path.join(self.dataset_path, "embeddings"))
        # Check if the embeddings already exist
        store_load = [False, True] if os.path.exists(embeddings_path) else [True, False]
        pf.session.load_embedding(save=store_load[0], path=embeddings_path, load=store_load[1])

        # Load training data
        pf.session.load_training_data(multiplier=None, difference=True)

        # Learn structure
        # set sparsity to 0 for exp_reproduce
        autoregress_matrix = pf.session.learn_structure(sparsity=0, infer_order=True)

        # score:
        #      * "fit_error": mse for fitting y = B'X + c for each atttribute y
        #      * "training_data_fd_vio_ratio": the higher the score, the more violations of FDs in the training
        #         data. (bounded: [0,1])
        parent_sets = pf.session.get_dependencies(score="fit_error")

    def get_actual_errors_old(self, dirtyDF, groundTruthDF):
        """
        This method estimates the actual errors in a dataset and the error rate

        Arguments:
        dirtyDF (dataframe) -- dirty dataset
        groundTruthDF (dataframe) -- ground truth of the dataset

        Returns:
        actual_errors_dictionary (dictionary) -- keys represent i,j of dirty cells & values are constant string "DUUMY VALUE"
        error_reate -- error rate in dirtDF compared to groundtruthDF
        """

        # Create dictionary for the output
        actual_errors_dictionary = {}

        for col in dirtyDF.columns:
            # Get the location of the next column
           col_j = dirtyDF.columns.get_loc(col)

           for i, row in dirtyDF.iterrows():

                if dirtyDF.iat[i, col_j] != groundTruthDF.iat[i, col_j]:
                    actual_errors_dictionary[(i, col_j)] = "DUMMY VALUE"

        #actual_errors_dictionary = self.get_dataframes_difference(dirtyDF, groundTruthDF)
        error_rate = len(actual_errors_dictionary) / groundTruthDF.size

        # Store the error rate and actual errors for later use
        self.__store_actual_errors(actual_errors_dictionary, error_rate)

        return actual_errors_dictionary, error_rate

    def __inject_outlier(self, df, label, outlier_rate=0.1, multiplier=3):
        """injects errors in outlier_rate cases only in num columns
        injected value ist previous value + -/+1 * log(1+rand(0,1)*3) therefore
        the minimum outlier is always ol_thresholds stds away from current value but maybe further.
        """

        # Extract the name of the numerical columns
        num_cols = df.select_dtypes("number").columns
        # Exclcude the labels
        num_cols = [col for col in num_cols if col != label]

        # Create a mask of zeros
        mask = np.zeros(df[num_cols].size)
        mask[:int(outlier_rate * df[num_cols].size)] = 1
        np.random.shuffle(mask)
        mask = mask.reshape((df[num_cols].shape[0], df[num_cols].shape[1]))

        for i, col in enumerate(num_cols):
            std = np.std(df[col])
            df[col] = df[col] + mask.T[i] * np.random.normal(loc=0.0, scale=std * multiplier, size=df.shape[0])

        return df

    def __generate_errors(self, dataset, error_types, percent, muted_columns):
        """
        This method is used to inject various type of errors, including
        - typos based on keyboards
            + Duplicate the character
            + Delete the character
            + Shift the character one keyboard space

        - typos base on butter-fingers
            + A python library to generate highly realistic typos (fuzz-testing)

        - explicit missing value
            + randomly one value will be removed

        - implicit missing value
            + one of median or mode of the active domain, randomly pick and
                replace with the selected value

        - Random Active domain
            + randomly one value from active domain will be replaced with the selected value

        - Similar based Active domain
            + the most similar value from the active domain is picked and replaced with the selected value

        - White noise (min=0, var=1)
            + white noise added to the selected value(for string value the noise add to asci code of them)

        - gaussian noise:
            + the Gaussian noise added to the selected value (for string value the noise add to asci code of them). in this method, you can specify the noise rate as well.
        """

        dataset_dataframe = dataset
        dataset_dataframe = dataset_dataframe.apply(lambda x: x.str.strip())
        dataset = [list(dataset_dataframe.columns.to_numpy())] + list(dataset_dataframe.to_numpy())
        # Obtain a copy of the original dataset
        dirty_dataset = deepcopy(dataset)

        # Choose a selector, either list_selected or value_selected
        selector = List_selected()
        # Initialize an error generator
        error_generator = Error_Generator()

        # Loop over all error types
        for error_type in error_types:
            # Choose a strategy
            strategy = error_type
            print(strategy)
            # Inject errors
            # Percentage : The amount of errors to be injected
            # Mute column: Columns that should be safe, i.e. away from the error generator proccess
            dirty_dataset = error_generator.error_generator(method_gen=strategy, selector=selector, percentage=percent*100,
                                            dataset=dirty_dataset, mute_column=muted_columns)

        return dirty_dataset

    def __inject_mv(self, dataset, mv_prob, mv_type='MCAR'):
        """This method inject a number of nulls in the dataseto

        ** This method removes the label attributes from the output dataset **

        :argument
        mv_prob -- float, the probability of missing for a cell
        mv_type -- string, type of missing (MCAR or MAR)
        dirty_data -- dictionary, dirty data for different ML tasks
        """
        # Instatntiate a dictionary for the output dirty data
        dirty_data = {}
        # Instansiate a model instance to prepare the dataset and obtain the feature importance
        model = Models(self.name)
        # Identify the ML task
        ml_task = datasets_dictionary[self.name]["ml_tasks"]

        for task in ml_task:
            # Get a veriosn of the train data
            labels = "labels_reg" if task == "regression" else "labels_clf"
            if task != 'clustering':
                x_train = dataset.drop(datasets_dictionary[self.name][labels], axis=1)
                x_train_mv = deepcopy(x_train)
            else:
                x_train_mv = deepcopy(dataset)

            # Get missing prob matrix
            if mv_type == "MCAR":
                m_prob_matrix = np.ones(x_train_mv.shape) * mv_prob
            elif mv_type == "MAR":
                feature_importance = model.feature_importance(model.prepare_datasets(dataset,task))
                probs = np.array([feature_importance[task][c] for c in x_train.columns])
                probs[probs < 0] = 0
                probs += 1e-100
                probs = probs / np.sum(probs)
                col_missing_prob = probs * mv_prob * x_train.shape[1]
                col_missing_prob[col_missing_prob > 0.95] = 0.95
                m_prob_matrix = np.tile(col_missing_prob, (x_train_mv.shape[0], 1))

            # Create a mask matrix for the cells which will be changed
            mask = np.random.rand(*x_train_mv.shape) <= m_prob_matrix

            # Avoid injecting in all columns for one row
            for i in range(len(mask)):
                if mask[i].all():
                    non_mv = int(mask.shape[1] * (1 - mv_prob))
                    non_mv_indices = np.random.choice(mask.shape[1], size=non_mv, replace=False)
                    mask[i, non_mv_indices] = False

            #Inject missing values
            x_train_mv[mask] = np.nan
            #ind_mv = pd.DataFrame(mask, columns=x_train_mv.columns)

        return pd.DataFrame(x_train_mv)

    def inject_errors(self, error_types, configs, muted_columns, store_postgres=False):
        """
        This method injects different types of errors in a dataset

        Arguments:
             error_types -- list comprising the error types to be injected
             configs -- list encapsulating the different configurations for each error type
             muted_columns -- list of strings denoting the columns which should remain clean (not injected)
             store_postgres -- bool to decide whether to write the data to the database

        Returns:
            dirtyDF -- dataframe of the dirty dataset saved to the dataset directory
        """
        # Read the ground truth
        df = self.groundTruthDF.copy()

        # Extract the configurations
        all_errors_percent, outlier_multiplier = configs

        #assert input("Is it OK to inject errors into the *{}* dataset [Y/N]? "
        #             .format(self.name)).lower() == 'y'

        # Drop muted columns, which must not change, e.g., labels
        if muted_columns:
            df = df.drop(muted_columns, axis=1)

        # Extimate rate of each error type
        number_error_types = len(error_types)
        error_rate = all_errors_percent / number_error_types

        # Initialize the dirty dataframe
        dirtyDF = df.copy()

        for error_type in error_types:
            if error_type == outliers:
                dirtyDF = self.__inject_outlier(dirtyDF, muted_columns, error_rate, outlier_multiplier)
            elif error_type == missing_values:
                dirtyDF = self.__inject_mv(dirtyDF, error_rate)
            else:
                dirtyDF = self.__generate_errors(dirtyDF.astype(str), error_type, error_rate, muted_columns)
                dirtyDF = pd.DataFrame(dirtyDF[1:],columns=dirtyDF[0])

        # Restore the muted columns
        if muted_columns:
            muted_data = self.groundTruthDF[muted_columns]
            dirtyDF = dirtyDF.join(muted_data)

        # Save the output
        if store_postgres:
            db_object = Database()
            db_object.write_df_postgresql(dirtyDF, 'dirty_{}'.format(self.name))
            db_object.write_df_postgresql(self.groundTruthDF, 'ground_truth_{}'.format(self.name))
        else:
            output_path = os.path.abspath(os.path.join(self.dataset_path, "dirty.csv"))
            dirtyDF.to_csv(output_path, sep=",", index=False, encoding="utf-8")


class Database:
    """This class comrprises several methods for dealing with REIN database"""

    def __init__(self):
        """ Default constructor """
        # Create an engine
        self.engine = create_engine('postgresql://reinuser:abcd1234@localhost:5432/rein')
        self.database_info = "dbname=rein user=reinuser password=abcd1234"

    def write_df_postgresql(self, data_df, table_name):
        """
        This method is used to store a dataset in a postgresql database
            data_df -- dataframe of the table to be inserted in the postgreSQL database
            table_name -- string denoting the name of the table
        """
        # Connect to the PostgreSQL database
        data_df.to_sql(table_name, self.engine, if_exists='replace')

    def load_df_postgresql(self, table_name):
        """
            This method loads a table from a postgreSQL database and returns a dataframe
        """
        # Return a dataframe of the loaded table
        return pd.read_sql_query('select * from "{}"'.format(table_name), con=self.engine)

    def db_table_exists_postgresql(self, table_name):
        """
        This method checks if a table already exists in the PostgreSQL database
        :param table_name: string denoting the name of the table
        :return: bool denoting whether the table exists in the database
        """
        # Connect to the PostgreSQL database
        conn = psycopg2.connect(self.database_info)
        # Define a SQL query
        sql = f"select * from information_schema.tables where table_name='{table_name}'"

        # return results of sql query from conn as a pandas dataframe
        results_df = pd.read_sql_query(sql, conn)

        # Close the connection
        conn.close()

        # True if we got any results back, False if we didn't
        return bool(len(results_df))

    def db_exists_postgresql(self):
        """
        This method checks whether REIN database exists in PostgreSQL
        :return: bool denting whether the database exists
        """
        return database_exists(self.engine.url)

    def get_table_names_postgresql(self):
        """This method retrieves a list of table names exist in REIN database"""
        conn = psycopg2.connect(self.database_info)
        conn_cursor = conn.cursor()
        # SQL query
        sql_query = "SELECT table_name FROM information_schema.tables" \
                    " WHERE table_schema='public' AND table_type='BASE TABLE';"
        # Execute the SQL query
        conn_cursor.execute(sql_query)
        # Read the list of tables
        list_tables = conn_cursor.fetchall()

        # Flatten the list of table names
        table_names_flatten = []
        for table_name in list_tables:
            # Max name length in Postgresql is 63
            # Repaired tables will have name of length greater than 60
            # Other tables can be skipped
            # len(table_name[0]) > 60 and table_names_flatten.append(table_name[0])
            table_names_flatten.append(table_name[0])

        # Close the connection
        conn_cursor.close()
        conn.close()

        return table_names_flatten

    def rename_table_postgrsql(self, old_name, new_name):
        """This method renames a relation in the rein database"""

        if self.db_table_exists_postgresql(old_name):
            # SQL query
            sql_query = "ALTER TABLE {} " \
                        "RENAME TO {};".format(old_name, new_name)
            # Execute the SQL query
            self.engine.execute(sql_query)

    def remove_table_postgres(self, table_name):
        """This method removes a table from the rein database"""
        sql_query = "DROP TABLE IF EXISTS \"{}\";".format(table_name)
        self.engine.execute(sql_query)

###################################################
if __name__ == "__main__":
    dataset_name = nursery
    dataset_dict = {
        "name": dataset_name,
        "dirty_path": os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "dirty.csv")),
        "groundTruth_path": os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir, "datasets", dataset_name, "clean.csv"))
    }
    logging.info('Loading the dataset and instantiating a Dataset object..')
    app = Datasets(dataset_dict)

    # ======= Error injection ==========
    # Set the configurations of all error types: error rate, outlier multiplier
    configurations = [0.6, 4]
    # Define the error types to be injected, e.g., missing values, outliers, [implicit_mv, swapping]
    error_list = [[ErrorType.typos.func]]
    # Define the columns which should not be injected
    muted_columns = ['class']
    # Inject the errors
    #app.inject_errors(error_list, configurations, muted_columns)

    app2 = Database()
    clean_df = app.load_data(dataset_dict['groundTruth_path'])
    dirty_df = app.load_data(dataset_dict['dirty_path'])
    app2.remove_table_postgres("k/datasets/nursery/fahes_ALL/standardImputerdelete/repaired.cs")


    #print(app.load_df_postgresql('dirty').head())

    # ====== Clustering ============
    # print("Hopkins metric: {}".format(app.get_clus_tendency()))
    # app.get_gmm_components()
    # app.get_n_clusters()

    # ========== EDA analysis =======
    #app.explore_dataset(plot=False)
    # Find correlation with a target label
    #app.find_correlation(label="roughness", plot=True)
    # Plot a feature vs. the label
    #app.plot_versus_label(label="Price", feature="Bedrooms")

    # ======= FD rules: Profiler tool =======
    # app.generate_fd_rules()

    # ===== Drop Duplicates =====
    #app.drop_duplicates()

