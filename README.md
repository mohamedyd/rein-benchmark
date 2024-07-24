REIN: A Comprehensive Benchmark Framework for Data Cleaning Methods in ML Pipelines
==============================

This repository comprises the source codes, scripts, and datasets for our data cleaning benchmark, called REIN. 
REIN is a comprehensive benchmark to thoroughly investigate the impact of data cleaning methods on various ML models.
Through the benchmark, we provide answers to important research questions, e.g., where and whether data cleaning is a 
necessary step in ML pipelines. To this end, REIN examines 38 simple and advanced error detection and repair methods. 
To evaluate these methods, we utilized 14 publicly-available datasets covering different domains together with a wide
collection of ML models. Furthermore, REIN can be easily extended to evaluate new error detection and repair methods. 

<!-- TODO: Add a link to the paper -->
For more details about the benchmark and the data cleaning methods, feel free to download our paper:
 
Please, cite our paper if you use the code.
```
@article{rein2022,
  title={REIN: A Comprehensive Benchmark Framework for Data Cleaning Methods in ML Pipelines},
  author={Abdelaal, Mohamed and Hammacher, Christian and Schoening, Harald},
  journal={Proceedings of the VLDB Endowment (PVLDB)},
  volume={},
  number={},
  pages={},
  year={2023},
  publisher={VLDB Endowment}
}
```

## Setup

Clone with submodules
```shell script
git clone https://github.com/mohamedyd/rein-benchmark.git --recurse-submodules
```

Create a virtual environment and install requirements

```shell script
python3 -m venv venv 
source venv/bin/activate
pip3 install --upgrade setuptools
pip3 install --upgrade pip
pip3 install -e .
```

Install PostgreSQL 

```shell script
# Create the file repository configuration:
sudo sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'

# Import the repository signing key:
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -

# Update the package lists:
sudo apt-get update

# Install the latest version of PostgreSQL.
# If you want a specific version, use 'postgresql-12' or similar instead of 'postgresql':
sudo apt-get -y install postgresql

Start the postgreSQL server
sudo service postgresql start
```

Install the necessary tools
```shell script
cd tools/error-generator 

cd ../Profiler
bash create_db_user.sh
bash setup_linux.sh 
```

#### Install data cleaning tools and their dependencies

Install RAHA and BARAN
```shell script
pip3 install raha
```

Install Picket
```shell script
cd cleaners/Picket
# If GPU is available, replace "cpu" in the command below with "gpu" 
bash install.sh cpu
```
Install C++ files of FAHES
```shell script
cd cleaners/FAHES/src/
make clean && make
```

Set up PostgreSQL for HoloClean
```shell script
cd cleaners/holoclean/
bash create_pd_user_ubuntu.sh
```

Download the knowledge base of KATARA ([link](https://bit.ly/3hPkpWX)) and unzip the file
```shell script
cd cleaners/katara/knowedge-base
```

### Data
You can download more datasets used in the benchmark from this [link](https://www.researchgate.net/publication/382496079_Datasets_of_REIN_Benchmark).

## Usage

Start the Postgresql database server, if it is not already started
```shell script
sudo service postgresql start
```

If postgres will be used to store the various data versions, the `rein` database has to be first created.
```shell script
bash create_db_user.sh
```

### Inject errors into the dataset
This command injects different error types into the dataset. The term `muted_attribs` defines the attributes which should remain clean, e.g., the labels. The term `error_type` enables injecting several error types simultaneously. The current list of errors include: `outliers`, `cell_swapping`, `implicit_mv`, `explicit_mv`, `typo_keyboard`, `gaussian_noise`. If the option `store_postgres` is used, the datasets will be written to the `rein` database. 
```shell script
 python3 scripts/create_dirty.py 
        --dataset_name beers
        -- muted_attribs labels
        -- error_rate 0.2
        -- outlier_degree 3
        -- store_postgres 
        --error_type outliers implicit_mv
```

### Detect errors
The term `detect_method` defines the detection method, where REIN executes all available detection methods, if this argument is not defined in the command. The term `n_iterations` defines how many times the experiment will be repeated. The list of available error detection methods include: `mvdetector`, `raha`, `ed2`, `max_entropy`, `min_k`, `metadata_driven`, `holoclean`, `fahes`, `katara`, `openrefine`, `cleanlab`, `outlierdetector`, `zeroer`, `duplicatesdetector`, `picket`, `nadeef`. The script initially searches for the data, named `dirty_{dataset_name}` and `ground_truth_{dataset_name}`, in the `rein` database. If not found, the script looks for CSV files, named `clean.csv` and `dirty.csv` in the dataset directory. If found, the script will insert them into the `rein` database.   
```shell script
python3 scripts/detect_errors.py 
        --dataset_name nursery 
        --detect_method max_entropy fahes min_k 
        --n_iterations 10
```

### Test robustness of the error detection methods
The following two commands test the robustness of the error detection methods. The robustness is examined in terms of the `error rate` and the `outlier degree`.  For the error rate, the experiment iteratively increases the error rate from 10% to 100, while fixing the outlier degree. Conversely, the outlier degree experiment increases the outlier degree from 1 to 10, while fixing the error rate. If the option `store_postgre` is used, the various generated dirty datasets will be inserted into the `rein` database.
```shell script
python3 scripts/test_robustness.py
        --dataset_name nursery
        --detect_method raha picket ed2 holoclean
        --n_iterations 10
        --run_error_rate
        --store_postgres 

python3 scripts/test_robustness.py
        --dataset_name nursery
        --detect_method raha outlierdetector dboost
        --n_iterations 10
        --run_outlier_degree
        --store_postgres  
```

### Test scalability of the error detection methods
This command runs the error detection methods over different fractions of each dataset, i.e., starting from 10% of the data up to 100%. If the option `store_postgres` is used, the generated datasets will be stored in the `rein` dataset. 
```shell script
python3 scripts/test_scalability.py 
        --dataset_name nursery 
        --detect_method mvdetector 
        --n_iterations 10
        --store_postgres
```

### Repair errors 
This command is used to repair the detected erroneous cells. The ML-agnostic repair methods generate repaired versions of the dirty data. The ML-oriented repair methods jointly
optimize the cleaning and modeling tasks. The list of available repair methods comprises: `baran`, `cleanwithGroundTruth`, `dcHoloCleaner`, `standardImputer`, `mlImputer`, `boostClean`, `cpClean`, `activecleanCleaner`, `duplicatescleaner`, `openrefine`. Like the script of error detection, this script first looks for the data in the `rein` database and if no data stored there, it will search for CSV files in the dataset directory. If the option `store_postgres` is used, the repaired data will be written to the `rein` database.
```shell script
python3 scripts/repair_errors.py 
        --dataset_name nursery 
        --repair_method cleanWithGroundTruth baran 
        --n_iterations 10
        --store_postgres
```

### Train ML models
This command trains ML models on various data versions, i.e., ground truth, dirty, and repaired. Run `python3 scripts/train_model.py -h` to get the list of available classification, regression, and clustering methods. The option `hyperopt` is used to enable hyper-parameters optimization using `Optuna`. The option `early_termination` enables the early termination of the experiment if the obtained results are extremely similar over five iterations. If the option `store_postgres` is used, the repaired data will be fetched from the `rein` database. 
```shell script
python3 scripts/train_model.py
        --dataset_name nursery
        --ml_model forest_clf mlp_clf
        --n_iterations 10
        --hyperopt False
        --early_termination False
        --store_postgres
```

### Lack of ground truth
If the ground truth is not available, you can still use REIN for detecting and repairing the dirty datasets. In this case, the performance is measured in terms of the performance difference between the models trained on repaired data and the models trained on the dirty data version. To run REIN in this mode, add the option `--no_ground_truth` to the above error detection script. However, some detection and repair methods will not work in this mode.  

## Extending the benchmark

### Adding a dataset
REIN takes as an input either the CSV files or PostgreSQL relations stored in the `rein` database. To add a new dataset, follow the steps below:  
* Create a folder named according to the new dataset, e.g., `covid_data`, in the `datasets/` directory. 
* Insert the CSV files of the dirty and the ground truth (if available) into the directory `datasets/covid_data`. Rename the CSV file of the ground truth, if available, to `clean.csv` and the CSV file of the dirty version to `dirty.csv`. Another option is to insert the new dataset directly into PostgreSQL, e.g., `dirty_covid_data` and `ground_truth_covid_data` in the `rein` database. 
* Add metadata about the newly-added dataset to the dictionary of datasets in the `auxiliaries/` directory. The key `error_types` can have values from the list of considered error types, including: `missing_values`, `pattern_violation`, `rule_violation`, `duplicates`, `outliers`, `mislabels`, `typos`, `inconsistency`. The key `excluded_attribs` can be used to exclude any irrelevant columns in the dataset. Finally, the keys `lables_reg` and `labels_clf` define the columns serving as labels in case of regression and classification, respectively.   
```python
datasets_dictionary = 
    {"covid_data": {
        "name": "covid_data",
        "dataset_path": os.path.abspath(os.path.join(datasets_path, "covid_data")),
        "dirty_path": os.path.abspath(os.path.join(datasets_path, covid_data, "dirty.csv")),
        "groundTruth_path": os.path.abspath(os.path.join(datasets_path, covid_data, "clean.csv")),
        "error_types": [missing_values, duplicates],
        "ml_tasks": [regression, classification],
        "excluded_attribs": ["ID"],
        "labels_reg": ["infection_rate"],
        "labels_clf": ["dangerous_region"],
        "notes": "space for adding general notes about the data, e.g., data version, data source, etc."
}}
```

* The following cleaning signals have to be provided for the new dataset:
	- Functional dependency (FD rules) and patterns to run NADEEF
	- Denial constraints (DC rules) for HoloClean
	- Key attributes for the key collision method
	- Blocking functions to efficiently run the ZeroER method
	- Clusters to run the OpenRefine method


### Adding an error detection method
To add a new detection method, follow the steps below:
* Create a new function in `detectors.py` named according to the new error detection method. If you already have the source code of the new error detection method, you can clone it to the `cleaners/` directory. Otherwise, you can implement the algorithm in the new function which should have the following footprint:

```python
    def new_detector(self, dirtydf, dataset, configs):
        """
        This method calls a new error detection algorithm to detect incorrect cells.

        Arguments:
        dirtyDF -- dataframe of shape n_R (# of records) x n_A (# of attributes) - containing a dirty version of a dataset
        dataset (String) -- name of the dataset
        configs (dict) -- any configurations, parameters, or signals, such as labels or clusters, needed to run the new error detection method 
        
        Returns:
        detection_dictionary -- dictionary - keys represent i,j of dirty cells & values are constant string "JUST A DUUMY VALUE"
        evaluation_dict -- dictionary - it holds evaluation information.
        """ 
        
        # Write here either a call to the algorithm or the algorithm itself
        ....

        return detection_dictionary, evaluation_dict 
```

* In `auxiliaries/configurations.py`, add the name of the new error detection method to the `DetectMethod` class.  
* In `auxiliaries/detectors_dictionary.py`, add `DetectMethod.new_detector` to each relevant key, i.e., error type.  


### Adding a data repair method
To add a new detection method, follow the steps below:
* Create a new function in `cleaners.py` named according to the new data repair method. If you already have the source code of the new method, you can clone it to the `cleaners/` directory. Otherwise, you can implement the algorithm in the new function which should have the following footprint:
```python
    def new_repair_method(self, dirtyDF, detections, configs):
        """
        This method repairs detected errors using the new data repair method.

        Arguments:
        dirtyDF -- dataframe of shape n_R (# of records) x n_A (# of attributes) - containing a dirty version of a dataset
        detections -- dictionary - keys represent i,j of dirty cells & values are constant string "JUST A DUUMY VALUE"

        Returns:
        repairedDF -- dataframe of shape n_R x n_A - containing a cleaned version of the dirty dataset 
        results (dict) -- dictionary with results from evaluation
        """

        # Write here either a call to the algorithm or the algorithm itself
        ....

        return repairedDF, results
```
* In `auxiliaries/configurations.py`, add the name of the new data repair method to the `RepairMethod` class.  
* In `auxiliaries/detectors_dictionary.py`, add `RepairMethod.new_repair_method` to each relevant key, i.e., error type.  
* In `auxiliaries/cleaners_configurations.py`, add all configurations of the new data repair method to the class `cleaners_configurations`.  


### Adding an ML model

* In `auxiliaries/models_dictionary.py`, add the ML model information to the models dictionary. In this dictionay, the value of the key `fn` represents the function implementing the new ML model For instance, it can be an imported function from Sci-kit Learn. The key `hyperopt` is used to define the hyperparamter tunning method. The key `fixed_params` is used to define static values for some hyperparameters. Finally, the key `hyperparams` is used to define the hyperparameters which can be adjusted. 
```python
import NewModel
models_dictionary = {
        "classification": {
            "new_model": {
                "name": "new_model",
                "fn": NewModel,
                'hyperopt': Hyperoptimization,
                "fixed_params": {},
                "hyperparams": {
                    "max_depth": [1, 20, 40, 80, 100, 150, 200],
                    "n_estimators": [10, 100, 1000, 1500, 2000]}
        },}}
```  
* In `hyperopt_functions`, implement a function to run Optuna on the new ML model to fine tune the hyperparameters of the new ML model. If hyperparameter tunning is not of interest, you can ignore this step and remove the key `hyperopt` from the above dictionary.
* In `auxiliaries/configurations.py`, add the name of the new ML model to the `MLModel` class.
