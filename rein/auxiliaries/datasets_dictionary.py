####################################################
# Benchmark: configuration dictionaries 
# Authors: Christian Hammacher, Mohamed Abdelaal
# Date: February 2021
# Software AG
# All Rights Reserved
####################################################

from rein.auxiliaries.configurations import *

####################################################


# Define the path to the datasets directory
#datasets_path = os.path.abspath(
#    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.pardir, "datasets"))
datasets_path = 'datasets'

datasets_dictionary = {
    print3d: {
        'name': print3d,
        "dataset_path": os.path.abspath(os.path.join(datasets_path, print3d)),
        "dirty_path": os.path.abspath(os.path.join(datasets_path, print3d, "dirty.csv")),
        "groundTruth_path": os.path.abspath(os.path.join(datasets_path, print3d, "clean.csv")),
        "error_types": [missing_values, duplicates],
        'dboost_configs': [],
        "ml_tasks": [regression],
        "excluded_attribs": [],
        "labels_reg": ["tension_strenght"],
        'notes': "actual errors generated manually"
    },
    soccer: {
        'name': soccer,
        "dataset_path": os.path.abspath(os.path.join(datasets_path, soccer)),
        "dirty_path": os.path.abspath(os.path.join(datasets_path, soccer, "dirty.csv")),
        "groundTruth_path": os.path.abspath(os.path.join(datasets_path, soccer, "clean.csv")),
        "error_types": [missing_values, outliers, rule_violation, typos],
        'dboost_configs': [],
        "ml_tasks": [regression],
        "excluded_attribs": [],
        "labels_reg": ["overall_rating"],
    },
    power: {
        'name': power,
        "dataset_path": os.path.abspath(os.path.join(datasets_path, power)),
        "dirty_path": os.path.abspath(os.path.join(datasets_path, power, "dirty.csv")),
        "groundTruth_path": os.path.abspath(os.path.join(datasets_path, power, "clean.csv")),
        "error_types": [outliers],
        #"error_types": [missing_values, typos],
        'dboost_configs': [],
        "ml_tasks": [clustering],
        "n_clusters": 2
    },
    water: {
        'name': water,
        "dataset_path": os.path.abspath(os.path.join(datasets_path, water)),
        "dirty_path": os.path.abspath(os.path.join(datasets_path, water, "dirty.csv")),
        "groundTruth_path": os.path.abspath(os.path.join(datasets_path, water, "clean.csv")),
        "error_types": [outliers, missing_values],
        'dboost_configs': [],
        "ml_tasks": [clustering],
        "excluded_attribs": ['index', 'gt'],
        "n_clusters": 4
    },
    har: {
        'name': har,
        "dataset_path": os.path.abspath(os.path.join(datasets_path, har)),
        "dirty_path": os.path.abspath(os.path.join(datasets_path, har, "dirty.csv")),
        "groundTruth_path": os.path.abspath(os.path.join(datasets_path, har, "clean.csv")),
        "error_types": [outliers, missing_values],
        'dboost_configs': [],
        "ml_tasks": [clustering],
        "excluded_attribs": ['Index'],
        "n_clusters": 2,
        'labels_cls': ['gt']
    },
    soilmoisture: {
        'name': soilmoisture,
        "dataset_path": os.path.abspath(os.path.join(datasets_path, soilmoisture)),
        "dirty_path": os.path.abspath(os.path.join(datasets_path, soilmoisture, "dirty.csv")),
        "groundTruth_path": os.path.abspath(os.path.join(datasets_path, soilmoisture, "clean.csv")),
        "error_types": [outliers, missing_values],
        "labels_reg": ["soil_moisture"],
        "ml_tasks": [regression],
        "excluded_attribs": []
    },
    citation: {
        'name': citation,
        "dataset_path": os.path.abspath(os.path.join(datasets_path, citation)),
        "dirty_path": os.path.abspath(os.path.join(datasets_path, citation, "dirty.csv")),
        "groundTruth_path": os.path.abspath(os.path.join(datasets_path, citation, "clean.csv")),
        "error_types": [duplicates, mislabels],
        "keys": ['CS', 'title'],
        "labels_clf": ["CS"],
        'dboost_configs': [],
        "ml_tasks": [classification],
        'classes': classes_list[0],
        "notes": "Actual errors detected manually, some duplicates have also mislabels"
    },
    bike: {
        'name': bike,
        "dataset_path": os.path.abspath(os.path.join(datasets_path, bike)),
        "dirty_path": os.path.abspath(os.path.join(datasets_path, bike, "dirty.csv")),
        "groundTruth_path": os.path.abspath(os.path.join(datasets_path, bike, "clean.csv")),
        "error_types": [outliers, rule_violation, pattern_violation],
        "labels_reg": ["cnt"],
        'dboost_configs': [],
        "ml_tasks": [regression],
        'excluded_attribs': []
    },
    nasa: {
        'name': nasa,
        "dataset_path": os.path.abspath(os.path.join(datasets_path, nasa)),
        "dirty_path": os.path.abspath(os.path.join(datasets_path, nasa, "dirty.csv")),
        "groundTruth_path": os.path.abspath(os.path.join(datasets_path, nasa, "clean.csv")),
        "error_types": [outliers, missing_values],
        "labels_reg": ["sound_pressure_level"],
        "ml_tasks": [regression],
        'excluded_attribs': []
    },
    smartfactory: {
        'name': smartfactory,
        "dataset_path": os.path.abspath(os.path.join(datasets_path, smartfactory)),
        "dirty_path": os.path.abspath(os.path.join(datasets_path, smartfactory, "dirty.csv")),
        "groundTruth_path": os.path.abspath(os.path.join(datasets_path, smartfactory, "clean.csv")),
        #"error_types": [outliers, missing_values],
        "error_types": [outliers],
        "labels_clf": ["labels"],
        "ml_tasks": [classification],
        'classes': classes_list[0],
        'excluded_attribs': []
    },
    mercedes: {
        'name': mercedes,
        "dataset_path": os.path.abspath(os.path.join(datasets_path, mercedes)),
        "dirty_path": os.path.abspath(os.path.join(datasets_path, mercedes, "dirty.csv")),
        "groundTruth_path": os.path.abspath(os.path.join(datasets_path, mercedes, "clean.csv")),
        "error_types": [missing_values, outliers],
        "labels_reg": ["y"],
        "ml_tasks": [regression],
        'excluded_attribs': []
    },
    adult: {
        'name': adult,
        "dataset_path": os.path.abspath(os.path.join(datasets_path, adult)),
        "dirty_path": os.path.abspath(os.path.join(datasets_path, adult, "dirty.csv")),
        "groundTruth_path": os.path.abspath(os.path.join(datasets_path, adult, "clean.csv")),
        "error_types": [outliers, rule_violation],
        'dboost_configs': ["histogram", "0.1", "0.7"],
        "labels_clf": ["income"],
        "ml_tasks": [classification],
        'classes': classes_list[0],
        "n_clusters": 3,
        'excluded_attribs': []
    },
    beers: {
        'name': beers,
        'dataset_path': os.path.abspath(os.path.join(datasets_path, beers)),
        'dirty_path': os.path.abspath(os.path.join(datasets_path, beers, 'dirty.csv')),
        'groundTruth_path': os.path.abspath(os.path.join(datasets_path, beers, 'clean.csv')),
        'error_types': [missing_values, rule_violation, typos],
        'dboost_configs': ['histogram', '0.9', '0.3'],
        'labels_clf': ['brewery_name'],
        'ml_tasks': [classification],
        'classes': classes_list[1],  # multi-class classification
        'text_features': ["beer_name", "state"],
        'excluded_attribs': ['index', 'id']
    },
    breast_cancer: {
        'name': breast_cancer,
        'dataset_path': os.path.abspath(os.path.join(datasets_path, breast_cancer)),
        'dirty_path': os.path.abspath(os.path.join(datasets_path, breast_cancer, 'dirty.csv')),
        'groundTruth_path': os.path.abspath(os.path.join(datasets_path, breast_cancer, 'clean.csv')),
        'error_types': [outliers, typos, missing_values],
        'ml_tasks': [classification],
        'labels_clf': ['class'],
        'classes': classes_list[0],
        'n_clusters': 2
    },
    nursery: {
        'name': nursery,
        'dataset_path': os.path.abspath(os.path.join(datasets_path, nursery)),
        'dirty_path': os.path.abspath(os.path.join(datasets_path, nursery, 'dirty.csv')),
        'groundTruth_path': os.path.abspath(os.path.join(datasets_path, nursery, 'clean.csv')),
        'error_types': [outliers, typos, missing_values],
        'ml_tasks': [classification],
        'labels_clf': ['class'],
        'classes': classes_list[0],
    }
}


if __name__ == "__main__":
    #print(os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.pardir)))
    #print(datasets_path)
    #datasets_dictionary = '/mnt/c/Users/moabd/Documents/Projects/benchmark/venv/lib/python3.8/site-packages/rein-0.1.0-py3.8.egg/datasets/print3d'
    print(os.path.abspath(os.path.dirname(__file__)))