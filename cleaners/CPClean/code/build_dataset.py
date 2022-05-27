"""Split dataset into training set and test set
   Inject missing values
   Repair missing values
"""
import numpy as np
import pandas as pd
import utils
import argparse
import os
import shutil
import json
from itertools import product
from collections import Counter
import time
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from copy import deepcopy
from training.preprocess import Preprocessor
from scipy.stats import mode

def split(X, y, val_size, test_size=None, max_size=None, random_state=1):
    """Shuffle and split data to train / val / test
    
    Args:
        X (pd.DataFrame): features
        y (pd.DataFrame): label
        test_ratio (flaot): percent of test data
        random_state (int): random_seed
        max_size (int): maximum size of dataset
    """
    # random shuffle 
    np.random.seed(random_state)
    N = X.shape[0]
    idx = np.random.permutation(N)

    # only use first max_size data if N > max_size
    if max_size is not None:
        N = int(min(N, max_size))

    if test_size is None:
        test_size = int((N - val_size) * 0.3)

    
    # split train and test
    idx_test = idx[:test_size]
    idx_val = idx[test_size: test_size+val_size]
    idx_train = idx[test_size+val_size: N]



    # split X and y
    X_train = X.iloc[idx_train].reset_index(drop=True)
    y_train = y.iloc[idx_train].reset_index(drop=True)
    X_val = X.iloc[idx_val].reset_index(drop=True)
    y_val = y.iloc[idx_val].reset_index(drop=True)
    X_test = X.iloc[idx_test].reset_index(drop=True)
    y_test = y.iloc[idx_test].reset_index(drop=True)

    return X_train, y_train, X_val, y_val, X_test, y_test

def load_df(file_path, info):
    """Load data file into pandas dataframe, preliminarily preprocess data.

    Args: 
        file_path (string): path of data
        info (dict): info
    """
    df = pd.read_csv(file_path)
    if 'categorical_variables' in info:
        categories = info['categorical_variables']
        for cat in categories:
            if cat in df.columns:
                df[cat] = df[cat].astype(str).replace('nan', np.nan) 

    if "drop_variables" in info:
        df = df.drop(columns=info["drop_variables"])

    return df

def load_raw_data(data_dir, dataset, dropna=True):
    """Load dataset.

    Args:
        data_dir (string): directory of data
        dataset (string): name of dataset
        dropna (bool): whether drop missing values in raw data

    Return:
        X (pd.Dataframe): features 
        y (pd.Dataframe): label
        info (dict): information of this dataset
    """
    
    # load info 
    info_path = os.path.join(data_dir, dataset, "info.json")
    with open(info_path) as info_data:
        info = json.load(info_data)
    

    # load data
    data_path = os.path.join(data_dir, dataset, "data.csv")
    data = load_df(data_path, info)

    if dropna:
        data = data.dropna()

    # split feature and label
    label_column = info["label"]
    feature_column = [c for c in data.columns if c != label_column]
    X = data[feature_column]
    y = data[[label_column]]
    return X, y

def get_feature_importance(X, y, random_state=1):
    """Compute the importance of features.
       The importance of a feature is computed as the acc drop before
       and after dropping this feature
    """
    np.random.seed(random_state)
    knn = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
    preprocessor = Preprocessor()

    preprocessor.fit(X, y)
    X_full, y_full = preprocessor.transform(X, y)

    acc_full = cross_val_score(knn, X_full, y_full, cv=5, n_jobs=-1).mean()
    acc_diff = {}

    for c in X.columns:
        X_c = X.drop(columns=c)
        preprocessor.fit(X_c, y)
        X_c, y_c = preprocessor.transform(X_c, y)
        acc_c = cross_val_score(knn, X_c, y_c, cv=5, n_jobs=-1).mean()
        acc_diff[c] = acc_full - acc_c

    return acc_diff

def inject_mv(X_train, y_train, mv_prob, mv_type, feature_importance):
    X_train_mv = deepcopy(X_train)

    # get missing prob matrix
    if mv_type == "MCAR":
        m_prob_matrix = np.ones(X_train_mv.shape)*mv_prob
    elif mv_type == "MAR":
        probs = np.array([feature_importance[c] for c in X_train.columns])
        probs[probs<0] = 0
        probs += 1e-100
        probs = probs / np.sum(probs)
        col_missing_prob = probs * mv_prob * X_train.shape[1]
        col_missing_prob[col_missing_prob>0.95] = 0.95
        m_prob_matrix = np.tile(col_missing_prob, (X_train_mv.shape[0], 1))

    # inject missing values
    mask = np.random.rand(*X_train_mv.shape) <= m_prob_matrix

    # avoid injecting in all columns for one row
    for i in range(len(mask)):
        if mask[i].all():
            print("Bad Luck")
            non_mv = int(mask.shape[1] * (1-mv_prob))
            non_mv_indices = np.random.choice(mask.shape[1], size=non_mv, replace=False)
            mask[i, non_mv_indices] = False

    X_train_mv[mask] = np.nan
    ind_mv = pd.DataFrame(mask, columns=X_train_mv.columns)
    return X_train_mv, ind_mv

def save_data(data_dict, info, save_dir):
    for name, data in data_dict.items():
        if isinstance(data, pd.DataFrame):
            data.to_csv(utils.makedir([save_dir], "{}.csv".format(name)), index=False)

    with open(os.path.join(save_dir, 'info.json'), 'w') as f:
        json.dump(info, f, indent=4)


def get_info(data, dataset, mv_type):
    n_train = len(data["X_train_clean"])
    n_val = len(data["X_val"])
    n_test = len(data["X_test"])
    n_clean_train = len(data["X_train_dirty"].dropna())
    percent_mv_rows = (n_train - n_clean_train) / n_train
    percent_mv_cells = data["X_train_dirty"].isna().values.mean()
    missing_columns = data["X_train_dirty"].columns[data["X_train_dirty"].isna().values.any(axis=0)]
    cat_columns = data["X_train_dirty"].select_dtypes(exclude="number").columns
    mv_columns = ""
    for c in missing_columns:
        if c in cat_columns:
            mv_columns += c + "(cat), "
        else:
            mv_columns += c + "(num), "

    info = {
        "dataset": dataset,
        "mv_type": mv_type,
        "train/val/test": "{}/{}/{}".format(n_train, n_val, n_test),
        "num_feature": data["X_train_clean"].shape[1],
        "percent_mv_cells": percent_mv_cells,
        "percent_mv_rows": percent_mv_rows,
        "mv_columns": mv_columns
    }
    return info


    

def build_dataset(data_dir, dataset, val_size=1000, test_size=None, 
                  max_size=None, mv_prob=0.2, mv_type="MAR",  
                  max_num_feature=None, save_dir=None, random_state=1):

    """ Generate dataset for CP experiment

    Args:
        data_dir (string): raw dataset directory
        dataset (string): name of the dataset
        val_size (int): size of test set
        test_size (int): size of test set
        max_num_feature (int): maximum number of features
        max_size (int): max size of train + val + test
        mv_prob (float): probability of missing for a cell
        mv_type (string): type of missing (MCAR or MAR)
        save_dir (string): directory to save data
        random_state (int): random seed

    Returns:
        data_dict (dict): dict containing data needed for experiments
    """
    # load raw data and data info
    X_full, y_full = load_raw_data(data_dir, dataset)

    #select important features
    f_importance = get_feature_importance(X_full, y_full)
    important_features = sorted(f_importance.keys(), key=lambda c: -f_importance[c])
    if max_num_feature is not None:
        important_features = important_features[:max_num_feature]
        X_full = X_full[important_features]

    X_train, y_train, X_val, y_val, X_test, y_test = \
        split(X_full, y_full, val_size, test_size, max_size=max_size, random_state=random_state)
    X_train_mv, ind_mv = inject_mv(X_train, y_train, mv_prob, mv_type, f_importance)

    data_dict = {
        "X_train_clean": X_train, "y_train": y_train,
        "X_train_dirty": X_train_mv, "indicator": ind_mv, 
        "X_full": X_full, "y_full": y_full,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }

    info = get_info(data_dict, dataset, mv_type)
    info["seed"] = random_state

    if save_dir is not None:
        save_data(data_dict, info, save_dir)

    return data_dict, info

def build_real_dataset(data_dir, dataset, info, val_size=1000, test_size=None, 
                       max_size=None, save_dir=None, random_state=1):
    
    #info_path = os.path.join(data_dir, dataset, "info.json")
    #with open(info_path) as info_data:
    #    info = json.load(info_data)

    data_clean = load_df(os.path.join(data_dir, dataset, "clean.csv"), info)
    data_dirty = load_df(os.path.join(data_dir, dataset, "dirty.csv"), info)

    # duplicates = data_clean.duplicated().values
    # data_clean = data_clean[~duplicates]
    # data_dirty = data_dirty[~duplicates]

    label_column = info["label"]
    feature_column = [c for c in data_clean.columns if c != label_column]

    X_full = data_clean[feature_column]
    y_full = data_clean[[label_column]]
    X_full_dirty = data_dirty[feature_column]
    y_full_dirty = data_dirty[[label_column]]

    X_train, y_train, X_val, y_val, X_test, y_test = \
        split(X_full, y_full, val_size, test_size, max_size=max_size, random_state=random_state)

    X_train_mv, _, _, _, _, _ = \
        split(X_full_dirty, y_full_dirty, val_size, test_size, max_size=max_size, random_state=random_state)

    ind_mv = X_train_mv.isnull()

    data_dict = {
        "X_train_clean": X_train, "y_train": y_train,
        "X_train_dirty": X_train_mv, "indicator": ind_mv, 
        "X_full": X_full, "y_full": y_full,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }

    #info = get_info(data_dict, dataset, "real")
    #info["seed"] = random_state

    #if save_dir is not None:
    #    save_data(data_dict, info, save_dir)

    return data_dict#, info

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=None)
    parser.add_argument('--data_dir', default="data/CC-raw")
    parser.add_argument('--save_dir', default="data/CC-01")
    parser.add_argument('--mv_prob', default=0.1, type=float)
    parser.add_argument('--mv_type', default=["rand", "sys_class_depend"], nargs="+")
    parser.add_argument('--max_size', default=float("inf"), type=float)
    args = parser.parse_args()

    for mv_type in args.mv_type:
        print("Build dataset {}_{}_{}".format(args.dataset, mv_type, args.mv_prob))
        data_dict, new_info = build_dataset(args.data_dir, args.dataset, mv_prob=args.mv_prob, max_size=args.max_size, mv_type=mv_type, save_dir=args.save_dir)