import random

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
#import utils
import os
import json
import pickle
from collections import Counter
import time
from itertools import product

class Preprocessor(object):
    """docstring for Preprocessor"""
    def __init__(self, num_strategy="mean"):
        super(Preprocessor, self).__init__()
        self.num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=num_strategy)),
            ('scaler', MinMaxScaler())
        ])
        self.feature_enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.cat_imputer = SimpleImputer(strategy="constant", fill_value="missing")
        self.label_enc = LabelEncoder()

    def fit(self, X_train, y_train, X_full=None):
        self.num_features = X_train.select_dtypes(include='number').columns
        self.cat_features = X_train.select_dtypes(exclude='number').columns

        if len(self.num_features) > 0:
            self.num_transformer.fit(X_train[self.num_features].values)

        if len(self.cat_features) > 0:
            if X_full is None:
                X_full = X_train
            self.feature_enc.fit(X_full[self.cat_features].values)
            self.cat_imputer.fit(X_train[self.cat_features].values)
            self.cat_transformer = Pipeline(steps=[
                ('imputer', self.cat_imputer),
                ('onehot', self.feature_enc)
            ])

        self.label_enc.fit(y_train.values.ravel())

    def transform(self, X=None, y=None):
        if X is not None:
            X_after = []
            if len(self.num_features) > 0:
                X_arr = X[self.num_features].values
                if len(X_arr.shape)==1:
                    X_arr = X_arr.reshape(1, -1)
                X_num = self.num_transformer.transform(X_arr)
                X_after.append(X_num)

            if len(self.cat_features) > 0:
                X_arr = X[self.cat_features].values
                if len(X_arr.shape)==1:
                    X_arr = X_arr.reshape(1, -1)
                X_cat = self.cat_transformer.transform(X_arr)
                X_after.append(X_cat)

            X = np.hstack(X_after)

        if y is not None:
            y = self.label_enc.transform(y.values.ravel())

        if X is None:
            return y
        elif y is None:
            return X
        else:
            return X, y

def preprocess(data):
    X_full, y_full = data["X_full"], data["y_full"]
    X_train_dirty = data["X_train_dirty"]
    X_train_clean = data["X_train_clean"]
    y_train = data["y_train"]
    indicator = data["indicator"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    
    # preprocess data
    preprocessor = Preprocessor()
    preprocessor.fit(X_train_dirty, y_train, X_full)

    X_val, y_val = preprocessor.transform(X_val, y_val)
    X_test, y_test = preprocessor.transform(X_test, y_test)
    X_train_clean, y_train = preprocessor.transform(X_train_clean, y_train)

    X_train_repairs = {}
    for name, X in data["X_train_repairs"].items():
        X_train_repairs[name] = preprocessor.transform(X=X)

    data_after = {}
    data_after["X_train_mv"] = X_train_dirty 
    data_after["X_train_repairs"] = X_train_repairs
    data_after["X_train_clean"] = X_train_clean
    data_after["y_train"] = y_train
    data_after["X_val"] = X_val
    data_after["y_val"] = y_val
    data_after["X_test"] = X_test
    data_after["y_test"] = y_test
    data_after["indicator"] = indicator

    d_train_repairs = []
    repair_methods = sorted(X_train_repairs.keys())
    X_train_repairs_sorted = [X_train_repairs[m] for m in repair_methods]
    for X in X_train_repairs_sorted:
        d = np.sum((X - X_train_clean)**2, axis=1)
        d_train_repairs.append(d)
    d_train_repairs = np.array(d_train_repairs).T
    gt_indices = np.argmin(d_train_repairs, axis=1)
    X_train_gt = []
    
    for i, gt_i in enumerate(gt_indices):
        X_train_gt.append(X_train_repairs_sorted[gt_i][i])
    X_train_gt = np.array(X_train_gt)
    
    data_after["X_train_gt"] = X_train_gt
    data_after["repair_methods"] = repair_methods
    data_after["gt_indices"] = gt_indices

    X_train_gt_raw = []
    for i, gt_i in enumerate(gt_indices):
        gt_method = repair_methods[gt_i]
        gt_i = data["X_train_repairs"][gt_method].iloc[i:i+1]
        X_train_gt_raw.append(gt_i)

    X_train_gt_raw = pd.concat(X_train_gt_raw, axis=0)
    data_after["X_train_gt_raw"] = X_train_gt_raw
    return data_after

