from sklearn.impute import KNNImputer as sklearnKNNImputer
from sklearn.impute import SimpleImputer as sklearnSimpleImputer
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer as sklearnIterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
# from fancyimpute import SoftImpute as fancySoftImpute, BiScaler as fancyBiScaler, IterativeSVD as fancyIterativeSVD
import sys
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base

from missingpy import MissForest
import datawig
from impyute.imputation.cs import em as impEM

from sklearn.preprocessing import OrdinalEncoder
import numpy as np
from collections import Counter

class SimpleImputer(object):
    def __init__(self, num="mean", cat="most_frequent"):
        self.num_imputer = sklearnSimpleImputer(strategy=num)

        if cat == "most_frequent":
            self.cat_imputer = sklearnSimpleImputer(strategy="most_frequent")
        elif cat == "dummy":
            self.cat_imputer = sklearnSimpleImputer(strategy="constant", fill_value="missing")
        else:
            raise Exception(cat + "is not a valid imputing method")

    def fit_transform(self, X):
        num_X = X.select_dtypes(include='number')
        cat_X = X.select_dtypes(exclude='number')

        if num_X.shape[1] > 0:
            num_X_imp = self.num_imputer.fit_transform(num_X.values)
            num_X_imp = pd.DataFrame(num_X_imp, columns=num_X.columns)
        if cat_X.shape[1] > 0:
            cat_X_imp = self.cat_imputer.fit_transform(cat_X.values)
            cat_X_imp = pd.DataFrame(cat_X_imp, columns=cat_X.columns)

        if num_X.shape[1] == 0:
            X_imp = cat_X_imp
        elif cat_X.shape[1] == 0:
            X_imp = num_X_imp
        else:
            X_imp = pd.concat([num_X_imp, cat_X_imp], axis=1)
        X_imp = X_imp[X.columns]
        return X_imp

class NumImputer(object):
    def __init__(self, **kwargs):
        self.imputer = None

    def fit_transform(self, X):
        num_X = X.select_dtypes(include='number')
        cat_X = X.select_dtypes(exclude='number')

        # impute categorical columns with dummy variable only
        if cat_X.shape[1] > 0:
            assert cat_X.isnull().values.any() == False
            cat_X_enc = pd.get_dummies(cat_X, drop_first=True)
            X_clean_cat = pd.concat([num_X, cat_X_enc], axis=1)
        else:
            X_clean_cat = X
            
        # impute numerical columns
        X_clean_cat_imp = self.imputer.fit_transform(X_clean_cat.values)
        X_clean_cat_imp = pd.DataFrame(X_clean_cat_imp, columns=X_clean_cat.columns)
        num_X_imp = X_clean_cat_imp[num_X.columns]

        # combine num and cat
        if cat_X.shape[1] > 0:
            X_imp = pd.concat([num_X_imp, cat_X], axis=1)
        else:
            X_imp = num_X_imp
            
        X_imp = X_imp[X.columns]
        assert X.shape == X_imp.shape
        return X_imp

class KNNImputer(NumImputer):
    """docstring for KNNImputer"""
    def __init__(self, n_neighbors=5):
        self.imputer = sklearnKNNImputer(n_neighbors=n_neighbors)

class IterativeImputer(NumImputer):
    def __init__(self, model="bayesian_ridge", random_state=1):
        estimators = {
            "bayesian_ridge": BayesianRidge(),
            "decision_tree": DecisionTreeRegressor(max_features='sqrt', random_state=0),
            "extra_trees": ExtraTreesRegressor(n_estimators=10, random_state=0),
            "knn": KNeighborsRegressor(n_neighbors=15)
        }
        self.imputer = sklearnIterativeImputer(estimator=estimators[model], random_state=random_state)

class SoftImputerWrapper(object):
    def __init__(self):
        pass

    def fit_transform(self, X):
        biscaler = fancyBiScaler(verbose=False)
        X_normalized = biscaler.fit_transform(X)
        X_filled_normalized = fancySoftImpute(verbose=False).fit_transform(X_normalized)
        X_filled = biscaler.inverse_transform(X_filled_normalized)
        return X_filled

class SoftImputer(NumImputer):
    def __init__(self):
        self.imputer = SoftImputerWrapper()

class EMImputeWrapper(object):
    def __init__(self):
        pass

    def fit_transform(self, X):
        X_imp = impEM(X)
        return X_imp

class EMImputer(NumImputer):
    def __init__(self):
        self.imputer = EMImputeWrapper()

class DataWigImputer(object):
    def __init__(self):
        pass

    def fit_transform(self, X):
        return datawig.SimpleImputer.complete(X)

class MissForestImputer(object):
    def __init__(self):
        self.imputer = MissForest(verbose=0)

    def encode_cat(self, X_c):
        data = X_c.copy()
        nonulls = data.dropna().values
        impute_reshape = nonulls.reshape(-1,1)
        encoder = OrdinalEncoder()
        impute_ordinal = encoder.fit_transform(impute_reshape)
        data.loc[data.notnull()] = np.squeeze(impute_ordinal)
        return data, encoder

    def decode_cat(self, X_c, encoder):
        data = X_c.copy()
        nonulls = data.dropna().values.reshape(-1,1)
        n_cat = len(encoder.categories_[0])
        nonulls = np.round(nonulls).clip(0, n_cat-1)
        nonulls = encoder.inverse_transform(nonulls)
        data.loc[data.notnull()] = np.squeeze(nonulls)
        return data

    def fit_transform(self, X):
        num_X = X.select_dtypes(include='number')
        cat_X = X.select_dtypes(exclude='number')

        # encode the categorical columns to numeric columns
        if cat_X.shape[1] > 0:
            cat_encoders = {}
            cat_X_enc = []
            for c in cat_X.columns:
                X_c_enc, encoder = self.encode_cat(cat_X[c])
                cat_X_enc.append(X_c_enc)
                cat_encoders[c] = encoder
            cat_X_enc = pd.concat(cat_X_enc, axis=1)
            X_enc = pd.concat([num_X, cat_X_enc], axis=1)
            cat_columns = cat_X.columns
            cat_indices = [i for i, c in enumerate(X_enc.columns) if c in cat_columns]
        else:
            X_enc = X
            cat_indices = None

        X_imp = self.imputer.fit_transform(X_enc.values.astype(float), cat_vars=cat_indices)
        X_imp = pd.DataFrame(X_imp, columns=X_enc.columns)

        if cat_X.shape[1] > 0:       
            num_X_imp = X_imp[num_X.columns]
            cat_X_imp = X_imp[cat_X.columns]
            cat_X_dec = []
            for c in cat_X.columns:
                X_c_dec = self.decode_cat(cat_X_imp[c], cat_encoders[c])
                cat_X_dec.append(X_c_dec)
            cat_X_dec = pd.concat(cat_X_dec, axis=1)
            X_imp = pd.concat([num_X_imp, cat_X_dec], axis=1)
        
        X_imp = X_imp[X.columns]
        return X_imp

class PercentileImputer(object):
    def __init__(self, p):
        self.p = p
        pass

    def fit_transform(self, X_train_mv):
        mv_columns = X_train_mv.isnull().any(axis=0)
        mv_columns = list(mv_columns[mv_columns==True].index)

        imp_dict = {}
        for c in mv_columns:
            imp_dict[c] = np.percentile(X_train_mv[c].dropna().values, self.p)

        X_imp = X_train_mv.fillna(value=imp_dict)
        return X_imp     

class GridImputer(object):
    def __init__(self, p):
        self.p = p
        pass

    def fit_transform(self, X_train_mv):
        mv_columns = X_train_mv.isnull().any(axis=0)
        mv_columns = list(mv_columns[mv_columns==True].index)

        imp_dict = {}
        for c in mv_columns:
            min_c = X_train_mv[c].dropna().values.min()
            max_c = X_train_mv[c].dropna().values.max()
            repair_value = min_c + (max_c - min_c) * (self.p / 100)
            imp_dict[c] = repair_value

        X_imp = X_train_mv.fillna(value=imp_dict)
        return X_imp     

class CatGridImputer(object):
    def __init__(self, p):
        self.p = p
        pass

    def fit_transform(self, X_train_mv):
        mv_columns = X_train_mv.isnull().any(axis=0)
        mv_columns = list(mv_columns[mv_columns==True].index)

        imp_dict = {}
        for c in mv_columns:
            counter = Counter(X_train_mv[c].dropna().values)
            repair_value = [k for k, n in counter.most_common(self.p)][-1]
            imp_dict[c] = repair_value

        X_imp = X_train_mv.fillna(value=imp_dict)
        return X_imp  

# all_imputers = {
#     "em": EMImputer(),
#     "knn": KNNImputer(),
#     "mean": SimpleImputer(num="mean"),
#     "median": SimpleImputer(num="median"),
#     "mode": SimpleImputer(num="most_frequent"),
#     "knn_iterative": IterativeImputer("knn"),
#     "br_iterative": IterativeImputer("bayesian_ridge"),
#     "dt_iterative": IterativeImputer("decision_tree"),
#     "soft": SoftImputer(),
#     "biscaler": BiScalerImputer(),
#     "missForest": MissForestImputer(),
#     # "datawig": DataWigImputer()
#     # "svd_iterative": IterativeSVDImputer(),
# }

# for name, imputer in all_imputers.items():
#     data = pd.read_csv("test.csv")
#     imp = imputer.fit_transform(data)
#     imp.to_csv("{}.csv".format(name), index=False)

