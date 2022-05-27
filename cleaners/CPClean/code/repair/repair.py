from .imputers import *
import os
import os

def makedir(dir_list, file=None):
    save_dir = os.path.join(*dir_list)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if file is not None:
        save_dir = os.path.join(save_dir, file)
    return save_dir

num_imputers = {
    "em": EMImputer(),
    "knn": KNNImputer(),
    "mean": SimpleImputer(num="mean"),
    # "median": SimpleImputer(num="median"),
    # "mode": SimpleImputer(num="most_frequent"),
    # "br_iterative": IterativeImputer("bayesian_ridge"),
    "dt_iterative": IterativeImputer("decision_tree"),
    # "soft": SoftImputer(),
    "missForest": MissForestImputer(),
    "datawig": DataWigImputer(),
    "min": PercentileImputer(0),
    # "25_percent": PercentileImputer(25),
    # "75_percent": PercentileImputer(75),
    "max": PercentileImputer(100),
    "25-grid": GridImputer(25),
    "75-grid": GridImputer(75)
}

cat_imputers = {
    "mean": SimpleImputer(cat="most_frequent"),
    "dummy": SimpleImputer(cat="dummy"),
    #"missForest": MissForestImputer(),
    "datawig": DataWigImputer(),
    # "2_frequent": CatGridImputer(2),
    # "3_frequent": CatGridImputer(3),
    # "4_frequent": CatGridImputer(4),
    # "5_frequent": CatGridImputer(5)
}

mix_imputers = {
    "mean_mode": SimpleImputer(num="mean", cat="most_frequent"),
    "mean_dummy": SimpleImputer(num="mean", cat="dummy"),
    "median_mode": SimpleImputer(num="median", cat="most_frequent"),
    "median_dummy": SimpleImputer(num="median", cat="dummy"),
    "mode_mode": SimpleImputer(num="most_frequent", cat="most_frequent"),
    "mode_dummy": SimpleImputer(num="most_frequent", cat="dummy"),
    "missForest": MissForestImputer(),
    "datawig": DataWigImputer()
}

#takes dataframe with true where cell is errror
#viele imputer machen repairs, diese werden dann spater zum stacken verwendet
def repair(X_train_mv, save_dir=None):
    num_X = X_train_mv.select_dtypes(include='number')
    cat_X = X_train_mv.select_dtypes(exclude='number')

    is_num = num_X.isnull().values.any()
    is_cat = cat_X.isnull().values.any()

    if is_num and is_cat:
        all_imputers = mix_imputers
    elif is_num:
        all_imputers = num_imputers
    elif is_cat:
        all_imputers = cat_imputers
        mv_columns = X_train_mv.isnull().any(axis=0)
        c = list(mv_columns[mv_columns==True].index)[0]
        domain = set(X_train_mv[c].dropna().values)
        for i in range(len(domain) - 1):
            all_imputers["{}_frequent".format(i+2)] = CatGridImputer(i+2)
    else:
        raise Exception("no missing values")

    X_train_repairs = {}

    for name, imputer in all_imputers.items():
        X_imp = imputer.fit_transform(X_train_mv)
        
        for c in X_imp.columns:
            nonnull = np.argwhere(X_train_mv[c].notnull().values).ravel()
            X_imp[c].iloc[nonnull] = X_train_mv[c].iloc[nonnull]

        X_train_repairs[name] = X_imp

    if save_dir is not None:
        for name, X_imp in X_train_repairs.items():
            X_imp.to_csv(utils.makedir([save_dir], "{}.csv".format(name)), index=False)
    
    return X_train_repairs
