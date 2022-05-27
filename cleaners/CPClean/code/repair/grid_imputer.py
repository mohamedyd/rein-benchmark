from sklearn.impute import SimpleImputer
import numpy as np
import utils

def repair(X_train_mv, save_dir=None):
    mv_columns = X_train_mv.isnull().any(axis=0)
    mv_columns = list(mv_columns[mv_columns==True].index)

    repair_dict = {}

    for c in mv_columns:
        X_c = X_train_mv[c].dropna().values
        cand = set(np.linspace(min(X_c), max(X_c), 4))
        cand.add(X_c.mean())
        repair_dict[c] = sorted(list(cand))
    
    c1, c2 = mv_columns

    X_train_repairs = {}

    for i, v1 in enumerate(repair_dict[c1]):
        for j, v2 in enumerate(repair_dict[c2]):
            name = "{}_{}".format(i, j)
            if name == "2_2":
                name = "mean"
            imp_dict = {c1:v1, c2:v2}
            X_train_repairs[name] = X_train_mv.fillna(value=imp_dict)

    if save_dir is not None:
        for name, X_imp in X_train_repairs.items():
            X_imp.to_csv(utils.makedir([save_dir], "{}.csv".format(name)), index=False)

    return X_train_repairs
