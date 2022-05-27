from training.train import train_evaluate
from cleaner.boost_clean import boost_clean
import pandas as pd
import numpy as np
import os
from cleaner.CPClean.clean import CPClean
from cleaner.CPClean.debugger import Debugger
import utils

def run_classic_clean(data, model):
    result = {}
    # evaluate clean and dirty
    clean_test_acc = train_evaluate(model, data["X_train_clean"], data["y_train"], data["X_test"], data["y_test"])
    clean_val_acc = train_evaluate(model, data["X_train_clean"], data["y_train"], data["X_val"], data["y_val"])
    result["test_acc_clean"] = round(clean_test_acc, 3)
    result["val_acc_clean"] = round(clean_val_acc, 3)
    gt_test_acc = train_evaluate(model, data["X_train_gt"], data["y_train"], data["X_test"], data["y_test"])
    gt_val_acc = train_evaluate(model, data["X_train_gt"], data["y_train"], data["X_val"], data["y_val"])
    result["test_acc_gt"] = round(gt_test_acc, 3)
    result["val_acc_gt"] = round(gt_val_acc, 3)

    name = "mean"
    test_acc = train_evaluate(model, data["X_train_repairs"][name], data["y_train"], data["X_test"], data["y_test"])
    val_acc = train_evaluate(model, data["X_train_repairs"][name], data["y_train"], data["X_val"], data["y_val"])
    result["test_acc_" + name] = round(test_acc, 3)
    result["val_acc_" + name] = round(val_acc, 3)
    for name, X_train_repair in data["X_train_repairs"].items():
        test_acc = train_evaluate(model, X_train_repair, data["y_train"], data["X_test"], data["y_test"])
        val_acc = train_evaluate(model, X_train_repair, data["y_train"], data["X_val"], data["y_val"])
        result["test_acc_" + name] = round(test_acc, 3)
        result["val_acc_" + name] = round(val_acc, 3)
    return result

def run_boost_clean(data, model):
    X_train_repairs = list(data["X_train_repairs"].values())
    bc_1_test_acc, bc_1_val_acc = boost_clean(model, X_train_repairs, data["y_train"], data["X_val"], data["y_val"], data["X_test"], data["y_test"], T=1)
    bc_5_test_acc, bc_5_val_acc = boost_clean(model, X_train_repairs, data["y_train"], data["X_val"], data["y_val"], data["X_test"], data["y_test"], T=5)
    bc_result = {"test_acc_bc_1":bc_1_test_acc, "val_acc_bc_1": bc_1_val_acc,
                 "test_acc_bc_5":bc_5_test_acc, "val_acc_bc_5": bc_5_val_acc}
    return bc_result

def run_cp_clean(data, model, n_jobs=4, debug_dir=None, restore=False, method="cpclean", sample_size=100):
    X_train_repairs = np.array([data["X_train_repairs"][m] for m in data["repair_methods"]])
    cleaner = CPClean(K=model["params"]["n_neighbors"], n_jobs=n_jobs, random_state=1)

    #debugger = Debugger(data, model, utils.makedir([debug_dir, method]))

    cleaner.fit(X_train_repairs, data["y_train"], data["X_val"], data["y_val"], 
                gt=data["X_train_gt"], X_train_mean=data["X_train_repairs"]["mean"], 
                restore=restore, method=method, sample_size=sample_size)

    val_acc = cleaner.score(data["X_val"], data["y_val"])
    test_acc = cleaner.score(data["X_test"], data["y_test"])
    cp_result = {"test_acc_cp": test_acc, "val_acc_cp": val_acc, "percent_clean": debugger.percent_clean}
    return cp_result

def run_random(data, model, n_jobs=4, debug_dir=None, seed=1):
    X_train_repairs = np.array([data["X_train_repairs"][m] for m in data["repair_methods"]])

    cleaner = CPClean(K=model["params"]["n_neighbors"], n_jobs=n_jobs, random_state=seed)
    debugger = Debugger(data, model, utils.makedir([debug_dir, "random_clean", str(seed)]))
    cleaner.fit(X_train_repairs, data["y_train"], data["X_val"], data["y_val"], 
                gt=data["X_train_gt"], X_train_mean=data["X_train_repairs"]["mean"], 
                debugger=debugger, method="random")
