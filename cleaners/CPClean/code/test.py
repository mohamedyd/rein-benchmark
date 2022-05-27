import numpy as np
import pandas as pd
import os
import utils
import argparse
from training.preprocess import preprocess
import warnings
import time
import pickle
from experiment import *
from training.knn import KNN
from repair.repair import repair
from build_dataset import build_dataset, build_real_dataset

model = {
    "fn": KNN,
    "params": {"n_neighbors":3}
}


info = {
    "label": "binaryClass", # name of label column
}
val_size = 1000
data = build_real_dataset("/home/chrha/Desktop", "testdata", info, val_size=val_size, test_size=None, 
                       max_size=None, save_dir=None, random_state=1)

data["X_train_repairs"] = repair(data["X_train_dirty"])
        
data = preprocess(data)
        
data["X_val"] = data["X_val"][:val_size]
data["y_val"] = data["y_val"][:val_size]

print("Preprocess Finished")

results = run_boost_clean(data, model)
#results = run_cp_clean(data,model, n_jobs=1)
print(results)