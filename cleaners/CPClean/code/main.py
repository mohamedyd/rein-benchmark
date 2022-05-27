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

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default=None, nargs='+')
parser.add_argument('--data_dir', default="data/CP-new")
parser.add_argument('--save_dir', default="result_revision/")
parser.add_argument('--cache_dir', default="data/CP-revision-cache")
parser.add_argument('--mv_type', default=["MAR", "MCAR"], nargs='+')
parser.add_argument('--val_size', default=1000, type=int)
parser.add_argument('--n_jobs', default=8, type=int)
parser.add_argument('--cp_seed', default=1, type=int)
parser.add_argument('--run_random', action='store_true', default=False)
parser.add_argument('--run_pre', action='store_true', default=False)
parser.add_argument('--sample_size', default=32, type=int)
parser.add_argument('--restore', action='store_true', default=False)
args = parser.parse_args()

# model
model = {
    "fn": KNN,
    "params": {"n_neighbors":3}
}

if args.dataset is None:
    args.dataset = sorted([d for d in os.listdir(args.data_dir) if d[0]!="."])

for dataset in args.dataset:
    for mv_type in args.mv_type:
        print("Running", dataset, mv_type)

        # build data
        cache_dir = os.path.join(args.cache_dir, dataset, mv_type)
        data, info = utils.load_cache(cache_dir)
        data = preprocess(data)
        
        data["X_val"] = data["X_val"][:args.val_size]
        data["y_val"] = data["y_val"][:args.val_size]

        print("Preprocess Finished")

        info["val_size"] = len(data["X_val"])

        result_classic = run_classic_clean(data, model)
        result_bc = run_boost_clean(data, model)
        save_path = utils.makedir([args.save_dir, dataset, mv_type, "_" + str(args.val_size)], "baseline.csv")
        utils.dicts_to_csv([info, result_classic, result_bc], save_path)

        if args.run_pre:
            continue
        elif not args.run_random:
            result_cp = run_cp_clean(data, model, n_jobs=args.n_jobs, debug_dir=utils.makedir([args.save_dir, dataset, mv_type, "_" + str(args.val_size)]), sample_size=args.sample_size, method="sgd_cpclean")
        else:
            result_random = run_random(data, model, n_jobs=args.n_jobs, debug_dir=utils.makedir([args.save_dir, dataset, mv_type, "_" + str(args.val_size)]), seed=args.cp_seed)