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

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default=None, nargs='+')
parser.add_argument('--data_dir', default="data/CP-new")
parser.add_argument('--cache_dir', default="data/CP-search-cache")
parser.add_argument('--save_dir', default="result_search/")
parser.add_argument('--max_size', default=None, type=int)
parser.add_argument('--max_num_feature', default=None, type=int)
parser.add_argument('--mv_prob', default=0.2, type=float)
parser.add_argument('--mv_type', default=["MAR"], nargs='+')
parser.add_argument('--cache_val_size', default=1400, type=int)
parser.add_argument('--test_size', default=None, type=int)
parser.add_argument('--val_size', default=1000, type=int)
parser.add_argument('--n_jobs', default=8, type=int)
parser.add_argument('--cp_seed', default=1, type=int)
parser.add_argument('--mv_seed', default=1, type=int)
parser.add_argument('--build_cache', action='store_true', default=False)
parser.add_argument('--real', action='store_true', default=False)
parser.add_argument('--run_random', action='store_true', default=False)
parser.add_argument('--sample_size', default=32, type=int)

args = parser.parse_args()

# model
model = {
    "fn": KNN,
    "params": {"n_neighbors":3}
}

if args.dataset is None:
    args.dataset = sorted([d for d in os.listdir(args.data_dir) if d[0]!="."])

mv_seed = args.mv_seed

for dataset in args.dataset:
    for mv_type in args.mv_type:
        print("Running", dataset, mv_type, mv_seed)

        # build data
        cache_dir = os.path.join(args.cache_dir, dataset, str(mv_seed), mv_type)
        if not args.real:
            data, info = build_dataset(
                args.data_dir, dataset, val_size=args.cache_val_size, 
                test_size=args.test_size, max_size=args.max_size,
                mv_prob=args.mv_prob, mv_type=mv_type, 
                random_state=mv_seed, save_dir=cache_dir)
        else:
            data, info = build_real_dataset(
                args.data_dir, dataset, val_size=args.cache_val_size, 
                test_size=args.test_size, max_size=args.max_size,
                random_state=mv_seed, save_dir=cache_dir)

        cache_dir = os.path.join(cache_dir, "X_train_repairs")
        data["X_train_repairs"] = repair(
            data["X_train_dirty"], 
            save_dir=cache_dir)
        
        data = preprocess(data)
        
        data["X_val"] = data["X_val"][:args.val_size]
        data["y_val"] = data["y_val"][:args.val_size]

        print("Preprocess Finished")

        result_classic = run_classic_clean(data, model)

        if result_classic["test_acc_gt"] - result_classic["test_acc_mean"] < 0.05:
            continue

        result_bc = run_boost_clean(data, model)
        save_path = utils.makedir([args.save_dir, dataset, str(mv_seed), mv_type], "baseline.csv")
        utils.dicts_to_csv([info, result_classic, result_bc], save_path)

        if args.run_random:
            result_random = run_random(data, model, n_jobs=args.n_jobs, debug_dir=utils.makedir([args.save_dir, dataset, str(mv_seed), mv_type]), seed=args.cp_seed)
        else:
            result_cp = run_cp_clean(data, model, n_jobs=args.n_jobs, debug_dir=utils.makedir([args.save_dir, dataset, str(mv_seed), mv_type]), sample_size=args.sample_size, method="sgd_cpclean")
