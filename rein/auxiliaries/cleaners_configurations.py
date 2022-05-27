####################################################
# Benchmark: Dictionary of all configurations that are to be run for each cleaners.
# Authors: Mohamed Abdelaal, Christian Hammacher
# Date: July 2021
# Software AG
# All Rights Reserved
####################################################

from rein.auxiliaries.configurations import *

cleaners_configurations = {
    RepairMethod.standardImputer : [
        {"configs":{"method" : "delete"}, "kwargs" : {}},
        {"configs":{"method" : "impute"}, "kwargs" : {"num" : "mean", "cat" : "mode"}},
        {"configs":{"method" : "impute"}, "kwargs" : {"num" : "mean", "cat" : "dummy"}},
        {"configs":{"method" : "impute"}, "kwargs" : {"num" : "median", "cat" : "mode"}},
        {"configs":{"method" : "impute"}, "kwargs" : {"num" : "median", "cat" : "dummy"}},
        {"configs":{"method" : "impute"}, "kwargs" : {"num" : "mode", "cat" : "mode"}},
        {"configs":{"method" : "impute"}, "kwargs" : {"num" : "mode", "cat" : "dummy"}},
    ],
    RepairMethod.mlImputer : [
        {"configs": {"method" : "mix"}, "kwargs" : {"mix_method" : "missForest"}},
        {"configs":{"method" : "mix"}, "kwargs" : {"mix_method" : "datawig"}},
        {"configs": {"method" : "seperate"}, "kwargs" : {"num" : "missForest", "cat" : "missForest"}},
        {"configs": {"method" : "seperate"}, "kwargs" : {"num" : "missForest", "cat" : "datawig"}},
        {"configs":{"method" : "seperate"}, "kwargs" : {"num" : "knn", "cat" : "missForest"}},
        # {"configs":{"method" : "seperate"}, "kwargs" : {"num" : "knn", "cat" : "datawig"}},
        # {"configs":{"method" : "seperate"}, "kwargs" : {"num" : "em", "cat" : "missForest"}},
        # {"configs":{"method" : "seperate"}, "kwargs" : {"num" : "em", "cat" : "datawig"}},
        {"configs": {"method" : "seperate"}, "kwargs" : {"num" : "decisionTree", "cat" : "missForest"}},
        # {"configs":{"method" : "seperate"}, "kwargs" : {"num" : "decisionTree", "cat" : "datawig"}},
        {"configs": {"method" : "seperate"}, "kwargs" : {"num" : "bayesianRidge", "cat" : "missForest"}},
        # {"configs":{"method" : "seperate"}, "kwargs" : {"num" : "bayesianRidge", "cat" : "datawig"}},
        # {"configs":{"method" : "seperate"}, "kwargs" : {"num" : "extraTrees", "cat" : "missForest"}},
        # {"configs":{"method" : "seperate"}, "kwargs" : {"num" : "extraTrees", "cat" : "datawig"}},
        # {"configs":{"method" : "seperate"}, "kwargs" : {"num" : "datawig", "cat" : "missForest"}},
        # {"configs":{"method" : "seperate"}, "kwargs" : {"num" : "datawig", "cat" : "datawig"}},
    ],
    RepairMethod.dcHoloCleaner : [
        # {"configs":{"method" : "with_init"}, "kwargs" : {}},
        {"configs" :{"method" : "without_init"}, "kwargs" : {}},
    ],
    RepairMethod.cleanlab : [ # can be extended with other classifiers from models_dictionary["classification"]
        {"configs": {"model": "forest_clf"}, "kwargs": {}},
        {"configs": {"model": "logit_clf"}, "kwargs": {}}
    ],
    RepairMethod.baran : [
        {"configs":{}, "kwargs" : {}}
    ],
    RepairMethod.cleanWithGroundTruth : [
        {"configs":{}, "kwargs" : {}}
    ],
    RepairMethod.duplicatesCleaner : [
        {"configs":{}, "kwargs" : {}}
    ],
    RepairMethod.openrefine : [
        {"configs":{}, "kwargs" : {}}
    ],
    RepairMethod.boostClean : [
        {"configs":{}, "kwargs" : {}}
    ],
    RepairMethod.cpClean: [
        {"configs":{}, "kwargs" : {}}
    ],
    RepairMethod.activecleanCleaner : [
        {"configs":{"sampling_budget" : 0.2}, "kwargs" : {}}
    ],
}


if __name__ == "__main__":
    pass