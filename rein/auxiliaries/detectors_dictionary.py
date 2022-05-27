####################################################
# Benchmark: Dictionary of detect and repair methods.
# Authors: Mohamed Abdelaal, Christian Hammacher
# Date: March 2021
# Software AG
# All Rights Reserved
####################################################

# ** Requirements **
# OpenRefine: clusters
# Zeroer: Blocking function
# DetectMethod.holoclean: Denial Constraints
# NADEEF: Functional dependencies

from rein.auxiliaries.configurations import *

detectors_dictionary = {
    missing_values: {
        "name": missing_values,
        "detectors_list": [DetectMethod.raha, DetectMethod.ed2, 
                           DetectMethod.picket, DetectMethod.metadata_driven, 
                           DetectMethod.max_entropy, DetectMethod.min_k, 
                           DetectMethod.mvdetector, DetectMethod.fahes, DetectMethod.holoclean],
        "cleaners_list": [RepairMethod.cleanWithGroundTruth, RepairMethod.standardImputer,
                          RepairMethod.baran, RepairMethod.mlImputer, RepairMethod.dcHoloCleaner,
                          RepairMethod.boostClean, RepairMethod.cpClean, RepairMethod.activecleanCleaner]
    },
    pattern_violation: {
        "name": pattern_violation,
        "detectors_list": [DetectMethod.katara, DetectMethod.openrefine,
                           DetectMethod.raha, DetectMethod.ed2, DetectMethod.picket,
                           DetectMethod.min_k, DetectMethod.max_entropy, DetectMethod.metadata_driven,
                           DetectMethod.nadeef, DetectMethod.holoclean],
        "cleaners_list": [RepairMethod.cleanWithGroundTruth, RepairMethod.dcHoloCleaner, RepairMethod.baran,
                          RepairMethod.mlImputer, RepairMethod.boostClean, RepairMethod.cpClean,
                          RepairMethod.activecleanCleaner]
    },
    rule_violation: {
        "name": rule_violation,
        "detectors_list": [DetectMethod.nadeef, DetectMethod.ed2, DetectMethod.picket,
                           DetectMethod.min_k, DetectMethod.max_entropy, DetectMethod.metadata_driven,
                           DetectMethod.holoclean, DetectMethod.raha],
        "cleaners_list": [RepairMethod.cleanWithGroundTruth, RepairMethod.dcHoloCleaner,
                          RepairMethod.baran, RepairMethod.mlImputer, RepairMethod.boostClean,
                          RepairMethod.cpClean, RepairMethod.activecleanCleaner]
    },
    outliers: {
        "name": outliers,
        "detectors_list": [DetectMethod.raha, DetectMethod.dboost, DetectMethod.fahes,
                           DetectMethod.metadata_driven, DetectMethod.outlierdetector,
                           DetectMethod.max_entropy, DetectMethod.min_k, DetectMethod.ed2],
        "cleaners_list": [RepairMethod.cleanWithGroundTruth, RepairMethod.standardImputer,
                          RepairMethod.baran, RepairMethod.mlImputer, RepairMethod.boostClean,
                          RepairMethod.cpClean, RepairMethod.activecleanCleaner]
    },
    mislabels: {
        "name": mislabels,
        "detectors_list": [DetectMethod.cleanlab, DetectMethod.raha, DetectMethod.ed2,
                           DetectMethod.picket, DetectMethod.metadata_driven, DetectMethod.min_k,
                           DetectMethod.max_entropy, DetectMethod.mislabeldetector],
        "cleaners_list": [RepairMethod.cleanWithGroundTruth, RepairMethod.mlImputer,
                          RepairMethod.baran, RepairMethod.boostClean, RepairMethod.cpClean,
                          RepairMethod.activecleanCleaner]
    },
    duplicates: {
        "name": duplicates,
        "detectors_list": [DetectMethod.duplicatesdetector, DetectMethod.zeroer,
                           DetectMethod.openrefine, DetectMethod.min_k, DetectMethod.max_entropy,
                           DetectMethod.picket, DetectMethod.cleanlab],
        "cleaners_list": [RepairMethod.duplicatesCleaner, RepairMethod.openrefine]
    },
    typos: {
        "name": typos,
        "detectors_list": [DetectMethod.katara, DetectMethod.openrefine, DetectMethod.raha,
                           DetectMethod.ed2, DetectMethod.nadeef, DetectMethod.holoclean,
                           DetectMethod.min_k, DetectMethod.max_entropy, DetectMethod.picket,
                           DetectMethod.metadata_driven],
        "cleaners_list": [RepairMethod.cleanWithGroundTruth, RepairMethod.baran,
                          RepairMethod.dcHoloCleaner, RepairMethod.mlImputer, RepairMethod.boostClean,
                          RepairMethod.cpClean, RepairMethod.activecleanCleaner]
    },
    inconsistency: {
        "name": inconsistency,
        "detectors_list": [DetectMethod.katara, DetectMethod.raha, DetectMethod.openrefine,
                           DetectMethod.ed2, DetectMethod.picket, DetectMethod.min_k,
                           DetectMethod.max_entropy, DetectMethod.metadata_driven],
        "cleaners_list": [RepairMethod.cleanWithGroundTruth, RepairMethod.openrefine,
                          RepairMethod.baran, RepairMethod.mlImputer, RepairMethod.boostClean,
                          RepairMethod.cpClean, RepairMethod.activecleanCleaner]
    }
}

if __name__ == "__main__":
    detector_list = detectors_dictionary["ensemble"]["detectors_list"]
    print(detector_list)
