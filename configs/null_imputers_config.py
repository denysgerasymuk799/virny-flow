import source.null_imputers.datawig_imputer as datawig_imputer

from configs.constants import (GERMAN_CREDIT_DATASET, BANK_MARKETING_DATASET, CARDIOVASCULAR_DISEASE_DATASET,
                               DIABETES_DATASET, LAW_SCHOOL_DATASET, ACS_INCOME_DATASET,
                               ErrorRepairMethod, ErrorInjectionStrategy)
from source.null_imputers.imputation_methods import (impute_with_deletion, impute_with_simple_imputer, impute_with_automl,
                                                     impute_with_missforest, impute_with_kmeans)
from source.null_imputers.joint_cleaning_and_training_methods import prepare_cpclean


NULL_IMPUTERS_CONFIG = {
    ErrorRepairMethod.deletion.value: {"method": impute_with_deletion, "kwargs": {}},
    ErrorRepairMethod.median_mode.value: {"method": impute_with_simple_imputer, "kwargs": {"num": "median", "cat": "most_frequent"}},
    ErrorRepairMethod.median_dummy.value: {"method": impute_with_simple_imputer, "kwargs": {"num": "median", "cat": "constant"}},
    ErrorRepairMethod.miss_forest.value: {"method": impute_with_missforest, "kwargs": {}},
    ErrorRepairMethod.k_means_clustering.value: {"method": impute_with_kmeans, "kwargs": {}},
    ErrorRepairMethod.datawig.value: {"method": datawig_imputer.complete, "kwargs": {"precision_threshold": 0.0, "num_epochs": 100, "iterations": 1}},
    ErrorRepairMethod.automl.value: {"method": impute_with_automl, "kwargs": {"max_trials": 50, "tuner": None, "validation_split": 0.2, "epochs": 100}},
    ErrorRepairMethod.cp_clean.value: {"method": prepare_cpclean, "kwargs": {}},
}

NULL_IMPUTERS_HYPERPARAMS = {}
