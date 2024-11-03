import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier

from .constants import FairnessIntervention
from .constants import ErrorRepairMethod
import virny_flow.core.null_imputers.datawig_imputer as datawig_imputer
from virny_flow.core.null_imputers.imputation_methods import (impute_with_deletion, impute_with_simple_imputer, impute_with_automl,
                                                              impute_with_missforest, impute_with_kmeans)


NULL_IMPUTERS_CONFIG = {
    ErrorRepairMethod.deletion.value: {"method": impute_with_deletion, "kwargs": {}},
    ErrorRepairMethod.median_mode.value: {"method": impute_with_simple_imputer, "kwargs": {"num": "median", "cat": "most_frequent"}},
    ErrorRepairMethod.median_dummy.value: {"method": impute_with_simple_imputer, "kwargs": {"num": "median", "cat": "constant"}},
    ErrorRepairMethod.miss_forest.value: {"method": impute_with_missforest, "kwargs": {}},
    ErrorRepairMethod.k_means_clustering.value: {"method": impute_with_kmeans, "kwargs": {}},
    ErrorRepairMethod.datawig.value: {"method": datawig_imputer.complete, "kwargs": {"precision_threshold": 0.0, "num_epochs": 100, "iterations": 1}},
    ErrorRepairMethod.automl.value: {"method": impute_with_automl, "kwargs": {"max_trials": 50, "tuner": None, "validation_split": 0.2, "epochs": 100}},
}

FAIRNESS_INTERVENTION_HYPERPARAMS = {
    FairnessIntervention.DIR.value: {"repair_level": 0.7},
    FairnessIntervention.LFR.value: {"k": 5, "Ax": 0.01, "Ay": 1.0, "Az": 50.0},
    FairnessIntervention.AD.value: {"scope_name": "debiased_classifier",
                                    "adversary_loss_weight": 0.1, "num_epochs": 50, "batch_size": 128,
                                    "classifier_num_hidden_units": 200, "debias": True},
    FairnessIntervention.EGR.value: {"constraints": "DemographicParity",
                                     "eps": 0.01, "max_iter": 50, "nu": None, "eta0": 2.0,
                                     "run_linprog_step": True, "drop_prot_attr": True,
                                     "estimator_params": {"C": 1.0, "solver": "lbfgs"}},
    FairnessIntervention.EOP.value: {},
    FairnessIntervention.ROC.value: {"low_class_thresh": 0.01, "high_class_thresh": 0.99, "num_class_thresh": 100,
                                     "num_ROC_margin": 50, "metric_name": "Statistical parity difference",
                                     "metric_ub": 0.05, "metric_lb": -0.05},
}


def get_models_params_for_tuning(models_tuning_seed):
    return {
        'dt_clf': {
            'model': DecisionTreeClassifier(random_state=models_tuning_seed),
            'params': {
                "max_depth": [5, 10],
                # "max_depth": [5, 10, 20, 30],
                # 'min_samples_leaf': [5, 10, 20, 50, 100],
                # "max_features": [0.6, 'sqrt'],
                # "criterion": ["gini", "entropy"]
            }
        },
        'lr_clf': {
            'model': LogisticRegression(random_state=models_tuning_seed, max_iter=1000),
            'params': {
                'penalty': ['l1', 'l2'],
                'C' : [0.001, 0.01, 0.1, 1],
                'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
            }
        },
        'lgbm_clf': {
            'model': LGBMClassifier(random_state=models_tuning_seed, n_jobs=48, num_threads=48),
            'params': {
                'n_estimators': [50, 100, 200, 500],
                'max_depth' : [i for i in range(3, 10)] + [-1],
                'num_leaves' : [int(x) for x in np.linspace(start = 20, stop = 3000, num = 8)],
                'min_data_in_leaf' : [int(x) for x in np.linspace(start = 100, stop = 1000, num = 8)],
                'verbosity': [-1]
            }
        },
        'rf_clf': {
            'model': RandomForestClassifier(random_state=models_tuning_seed),
            'params': {
                'n_estimators': [50, 100],
                # 'n_estimators': [50, 100, 200, 500],
                # 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                # 'min_samples_split': [2, 5, 10],
                # 'min_samples_leaf': [1, 2, 4],
                # 'bootstrap': [True, False]
            }
        },
        'mlp_clf': {
            'model': MLPClassifier(hidden_layer_sizes=(100,100,), random_state=models_tuning_seed, max_iter=1000),
            'params': {
                'activation': ['logistic', 'tanh', 'relu'],
                'solver': ['lbfgs', 'sgd', 'adam'],
                'learning_rate': ['constant', 'invscaling', 'adaptive']
            }
        }
    }
