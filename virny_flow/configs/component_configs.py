import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from openbox.utils.config_space import UniformIntegerHyperparameter, UniformFloatHyperparameter, CategoricalHyperparameter
from pytorch_tabular.models import GANDALFConfig
from pytorch_tabular.config import OptimizerConfig, TrainerConfig

from .constants import FairnessIntervention, INIT_RANDOM_STATE
from .constants import ErrorRepairMethod
import virny_flow.core.null_imputers.datawig_imputer as datawig_imputer
from virny_flow.core.null_imputers.imputation_methods import (impute_with_deletion, impute_with_simple_imputer,
                                                              impute_with_missforest, impute_with_kmeans)


# ====================================================================
# Null Imputation Params
# ====================================================================
NULL_IMPUTATION_CONFIG = {
    ErrorRepairMethod.deletion.value: {
        "method": impute_with_deletion,
        "default_kwargs": {},
        "config_space": {}
    },
    ErrorRepairMethod.median_mode.value: {
        "method": impute_with_simple_imputer,
        "default_kwargs": {"num": "median", "cat": "most_frequent"},
        "config_space": {}
    },
    ErrorRepairMethod.median_dummy.value: {
        "method": impute_with_simple_imputer,
        "default_kwargs": {"num": "median", "cat": "constant"},
        "config_space": {}
    },
    ErrorRepairMethod.miss_forest.value: {
        "method": impute_with_missforest,
        "default_kwargs": {},
        "config_space": {
            # RandomForestClassifier
            "mvi__RandomForestClassifier__n_estimators": UniformIntegerHyperparameter("mvi__RandomForestClassifier__n_estimators", 50, 200, q=50),
            "mvi__RandomForestClassifier__max_depth": CategoricalHyperparameter("mvi__RandomForestClassifier__max_depth", [10, 25, 50, 75, 100, "None"]),
            "mvi__RandomForestClassifier__min_samples_split": CategoricalHyperparameter("mvi__RandomForestClassifier__min_samples_split", [2, 5, 10]),
            "mvi__RandomForestClassifier__min_samples_leaf": CategoricalHyperparameter("mvi__RandomForestClassifier__min_samples_leaf", [1, 2, 4]),
            "mvi__RandomForestClassifier__bootstrap": CategoricalHyperparameter("mvi__RandomForestClassifier__bootstrap", [True, False]),

            # RandomForestRegressor
            "mvi__RandomForestRegressor__n_estimators": UniformIntegerHyperparameter("mvi__RandomForestRegressor__n_estimators", 50, 200, q=50),
            "mvi__RandomForestRegressor__max_depth": CategoricalHyperparameter("mvi__RandomForestRegressor__max_depth", [10, 25, 50, 75, 100, "None"]),
            "mvi__RandomForestRegressor__min_samples_split": CategoricalHyperparameter("mvi__RandomForestRegressor__min_samples_split", [2, 5, 10]),
            "mvi__RandomForestRegressor__min_samples_leaf": CategoricalHyperparameter("mvi__RandomForestRegressor__min_samples_leaf", [1, 2, 4]),
            "mvi__RandomForestRegressor__bootstrap": CategoricalHyperparameter("mvi__RandomForestRegressor__bootstrap", [True, False]),
        }
    },
    ErrorRepairMethod.k_means_clustering.value: {
        "method": impute_with_kmeans,
        "default_kwargs": {},
        "config_space": {
            "mvi__n_clusters": UniformIntegerHyperparameter("mvi__n_clusters", 2, 10, q=1),
            "mvi__max_iter": CategoricalHyperparameter("mvi__max_iter", [100, 200]),
            "mvi__init": CategoricalHyperparameter("mvi__init", ["Huang", "Cao", "random"]),
            "mvi__n_init": CategoricalHyperparameter("mvi__n_init", [1, 5, 10]),
        }
    },
    ErrorRepairMethod.datawig.value: {
        "method": datawig_imputer.complete,
        "default_kwargs": {"precision_threshold": 0.0, "num_epochs": 100, "iterations": 1},
        "config_space": {
            "mvi__final_fc_hidden_units": CategoricalHyperparameter("mvi__final_fc_hidden_units", [1, 10, 50, 100]),
        }
    },
    # ErrorRepairMethod.automl.value: {
    #     "method": impute_with_automl,
    #     "default_kwargs": {"max_trials": 50, "tuner": None, "validation_split": 0.2, "epochs": 100},
    #     "config_space": {}
    # },
}

FAIRNESS_INTERVENTION_CONFIG_SPACE = {
    FairnessIntervention.DIR.value: {
        "fi__repair_level": UniformFloatHyperparameter("fi__repair_level", 0.1, 1.0),
    },
    # FairnessIntervention.LFR.value: {"k": 5, "Ax": 0.01, "Ay": 1.0, "Az": 50.0},
    # FairnessIntervention.AD.value: {"scope_name": "debiased_classifier",
    #                                 "adversary_loss_weight": 0.1, "num_epochs": 50, "batch_size": 128,
    #                                 "classifier_num_hidden_units": 200, "debias": True},
    # FairnessIntervention.EGR.value: {"constraints": "DemographicParity",
    #                                  "eps": 0.01, "max_iter": 50, "nu": None, "eta0": 2.0,
    #                                  "run_linprog_step": True, "drop_prot_attr": True,
    #                                  "estimator_params": {"C": 1.0, "solver": "lbfgs"}},
    # FairnessIntervention.EOP.value: {},
    # FairnessIntervention.ROC.value: {"low_class_thresh": 0.01, "high_class_thresh": 0.99, "num_class_thresh": 100,
    #                                  "num_ROC_margin": 50, "metric_name": "Statistical parity difference",
    #                                  "metric_ub": 0.05, "metric_lb": -0.05},
}


def get_models_params_for_tuning(models_tuning_seed: int = INIT_RANDOM_STATE):
    return {
        'dt_clf': {
            'model': DecisionTreeClassifier,
            'default_kwargs': {'random_state': models_tuning_seed},
            'config_space': {
                'model__max_depth': CategoricalHyperparameter("model__max_depth", [5, 10]),
                # "max_depth": [5, 10, 20, 30],
                # 'min_samples_leaf': [5, 10, 20, 50, 100],
                # "max_features": [0.6, 'sqrt'],
                # "criterion": ["gini", "entropy"]
            }
        },
        'lr_clf': {
            'model': LogisticRegression,
            'default_kwargs': {'random_state': models_tuning_seed, 'max_iter': 1000},
            'config_space': {
                'model__penalty': CategoricalHyperparameter("model__penalty", ['l2', 'None']),
                'model__C': CategoricalHyperparameter("model__C", [0.001, 0.01, 0.1, 1]),
                'model__solver': CategoricalHyperparameter("model__solver", ['newton-cg', 'lbfgs', 'sag', 'saga']),
            }
        },
        'rf_clf': {
            'model': RandomForestClassifier,
            'default_kwargs': {'random_state': models_tuning_seed},
            'config_space': {
                # 'model__n_estimators': CategoricalHyperparameter("model__n_estimators", [50, 100]),
                'model__n_estimators': CategoricalHyperparameter("model__n_estimators", [50, 100, 200, 500]),
                'model__max_depth': CategoricalHyperparameter("model__max_depth", [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 'None']),
                'model__min_samples_split': CategoricalHyperparameter("model__min_samples_split", [2, 5, 10]),
                'model__min_samples_leaf': CategoricalHyperparameter("model__min_samples_leaf", [1, 2, 4]),
                'model__bootstrap': CategoricalHyperparameter("model__bootstrap", [True, False]),
            }
        },
        'lgbm_clf': {
            'model': LGBMClassifier,
            'default_kwargs': {'random_state': models_tuning_seed, 'n_jobs': 48, 'num_threads': 48},
            'config_space': {
                'model__n_estimators': CategoricalHyperparameter("model__n_estimators", [50, 100, 200, 500]),
                'model__max_depth': CategoricalHyperparameter("model__max_depth", [3, 4, 5, 6, 7, 8, 9, -1]),
                'model__num_leaves': CategoricalHyperparameter("model__num_leaves", [int(x) for x in np.linspace(start = 20, stop = 3000, num = 8)]),
                'model__min_data_in_leaf': CategoricalHyperparameter("model__min_data_in_leaf", [int(x) for x in np.linspace(start = 100, stop = 1000, num = 8)]),
                'model__verbosity': CategoricalHyperparameter("model__verbosity", [-1]),
            }
        },
        ####################################################################
        # Use Pytorch Tabular API to work with tabular neural networks
        ####################################################################
        'gandalf_clf': {
            'model': GANDALFConfig,
            'default_kwargs': {'seed': models_tuning_seed, 'task': 'classification'},
            'optimizer_config': OptimizerConfig(),
            'trainer_config': TrainerConfig(batch_size=512,
                                            max_epochs=100,
                                            seed=models_tuning_seed,
                                            load_best=False,
                                            trainer_kwargs=dict(enable_model_summary=False, # Turning off model summary
                                                                log_every_n_steps=None,
                                                                enable_progress_bar=False)),
            'config_space': {
                'model__gflu_stages': CategoricalHyperparameter("model__gflu_stages", [i for i in range(2, 31)]),
                'model__gflu_dropout': CategoricalHyperparameter("model__gflu_dropout", [0.01 * i for i in range(6)]),
                'model__gflu_feature_init_sparsity': CategoricalHyperparameter("model__gflu_feature_init_sparsity", [0.1 * i for i in range(6)]),
                'model__learning_rate': CategoricalHyperparameter("model__learning_rate", [1e-3, 1e-4, 1e-5, 1e-6]),
            },
        },
    }
