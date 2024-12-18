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
    FairnessIntervention.LFR.value: {
        "fi__k": CategoricalHyperparameter("fi__k", [5]),
        "fi__Ax": CategoricalHyperparameter("fi__Ax", [0.01]),
        "fi__Ay": CategoricalHyperparameter("fi__Ay", [0.1, 0.5, 1.0, 5.0, 10.0]),
        "fi__Az": CategoricalHyperparameter("fi__Az", [0.0, 0.1, 0.5, 1.0, 5.0, 10.0]),
    },
    FairnessIntervention.AD.value: {
        "fi__scope_name": CategoricalHyperparameter("fi__scope_name", ["adversarial_debiasing"]),
        "fi__adversary_loss_weight": UniformFloatHyperparameter("fi__adversary_loss_weight", 0.1, 0.5),
        "fi__num_epochs": UniformIntegerHyperparameter("fi__num_epochs", 50, 100, q=10),
        "fi__batch_size": CategoricalHyperparameter("fi__batch_size", [64, 128, 256]),
        "fi__classifier_num_hidden_units": CategoricalHyperparameter("fi__classifier_num_hidden_units", [100, 200, 300]),
        "fi__debias": CategoricalHyperparameter("fi__debias", [True, False]),
    },
    FairnessIntervention.EGR.value: {
        "fi__constraints": CategoricalHyperparameter("fi__constraints", ["DemographicParity", "EqualizedOdds"]),
        "fi__eps": UniformFloatHyperparameter("fi__eps", 0.01, 0.1),
        "fi__max_iter": UniformIntegerHyperparameter("fi__max_iter", 50, 100, q=10),
        "fi__nu": CategoricalHyperparameter("fi__nu", [0.1, 0.2, 0.3]),
        "fi__eta0": UniformFloatHyperparameter("fi__eta0", 1.0, 2.0),
        "fi__run_linprog_step": CategoricalHyperparameter("fi__run_linprog_step", [True, False]),
        "fi__drop_prot_attr": CategoricalHyperparameter("fi__drop_prot_attr", [True, False]),
    },           
    FairnessIntervention.EOP.value: {},
    FairnessIntervention.ROC.value: {
        "low_class_thresh": UniformFloatHyperparameter("low_class_thresh", 0.01, 0.1),
        "high_class_thresh": UniformFloatHyperparameter("high_class_thresh", 0.9, 0.99),
        "num_class_thresh": UniformIntegerHyperparameter("num_class_thresh", 50, 100, q=10),
        "num_ROC_margin": UniformIntegerHyperparameter("num_ROC_margin", 20, 50, q=5),
        "metric_name": CategoricalHyperparameter("metric_name", ["Statistical parity difference", "Average odds difference", "Equal opportunity difference"]),
        "metric_ub": UniformFloatHyperparameter("metric_ub", 0.05, 0.1),
        "metric_lb": UniformFloatHyperparameter("metric_lb", -0.1, -0.05),
    }
}


def get_models_params_for_tuning(models_tuning_seed: int = INIT_RANDOM_STATE):
    return {
        'dt_clf': {
            'model': DecisionTreeClassifier,
            'default_kwargs': {'random_state': models_tuning_seed},
            'config_space': {
                'model__max_depth': CategoricalHyperparameter("model__max_depth", [5, 10]),
                'model__min_samples_leaf': CategoricalHyperparameter("model__min_samples_leaf", [5, 10, 20, 50, 100]),
                'model__max_features': CategoricalHyperparameter("model__max_features", [0.6, 'sqrt']),
                'model__criterion': CategoricalHyperparameter("model__criterion", ["gini", "entropy"])
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
                'model__n_estimators': UniformIntegerHyperparameter("model__n_estimators", 50, 1000, q=50),
                'model__max_depth': CategoricalHyperparameter("model__max_depth", [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 'None']),
                'model__min_samples_split': UniformIntegerHyperparameter("model__min_samples_split", 2, 10, q=1),
                'model__min_samples_leaf': UniformIntegerHyperparameter("model__min_samples_leaf", 1, 5, q=1),
                'model__bootstrap': CategoricalHyperparameter("model__bootstrap", [True, False]),
            }
        },
        'mlp_clf': {
            'model': MLPClassifier,
            'default_kwargs': {'hidden_layer_sizes': (100,100,). 'random_state': models_tuning_seed. 'max_iter': 1000},
            'config_space': {
                'model__activation': CategoricalHyperparameter("model__activation", ['logistic', 'tanh', 'relu']),
                'model__solver': CategoricalHyperparameter("model__solver", ['lbfgs', 'sgd', 'adam']),
                'model__learning_rate': CategoricalHyperparameter("model__learning_rate", ['constant', 'invscaling', 'adaptive'])
            }
        },
        'lgbm_clf': {
            'model': LGBMClassifier,
            'default_kwargs': {'random_state': models_tuning_seed, 'n_jobs': 8, 'num_threads': 8, 'verbosity': -1},
            'config_space': {
                'model__n_estimators': UniformIntegerHyperparameter("model__n_estimators", 50, 1000, q=50),
                'model__max_depth': CategoricalHyperparameter("model__max_depth", [3, 4, 5, 6, 7, 8, 9, -1]),
                'model__num_leaves': CategoricalHyperparameter("model__num_leaves", [int(x) for x in np.linspace(start = 20, stop = 3000, num = 20)]),
                'model__min_data_in_leaf': UniformIntegerHyperparameter("model__min_data_in_leaf", 100, 1000, q=50),
            }
        },
        ####################################################################
        # Use Pytorch Tabular API to work with tabular neural networks
        ####################################################################
        'gandalf_clf': {
            'model': GANDALFConfig,
            'default_kwargs': {'seed': models_tuning_seed, 'task': 'classification'},
            'optimizer_config': OptimizerConfig(),
            'trainer_config': TrainerConfig(accelerator="cpu",
                                            batch_size=256,
                                            max_epochs=100,
                                            seed=models_tuning_seed,
                                            early_stopping=None,
                                            checkpoints=None,
                                            load_best=False,
                                            trainer_kwargs=dict(enable_model_summary=False, # Turning off model summary
                                                                log_every_n_steps=None,
                                                                enable_progress_bar=False)),
            'config_space': {
                'model__gflu_stages': UniformIntegerHyperparameter("model__gflu_stages", 2, 30, q=1),
                'model__gflu_dropout': CategoricalHyperparameter("model__gflu_dropout", [0.01 * i for i in range(6)]),
                'model__gflu_feature_init_sparsity': CategoricalHyperparameter("model__gflu_feature_init_sparsity", [0.1 * i for i in range(6)]),
                'model__learning_rate': CategoricalHyperparameter("model__learning_rate", [1e-3, 1e-4, 1e-5, 1e-6]),
            },
        },
    }
