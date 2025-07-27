import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMClassifier
from openbox.utils.config_space import (UniformIntegerHyperparameter, UniformFloatHyperparameter,
                                        CategoricalHyperparameter, Constant)
from pytorch_tabular.models import (GANDALFConfig, DANetModel, MDNModel, NODEModel, FTTransformerModel,
                                    GatedAdditiveTreeEnsembleModel)
from pytorch_tabular.config import OptimizerConfig, TrainerConfig
from xgboost import XGBClassifier

from .constants import FairnessIntervention, INIT_RANDOM_STATE, ErrorRepairMethod
import virny_flow.core.null_imputers.datawig_imputer as datawig_imputer
from virny_flow.core.null_imputers.imputation_methods import (impute_with_deletion, impute_with_simple_imputer,
                                                              impute_with_missforest, impute_with_kmeans, impute_with_automl)


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
    ErrorRepairMethod.automl.value: {
        "method": impute_with_automl,
        "default_kwargs": {"max_trials": 50, "tuner": None, "validation_split": 0.2, "epochs": 100},
        "config_space": {}
    },
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
        "fi__scope_name": Constant("fi__scope_name", "adversarial_debiasing"),
        "fi__adversary_loss_weight": UniformFloatHyperparameter("fi__adversary_loss_weight", 0.1, 0.5),
        "fi__num_epochs": UniformIntegerHyperparameter("fi__num_epochs", 50, 100, q=10),
        "fi__batch_size": CategoricalHyperparameter("fi__batch_size", [64, 128, 256]),
        "fi__classifier_num_hidden_units": CategoricalHyperparameter("fi__classifier_num_hidden_units", [100, 200, 300]),
        "fi__debias": CategoricalHyperparameter("fi__debias", [True]),
    },
    FairnessIntervention.EGR.value: {
        "fi__max_iter": UniformIntegerHyperparameter("fi__max_iter", 50, 100, q=10),
        "fi__run_linprog_step": CategoricalHyperparameter("fi__run_linprog_step", [True, False]),
        "fi__drop_prot_attr": CategoricalHyperparameter("fi__drop_prot_attr", [True]),
    },           
    FairnessIntervention.EOP.value: {},
    FairnessIntervention.ROC.value: {
        "fi__low_class_thresh": UniformFloatHyperparameter("fi__low_class_thresh", 0.01, 0.1),
        "fi__high_class_thresh": UniformFloatHyperparameter("fi__high_class_thresh", 0.9, 0.99),
        "fi__num_class_thresh": UniformIntegerHyperparameter("fi__num_class_thresh", 50, 100, q=10),
        "fi__num_ROC_margin": UniformIntegerHyperparameter("fi__num_ROC_margin", 20, 50, q=5),
        "fi__metric_name": CategoricalHyperparameter("fi__metric_name", ["Statistical parity difference", "Average odds difference", "Equal opportunity difference"]),
        "fi__metric_ub": UniformFloatHyperparameter("fi__metric_ub", 0.05, 0.1),
        "fi__metric_lb": UniformFloatHyperparameter("fi__metric_lb", -0.1, -0.05),
    }
}


def get_models_params_for_tuning(models_tuning_seed: int = INIT_RANDOM_STATE):
    return {
        'dt_clf': {
            'model': DecisionTreeClassifier,
            'default_kwargs': {'random_state': models_tuning_seed},
            'config_space': {
                'model__criterion': CategoricalHyperparameter("model__criterion", ["gini", "entropy"], default_value="gini"),
                'model__max_depth': UniformIntegerHyperparameter("model__max_depth", lower=1, upper=10, default_value=3),
                'model__min_samples_split': UniformIntegerHyperparameter("model__min_samples_split", 2, 20, default_value=2),
                'model__min_samples_leaf': UniformIntegerHyperparameter("model__min_samples_leaf", 1, 20, default_value=1),
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
            'default_kwargs': {'hidden_layer_sizes': (100,100,), 'random_state': models_tuning_seed, 'max_iter': 1000},
            'config_space': {
                'model__activation': CategoricalHyperparameter("model__activation", ['logistic', 'tanh', 'relu']),
                'model__solver': CategoricalHyperparameter("model__solver", ['lbfgs', 'sgd', 'adam']),
                'model__learning_rate': CategoricalHyperparameter("model__learning_rate", ['constant', 'invscaling', 'adaptive'])
            }
        },
        'xgb_clf': {
            'model': XGBClassifier,
            'default_kwargs': {'random_state': models_tuning_seed},
            'config_space': {
                'model__max_depth': UniformIntegerHyperparameter("model__max_depth", lower=1, upper=10, default_value=3),
                'model__learning_rate': UniformFloatHyperparameter("model__learning_rate", lower=0.01, upper=1, default_value=0.1, log=True),
                'model__n_estimators': UniformIntegerHyperparameter("model__n_estimators", 50, 500, default_value=100),
                'model__subsample': UniformFloatHyperparameter("model__subsample", lower=0.01, upper=1.0, default_value=1.0, log=False),
                'model__min_child_weight': UniformIntegerHyperparameter("model__min_child_weight", lower=1, upper=20, default_value=1, log=False),
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
        'knn_clf': {
            'model': KNeighborsClassifier,
            'default_kwargs': {},
            'config_space': {
                'model__n_neighbors': UniformIntegerHyperparameter("model__n_neighbors", lower=1, upper=30, default_value=5),
                'model__weights': CategoricalHyperparameter("model__weights", ["uniform", "distance"], default_value="uniform"),
                'model__p': CategoricalHyperparameter("model__p", [1, 2], default_value=2),  # 1 = manhattan, 2 = euclidean
            }
        },
        'svc_clf': {
            'model': SVC,
            'default_kwargs': {'random_state': models_tuning_seed, 'probability': True},
            'config_space': {
                'model__C': UniformFloatHyperparameter("model__C", lower=0.01, upper=10.0, default_value=1.0, log=True),
                'model__kernel': CategoricalHyperparameter("model__kernel", ["linear", "poly", "rbf", "sigmoid"]),
                'model__gamma': CategoricalHyperparameter("model__gamma", ["scale", "auto"]),
            }
        },
        'lr_reg': {
            'model': LinearRegression,
            'default_kwargs': {},
            'config_space': {
                'model__fit_intercept': CategoricalHyperparameter("model__fit_intercept", [True, False]),
                'model__normalize': CategoricalHyperparameter("model__normalize", [True, False]),  # deprecated in latest sklearn but kept for backward compatibility
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
                                            batch_size=512,
                                            max_epochs=100,
                                            seed=models_tuning_seed,
                                            early_stopping=None,
                                            checkpoints=None,
                                            load_best=False,
                                            trainer_kwargs=dict(enable_model_summary=False, # Turning off model summary
                                                                log_every_n_steps=None,
                                                                enable_progress_bar=True,
                                                                enable_checkpointing=False,
                                                                default_root_dir=None)),
            'config_space': {
                'model__gflu_stages': UniformIntegerHyperparameter("model__gflu_stages", 2, 30, q=1),
                'model__gflu_dropout': CategoricalHyperparameter("model__gflu_dropout", [0.01 * i for i in range(6)]),
                'model__gflu_feature_init_sparsity': CategoricalHyperparameter("model__gflu_feature_init_sparsity", [0.1 * i for i in range(6)]),
                'model__learning_rate': CategoricalHyperparameter("model__learning_rate", [1e-3, 1e-4, 1e-5, 1e-6]),
            },
        },
        'danet_clf': {
            'model': DANetModel,
            'default_kwargs': {'seed': models_tuning_seed, 'task': 'classification'},
            'optimizer_config': OptimizerConfig(),
            'trainer_config': TrainerConfig(accelerator="cpu",
                                            batch_size=512,
                                            max_epochs=100,
                                            seed=models_tuning_seed,
                                            early_stopping=None,
                                            checkpoints=None,
                                            load_best=False,
                                            trainer_kwargs=dict(enable_model_summary=False,
                                                                log_every_n_steps=None,
                                                                enable_progress_bar=True,
                                                                enable_checkpointing=False,
                                                                default_root_dir=None)),
            'config_space': {
                'model__num_layers': UniformIntegerHyperparameter("model__num_layers", 1, 10, q=1),
                'model__dropout': UniformFloatHyperparameter("model__dropout", 0.0, 0.5, q=0.05),
                'model__learning_rate': CategoricalHyperparameter("model__learning_rate", [1e-3, 1e-4, 1e-5]),
            }
        },
        'mdn_clf': {
            'model': MDNModel,
            'default_kwargs': {'seed': models_tuning_seed, 'task': 'classification'},
            'optimizer_config': OptimizerConfig(),
            'trainer_config': TrainerConfig(accelerator="cpu",
                                            batch_size=512,
                                            max_epochs=100,
                                            seed=models_tuning_seed,
                                            early_stopping=None,
                                            checkpoints=None,
                                            load_best=False,
                                            trainer_kwargs=dict(enable_model_summary=False,
                                                                log_every_n_steps=None,
                                                                enable_progress_bar=True,
                                                                enable_checkpointing=False,
                                                                default_root_dir=None)),
            'config_space': {
                'model__num_gaussian': UniformIntegerHyperparameter("model__num_gaussian", 1, 10, q=1),
                'model__learning_rate': CategoricalHyperparameter("model__learning_rate", [1e-3, 1e-4, 1e-5]),
            }
        },
        'node_clf': {
            'model': NODEModel,
            'default_kwargs': {'seed': models_tuning_seed, 'task': 'classification'},
            'optimizer_config': OptimizerConfig(),
            'trainer_config': TrainerConfig(accelerator="cpu",
                                            batch_size=512,
                                            max_epochs=100,
                                            seed=models_tuning_seed,
                                            early_stopping=None,
                                            checkpoints=None,
                                            load_best=False,
                                            trainer_kwargs=dict(enable_model_summary=False,
                                                                log_every_n_steps=None,
                                                                enable_progress_bar=True,
                                                                enable_checkpointing=False,
                                                                default_root_dir=None)),
            'config_space': {
                'model__num_layers': UniformIntegerHyperparameter("model__num_layers", 1, 5, q=1),
                'model__num_trees': UniformIntegerHyperparameter("model__num_trees", 32, 512, q=32),
                'model__learning_rate': CategoricalHyperparameter("model__learning_rate", [1e-3, 1e-4, 1e-5]),
            }
        },
        'ftt_clf': {
            'model': FTTransformerModel,
            'default_kwargs': {'seed': models_tuning_seed, 'task': 'classification'},
            'optimizer_config': OptimizerConfig(),
            'trainer_config': TrainerConfig(accelerator="cpu",
                                            batch_size=512,
                                            max_epochs=100,
                                            seed=models_tuning_seed,
                                            early_stopping=None,
                                            checkpoints=None,
                                            load_best=False,
                                            trainer_kwargs=dict(enable_model_summary=False,
                                                                log_every_n_steps=None,
                                                                enable_progress_bar=True,
                                                                enable_checkpointing=False,
                                                                default_root_dir=None)),
            'config_space': {
                'model__attn_dropout': UniformFloatHyperparameter("model__attn_dropout", 0.0, 0.5, q=0.05),
                'model__learning_rate': CategoricalHyperparameter("model__learning_rate", [1e-3, 1e-4, 1e-5]),
                'model__num_heads': CategoricalHyperparameter("model__num_heads", [1, 2, 4, 8]),
            }
        },
        'gate_clf': {
            'model': GatedAdditiveTreeEnsembleModel,
            'default_kwargs': {'seed': models_tuning_seed, 'task': 'classification'},
            'optimizer_config': OptimizerConfig(),
            'trainer_config': TrainerConfig(accelerator="cpu",
                                            batch_size=512,
                                            max_epochs=100,
                                            seed=models_tuning_seed,
                                            early_stopping=None,
                                            checkpoints=None,
                                            load_best=False,
                                            trainer_kwargs=dict(enable_model_summary=False,
                                                                log_every_n_steps=None,
                                                                enable_progress_bar=True,
                                                                enable_checkpointing=False,
                                                                default_root_dir=None)),
            'config_space': {
                'model__num_trees': UniformIntegerHyperparameter("model__num_trees", 64, 1024, q=64),
                'model__tree_depth': UniformIntegerHyperparameter("model__tree_depth", 2, 6, q=1),
                'model__learning_rate': CategoricalHyperparameter("model__learning_rate", [1e-2, 1e-3, 1e-4]),
            }
        },
    }
