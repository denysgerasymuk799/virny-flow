import numpy as np
from pprint import pprint
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier

from openbox.utils.config_space import UniformFloatHyperparameter
from virny_flow.configs.constants import FairnessIntervention


FAIRNESS_INTERVENTION_CONFIG_SPACE = {
    FairnessIntervention.DIR.value: {
        "repair_level": UniformFloatHyperparameter("repair_level", 0.1, 1.0)
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


if __name__ == '__main__':
    model_tuning_params = get_models_params_for_tuning(models_tuning_seed=200)
    pprint(model_tuning_params)
