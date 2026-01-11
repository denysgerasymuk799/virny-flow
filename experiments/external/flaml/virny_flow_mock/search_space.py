import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from flaml import tune


def get_model_search_space():
    """
    Define hyperparameter search space for multiple model types.
    Using a flattened structure to ensure proper sampling across all model types.
    """
    search_space = {
        "model_type": tune.choice(["dt_clf", "lr_clf", "rf_clf", "xgb_clf", "lgbm_clf"]),
        # Decision Tree parameters
        "dt_criterion": tune.choice(["gini", "entropy"]),
        "dt_max_depth": tune.randint(1, 10),
        "dt_min_samples_split": tune.randint(2, 20),
        "dt_min_samples_leaf": tune.randint(1, 20),
        # Logistic Regression parameters
        "lr_penalty": tune.choice(['l2', None]),
        "lr_C": tune.choice([0.001, 0.01, 0.1, 1]),
        "lr_solver": tune.choice(['newton-cg', 'lbfgs', 'sag', 'saga']),
        # Random Forest parameters
        "rf_n_estimators": tune.qrandint(50, 1000, 50),
        "rf_max_depth": tune.choice([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]),
        "rf_min_samples_split": tune.randint(2, 10),
        "rf_min_samples_leaf": tune.randint(1, 5),
        "rf_bootstrap": tune.choice([True, False]),
        # XGBoost parameters
        "xgb_max_depth": tune.randint(1, 10),
        "xgb_learning_rate": tune.loguniform(0.01, 1),
        "xgb_n_estimators": tune.randint(50, 500),
        "xgb_subsample": tune.uniform(0.01, 1.0),
        "xgb_min_child_weight": tune.randint(1, 20),
        # LightGBM parameters
        "lgbm_n_estimators": tune.qrandint(50, 1000, 50),
        "lgbm_max_depth": tune.choice([3, 4, 5, 6, 7, 8, 9, -1]),
        "lgbm_num_leaves": tune.choice([int(x) for x in np.linspace(start = 20, stop = 3000, num = 20)]),
        "lgbm_min_data_in_leaf": tune.qrandint(100, 1000, 50),
    }

    return search_space


def create_model_from_config(config, preprocessor, random_state):
    """
    Create a model pipeline based on config.
    Works with flattened search space structure.
    """
    params = config
    model_type = config["model_type"]

    # 1. Manual Parameter Assignment using If-Statements
    if model_type == 'dt_clf':
        model = DecisionTreeClassifier(
            criterion=params["dt_criterion"],
            max_depth=params["dt_max_depth"],
            min_samples_split=params["dt_min_samples_split"],
            min_samples_leaf=params["dt_min_samples_leaf"],
            random_state=random_state
        )

    elif model_type == 'lr_clf':
        model = LogisticRegression(
            penalty=params["lr_penalty"],
            C=params["lr_C"],
            solver=params["lr_solver"],
            random_state=random_state,
            max_iter=1000
        )

    elif model_type == 'rf_clf':
        model = RandomForestClassifier(
            n_estimators=params["rf_n_estimators"],
            max_depth=params["rf_max_depth"],
            min_samples_split=params["rf_min_samples_split"],
            min_samples_leaf=params["rf_min_samples_leaf"],
            bootstrap=params["rf_bootstrap"],
            random_state=random_state
        )

    elif model_type == 'xgb_clf':
        model = XGBClassifier(
            max_depth=params["xgb_max_depth"],
            learning_rate=params["xgb_learning_rate"],
            n_estimators=params["xgb_n_estimators"],
            subsample=params["xgb_subsample"],
            min_child_weight=params["xgb_min_child_weight"],
            random_state=random_state
        )

    elif model_type == 'lgbm_clf':
        model = LGBMClassifier(
            n_estimators=params["lgbm_n_estimators"],
            max_depth=params["lgbm_max_depth"],
            num_leaves=params["lgbm_num_leaves"],
            min_data_in_leaf=params["lgbm_min_data_in_leaf"],
            random_state=random_state,
            n_jobs=8,
            num_threads=8,
            verbosity=-1
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return Pipeline([("preprocessor", preprocessor), ("model", model)])
