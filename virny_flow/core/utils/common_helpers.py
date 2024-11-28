import copy
import hashlib
import secrets
import base64
import yaml
import pandas as pd

from pprint import pprint
from cerberus import Validator
from munch import DefaultMunch
from virny.custom_classes.base_dataset import BaseFlowDataset


def generate_guid(ordered_hierarchy_lst: list):
    identifier = '|'.join([str(val) for val in ordered_hierarchy_lst])
    return hashlib.md5(identifier.encode()).hexdigest()


def generate_base64_hash(length=8):
    # Generate random bytes. Since Base64 encodes each 3 bytes into 4 characters, calculate bytes needed.
    bytes_needed = (length * 3) // 4
    random_bytes = secrets.token_bytes(bytes_needed)

    # Encode bytes in base64 and decode to utf-8 string
    random_hash = base64.urlsafe_b64encode(random_bytes).decode('utf-8')

    # Return the required length
    return random_hash[:length]


def create_exp_config_obj(exp_config_yaml_path: str):
    """
    Return a config object created based on a exp config yaml file.

    Parameters
    ----------
    exp_config_yaml_path
        Path to a config yaml file

    """
    with open(exp_config_yaml_path) as f:
        config_dct = yaml.load(f, Loader=yaml.FullLoader)

    config_dct = validate_config(config_dct)
    print('Input experiment config:')
    pprint(config_dct)
    print()
    config_obj = DefaultMunch.fromDict(config_dct)
    config_obj.objectives = [dict(obj) for obj in config_obj.objectives]

    return config_obj


def validate_config(config_obj):
    """
    Validate parameters types, values, ranges and set optional parameters in config yaml file.
    """
    # Define the schema for the configuration
    schema = {
        # General experiment parameters
        "exp_config_name": {"type": "string", "required": True},
        "dataset": {"type": "string", "required": True},
        "sensitive_attrs_for_intervention": {"type": "list", "required": True},
        "null_imputers": {"type": "list", "required": True},
        "fairness_interventions": {"type": "list", "required": True},
        "models": {"type": "list", "required": True},
        "random_state": {"type": "integer", "min": 1, "required": True},
        "secrets_path": {"type": "string", "required": True},

        # Parameters for multi-objective optimisation
        "ref_point": {"type": "list", "required": False},
        "objectives": {
            "type": "list",
            "required": True,
            "schema": {  # Schema for each dictionary in the list
                "type": "dict",
                "schema": {
                    "name": {"type": "string", "required": True},
                    "metric": {"type": "string", "required": True},
                    "group": {"type": "string", "required": True},
                    "weight": {"type": "float", "required": False, "default": 0.5},
                }
            }
        },
        "max_trials": {"type": "integer", "min": 1, "required": True},
        "num_workers": {"type": "integer", "min": 1, "required": True},
        "num_pp_candidates": {"type": "integer", "min": 2, "required": False, "default": 10},
        "queue_size": {"type": "integer", "min": config_obj['num_workers'],
                       "required": False, "default": 3 * config_obj['num_workers']},
        "training_set_fractions_for_halting": {"type": "list", "required": False, "default": [0.5, 0.75, 1.0]},
        "exploration_factor": {"type": "float", "min": 0.0, "max": 1.0, "required": False, "default": 0.5},
        "risk_factor": {"type": "float", "min": 0.0, "max": 1.0, "required": False, "default": 0.5},
    }

    # Initialize the validator
    v = Validator(schema)

    # Validate the configuration
    if v.validate(config_obj):
        config_obj = v.normalized(config_obj)  # Enforce defaults
    else:
        raise ValueError("Validation errors in exp_config.yaml:", v.errors)

    # Default arguments
    if len(config_obj['null_imputers']) == 0:
        config_obj['null_imputers'] = ['None']

    # Other checks
    objective_total_weight = 0.0
    for objective in config_obj['objectives']:
        objective_total_weight += objective['weight']

    if objective_total_weight != 1.0:
        raise ValueError("Objective weights must sum to 1.0")

    if config_obj['num_workers'] >= config_obj['num_pp_candidates']:
        raise ValueError("Number of workers should be much larger than number of physical pipeline candidates for each round")

    return config_obj


def create_base_flow_dataset(data_loader, dataset_sensitive_attrs,
                             X_train_val_wo_sensitive_attrs, X_test_wo_sensitive_attrs,
                             y_train_val, y_test, numerical_columns_wo_sensitive_attrs,
                             categorical_columns_wo_sensitive_attrs):
    # Create a dataframe with sensitive attributes and initial dataset indexes
    sensitive_attrs_df = data_loader.full_df[dataset_sensitive_attrs]

    # Ensure correctness of indexes in X and sensitive_attrs sets
    if X_train_val_wo_sensitive_attrs is not None:
        assert X_train_val_wo_sensitive_attrs.index.isin(sensitive_attrs_df.index).all(), \
            "Not all indexes of X_train_val_wo_sensitive_attrs are present in sensitive_attrs_df"
    assert X_test_wo_sensitive_attrs.index.isin(sensitive_attrs_df.index).all(), \
        "Not all indexes of X_test_wo_sensitive_attrs are present in sensitive_attrs_df"

    # Ensure correctness of indexes in X and y sets
    if X_train_val_wo_sensitive_attrs is not None and y_train_val is not None:
        assert X_train_val_wo_sensitive_attrs.index.equals(y_train_val.index) is True, \
            "Indexes of X_train_val_wo_sensitive_attrs and y_train_val are different"
    assert X_test_wo_sensitive_attrs.index.equals(y_test.index) is True, \
        "Indexes of X_test_wo_sensitive_attrs and y_test are different"

    return BaseFlowDataset(init_sensitive_attrs_df=sensitive_attrs_df,  # keep only sensitive attributes with original indexes to compute group metrics
                           X_train_val=X_train_val_wo_sensitive_attrs,
                           X_test=X_test_wo_sensitive_attrs,
                           y_train_val=y_train_val,
                           y_test=y_test,
                           target=data_loader.target,
                           numerical_columns=numerical_columns_wo_sensitive_attrs,
                           categorical_columns=categorical_columns_wo_sensitive_attrs)


def create_virny_base_flow_datasets(data_loader, dataset_sensitive_attrs,
                                    X_train_val_wo_sensitive_attrs, X_tests_wo_sensitive_attrs_lst,
                                    y_train_val, y_test, numerical_columns_wo_sensitive_attrs,
                                    categorical_columns_wo_sensitive_attrs):
    main_X_test_wo_sensitive_attrs, extra_X_tests_wo_sensitive_attrs_lst = \
        X_tests_wo_sensitive_attrs_lst[0], X_tests_wo_sensitive_attrs_lst[1:]

    # Create a main base flow dataset for Virny
    main_base_flow_dataset = create_base_flow_dataset(data_loader=copy.deepcopy(data_loader),
                                                      dataset_sensitive_attrs=dataset_sensitive_attrs,
                                                      X_train_val_wo_sensitive_attrs=X_train_val_wo_sensitive_attrs,
                                                      X_test_wo_sensitive_attrs=main_X_test_wo_sensitive_attrs,
                                                      y_train_val=y_train_val,
                                                      y_test=copy.deepcopy(y_test),
                                                      numerical_columns_wo_sensitive_attrs=numerical_columns_wo_sensitive_attrs,
                                                      categorical_columns_wo_sensitive_attrs=categorical_columns_wo_sensitive_attrs)

    # Create extra base flow datasets for Virny
    extra_base_flow_datasets = list(map(
        lambda extra_X_test_wo_sensitive_attrs: \
            create_base_flow_dataset(data_loader=copy.deepcopy(data_loader),
                                     dataset_sensitive_attrs=dataset_sensitive_attrs,
                                     X_train_val_wo_sensitive_attrs=pd.DataFrame(),
                                     X_test_wo_sensitive_attrs=extra_X_test_wo_sensitive_attrs,
                                     y_train_val=pd.DataFrame(),
                                     y_test=copy.deepcopy(y_test),
                                     numerical_columns_wo_sensitive_attrs=numerical_columns_wo_sensitive_attrs,
                                     categorical_columns_wo_sensitive_attrs=categorical_columns_wo_sensitive_attrs),
        extra_X_tests_wo_sensitive_attrs_lst
    ))

    return main_base_flow_dataset, extra_base_flow_datasets
