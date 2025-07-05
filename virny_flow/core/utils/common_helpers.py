import copy
import hashlib
import secrets
import base64
import yaml
import pandas as pd

from datetime import datetime
from pprint import pprint
from cerberus import Validator
from munch import DefaultMunch
from virny.custom_classes.base_dataset import BaseFlowDataset
from virny.utils.common_helpers import validate_config as validate_virny_config
from virny.configs.constants import ComputationMode

from virny_flow.core.custom_classes.core_db_client import CoreDBClient
from virny_flow.configs.constants import STABILITY_AND_UNCERTAINTY_METRICS, PHYSICAL_PIPELINE_OBSERVATIONS_TABLE, \
    LOGICAL_PIPELINE_SCORES_TABLE, EXP_CONFIG_HISTORY_TABLE


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
    config_obj.optimisation_args.objectives = [dict(obj) for obj in config_obj.optimisation_args.objectives]
    config_obj.virny_args.dataset_name = config_obj.pipeline_args.dataset
    validate_virny_config(config_obj.virny_args)

    return config_obj


def validate_config(config_obj):
    """
    Validate parameters types, values, ranges and set optional parameters in config yaml file.
    """
    # Define the schema for the configuration
    schema = {
        "common_args": {
            "type": "dict",
            "schema": {
                "exp_config_name": {"type": "string", "required": True},
                "num_runs": {"type": "integer", "required": False, "min": 1},
                "run_nums": {"type": "list", "required": False, "schema": {"type": "integer", "min": 1}},
                "save_storage": {"type": "boolean", "required": False, "default": False},
                "secrets_path": {"type": "string", "required": True},
            }
        },
        "pipeline_args": {
            "type": "dict",
            "schema": {
                "dataset": {"type": "string", "required": True},
                "sensitive_attrs_for_intervention": {"type": "list", "required": True},
                "null_imputers": {"type": "list", "required": True},
                "fairness_interventions": {"type": "list", "required": True},
                "models": {"type": "list", "required": True},
            }
        },
        "optimisation_args": {
            "type": "dict",
            "schema": {
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
                            "constraint": {"type": "list", "required": False, "schema": {"type": "string"}}
                        }
                    }
                },
                "max_trials": {"type": "integer", "min": 1, "default": 1000},
                "max_time_budget": {"type": "integer", "min": 60},
                "max_total_pipelines_num": {"type": "integer", "min": 1},
                "num_workers": {"type": "integer", "min": 1, "required": True},
                "num_pp_candidates": {"type": "integer", "min": 1, "required": False, "default": 10},
                "queue_size": {"type": "integer", "min": config_obj['optimisation_args']['num_workers'],
                               "required": False, "default": 3 * config_obj['optimisation_args']['num_workers']},
                "training_set_fractions_for_halting": {"type": "list", "required": False, "default": [0.5, 0.75, 1.0]},
                "exploration_factor": {"type": "float", "min": 0.0, "max": 1.0, "required": False, "default": 0.5},
                "risk_factor": {"type": "float", "min": 0.0, "max": 1.0, "required": False, "default": 0.5},
            }
        },
        "virny_args": {
            "type": "dict",
            "schema": {
                "bootstrap_fraction": {"type": "float", "min": 0.0, "max": 1.0, "required": False},
                "n_estimators": {"type": "integer", "min": 2, "required": False},
                "sensitive_attributes_dct": {"type": "dict", "allow_unknown": True, "schema": {}, "required": True},
            }
        },
    }

    # Initialize the validator
    v = Validator(schema)

    # Validate the configuration
    if v.validate(config_obj):
        config_obj = v.normalized(config_obj)  # Enforce defaults
    else:
        raise ValueError("Validation errors in exp_config.yaml:", v.errors)

    # Other checks
    if (config_obj["common_args"].get("num_runs", None) is not None and
            config_obj["common_args"].get("run_nums", None) is not None):
        raise ValueError("Only one of two arguments (num_runs, run_nums) should be defined in a config.")
    if (config_obj["common_args"].get("num_runs", None) is None and
            config_obj["common_args"].get("run_nums", None) is None):
        raise ValueError("One of two arguments (num_runs, run_nums) should be defined in a config.")

    # Disable bootstrap if stability and uncertainty metrics are not in objectives
    objective_names = set([objective["metric"].strip().lower() for objective in config_obj["optimisation_args"]["objectives"]])
    if len(objective_names.intersection(set([m.lower() for m in STABILITY_AND_UNCERTAINTY_METRICS]))) == 0:
        config_obj['virny_args']['computation_mode'] = ComputationMode.NO_BOOTSTRAP.value
    else:
        print('Enable bootstrap in Virny')
        if config_obj["virny_args"].get("bootstrap_fraction", None) is None \
                    or config_obj['virny_args']['bootstrap_fraction'] < 0.0 \
                    or config_obj['virny_args']['bootstrap_fraction'] > 1.0:
            raise ValueError('virny_args.bootstrap_fraction must be float in [0.0, 1.0] range')

        if config_obj["virny_args"].get("n_estimators", None) is None \
                or config_obj['virny_args']['n_estimators'] <= 1:
            raise ValueError('virny_args.n_estimators must be integer greater than 1')

    objective_total_weight = 0.0
    for objective in config_obj['optimisation_args']['objectives']:
        objective_total_weight += objective['weight']

    if not (0.99 <= objective_total_weight <= 1.0):
        raise ValueError("Objective weights must sum to 1.0")

    if config_obj['optimisation_args']['num_workers'] < config_obj['optimisation_args']['num_pp_candidates']:
        raise ValueError("The number of workers should be greater or equal than the number of physical pipeline candidates for each round")

    # Default arguments
    if len(config_obj['pipeline_args']['null_imputers']) == 0:
        config_obj['pipeline_args']['null_imputers'] = ['None']

    if config_obj["common_args"].get("num_runs") is not None:
        config_obj["common_args"]["run_nums"] = [run_num for run_num in range(1, config_obj["common_args"]["num_runs"] + 1)]

    if config_obj["optimisation_args"].get("max_total_pipelines_num") is not None:
        config_obj["optimisation_args"]["max_trials"] = config_obj["optimisation_args"]["max_total_pipelines_num"]

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


def flatten_dict(d, parent_key='', sep='.'):
    """
    Flattens a nested dictionary into a single-level dictionary.

    Args:
    - d (dict): The dictionary to flatten.
    - parent_key (str): The base key to prefix (used in recursion).
    - sep (str): Separator for concatenating nested keys.

    Returns:
    - dict: A flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def transform_row_to_observation(row: dict):
    """Transform a MongoDB row into the desired observation format.
    
    Args:
    - row (dict): A MongoDB row
    Returns:
    - dict: An observation
    """
    return {
        "config": row["config"],
        "objectives": row["objectives"],
        "constraints": row.get("constraints"),
        "trial_state": row["trial_state"],
        "elapsed_time": row["elapsed_time"],
        "create_time": row["create_time"],
        "extra_info": {
            "objectives": row["extra_info"].get("exp_config_objectives", [])
        }
    }


def read_history_from_db(secrets_path: str, exp_config_name: str, lp_name: str, run_num: int, ref_point = [0.2, 0.1]):
    """Fetch all rows and transform them into the final desired structure.
    
    Args:
    - secrets_path (str): Path to the secrets file for the database connection.
    - exp_config_name (str): Name of the experiment config to get observations from.
    - lp_name (str): Name of the logical pipeline to get observations from.
    """
    db = CoreDBClient(secrets_path)
    db.connect()

    exp_config = db.read_one_query(collection_name=EXP_CONFIG_HISTORY_TABLE,
                                   query={"exp_config_name": exp_config_name,
                                          "run_num": run_num})
    defined_objectives = exp_config['optimisation_args.objectives']
    print("defined_objectives:", defined_objectives)

    logical_pipeline_metadata = db.read_one_query(collection_name=LOGICAL_PIPELINE_SCORES_TABLE,
                                                  query={"exp_config_name": exp_config_name,
                                                         "logical_pipeline_name": lp_name,
                                                         "run_num": run_num})
    surrogate_model_type = logical_pipeline_metadata['surrogate_type']
    print("surrogate_model_type:", surrogate_model_type)

    all_rows = db.execute_read_query(collection_name=PHYSICAL_PIPELINE_OBSERVATIONS_TABLE,
                                     query={"exp_config_name": exp_config_name,
                                            "logical_pipeline_name": lp_name,
                                            "run_num": run_num,
                                            "deletion_flag": False})
    
    # Transform each row into an observation
    observations = [transform_row_to_observation(row) for row in all_rows]
    
    # Final structured output
    structured_data = {
        "task_id": "OpenBox",
        "num_objectives": len(observations[0]["objectives"]) if observations else 0,
        "num_constraints": 0,
        "ref_point": ref_point,  # Default or computed ref point
        "meta_info": {},
        "global_start_time": datetime.now().isoformat(),
        "observations": observations,
    }
        
    db.close()
    
    return structured_data, defined_objectives, surrogate_model_type
