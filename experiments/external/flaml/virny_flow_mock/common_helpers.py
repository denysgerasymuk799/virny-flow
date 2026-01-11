from virny.utils.common_helpers import validate_config as validate_virny_config
from munch import DefaultMunch
from cerberus import Validator
from pprint import pprint
import yaml

from virny.configs.constants import *


STABILITY_AND_UNCERTAINTY_METRICS = [
    STD,
    IQR,
    JITTER,
    LABEL_STABILITY,
    ALEATORIC_UNCERTAINTY,
    EPISTEMIC_UNCERTAINTY,
    OVERALL_UNCERTAINTY,
    LABEL_STABILITY_RATIO,
    LABEL_STABILITY_DIFFERENCE,
    IQR_DIFFERENCE,
    STD_DIFFERENCE,
    STD_RATIO,
    JITTER_DIFFERENCE,
    OVERALL_UNCERTAINTY_DIFFERENCE,
    OVERALL_UNCERTAINTY_RATIO,
    EPISTEMIC_UNCERTAINTY_DIFFERENCE,
    EPISTEMIC_UNCERTAINTY_RATIO,
    ALEATORIC_UNCERTAINTY_DIFFERENCE,
    ALEATORIC_UNCERTAINTY_RATIO,
]

# Key - metric name from Virny, value - an operation to create a loss for minimization
METRIC_TO_LOSS_ALIGNMENT = {
    # Accuracy metrics
    F1: "reverse",
    ACCURACY: "reverse",
    # Stability metrics
    STD: None,
    IQR: None,
    JITTER: None,
    LABEL_STABILITY: "reverse",
    # Uncertainty metrics
    ALEATORIC_UNCERTAINTY: None,
    EPISTEMIC_UNCERTAINTY: None,
    OVERALL_UNCERTAINTY: None,
    # Error disparity metrics
    EQUALIZED_ODDS_TPR: "abs",
    EQUALIZED_ODDS_TNR: "abs",
    EQUALIZED_ODDS_FPR: "abs",
    EQUALIZED_ODDS_FNR: "abs",
    DISPARATE_IMPACT: "reverse&abs",
    STATISTICAL_PARITY_DIFFERENCE: "abs",
    ACCURACY_DIFFERENCE: "abs",
    # Stability disparity metrics
    LABEL_STABILITY_RATIO: "reverse&abs",
    LABEL_STABILITY_DIFFERENCE: "abs",
    IQR_DIFFERENCE: "abs",
    STD_DIFFERENCE: "abs",
    STD_RATIO: "reverse&abs",
    JITTER_DIFFERENCE: "abs",
    # Uncertainty disparity metrics
    OVERALL_UNCERTAINTY_DIFFERENCE: "abs",
    OVERALL_UNCERTAINTY_RATIO: "reverse&abs",
    EPISTEMIC_UNCERTAINTY_DIFFERENCE: "abs",
    EPISTEMIC_UNCERTAINTY_RATIO: "reverse&abs",
    ALEATORIC_UNCERTAINTY_DIFFERENCE: "abs",
    ALEATORIC_UNCERTAINTY_RATIO: "reverse&abs",
}


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
    print("Input experiment config:")
    pprint(config_dct)
    print()
    config_obj = DefaultMunch.fromDict(config_dct)
    config_obj.optimisation_args.objectives = [
        dict(obj) for obj in config_obj.optimisation_args.objectives
    ]
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
                "run_nums": {
                    "type": "list",
                    "required": False,
                    "schema": {"type": "integer", "min": 1},
                },
                "save_storage": {
                    "type": "boolean",
                    "required": False,
                    "default": False,
                },
                "secrets_path": {"type": "string", "required": True},
            },
        },
        "pipeline_args": {
            "type": "dict",
            "schema": {
                "dataset": {"type": "string", "required": True},
                "sensitive_attrs_for_intervention": {"type": "list", "required": True},
                "null_imputers": {"type": "list", "required": True},
                "fairness_interventions": {"type": "list", "required": True},
                "models": {"type": "list", "required": True},
            },
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
                            "weight": {
                                "type": "float",
                                "required": False,
                                "default": 0.5,
                            },
                            "constraint": {
                                "type": "list",
                                "required": False,
                                "schema": {"type": "string"},
                            },
                        },
                    },
                },
                "optimizer": {
                    "type": "dict",
                    "required": False,
                    "default": {
                        "surrogate_type": "auto",
                        "acq_type": "auto",
                        "acq_optimizer_type": "auto",
                    },
                    "schema": {
                        "surrogate_type": {
                            "type": "string",
                            "required": False,
                            "default": "auto",
                        },
                        "acq_type": {
                            "type": "string",
                            "required": False,
                            "default": "auto",
                        },
                        "acq_optimizer_type": {
                            "type": "string",
                            "required": False,
                            "default": "auto",
                        },
                    },
                },
                "max_trials": {"type": "integer", "min": 1, "default": 1000},
                "max_time_budget": {"type": "integer", "min": 60},
                "max_total_pipelines_num": {"type": "integer", "min": 1},
                "num_workers": {"type": "integer", "min": 1, "required": True},
                "num_pp_candidates": {
                    "type": "integer",
                    "min": 1,
                    "required": False,
                    "default": 10,
                },
                "queue_size": {
                    "type": "integer",
                    "min": config_obj["optimisation_args"]["num_workers"],
                    "required": False,
                    "default": 3 * config_obj["optimisation_args"]["num_workers"],
                },
                "training_set_fractions_for_halting": {
                    "type": "list",
                    "required": False,
                    "default": [0.5, 0.75, 1.0],
                },
                "exploration_factor": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "required": False,
                    "default": 0.5,
                },
                "risk_factor": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "required": False,
                    "default": 0.5,
                },
            },
        },
        "virny_args": {
            "type": "dict",
            "schema": {
                "bootstrap_fraction": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "required": False,
                },
                "n_estimators": {"type": "integer", "min": 2, "required": False},
                "sensitive_attributes_dct": {
                    "type": "dict",
                    "allow_unknown": True,
                    "schema": {},
                    "required": True,
                },
            },
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
    if (
        config_obj["common_args"].get("num_runs", None) is not None
        and config_obj["common_args"].get("run_nums", None) is not None
    ):
        raise ValueError(
            "Only one of two arguments (num_runs, run_nums) should be defined in a config."
        )
    if (
        config_obj["common_args"].get("num_runs", None) is None
        and config_obj["common_args"].get("run_nums", None) is None
    ):
        raise ValueError(
            "One of two arguments (num_runs, run_nums) should be defined in a config."
        )

    # Disable bootstrap if stability and uncertainty metrics are not in objectives
    objective_names = set(
        [
            objective["metric"].strip().lower()
            for objective in config_obj["optimisation_args"]["objectives"]
        ]
    )
    if (
        len(
            objective_names.intersection(
                set([m.lower() for m in STABILITY_AND_UNCERTAINTY_METRICS])
            )
        )
        == 0
    ):
        config_obj["virny_args"][
            "computation_mode"
        ] = ComputationMode.NO_BOOTSTRAP.value
    else:
        print("Enable bootstrap in Virny")
        if (
            config_obj["virny_args"].get("bootstrap_fraction", None) is None
            or config_obj["virny_args"]["bootstrap_fraction"] < 0.0
            or config_obj["virny_args"]["bootstrap_fraction"] > 1.0
        ):
            raise ValueError(
                "virny_args.bootstrap_fraction must be float in [0.0, 1.0] range"
            )

        if (
            config_obj["virny_args"].get("n_estimators", None) is None
            or config_obj["virny_args"]["n_estimators"] <= 1
        ):
            raise ValueError("virny_args.n_estimators must be integer greater than 1")

    objective_total_weight = 0.0
    for objective in config_obj["optimisation_args"]["objectives"]:
        objective_total_weight += objective["weight"]

    if not (0.99 <= objective_total_weight <= 1.0):
        raise ValueError("Objective weights must sum to 1.0")

    if (
        config_obj["optimisation_args"]["num_workers"]
        < config_obj["optimisation_args"]["num_pp_candidates"]
    ):
        raise ValueError(
            "The number of workers should be greater or equal than the number of physical pipeline candidates for each round"
        )

    # Default arguments
    if len(config_obj["pipeline_args"]["null_imputers"]) == 0:
        config_obj["pipeline_args"]["null_imputers"] = ["None"]

    if config_obj["common_args"].get("num_runs") is not None:
        config_obj["common_args"]["run_nums"] = [
            run_num for run_num in range(1, config_obj["common_args"]["num_runs"] + 1)
        ]

    if config_obj["optimisation_args"].get("max_total_pipelines_num") is not None:
        config_obj["optimisation_args"]["max_trials"] = config_obj["optimisation_args"][
            "max_total_pipelines_num"
        ]

    return config_obj
