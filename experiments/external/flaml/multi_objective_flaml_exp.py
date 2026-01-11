import ray
import logging
import argparse
import uuid
import time
import warnings
import os
import pandas as pd

# Disable FLAML warnings
warnings.simplefilter('ignore')

# Suppress Ray warnings
os.environ['RAY_DEDUP_LOGS'] = '1'

from flaml import tune
from munch import DefaultMunch
from datetime import datetime, timezone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from virny.utils.common_helpers import validate_config
from virny.custom_classes.base_dataset import BaseFlowDataset
from virny.custom_classes.metrics_composer import MetricsComposer
from virny.user_interfaces.multiple_models_api import compute_metrics_with_config

from virny_flow_mock.common_helpers import create_exp_config_obj, METRIC_TO_LOSS_ALIGNMENT
from virny_flow_mock.search_space import get_model_search_space, create_model_from_config
from exps.core_db_client import CoreDBClient
from exps.datasets_config import DATASET_CONFIG


# Global constants
INIT_RANDOM_STATE = 100
SYSTEM_NAME = "flaml"


def get_objective_losses(metrics_dct: dict, objectives: list, model_name: str, sensitive_attributes_dct: dict):
    model_overall_metrics_df = metrics_dct[model_name]

    metrics_composer = MetricsComposer(metrics_dct, sensitive_attributes_dct)
    models_composed_metrics_df = metrics_composer.compose_metrics()
    models_composed_metrics_df = models_composed_metrics_df[models_composed_metrics_df.Model_Name == model_name]

    # OpenBox minimizes the objective
    original_objectives = []
    weighted_losses = []
    for objective in objectives:
        metric, group = objective['metric'], objective['group']
        if group == "overall":
            metric_value = model_overall_metrics_df[model_overall_metrics_df.Metric == metric][group].values[0]
        else:
            metric_value = models_composed_metrics_df[models_composed_metrics_df.Metric == metric][group].values[0]

        # Create a loss to minimize based on the metric
        loss = None
        operation = METRIC_TO_LOSS_ALIGNMENT[metric]
        if operation == "abs":
            loss = abs(metric_value)
        elif operation == "reverse":
            loss = 1 - metric_value
        elif operation == "reverse&abs":
            loss = abs(1 - metric_value)

        original_objectives.append(metric_value)
        weighted_losses.append(objective['weight'] * loss)

    result = dict(losses=weighted_losses, original_objectives=original_objectives)
    return result


def compute_metrics_with_virny(
    pipeline,
    pipeline_name,
    virny_config,
    data_loader,
    X_train_val,
    X_test,
    y_train_val,
    y_test,
):
    """
    Compute Virny metrics for a pipeline.
    """
    models_dct = {pipeline_name: pipeline}

    # Calculate subgroup metrics
    dataset_sensitive_attrs = [
        k for k in virny_config.sensitive_attributes_dct.keys() if "&" not in k
    ]
    main_base_flow_dataset = create_base_flow_dataset(
        data_loader=data_loader,
        dataset_sensitive_attrs=dataset_sensitive_attrs,
        X_train_val_wo_sensitive_attrs=X_train_val,
        X_test_wo_sensitive_attrs=X_test,
        y_train_val=y_train_val,
        y_test=y_test,
        numerical_columns_wo_sensitive_attrs=data_loader.numerical_columns,
        categorical_columns_wo_sensitive_attrs=data_loader.categorical_columns,
    )

    metrics_dct = compute_metrics_with_config(
        dataset=main_base_flow_dataset,
        config=virny_config,
        models_config=models_dct,
        notebook_logs_stdout=None,
        verbose=0,
    )

    return metrics_dct


def create_base_flow_dataset(
    data_loader,
    dataset_sensitive_attrs,
    X_train_val_wo_sensitive_attrs,
    X_test_wo_sensitive_attrs,
    y_train_val,
    y_test,
    numerical_columns_wo_sensitive_attrs,
    categorical_columns_wo_sensitive_attrs,
):
    """
    Create BaseFlowDataset for Virny metrics computation.
    """
    sensitive_attrs_df = data_loader.full_df[dataset_sensitive_attrs]

    if X_train_val_wo_sensitive_attrs is not None:
        assert X_train_val_wo_sensitive_attrs.index.isin(
            sensitive_attrs_df.index
        ).all(), "Not all indexes of X_train_val_wo_sensitive_attrs are present in sensitive_attrs_df"
    assert X_test_wo_sensitive_attrs.index.isin(
        sensitive_attrs_df.index
    ).all(), (
        "Not all indexes of X_test_wo_sensitive_attrs are present in sensitive_attrs_df"
    )

    if X_train_val_wo_sensitive_attrs is not None and y_train_val is not None:
        assert (
            X_train_val_wo_sensitive_attrs.index.equals(y_train_val.index) is True
        ), "Indexes of X_train_val_wo_sensitive_attrs and y_train_val are different"
    assert (
        X_test_wo_sensitive_attrs.index.equals(y_test.index) is True
    ), "Indexes of X_test_wo_sensitive_attrs and y_test are different"

    return BaseFlowDataset(
        init_sensitive_attrs_df=sensitive_attrs_df,
        X_train_val=X_train_val_wo_sensitive_attrs,
        X_test=X_test_wo_sensitive_attrs,
        y_train_val=y_train_val,
        y_test=y_test,
        target=data_loader.target,
        numerical_columns=numerical_columns_wo_sensitive_attrs,
        categorical_columns=categorical_columns_wo_sensitive_attrs,
    )


def save_virny_metrics_in_db(best_model_config: dict, model_metrics_df: pd.DataFrame, exp_config_name: str, virny_config,
                             session_uuid: str, run_num: int, experiment_seed: int, secrets_path: str,
                             optimization_time: float, total_execution_time: float):
    db_client = CoreDBClient(secrets_path)
    db_client.connect()

    # Create a dict with custom fields to store in DB
    custom_tbl_fields_dct = dict()
    custom_tbl_fields_dct['session_uuid'] = session_uuid
    custom_tbl_fields_dct['system_name'] = SYSTEM_NAME
    custom_tbl_fields_dct['logical_pipeline_name'] = best_model_config['model_type']
    custom_tbl_fields_dct['dataset_split_seed'] = experiment_seed
    custom_tbl_fields_dct['model_init_seed'] = experiment_seed
    custom_tbl_fields_dct['experiment_seed'] = experiment_seed
    custom_tbl_fields_dct['exp_config_name'] = exp_config_name
    custom_tbl_fields_dct['run_num'] = run_num
    custom_tbl_fields_dct['optimization_time'] = optimization_time
    custom_tbl_fields_dct['total_execution_time'] = total_execution_time
    custom_tbl_fields_dct["model_config"] = str(best_model_config)

    # Concatenate current run metrics with previous results and
    # create melted_model_metrics_df to save it in a database
    model_metrics_df['Dataset_Name'] = virny_config.dataset_name
    model_metrics_df['Num_Estimators'] = virny_config.n_estimators

    # Extend df with technical columns
    model_metrics_df['Tag'] = 'OK'
    model_metrics_df['Record_Create_Date_Time'] = datetime.now(timezone.utc)

    for column, value in custom_tbl_fields_dct.items():
        model_metrics_df[column] = value

    subgroup_names = [col for col in model_metrics_df.columns if '_priv' in col or '_dis' in col] + ['overall']
    melted_model_metrics_df = model_metrics_df.melt(
        id_vars=[col for col in model_metrics_df.columns if col not in subgroup_names],
        value_vars=subgroup_names,
        var_name="Subgroup",
        value_name="Metric_Value")

    melted_model_metrics_df.columns = melted_model_metrics_df.columns.str.lower()
    db_client.execute_write_query(records=melted_model_metrics_df.to_dict('records'),
                                  collection_name="all_experiment_metrics")
    db_client.close()


def evaluate_model_multiobjective(
    config,
    X_train,
    y_train,
    X_test,
    y_test,
    data_loader,
    virny_config,
    preprocessor,
    objectives: list,
):
    """
    Evaluation function using Virny metrics.
    """
    model = create_model_from_config(config, preprocessor, virny_config.random_state)

    # Compute metrics on validation set
    subgroup_metrics_dct = compute_metrics_with_virny(
        pipeline=model,
        pipeline_name=config["model_type"],
        virny_config=virny_config,
        data_loader=data_loader,
        X_train_val=X_train,
        X_test=X_test,
        y_train_val=y_train,
        y_test=y_test,
    )

    # Extract metrics
    objective_dct = get_objective_losses(metrics_dct=subgroup_metrics_dct,
                                         model_name=config["model_type"],
                                         objectives=objectives,
                                         sensitive_attributes_dct=virny_config.sensitive_attributes_dct)

    losses = objective_dct['losses']
    if len(objectives) == 2:
        tune.report(
            first_metric=losses[0],
            second_metric=losses[1],
        )
    elif len(objectives) == 3:
        tune.report(
            first_metric=losses[0],
            second_metric=losses[1],
            third_metric=losses[2],
        )
    else:
        raise ValueError(f"The script supports only two and three objectives. The input number of objectives is {len(objectives)}.")


def run_exp(
    exp_config,
    exp_config_name: str,
    dataset_name: str,
    run_num: int,
    num_workers: int,
    objectives: list,
    max_time_budget: int,
    max_total_pipelines_num: int,
    storage_path: str = None,
):
    """
    Run FLAML experiment with multi-objective optimization using Virny metrics.
    """
    if max_total_pipelines_num is None and max_time_budget is None:
        raise ValueError(
            "max_total_pipelines_num and max_time_budget cannot be both None"
        )

    # Define configs
    session_uuid = str(uuid.uuid1())
    experiment_seed = INIT_RANDOM_STATE + run_num
    virny_config = DefaultMunch.fromDict(DATASET_CONFIG[dataset_name]["virny_config"])
    virny_config.dataset_name = dataset_name
    virny_config.random_state = experiment_seed
    validate_config(virny_config)

    # Read data
    test_set_fraction = DATASET_CONFIG[dataset_name]["test_set_fraction"]
    data_loader = DATASET_CONFIG[dataset_name]["data_loader"](
        **DATASET_CONFIG[dataset_name]["data_loader_kwargs"]
    )
    X_train_val, X_test, y_train_val, y_test = train_test_split(data_loader.X_data, data_loader.y_data,
                                                                test_size=test_set_fraction,
                                                                random_state=experiment_seed)

    # Build a preprocessor
    preprocessor = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(categories='auto', handle_unknown='ignore', sparse_output=False), data_loader.categorical_columns),
        ('num', StandardScaler(), data_loader.numerical_columns),
    ])

    def evaluation_fn(config):
        return evaluate_model_multiobjective(
            config=config,
            X_train=X_train_val,
            y_train=y_train_val,
            X_test=X_test,
            y_test=y_test,
            data_loader=data_loader,
            virny_config=virny_config,
            preprocessor=preprocessor,
            objectives=objectives,
        )

    search_space = get_model_search_space()

    if len(objectives) == 2:
        lexico_objectives = {
            "metrics": ["first_metric", "second_metric"],
            "modes": ["min", "min"],
            "targets": {"first_metric": 0.0, "second_metric": 0.0},
            "tolerances": {},
        }
    elif len(objectives) == 3:
        lexico_objectives = {
            "metrics": ["first_metric", "second_metric", "third_metric"],
            "modes": ["min", "min", "min"],
            "targets": {"first_metric": 0.0, "second_metric": 0.0, "third_metric": 0.0},
            "tolerances": {},
        }
    else:
        raise ValueError(f"The script supports only two and three objectives. The input number of objectives is {len(objectives)}.")

    print("Starting multi-objective optimization with FLAML tune...", flush=True)
    start = time.time()

    # Convert storage_path to URI format if provided
    if storage_path is not None:
        # Check if it's already a URI (has a scheme like file://, s3://, etc.)
        if "://" not in storage_path:
            # Convert local path to absolute path and add file:// scheme
            abs_path = os.path.abspath(storage_path)
            storage_path = f"file://{abs_path}"

    analysis = tune.run(
        evaluation_fn,
        config=search_space,
        lexico_objectives=lexico_objectives,
        num_samples=max_total_pipelines_num if max_total_pipelines_num else -1,
        time_budget_s=max_time_budget,
        verbose=1,  # Show progress updates
        use_ray=True,
        resources_per_trial={"cpu": 1},
        storage_path=storage_path,
    )

    optimization_time = time.time() - start
    print(f"\nOptimization completed in {optimization_time:.2f} seconds", flush=True)

    # Extract solutions
    solutions = []
    num_trials_with_metrics = 0
    print(f"Number of conducted trials: {len(analysis.trials)}", flush=True)
    for trial in analysis.trials:
        if trial.status == "TERMINATED":
            if len(trial.metric_analysis) != 0:
                num_trials_with_metrics += 1
            if trial.metric_analysis:
                # Calculate pipeline quality and improvement based on the test losses
                compound_pp_quality = 0.0
                metric_names = ["first_metric", "second_metric", "third_metric"]
                for idx, objective in enumerate(objectives):
                    # Compute reversed_objective
                    metric_name = metric_names[idx]
                    objective_loss = trial.metric_analysis[metric_name]["last"]
                    reversed_objective = 1 - objective_loss / objective['weight']

                    compound_pp_quality += objective["weight"] * reversed_objective

                solutions.append({
                    "trial_id": trial.trial_id,
                    "config": trial.config,
                    "model_type": trial.config["model_type"],
                    "compound_pp_quality": compound_pp_quality,
                })

    print(f"Number of trials with metrics: {num_trials_with_metrics}", flush=True)

    if not solutions:
        print("No valid solutions found during optimization", flush=True)

    # Sort by compound_pp_quality
    solutions.sort(key=lambda x: x["compound_pp_quality"], reverse=True)
    best_solution = solutions[0]
    print(f"Selected solution: trial_id -- {best_solution['trial_id']}, model_type -- {best_solution['model_type']}, compound_pp_quality -- {best_solution['compound_pp_quality']}", flush=True)

    # Train final model on full train_val set
    print("Training final model on full train_val set...", flush=True)
    best_model = create_model_from_config(
        best_solution["config"], preprocessor, experiment_seed
    )

    # Compute final metrics on test set
    pipeline_name = f"{SYSTEM_NAME}_best_pipeline"
    best_pipeline_subgroup_metrics_dct = (
        compute_metrics_with_virny(
            pipeline=best_model,
            pipeline_name=pipeline_name,
            virny_config=virny_config,
            data_loader=data_loader,
            X_train_val=X_train_val,
            X_test=X_test,
            y_train_val=y_train_val,
            y_test=y_test,
        )
    )
    total_execution_time = time.time() - start
    print("Virny metrics are computed", flush=True)

    # Save results
    save_virny_metrics_in_db(
        best_model_config=best_solution["config"],
        model_metrics_df=best_pipeline_subgroup_metrics_dct[pipeline_name],
        exp_config_name=exp_config_name,
        virny_config=virny_config,
        session_uuid=session_uuid,
        run_num=run_num,
        experiment_seed=experiment_seed,
        optimization_time=optimization_time,
        total_execution_time=total_execution_time,
        secrets_path=exp_config.common_args.secrets_path,
    )
    print(f"Virny metrics are saved in DB. Session UUID: {session_uuid}", flush=True)


if __name__ == "__main__":
    """
    Example execution command:
    
    python3 ./external/flaml/multi_objective_flaml_exp.py \
        --exp-config-yaml-path ./external/flaml/exp_config.yaml \
        --storage-path ./logs/flaml/
    """

    parser = argparse.ArgumentParser(
        description="Run FLAML experiment with multi-objective optimization using Virny metrics"
    )
    parser.add_argument(
        "--exp-config-yaml-path",
        type=str,
        required=True,
        help="Path to the experiment configuration YAML file"
    )
    parser.add_argument(
        "--storage-path",
        type=str,
        default=None,
        help="Directory to save logs and checkpoints (default: None)"
    )

    args = parser.parse_args()

    # Read configuration from YAML file
    exp_config = create_exp_config_obj(exp_config_yaml_path=args.exp_config_yaml_path)

    # Extract parameters from config
    max_time_budget = getattr(exp_config.optimisation_args, 'max_time_budget', None)
    max_total_pipelines_num = getattr(exp_config.optimisation_args, 'max_total_pipelines_num', None)
    print("max_time_budget:", max_time_budget)
    print("max_total_pipelines_num:", max_total_pipelines_num)

    ray.init(
        num_cpus=exp_config.optimisation_args.num_workers,
        logging_level=logging.CRITICAL,
        log_to_driver=False
    )
    print("Ray resources:", ray.available_resources())

    objectives = exp_config.optimisation_args.objectives
    if len(objectives) not in (2, 3):
        raise ValueError("Two or three objectives are required in the YAML config.")

    run_exp(
        exp_config=exp_config,
        exp_config_name=exp_config.common_args.exp_config_name,
        dataset_name=exp_config.pipeline_args.dataset,
        run_num=exp_config.common_args.run_nums[0],  # Use first run number from list
        num_workers=exp_config.optimisation_args.num_workers,
        max_time_budget=max_time_budget,
        max_total_pipelines_num=max_total_pipelines_num,
        objectives=objectives,
        storage_path=args.storage_path,
    )
