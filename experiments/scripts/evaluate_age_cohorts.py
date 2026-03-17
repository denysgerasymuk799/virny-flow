"""
Evaluate top-K physical pipelines from VirnyFlow (trained on Cohort A) on Cohorts B and C.

Experiment 1: Distribution Shift via Age-Based Cohorts.

Usage:
python -m scripts.evaluate_age_cohorts \
    --exp-config-name <EXP_CONFIG_NAME> \
    --run-num <RUN_NUM> \
    --top-k-pipelines <K> \
    --secrets-path scripts/configs/secrets.env \
    --training-cohort cohort_a \
    --cohort-defs '{"cohort_a": {"max_age": 44}, "cohort_b": {"min_age": 45, "max_age": 54}, "cohort_c": {"min_age": 55}}'
"""

import os
import copy
import json
import random
import argparse
import uuid
import numpy as np
import pandas as pd

from datetime import datetime, timezone
from munch import DefaultMunch

from virny.utils.common_helpers import validate_config
from virny.custom_classes.base_dataset import BaseFlowDataset
from virny.custom_classes.metrics_composer import MetricsComposer
from virny.user_interfaces.multiple_models_api import compute_metrics_with_config
from virny.user_interfaces.inference_api import compute_metrics_with_fitted_bootstrap

from virny_flow.configs.constants import (
    PHYSICAL_PIPELINE_OBSERVATIONS_TABLE,
    ALL_EXPERIMENT_METRICS_TABLE,
    EXP_CONFIG_HISTORY_TABLE,
    STAGE_SEPARATOR,
    FairnessIntervention,
    INIT_RANDOM_STATE,
)
from virny_flow.visualizations.use_case_queries import (
    get_models_disparity_metric_df,
    DISPARITY_METRIC_METADATA,
)
from virny_flow.core.utils.pipeline_utils import get_dis_group_condition
from virny_flow.task_manager.domain_logic.bayesian_optimization import METRIC_TO_LOSS_ALIGNMENT
from virny_flow.configs.component_configs import get_models_params_for_tuning
from virny_flow.core.custom_classes.core_db_client import CoreDBClient
from virny_flow.core.utils.pipeline_utils import nested_dict_from_flat
from virny_flow.core.preprocessing import preprocess_base_flow_dataset
from virny_flow.core.fairness_interventions.preprocessors import remove_disparate_impact
from virny_flow.core.fairness_interventions.inprocessors import (
    get_adversarial_debiasing_wrapper_config,
)

from scripts.configs.data_loaders import CVDAgeCohortDataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CASE_STUDY_META_TABLE = "case_study_meta"
CASE_STUDY_SUBGROUP_METRICS_TABLE = "case_study_subgroup_metrics"
CASE_STUDY_DISPARITY_METRICS_TABLE = "case_study_disparity_metrics"
CASE_STUDY_RANKINGS_TABLE = "case_study_rankings"

SENSITIVE_ATTRIBUTES_DCT = {"gender": "1"}


# ---------------------------------------------------------------------------
# MongoDB helpers
# ---------------------------------------------------------------------------
def read_exp_config_from_db(db_client: CoreDBClient, exp_config_name: str, run_num: int) -> dict:
    """Read the experiment config record from exp_config_history."""
    record = db_client.read_one_query(
        collection_name=EXP_CONFIG_HISTORY_TABLE,
        query={"exp_config_name": exp_config_name, "run_num": run_num},
    )
    if record is None:
        raise ValueError(
            f"No exp_config_history record found for "
            f"exp_config_name={exp_config_name}, run_num={run_num}"
        )
    return record


def get_top_k_physical_pipelines(db_client: CoreDBClient, exp_config_name: str,
                                 run_num: int, top_k: int) -> pd.DataFrame:
    """
    Query physical_pipeline_observations for the given exp_config_name and run_num,
    return the top-K physical pipelines ranked by compound_pp_quality (descending).
    """
    pipeline = [
        {"$match": {
            "exp_config_name": exp_config_name,
            "run_num": run_num,
            "deletion_flag": False,
        }},
        {"$sort": {"compound_pp_quality": -1}},
        {"$limit": top_k},
        {"$project": {
            "_id": 0,
            "physical_pipeline_uuid": 1,
            "logical_pipeline_name": 1,
            "config": 1,
            "compound_pp_quality": 1,
            "run_num": 1,
            "exp_config_name": 1,
        }},
    ]
    results = list(
        db_client.client[db_client.db_name][PHYSICAL_PIPELINE_OBSERVATIONS_TABLE].aggregate(pipeline)
    )
    df = pd.DataFrame(results)
    if df.empty:
        raise ValueError(
            f"No physical pipelines found for exp_config_name={exp_config_name}, run_num={run_num}"
        )
    df["cohort_a_rank"] = range(1, len(df) + 1)
    return df


# ---------------------------------------------------------------------------
# Pipeline reconstruction
# ---------------------------------------------------------------------------
def parse_pipeline_config(logical_pipeline_name: str, config: dict):
    """Split logical_pipeline_name and config into stage-level components."""
    parts = logical_pipeline_name.split(STAGE_SEPARATOR)
    if len(parts) == 3:
        null_imputer_name, fairness_intervention_name, model_name = parts
    else:
        raise ValueError(f"Unexpected logical_pipeline_name format: {logical_pipeline_name}")

    null_imputer_params_flat = {k.replace("mvi__", "", 1): v for k, v in config.items() if k.startswith("mvi__")}
    fi_params_flat = {k.replace("fi__", "", 1): v for k, v in config.items() if k.startswith("fi__")}
    model_params_flat = {k.replace("model__", "", 1): v for k, v in config.items() if k.startswith("model__")}

    for d in (null_imputer_params_flat, fi_params_flat, model_params_flat):
        for k, v in list(d.items()):
            if v == "None":
                d[k] = None

    null_imputer_params = nested_dict_from_flat(null_imputer_params_flat)
    fi_params = nested_dict_from_flat(fi_params_flat)
    model_params = nested_dict_from_flat(model_params_flat)

    return null_imputer_name, fairness_intervention_name, model_name, null_imputer_params, fi_params, model_params


def build_model(model_name: str, model_params: dict, experiment_seed: int):
    """Instantiate a sklearn-compatible model from its name and hyperparameters."""
    models_config = get_models_params_for_tuning(models_tuning_seed=experiment_seed)
    if model_name not in models_config:
        raise ValueError(f"Unknown model name: {model_name}")

    cfg = models_config[model_name]
    all_params = {**model_params, **cfg["default_kwargs"]}
    return cfg["model"](**all_params)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def create_base_flow_dataset_for_cohorts(
    train_data_loader,
    test_data_loader,
    sensitive_attributes_dct: dict,
):
    """
    Build a BaseFlowDataset where:
      - X_train_val / y_train_val come from the full train_data_loader
      - X_test / y_test come from the full test_data_loader

    Sensitive attributes are removed from features but kept for Virny's group metrics.
    """
    dataset_sensitive_attrs = [k for k in sensitive_attributes_dct.keys() if "&" not in k]

    X_train_val = train_data_loader.X_data.drop(dataset_sensitive_attrs, axis=1, errors="ignore")
    y_train_val = train_data_loader.y_data

    X_test = test_data_loader.X_data.drop(dataset_sensitive_attrs, axis=1, errors="ignore")
    y_test = test_data_loader.y_data

    numerical_cols = [c for c in train_data_loader.numerical_columns if c not in dataset_sensitive_attrs]
    categorical_cols = [c for c in train_data_loader.categorical_columns if c not in dataset_sensitive_attrs]

    # Combine sensitive attrs from both loaders so Virny can map indexes for both train and test
    test_sensitive = test_data_loader.full_df[dataset_sensitive_attrs]

    return BaseFlowDataset(
        init_sensitive_attrs_df=test_sensitive,
        X_train_val=X_train_val,
        X_test=X_test,
        y_train_val=y_train_val,
        y_test=y_test,
        target=train_data_loader.target,
        numerical_columns=numerical_cols,
        categorical_columns=categorical_cols,
    )


def _prepare_pipeline_and_dataset(
    model_name: str,
    model_params: dict,
    fairness_intervention_name: str,
    fi_params: dict,
    train_data_loader,
    test_data_loader,
    experiment_seed: int,
    sensitive_attributes_dct: dict,
    sensitive_attrs_for_intervention: list,
):
    """
    Build the BaseFlowDataset, preprocess it, apply fairness interventions,
    and construct the models_dct. Returns (preprocessed_dataset, models_dct, column_transformer).
    """
    base_flow_dataset = create_base_flow_dataset_for_cohorts(
        train_data_loader=train_data_loader,
        test_data_loader=test_data_loader,
        sensitive_attributes_dct=sensitive_attributes_dct,
    )

    preprocessed_dataset, column_transformer = preprocess_base_flow_dataset(base_flow_dataset)

    binary_sensitive_attr = "&".join(sensitive_attrs_for_intervention) + "_binary"

    if fairness_intervention_name == FairnessIntervention.DIR.value:
        _add_binary_sensitive_attr(
            preprocessed_dataset, train_data_loader, test_data_loader,
            binary_sensitive_attr, sensitive_attributes_dct, sensitive_attrs_for_intervention,
        )
        preprocessed_dataset, _ = remove_disparate_impact(
            preprocessed_dataset,
            repair_level=fi_params["repair_level"],
            sensitive_attribute=binary_sensitive_attr,
        )

    model = build_model(model_name, model_params, experiment_seed)

    if fairness_intervention_name == FairnessIntervention.AD.value:
        _add_binary_sensitive_attr(
            preprocessed_dataset, train_data_loader, test_data_loader,
            binary_sensitive_attr, sensitive_attributes_dct, sensitive_attrs_for_intervention,
        )
        privileged_groups = [{binary_sensitive_attr: 1}]
        unprivileged_groups = [{binary_sensitive_attr: 0}]
        models_dct = get_adversarial_debiasing_wrapper_config(
            privileged_groups=privileged_groups,
            unprivileged_groups=unprivileged_groups,
            inprocessor_configs=fi_params,
            sensitive_attr_for_intervention=binary_sensitive_attr,
        )
        models_dct_key = list(models_dct.keys())[0]
        models_dct = {model_name: models_dct[models_dct_key]}
    else:
        models_dct = {model_name: model}

    return preprocessed_dataset, models_dct, column_transformer


def evaluate_pipeline_on_cohort(
    model_name: str,
    model_params: dict,
    fairness_intervention_name: str,
    fi_params: dict,
    train_data_loader,
    test_data_loader,
    experiment_seed: int,
    virny_config,
    sensitive_attributes_dct: dict,
    sensitive_attrs_for_intervention: list,
    is_first_eval_cohort: bool = False,
    models_fitted_bootstraps_dct: dict = None,
    column_transformer=None,
):
    """
    Evaluate a pipeline on a cohort.

    For the first evaluation cohort (is_first_eval_cohort=True):
      - Trains the model on train_data_loader, evaluates on test_data_loader.
      - Uses compute_metrics_with_config with return_fitted_bootstrap=True.
      - Returns (metrics_dct, models_fitted_bootstraps_dct, column_transformer).

    For subsequent cohorts (is_first_eval_cohort=False):
      - Reuses the fitted bootstrap and column_transformer from the first cohort.
      - Sets X_train_val and y_train_val to empty DataFrames.
      - Uses compute_metrics_with_fitted_bootstrap for speed.
      - Returns (metrics_dct, models_fitted_bootstraps_dct, column_transformer)
        where the latter two are passed through unchanged.
    """
    if is_first_eval_cohort:
        preprocessed_dataset, models_dct, column_transformer = _prepare_pipeline_and_dataset(
            model_name=model_name,
            model_params=model_params,
            fairness_intervention_name=fairness_intervention_name,
            fi_params=fi_params,
            train_data_loader=train_data_loader,
            test_data_loader=test_data_loader,
            experiment_seed=experiment_seed,
            sensitive_attributes_dct=sensitive_attributes_dct,
            sensitive_attrs_for_intervention=sensitive_attrs_for_intervention,
        )

        virny_config.random_state = experiment_seed
        print(f"  [Virny] Computing metrics with config (training + bootstrap fitting): {dict(virny_config)}")
        metrics_dct, models_fitted_bootstraps_dct = compute_metrics_with_config(
            dataset=preprocessed_dataset,
            config=virny_config,
            models_config=models_dct,
            notebook_logs_stdout=None,
            return_fitted_bootstrap=True,
            verbose=0,
        )

        return metrics_dct, models_fitted_bootstraps_dct, column_transformer

    # --- Subsequent cohorts: reuse fitted bootstrap ---
    # Build raw dataset (no preprocessing) so we can apply the first cohort's column_transformer
    raw_dataset = create_base_flow_dataset_for_cohorts(
        train_data_loader=train_data_loader,
        test_data_loader=test_data_loader,
        sensitive_attributes_dct=sensitive_attributes_dct,
    )

    binary_sensitive_attr = "&".join(sensitive_attrs_for_intervention) + "_binary"

    # Set train data to empty to ensure no re-training
    raw_dataset.X_train_val = pd.DataFrame()
    raw_dataset.y_train_val = pd.DataFrame()

    # Transform X_test with the column_transformer fitted on the first cohort
    raw_dataset.X_test = column_transformer.transform(raw_dataset.X_test)

    # AD keeps gender_binary as a feature (its predict() needs it).
    # DIR drops gender_binary after preprocessing, so the model never saw it.
    if fairness_intervention_name == FairnessIntervention.AD.value:
        _add_binary_sensitive_attr(
            raw_dataset, train_data_loader, test_data_loader,
            binary_sensitive_attr, sensitive_attributes_dct, sensitive_attrs_for_intervention,
        )

    print(f"  [Virny] Reusing fitted bootstrap from the first cohort for model '{model_name}': {dict(virny_config)}")
    metrics_df = compute_metrics_with_fitted_bootstrap(
        fitted_bootstrap=models_fitted_bootstraps_dct[model_name],
        test_base_flow_dataset=raw_dataset,
        config=virny_config,
        with_predict_proba=True,
    )
    metrics_df["Model_Name"] = model_name

    return {model_name: metrics_df}, models_fitted_bootstraps_dct, column_transformer


def _add_binary_sensitive_attr(
    preprocessed_dataset,
    train_data_loader,
    test_data_loader,
    attr_name: str,
    sensitive_attributes_dct: dict,
    sensitive_attrs_for_intervention: list,
):
    """
    Add a binary sensitive attribute column for fairness interventions.

    Convention (matching VirnyFlow):
      - sensitive_attributes_dct values are the *disadvantaged* group values
      - Rows matching the disadvantaged value → 0
      - Rows NOT matching (privileged) → 1
    """
    dis_values = [sensitive_attributes_dct[attr] for attr in sensitive_attrs_for_intervention]

    if len(preprocessed_dataset.X_train_val) > 0:
        train_sensitive_df = train_data_loader.full_df[sensitive_attrs_for_intervention]
        train_dis_mask = get_dis_group_condition(train_sensitive_df, attrs=sensitive_attrs_for_intervention, dis_values=dis_values)
        preprocessed_dataset.X_train_val[attr_name] = None
        preprocessed_dataset.X_train_val.loc[train_dis_mask, attr_name] = 0
        preprocessed_dataset.X_train_val.loc[~train_dis_mask, attr_name] = 1

    test_sensitive_df = test_data_loader.full_df[sensitive_attrs_for_intervention]
    test_dis_mask = get_dis_group_condition(test_sensitive_df, attrs=sensitive_attrs_for_intervention, dis_values=dis_values)
    preprocessed_dataset.X_test[attr_name] = None
    preprocessed_dataset.X_test.loc[test_dis_mask, attr_name] = 0
    preprocessed_dataset.X_test.loc[~test_dis_mask, attr_name] = 1


def get_cohort_a_objective_metrics_from_db(
    db_client: CoreDBClient,
    physical_pipeline_uuid: str,
    exp_config_name: str,
    run_num: int,
    objectives: list,
) -> dict:
    """
    Read Cohort A subgroup metrics from all_experiment_metrics, pivot them,
    and compute objective metric values (including disparity metrics via
    get_models_disparity_metric_df).

    Returns dict mapping metric name -> value.
    """
    records = list(
        db_client.client[db_client.db_name][ALL_EXPERIMENT_METRICS_TABLE].find(
            {
                "physical_pipeline_uuid": physical_pipeline_uuid,
                "exp_config_name": exp_config_name,
                "run_num": run_num,
            },
            {"_id": 0},
        )
    )
    if not records:
        print(f"  WARNING: No all_experiment_metrics records for pp_uuid={physical_pipeline_uuid}")
        return {}

    metrics_df = pd.DataFrame(records)

    subgroup_col = "subgroup"
    value_col = "metric_value"
    id_cols = [c for c in metrics_df.columns if c not in (subgroup_col, value_col)]
    pivoted_df = metrics_df.pivot(
        columns=subgroup_col,
        values=value_col,
        index=id_cols,
    ).reset_index()

    result = {}
    for obj in objectives:
        metric_name, group = obj["metric"], obj["group"]
        if group == "overall":
            row = pivoted_df[pivoted_df["metric"] == metric_name]
            if row.empty or "overall" not in row.columns:
                print(f"  WARNING: Metric '{metric_name}' with group 'overall' not found in all_experiment_metrics")
                continue
            result[metric_name] = row["overall"].values[0]
        else:
            if metric_name not in DISPARITY_METRIC_METADATA:
                print(f"  WARNING: Disparity metric '{metric_name}' not in DISPARITY_METRIC_METADATA")
                continue
            disparity_df = get_models_disparity_metric_df(pivoted_df, metric_name, group)
            if disparity_df.empty:
                print(f"  WARNING: Could not compute disparity metric '{metric_name}' for group '{group}'")
                continue
            result[metric_name] = disparity_df["disparity_metric_value"].values[0]

    return result


def compute_objective_metrics(metrics_dct: dict, model_name: str, objectives: list,
                              sensitive_attributes_dct: dict):
    """
    Extract objective metric values from Virny metrics based on the objectives list.
    Returns dict mapping metric name → value.
    """
    model_overall_df = metrics_dct[model_name]
    metrics_composer = MetricsComposer(metrics_dct, sensitive_attributes_dct)
    composed_df = metrics_composer.compose_metrics()
    composed_df = composed_df[composed_df.Model_Name == model_name]

    result = {}
    for obj in objectives:
        metric, group = obj["metric"], obj["group"]
        if group == "overall":
            metric_value = model_overall_df[model_overall_df.Metric == metric][group].values[0]
        else:
            metric_value = composed_df[composed_df.Metric == metric][group].values[0]
        result[metric] = metric_value

    return result


def compute_compound_pp_quality(objective_metrics: dict, objectives: list) -> float:
    """Compute compound_pp_quality consistent with VirnyFlow's formula."""
    quality = 0.0
    for obj in objectives:
        metric_value = objective_metrics[obj["metric"]]

        loss = None
        operation = METRIC_TO_LOSS_ALIGNMENT[obj["metric"]]
        if operation == "abs":
            loss = abs(metric_value)
        elif operation == "reverse":
            loss = 1 - metric_value
        elif operation == "reverse&abs":
            loss = abs(1 - metric_value)

        reversed_obj = 1 - loss
        quality += obj["weight"] * reversed_obj
    return quality


# ---------------------------------------------------------------------------
# MongoDB write helpers
# ---------------------------------------------------------------------------
def save_meta_to_db(db_client: CoreDBClient, record: dict):
    record["deletion_flag"] = False
    record["create_datetime"] = datetime.now(timezone.utc)
    db_client.execute_write_query(records=[record], collection_name=CASE_STUDY_META_TABLE)


def _assign_meta_fields(df: pd.DataFrame, meta_fields: dict):
    """Assign meta_fields to a DataFrame, handling dict/list values correctly."""
    for col, val in meta_fields.items():
        if isinstance(val, (dict, list)):
            df[col] = [val] * len(df)
        else:
            df[col] = val


def save_subgroup_metrics_to_db(db_client: CoreDBClient, metrics_dct: dict, model_name: str,
                                meta_fields: dict):
    """Save Virny subgroup metrics (per _priv, _dis, overall) in melted key-value format."""
    model_df = metrics_dct[model_name].copy()

    subgroup_cols = [c for c in model_df.columns if "_priv" in c or "_dis" in c] + ["overall"]
    existing_subgroup_cols = [c for c in subgroup_cols if c in model_df.columns]
    id_cols = [c for c in model_df.columns if c not in existing_subgroup_cols]

    melted = model_df.melt(
        id_vars=id_cols,
        value_vars=existing_subgroup_cols,
        var_name="subgroup",
        value_name="metric_value",
    )

    _assign_meta_fields(melted, meta_fields)
    melted["deletion_flag"] = False
    melted["create_datetime"] = datetime.now(timezone.utc)
    melted.columns = melted.columns.str.lower()

    db_client.execute_write_query(
        records=melted.to_dict("records"),
        collection_name=CASE_STUDY_SUBGROUP_METRICS_TABLE,
    )


def save_disparity_metrics_to_db(db_client: CoreDBClient, metrics_dct: dict, model_name: str,
                                  meta_fields: dict, sensitive_attributes_dct: dict):
    """Save Virny disparity metrics (from MetricsComposer) in melted key-value format."""
    metrics_composer = MetricsComposer(metrics_dct, sensitive_attributes_dct)
    composed_df = metrics_composer.compose_metrics()
    composed_df = composed_df[composed_df.Model_Name == model_name].copy()

    # composed_df columns: Metric, Model_Name, <sensitive_attr_name> (e.g. "gender")
    sensitive_attr_cols = [k for k in sensitive_attributes_dct.keys() if k in composed_df.columns]
    id_cols = [c for c in composed_df.columns if c not in sensitive_attr_cols]

    melted = composed_df.melt(
        id_vars=id_cols,
        value_vars=sensitive_attr_cols,
        var_name="sensitive_attribute",
        value_name="metric_value",
    )

    _assign_meta_fields(melted, meta_fields)
    melted["deletion_flag"] = False
    melted["create_datetime"] = datetime.now(timezone.utc)
    melted.columns = melted.columns.str.lower()

    db_client.execute_write_query(
        records=melted.to_dict("records"),
        collection_name=CASE_STUDY_DISPARITY_METRICS_TABLE,
    )


def save_rankings_to_db(db_client: CoreDBClient, rankings_df: pd.DataFrame, exp_config_name: str,
                         run_num: int, cohort_defs: dict, session_uuid: str):
    rankings_df = rankings_df.copy()
    rankings_df["session_uuid"] = session_uuid
    rankings_df["exp_config_name"] = exp_config_name
    rankings_df["run_num"] = run_num
    rankings_df["cohort_defs"] = json.dumps(cohort_defs)
    rankings_df["deletion_flag"] = False
    rankings_df["create_datetime"] = datetime.now(timezone.utc)

    db_client.execute_write_query(
        records=rankings_df.to_dict("records"),
        collection_name=CASE_STUDY_RANKINGS_TABLE,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(exp_config_name: str, run_num: int, top_k_pipelines: int, secrets_path: str,
         training_cohort: str, cohort_defs: dict):
    experiment_seed = INIT_RANDOM_STATE + run_num

    random.seed(experiment_seed)
    np.random.seed(experiment_seed)

    import tensorflow as tf
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.random.set_seed(experiment_seed)

    session_uuid = str(uuid.uuid1())

    print("=" * 80)
    print("Experiment 1: Distribution Shift Evaluation")
    print(f"  exp_config_name : {exp_config_name}")
    print(f"  run_num         : {run_num}")
    print(f"  top_k_pipelines : {top_k_pipelines}")
    print(f"  experiment_seed : {experiment_seed}")
    print(f"  session_uuid    : {session_uuid}")
    print(f"  cohort_defs     : {json.dumps(cohort_defs)}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Connect to MongoDB, read exp config, fetch top-K pipelines
    # ------------------------------------------------------------------
    print("\n[Step 1] Connecting to MongoDB and reading experiment config...")
    db_client = CoreDBClient(secrets_path)
    db_client.connect()

    exp_config_record = read_exp_config_from_db(db_client, exp_config_name, run_num)
    objectives = exp_config_record["optimisation_args.objectives"]
    sensitive_attributes_dct = SENSITIVE_ATTRIBUTES_DCT
    sensitive_attrs_for_intervention = exp_config_record.get(
        "pipeline_args.sensitive_attrs_for_intervention",
        [k for k in sensitive_attributes_dct.keys() if "&" not in k],
    )

    print(f"  Objectives from DB: {objectives}")
    print(f"  sensitive_attributes_dct: {sensitive_attributes_dct}")
    print(f"  sensitive_attrs_for_intervention: {sensitive_attrs_for_intervention}")

    print("\n[Step 1b] Fetching top-K physical pipelines...")
    top_k_df = get_top_k_physical_pipelines(db_client, exp_config_name, run_num, top_k_pipelines)
    print(f"  Found {len(top_k_df)} pipelines. Top pipeline compound_pp_quality: "
          f"{top_k_df['compound_pp_quality'].iloc[0]:.4f}")
    for _, row in top_k_df.iterrows():
        print(f"    Rank {row['cohort_a_rank']}: {row['logical_pipeline_name']} "
              f"(quality={row['compound_pp_quality']:.4f}, uuid={row['physical_pipeline_uuid']})")

    # ------------------------------------------------------------------
    # 2. Load cohort data
    # ------------------------------------------------------------------
    print("\n[Step 2] Loading cohort datasets...")
    cohort_loaders = {}
    for cohort_name, cohort_params in cohort_defs.items():
        print(f"  Loading {cohort_name} (params={cohort_params})...")
        cohort_loaders[cohort_name] = CVDAgeCohortDataset(**cohort_params)
        print(f"  {cohort_name}: {len(cohort_loaders[cohort_name].full_df)} rows")

    train_cohort_name = training_cohort
    eval_cohort_names = [cn for cn in cohort_defs.keys() if cn != train_cohort_name]
    train_loader = cohort_loaders[train_cohort_name]

    print(f"\n  Training cohort: {train_cohort_name}")
    print(f"  Evaluation cohorts: {eval_cohort_names}")

    # ------------------------------------------------------------------
    # 3. Set up Virny config
    # ------------------------------------------------------------------
    virny_config = DefaultMunch.fromDict({
        "computation_mode": "no_bootstrap",
        "sensitive_attributes_dct": sensitive_attributes_dct,
        "dataset_name": "heart_cohort_a",
        "random_state": experiment_seed,
    })
    validate_config(virny_config)

    # ------------------------------------------------------------------
    # 4. Evaluate each pipeline on evaluation cohorts
    # ------------------------------------------------------------------
    cohort_results = {cn: [] for cn in eval_cohort_names}

    for pp_idx, pp_row in top_k_df.iterrows():
        pp_uuid = pp_row["physical_pipeline_uuid"]
        lp_name = pp_row["logical_pipeline_name"]
        config = pp_row["config"]
        cohort_a_quality = pp_row["compound_pp_quality"]
        cohort_a_rank = pp_row["cohort_a_rank"]

        print(f"\n{'─' * 60}")
        print(f"[Step 4.{cohort_a_rank}] Evaluating pipeline {cohort_a_rank}/{len(top_k_df)}: {lp_name}")
        print(f"  physical_pipeline_uuid: {pp_uuid}")
        print(f"  {train_cohort_name} compound_pp_quality: {cohort_a_quality:.4f}")

        (null_imputer_name, fi_name, model_name,
         null_imputer_params, fi_params, model_params) = parse_pipeline_config(lp_name, config)
        print(f"  Model: {model_name}, Model Params: {model_params}, FI: {fi_name}, FI Params: {fi_params}")

        cohort_a_obj_metrics = get_cohort_a_objective_metrics_from_db(
            db_client=db_client,
            physical_pipeline_uuid=pp_uuid,
            exp_config_name=exp_config_name,
            run_num=run_num,
            objectives=objectives,
        )
        if cohort_a_obj_metrics:
            cohort_a_obj_str = ", ".join(f"{k}={v:.4f}" for k, v in cohort_a_obj_metrics.items())
            print(f"  {train_cohort_name} objective metrics (from DB): {cohort_a_obj_str}")

        meta_fields_base = {
            "session_uuid": session_uuid,
            "exp_config_name": exp_config_name,
            "run_num": run_num,
            "physical_pipeline_uuid": pp_uuid,
            "logical_pipeline_name": lp_name,
            "config": config,
            f"{train_cohort_name}_compound_pp_quality": cohort_a_quality,
        }

        models_fitted_bootstraps_dct = None
        column_transformer = None

        for eval_idx, eval_cohort_name in enumerate(eval_cohort_names):
            eval_loader = cohort_loaders[eval_cohort_name]
            eval_params = cohort_defs[eval_cohort_name]
            is_first = (eval_idx == 0)

            if is_first:
                print(f"\n  Evaluating on {eval_cohort_name} (params={eval_params}) — training model & fitting bootstrap...")
            else:
                print(f"\n  Evaluating on {eval_cohort_name} (params={eval_params}) — reusing fitted bootstrap...")

            metrics_dct, models_fitted_bootstraps_dct, column_transformer = evaluate_pipeline_on_cohort(
                model_name=model_name,
                model_params=model_params,
                fairness_intervention_name=fi_name,
                fi_params=fi_params,
                train_data_loader=train_loader,
                test_data_loader=eval_loader,
                experiment_seed=experiment_seed,
                virny_config=copy.deepcopy(virny_config),
                sensitive_attributes_dct=sensitive_attributes_dct,
                sensitive_attrs_for_intervention=sensitive_attrs_for_intervention,
                is_first_eval_cohort=is_first,
                models_fitted_bootstraps_dct=models_fitted_bootstraps_dct,
                column_transformer=column_transformer,
            )
            obj_metrics = compute_objective_metrics(metrics_dct, model_name, objectives, sensitive_attributes_dct)
            quality = compute_compound_pp_quality(obj_metrics, objectives)

            obj_metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in obj_metrics.items())
            print(f"  {eval_cohort_name} results: {obj_metrics_str}, compound_pp_quality={quality:.4f}")

            result_entry = {
                "physical_pipeline_uuid": pp_uuid,
                "logical_pipeline_name": lp_name,
                f"{train_cohort_name}_rank": cohort_a_rank,
                f"{train_cohort_name}_compound_pp_quality": cohort_a_quality,
                f"{eval_cohort_name}_compound_pp_quality": quality,
            }
            for metric_name, metric_val in cohort_a_obj_metrics.items():
                result_entry[f"{train_cohort_name}_{metric_name}"] = metric_val
            for metric_name, metric_val in obj_metrics.items():
                result_entry[f"{eval_cohort_name}_{metric_name}"] = metric_val
            cohort_results[eval_cohort_name].append(result_entry)

            meta = {
                **meta_fields_base,
                "pipeline_rank": cohort_a_rank,
                "cohort_name": eval_cohort_name,
                "cohort_params": eval_params,
                "compound_pp_quality": quality,
            }
            for metric_name, metric_val in obj_metrics.items():
                meta[metric_name] = metric_val
            save_meta_to_db(db_client, meta)
            metrics_meta = {
                **meta_fields_base,
                "pipeline_rank": cohort_a_rank,
                "cohort_name": eval_cohort_name,
                "cohort_params": eval_params,
                "compound_pp_quality": quality,
            }
            save_subgroup_metrics_to_db(db_client, metrics_dct, model_name, metrics_meta)
            save_disparity_metrics_to_db(db_client, metrics_dct, model_name, metrics_meta, sensitive_attributes_dct)
            print(f"  {eval_cohort_name} metrics saved to MongoDB.")

    # ------------------------------------------------------------------
    # 5. Compare rankings across cohorts
    # ------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("[Step 5] Comparing rankings across cohorts...")

    # Build per-cohort ranked DataFrames
    ranked_dfs = {}
    for eval_cohort_name in eval_cohort_names:
        df = pd.DataFrame(cohort_results[eval_cohort_name])
        quality_col = f"{eval_cohort_name}_compound_pp_quality"
        df = df.sort_values(quality_col, ascending=False)
        df[f"{eval_cohort_name}_rank"] = range(1, len(df) + 1)
        ranked_dfs[eval_cohort_name] = df

    # Merge all evaluation cohort rankings
    first_eval_df = ranked_dfs[eval_cohort_names[0]]
    base_cols = ["physical_pipeline_uuid", "logical_pipeline_name"]
    train_cols = [c for c in first_eval_df.columns if c.startswith(train_cohort_name)]
    eval_cols = [c for c in first_eval_df.columns if c.startswith(eval_cohort_names[0])]
    rankings_df = first_eval_df[base_cols + train_cols + eval_cols]
    for eval_cohort_name in eval_cohort_names[1:]:
        merge_cols = ["physical_pipeline_uuid"] + [
            c for c in ranked_dfs[eval_cohort_name].columns if c.startswith(eval_cohort_name)
        ]
        rankings_df = rankings_df.merge(ranked_dfs[eval_cohort_name][merge_cols], on="physical_pipeline_uuid")

    rankings_df = rankings_df.sort_values(f"{train_cohort_name}_rank")

    # Print ranking table
    rank_cols = [f"{train_cohort_name}_rank"] + [f"{cn}_rank" for cn in eval_cohort_names]
    quality_cols = [f"{train_cohort_name}_compound_pp_quality"] + [f"{cn}_compound_pp_quality" for cn in eval_cohort_names]

    header = f"{'Pipeline':<45}"
    for rc in rank_cols:
        header += f" {rc:>12}"
    for qc in quality_cols:
        header += f" {qc:>15}"
    print(f"\nRanking comparison (sorted by {train_cohort_name} rank):")
    print(header)
    print("─" * len(header))
    for _, row in rankings_df.iterrows():
        line = f"{row['logical_pipeline_name']:<45}"
        for rc in rank_cols:
            line += f" {row[rc]:>12}"
        for qc in quality_cols:
            line += f" {row[qc]:>15.4f}"
        print(line)

    # Save rankings to MongoDB (includes cohort_defs)
    save_rankings_to_db(db_client, rankings_df, exp_config_name, run_num, cohort_defs, session_uuid)
    print(f"\nRankings saved to MongoDB table '{CASE_STUDY_RANKINGS_TABLE}'.")

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("Summary:")
    train_rank_col = f"{train_cohort_name}_rank"
    for eval_cohort_name in eval_cohort_names:
        eval_rank_col = f"{eval_cohort_name}_rank"
        rank_changes = (rankings_df[train_rank_col] - rankings_df[eval_rank_col]).abs()
        print(f"  {train_cohort_name}→{eval_cohort_name}: "
              f"avg rank change={rank_changes.mean():.2f}, "
              f"max rank change={rank_changes.max()}, "
              f"pipelines with change >= 3: {(rank_changes >= 3).sum()}/{len(rankings_df)}")
    print("=" * 80)

    db_client.close()
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate top-K VirnyFlow pipelines on age-based cohorts"
    )
    parser.add_argument("--exp-config-name", type=str, required=True,
                        help="Experiment config name used in VirnyFlow run")
    parser.add_argument("--run-num", type=int, required=True,
                        help="Run number to evaluate")
    parser.add_argument("--top-k-pipelines", type=int, default=5,
                        help="Number of top pipelines to evaluate (default: 5)")
    parser.add_argument("--secrets-path", type=str,
                        default="scripts/configs/secrets.env",
                        help="Path to secrets.env for MongoDB connection")
    parser.add_argument("--training-cohort", type=str, required=True,
                        help="Name of the training cohort (must be a key in --cohort-defs)")
    parser.add_argument(
        "--cohort-defs", type=str, required=True,
        help=(
            'JSON string defining cohorts. '
            'Example: \'{"cohort_a": {"max_age": 44}, "cohort_b": {"min_age": 45, "max_age": 54}, '
            '"cohort_c": {"min_age": 55}}\''
        ),
    )
    args = parser.parse_args()

    cohort_defs = json.loads(args.cohort_defs)
    if len(cohort_defs) < 2:
        raise ValueError("At least two cohorts must be defined (one training + one evaluation)")
    if args.training_cohort not in cohort_defs:
        raise ValueError(
            f"Training cohort '{args.training_cohort}' not found in cohort-defs. "
            f"Available cohorts: {list(cohort_defs.keys())}"
        )

    main(
        exp_config_name=args.exp_config_name,
        run_num=args.run_num,
        top_k_pipelines=args.top_k_pipelines,
        secrets_path=args.secrets_path,
        training_cohort=args.training_cohort,
        cohort_defs=cohort_defs,
    )
