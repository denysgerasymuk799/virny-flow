import pandas as pd
from datetime import datetime

from virny_flow.configs.constants import ALL_EXPERIMENT_METRICS_TABLE
from virny_flow.core.custom_classes.core_db_client import CoreDBClient
from virny_flow.configs.constants import (PHYSICAL_PIPELINE_OBSERVATIONS_TABLE, LOGICAL_PIPELINE_SCORES_TABLE,
                                          EXP_CONFIG_HISTORY_TABLE)


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
    acq_type = logical_pipeline_metadata['acq_type']
    num_completed_pps = logical_pipeline_metadata['num_completed_pps']

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

    return structured_data, defined_objectives, surrogate_model_type, acq_type, num_completed_pps


def get_best_pps_per_lp_and_run_num_query(exp_config_name: str):
    return [
        # Step 1: Filter documents in PHYSICAL_PIPELINE_OBSERVATIONS_TABLE with the defined exp_config_name
        {
            "$match": {
                "exp_config_name": exp_config_name,
                "run_num": 1,
                "deletion_flag": False
            }
        },
        # Step 2: Sort by compound_pp_quality in descending order.
        # Sorting before grouping helps make the first document in each group to have max compound_pp_quality.
        {
            "$sort": { "compound_pp_quality": -1 },
        },
        # Step 3: Group to get the maximum `compound_pp_quality` per `exp_config_name`, `logical_pipeline_uuid`, and `run_num`
        {
            "$group": {
                "_id": {
                    "exp_config_name": "$exp_config_name",
                    "run_num": "$run_num",
                    "logical_pipeline_uuid": "$logical_pipeline_uuid",
                    "logical_pipeline_name": "$logical_pipeline_name",
                },
                "best_pipeline_doc": {
                    "$first": {
                        "physical_pipeline_uuid": "$physical_pipeline_uuid",
                        "run_num": "$run_num",
                        "compound_pp_quality": "$compound_pp_quality"
                    }
                }
            }
        },
        # Step 4: Replace root with the captured document
        {
            "$replaceRoot": {
                "newRoot": {
                    "exp_config_name": "$_id.exp_config_name",
                    "run_num": "$_id.run_num",
                    "logical_pipeline_uuid": "$_id.logical_pipeline_uuid",
                    "logical_pipeline_name": "$_id.logical_pipeline_name",
                    "physical_pipeline_uuid": "$best_pipeline_doc.physical_pipeline_uuid",
                    "compound_pp_quality": "$best_pipeline_doc.compound_pp_quality"
                }
            }
        },
        # Step 5: Join with `all_experiment_metrics` based on `physical_pipeline_uuid`
        {
            "$lookup": {
                "from": ALL_EXPERIMENT_METRICS_TABLE,
                "let": { "logical_name": "$logical_pipeline_name", "physical_uuid": "$physical_pipeline_uuid", "physical_run_num": "$run_num" },
                "pipeline": [
                    { "$match": {
                        "$expr": {
                            "$and": [
                                { "$eq": ["$exp_config_name", exp_config_name] },
                                { "$eq": ["$run_num", "$$physical_run_num"] },
                                { "$eq": ["$logical_pipeline_name", "$$logical_name"] },
                                { "$eq": ["$physical_pipeline_uuid", "$$physical_uuid"] },
                            ]
                        }
                    }}
                ],
                "as": "experiment_metrics"
            }
        },
        # Step 6: Unwind to get one document per metric in `all_experiment_metrics`
        {
            "$unwind": "$experiment_metrics"
        },
        # Step 7: Project only the fields you need (optional)
        {
            "$project": {
                "_id": 0,
                "exp_config_name": 1,
                "run_num": 1,
                "logical_pipeline_uuid": 1,
                "physical_pipeline_uuid": 1,
                "compound_pp_quality": 1,
                "experiment_metrics.logical_pipeline_name": 1,
                "experiment_metrics.dataset_name": 1,
                "experiment_metrics.null_imputer_name": 1,
                "experiment_metrics.fairness_intervention_name": 1,
                "experiment_metrics.model_name": 1,
                "experiment_metrics.subgroup": 1,
                "experiment_metrics.metric": 1,
                "experiment_metrics.metric_value": 1,
                "experiment_metrics.runtime_in_mins": 1,
            }
        }
    ]


def get_best_pps_per_lp_and_run_num(db_client: CoreDBClient, exp_config_name: str):
    pipeline = get_best_pps_per_lp_and_run_num_query(exp_config_name)

    # Run the aggregation pipeline
    results = list(db_client.client[db_client.db_name][PHYSICAL_PIPELINE_OBSERVATIONS_TABLE].aggregate(pipeline))

    # Convert results to a pandas DataFrame
    all_metrics_df = pd.json_normalize(results)
    all_metrics_df.columns = [col.replace("experiment_metrics.", "") for col in all_metrics_df.columns]

    # Create columns based on values in the Subgroup column
    pivoted_all_metrics_df = all_metrics_df.pivot(columns='subgroup', values='metric_value',
                                                  index=[col for col in all_metrics_df.columns
                                                         if col not in ('subgroup', 'metric_value')]).reset_index()

    return pivoted_all_metrics_df


def get_best_lps_for_exp_config(db_client: CoreDBClient, exp_config_name: str):
    best_pps_per_lp_and_run_num_df = get_best_pps_per_lp_and_run_num(db_client=db_client,
                                                                     exp_config_name=exp_config_name)

    # Find the maximum compound_pp_quality per run_num
    max_quality_per_run = best_pps_per_lp_and_run_num_df.groupby("run_num")["compound_pp_quality"].transform("max")

    # Filter rows where compound_pp_quality matches the maximum for that run_num
    best_lp_per_run_all = best_pps_per_lp_and_run_num_df[best_pps_per_lp_and_run_num_df["compound_pp_quality"] == max_quality_per_run]

    return best_lp_per_run_all


def get_best_lps_per_exp_config(secrets_path: str, exp_config_names: list):
    db_client = CoreDBClient(secrets_path)
    db_client.connect()

    best_lp_metrics_per_exp_config_df = pd.DataFrame()
    for exp_config_name in exp_config_names:
        print(f'Extracting metrics for {exp_config_name}...')
        best_lp_metrics_df = get_best_lps_for_exp_config(db_client=db_client, exp_config_name=exp_config_name)
        best_lp_metrics_per_exp_config_df = pd.concat([best_lp_metrics_per_exp_config_df, best_lp_metrics_df])
        print(f'Extracted metrics for {exp_config_name}\n')

    db_client.close()

    return best_lp_metrics_per_exp_config_df


def get_tuned_lps_for_exp_config(secrets_path: str, exp_config_name: str):
    db_client = CoreDBClient(secrets_path)
    db_client.connect()

    best_pps_per_lp_and_run_num_df = get_best_pps_per_lp_and_run_num(db_client=db_client,
                                                                     exp_config_name=exp_config_name)
    db_client.close()

    return best_pps_per_lp_and_run_num_df


def get_tuned_lp_for_exp_config(secrets_path: str, exp_config_name: str, run_num: int, lp_name: str):
    db_client = CoreDBClient(secrets_path)
    db_client.connect()

    pipeline = [
        # Step 1: Filter documents in PHYSICAL_PIPELINE_OBSERVATIONS_TABLE with the defined exp_config_name
        {
            "$match": {
                "exp_config_name": exp_config_name,
                "run_num": run_num,
                "logical_pipeline_name": lp_name,
                "deletion_flag": False,
            }
        },
        # Step 2: Sort by compound_pp_quality in descending order.
        # Sorting before grouping helps make the first document in each group to have max compound_pp_quality.
        {
            "$sort": { "compound_pp_quality": -1 },
        },
        # Step 3: Group to get the maximum `compound_pp_quality` per `exp_config_name`, `logical_pipeline_uuid`, and `run_num`
        {
            "$group": {
                "_id": {
                    "exp_config_name": "$exp_config_name",
                    "run_num": "$run_num",
                    "logical_pipeline_uuid": "$logical_pipeline_uuid",
                    "logical_pipeline_name": "$logical_pipeline_name",
                },
                "best_pipeline_doc": {
                    "$first": {
                        "physical_pipeline_uuid": "$physical_pipeline_uuid",
                        "run_num": "$run_num",
                        "compound_pp_quality": "$compound_pp_quality",
                        "config": "$config"
                    }
                }
            }
        },
        # Step 4: Replace root with the captured document
        {
            "$replaceRoot": {
                "newRoot": {
                    "exp_config_name": "$_id.exp_config_name",
                    "run_num": "$_id.run_num",
                    "logical_pipeline_uuid": "$_id.logical_pipeline_uuid",
                    "logical_pipeline_name": "$_id.logical_pipeline_name",
                    "physical_pipeline_uuid": "$best_pipeline_doc.physical_pipeline_uuid",
                    "compound_pp_quality": "$best_pipeline_doc.compound_pp_quality",
                    "config": "$best_pipeline_doc.config"
                }
            }
        },
    ]

    # Run the aggregation pipeline
    results = list(db_client.client[db_client.db_name][PHYSICAL_PIPELINE_OBSERVATIONS_TABLE].aggregate(pipeline))

    # Convert results to a pandas DataFrame
    pipeline_hyperparams = results[0].get('config', dict())
    db_client.close()

    return pipeline_hyperparams
