import pandas as pd

from virny_flow.core.custom_classes.core_db_client import CoreDBClient
from virny_flow.configs.constants import (PHYSICAL_PIPELINE_OBSERVATIONS_TABLE, ALL_EXPERIMENT_METRICS_TABLE,
                                          NO_FAIRNESS_INTERVENTION)


def prepare_metrics_for_virnyview(secrets_path: str, exp_config_name: str):
    db_client = CoreDBClient(secrets_path)
    db_client.connect()

    pipeline = [
        # Step 1: Filter documents in PHYSICAL_PIPELINE_OBSERVATIONS_TABLE with the defined exp_config_name
        {
            "$match": {
                "exp_config_name": exp_config_name,
            }
        },
        # Step 2: Group to get the maximum `compound_pp_quality` per `exp_config_name` and `logical_pipeline_uuid`
        {
            "$group": {
                "_id": {
                    "exp_config_name": "$exp_config_name",
                    "logical_pipeline_uuid": "$logical_pipeline_uuid"
                },
                "max_compound_pp_quality": { "$max": "$compound_pp_quality" },
                "doc": { "$first": "$$ROOT" }  # Capture the full document with max improvement
            }
        },
        # Step 3: Replace root with the captured document
        {
            "$replaceRoot": { "newRoot": "$doc" }
        },
        # Step 4: Join with `all_experiment_metrics` based on `physical_pipeline_uuid`
        {
            "$lookup": {
                "from": ALL_EXPERIMENT_METRICS_TABLE,
                "let": { "physical_uuid": "$physical_pipeline_uuid" },
                "pipeline": [
                    { "$match": {
                        "$expr": {
                            "$and": [
                                { "$eq": ["$physical_pipeline_uuid", "$$physical_uuid"] },
                                { "$eq": ["$exp_config_name", exp_config_name] }
                            ]
                        }
                    }}
                ],
                "as": "experiment_metrics"
            }
        },
        # Step 5: Unwind to get one document per metric in `all_experiment_metrics`
        {
            "$unwind": "$experiment_metrics"
        },
        # Step 6: Project only the fields you need (optional)
        {
            "$project": {
                "_id": 0,
                "exp_config_name": 1,
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

    # Run the aggregation pipeline
    results = list(db_client.client[db_client.db_name][PHYSICAL_PIPELINE_OBSERVATIONS_TABLE].aggregate(pipeline))

    # Convert results to a pandas DataFrame
    all_metrics_df = pd.json_normalize(results)
    all_metrics_df.columns = [col.replace("experiment_metrics.", "") for col in all_metrics_df.columns]

    # Capitalize column names to be consistent across the whole library
    new_column_names = []
    for col in all_metrics_df.columns:
        new_col_name = '_'.join([c.capitalize() for c in col.split('_')])
        new_column_names.append(new_col_name)

    all_metrics_df.columns = new_column_names
    all_metrics_df['Model_Name'] = all_metrics_df['Model_Name'] + '__' + all_metrics_df['Null_Imputer_Name'] + '&' + all_metrics_df['Fairness_Intervention_Name']
    all_metrics_df['Model_Name'] = all_metrics_df['Model_Name'].str.replace(NO_FAIRNESS_INTERVENTION, 'NO_FI')

    # Create columns based on values in the Subgroup column
    pivoted_all_metrics_df = all_metrics_df.pivot(columns='Subgroup', values='Metric_Value',
                                                  index=[col for col in all_metrics_df.columns
                                                         if col not in ('Subgroup', 'Metric_Value')]).reset_index()

    db_client.close()
    return pivoted_all_metrics_df
