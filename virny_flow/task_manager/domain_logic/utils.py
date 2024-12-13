import yaml
import base64
import pandas as pd
from munch import DefaultMunch

from virny_flow.configs.constants import ErrorRepairMethod, FairnessIntervention, MLModels, \
    PHYSICAL_PIPELINE_OBSERVATIONS_TABLE, ALL_EXPERIMENT_METRICS_TABLE, NO_FAIRNESS_INTERVENTION, STAGE_SEPARATOR
from virny_flow.task_manager.database.task_manager_db_client import TaskManagerDBClient
from virny_flow.visualizations.use_case_queries import get_best_pps_per_lp_and_run_num_query


def is_in_enum(val, enum_obj):
    enum_vals = [member.value for member in enum_obj]
    return val in enum_vals


def validate_config(exp_config_obj):
    """
    Validate parameter types and values in the exp_config_obj.
    """
    # ============================================================================================================
    # Required parameters
    # ============================================================================================================
    if not isinstance(exp_config_obj.exp_config_name, str):
        raise ValueError('exp_config_name must be string')

    if not isinstance(exp_config_obj.dataset, str):
        raise ValueError('dataset argument must be string')

    if not isinstance(exp_config_obj.sensitive_attrs_for_intervention, list):
        raise ValueError('sensitive_attrs_for_intervention must be a list')

    if not isinstance(exp_config_obj.random_state, int):
        raise ValueError('random_state must be integer')

    # Check list types
    if not isinstance(exp_config_obj.null_imputers, list):
        raise ValueError('null_imputers argument must be a list')

    if not isinstance(exp_config_obj.fairness_interventions, list):
        raise ValueError('fairness_interventions argument must be a list')

    if not isinstance(exp_config_obj.models, list):
        raise ValueError('models argument must be a list')

    for null_imputer_name in exp_config_obj.null_imputers:
        if not is_in_enum(val=null_imputer_name, enum_obj=ErrorRepairMethod):
            raise ValueError('null_imputers argument should include values from the ErrorRepairMethod enum in domain_logic/constants.py')

    for fairness_intervention in exp_config_obj.fairness_interventions:
        if not is_in_enum(val=fairness_intervention, enum_obj=FairnessIntervention):
            raise ValueError('fairness_interventions argument should include values from the FairnessIntervention enum in domain_logic/constants.py')

    for model_name in exp_config_obj.models:
        if not is_in_enum(val=model_name, enum_obj=MLModels):
            raise ValueError('models argument should include values from the MLModels enum in domain_logic/constants.py')

    return True


def create_exp_config_obj(config_yaml_path: str):
    """
    Return a config object created based on a config yaml file.

    Parameters
    ----------
    config_yaml_path
        Path to a config yaml file

    """
    with open(config_yaml_path) as f:
        config_dct = yaml.load(f, Loader=yaml.FullLoader)

    config_obj = DefaultMunch.fromDict(config_dct)
    validate_config(config_obj)

    return config_obj


def get_logical_pipeline_names(pipeline_args):
    logical_pipelines = []
    for null_imputer in pipeline_args.null_imputers:
        for fairness_intervention in pipeline_args.fairness_interventions + [NO_FAIRNESS_INTERVENTION]:
            for model in pipeline_args.models:
                logical_pipeline = f'{null_imputer}{STAGE_SEPARATOR}{fairness_intervention}{STAGE_SEPARATOR}{model}'
                logical_pipelines.append(logical_pipeline)

    return logical_pipelines


async def clean_unnecessary_metrics(db_client: TaskManagerDBClient, exp_config_name: str,
                                    lps: list, run_nums: list, groups: list):
    print('Cleaning unnecessary metrics...')
    num_groups = len(groups) * 2 + 1

    # Find uuids of the best pp per exp_config, lp, and run_num
    pipeline_query = get_best_pps_per_lp_and_run_num_query(exp_config_name)
    subquery = pipeline_query[:3] # We need just three first steps to get best pp uuids
    cursor = db_client.client[db_client.db_name][PHYSICAL_PIPELINE_OBSERVATIONS_TABLE].aggregate(subquery)
    results = await cursor.to_list(length=None)
    results_df = pd.json_normalize(results)

    for lp in lps:
        lp_uuid = base64.b64encode(lp.encode()).decode()
        for run_num in run_nums:
            filtered_df = results_df[(results_df["exp_config_name"] == exp_config_name) &
                                     (results_df["run_num"] == run_num) &
                                     (results_df["logical_pipeline_uuid"] == lp_uuid)]
            if filtered_df.empty:
                print(f"lp - {lp}, run_num - {run_num}: No pps", flush=True)
                continue

            # Delete all other pps for the defined exp_config, lp, and run_num
            pp_uuid = filtered_df["physical_pipeline_uuid"].iloc[0]
            num_deleted_records = await db_client.delete_query(collection_name=ALL_EXPERIMENT_METRICS_TABLE,
                                                               exp_config_name=exp_config_name,
                                                               run_num=run_num,
                                                               condition={"logical_pipeline_name": lp,
                                                                          "physical_pipeline_uuid": {"$ne": pp_uuid}})
            num_deleted_pipelines = num_deleted_records / 18 / num_groups
            print(f"lp: {lp}, run_num: {run_num}, num_deleted_pipelines: {num_deleted_pipelines}", flush=True)
