import sys
from pathlib import Path

# Define a correct root path
sys.path.append(str(Path(f"{__file__}").parent.parent.parent.parent))

import copy
import time
import uuid
import pathlib
import numpy as np
import pandas as pd

from datetime import datetime, timezone
from munch import DefaultMunch
from sklearn.model_selection import train_test_split
from virny.custom_classes.base_dataset import BaseFlowDataset
from virny.user_interfaces import compute_metrics_with_config
from virny.utils.common_helpers import validate_config

from alpine_meadow.common import Task, TaskKeyword, PerformanceMetric, Config
from alpine_meadow.common.metric import get_score
from alpine_meadow.core import Optimizer
from tools.benchmark.exps.core_db_client import CoreDBClient
from tools.benchmark.exps.datasets_config import DATASET_CONFIG


# Global constants
INIT_RANDOM_STATE = 100


class VirnyInprocessingWrapper:
    def __init__(self, pipeline, target, task):
        self.pipeline = pipeline
        self.target = target
        self.task = task

    def __copy__(self):
        new_pipeline = copy.copy(self.pipeline)
        return VirnyInprocessingWrapper(pipeline=new_pipeline, target=self.target, task=self.task)

    def __deepcopy__(self, memo):
        new_pipeline = copy.deepcopy(self.pipeline)
        return VirnyInprocessingWrapper(pipeline=new_pipeline, target=self.target, task=self.task)

    def get_params(self):
        return self.pipeline._pipeline.dumps()

    def set_params(self, **params):
        pass

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        df = copy.deepcopy(X)
        df[self.target] = y
        train_dataset = self.task.dataset.from_data_frame(df)
        self.pipeline.train([train_dataset])
        return self

    def predict(self, X):
        test_dataset = self.task.dataset.from_data_frame(X)
        preds = self.pipeline.test([test_dataset]).outputs.astype(np.int64).values.flatten()
        return preds


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


def convert_pipeline_to_readable_string(pipeline: list):
    class_names = []
    for step in pipeline:
        if hasattr(step, '__class__'):
            class_name = step.__class__.__name__
            class_names.append(class_name)
        else:
            class_names.append(str(step))

    class_names_str = ' & '.join(class_names)
    return class_names_str


def compute_metrics_with_virny(pipeline, virny_config, task, data_loader, X_train_val, X_test, y_train_val, y_test):
    dataset_sensitive_attrs = [k for k in virny_config.sensitive_attributes_dct.keys() if '&' not in k]
    pipeline_name = "alpine_meadow_best_pipeline"
    models_dct = {
        pipeline_name: VirnyInprocessingWrapper(pipeline, data_loader.target, task),
    }
    main_base_flow_dataset = create_base_flow_dataset(data_loader=data_loader,
                                                      dataset_sensitive_attrs=dataset_sensitive_attrs,
                                                      X_train_val_wo_sensitive_attrs=X_train_val,
                                                      X_test_wo_sensitive_attrs=X_test,
                                                      y_train_val=y_train_val,
                                                      y_test=y_test,
                                                      numerical_columns_wo_sensitive_attrs=data_loader.numerical_columns,
                                                      categorical_columns_wo_sensitive_attrs=data_loader.categorical_columns)
    test_metrics_dct = compute_metrics_with_config(dataset=main_base_flow_dataset,
                                                   config=virny_config,
                                                   models_config=models_dct,
                                                   notebook_logs_stdout=False,
                                                   with_predict_proba=False,
                                                   verbose=0)
    return test_metrics_dct[pipeline_name]


def save_virny_metrics_in_db(pipeline, model_metrics_df: pd.DataFrame, exp_config_name: str, virny_config,
                             session_uuid: str, run_num: int, experiment_seed: int,
                             optimization_time: float, total_execution_time: float):
    secrets_path = pathlib.Path(__file__).parent.joinpath('..', '..', '..', '..', '..',
                                                          'scripts', 'configs', 'secrets.env')
    db_client = CoreDBClient(secrets_path)
    db_client.connect()

    # Parse a pipeline
    pipeline_str = convert_pipeline_to_readable_string(pipeline._primitives)
    print("Best pipeline:", pipeline_str)
    pipeline_desc = pipeline._pipeline.to_pipeline_desc(human_readable=True)
    print("Pipeline description:\n", pipeline_desc)

    # Create a dict with custom fields to store in DB
    custom_tbl_fields_dct = dict()
    custom_tbl_fields_dct['session_uuid'] = session_uuid
    custom_tbl_fields_dct['system_name'] = 'alpine_meadow'
    custom_tbl_fields_dct['logical_pipeline_name'] = pipeline_str
    custom_tbl_fields_dct['dataset_split_seed'] = experiment_seed
    custom_tbl_fields_dct['model_init_seed'] = experiment_seed
    custom_tbl_fields_dct['experiment_seed'] = experiment_seed
    custom_tbl_fields_dct['exp_config_name'] = exp_config_name
    custom_tbl_fields_dct['run_num'] = run_num
    custom_tbl_fields_dct['optimization_time'] = optimization_time
    custom_tbl_fields_dct['total_execution_time'] = total_execution_time

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


def run_exp(exp_config_name: str, dataset_name: str, run_num: int, num_workers: int,
            max_time_budget: int = None, max_total_pipelines_num: int = None, num_pp_candidates: int = 5):
    if num_workers < 5:
        num_pp_candidates = 2
    if max_total_pipelines_num is None and max_time_budget is None:
        raise ValueError("max_total_pipelines_num and max_time_budget cannot be both None")

    print("============= Input arguments =============")
    print("exp_config_name:", exp_config_name)
    print("dataset_name:", dataset_name)
    print("run_num:", run_num)
    print("num_workers:", num_workers)
    print("max_time_budget:", max_time_budget)
    print("max_total_pipelines_num:", max_total_pipelines_num)
    print("num_pp_candidates:", num_pp_candidates, '\n')

    # Define configs
    session_uuid = str(uuid.uuid1())
    experiment_seed = INIT_RANDOM_STATE + run_num
    virny_config = DefaultMunch.fromDict(DATASET_CONFIG[dataset_name]["virny_config"])
    virny_config.dataset_name = dataset_name
    virny_config.random_state = experiment_seed  # Set random state for the metric computation with Virny
    validate_config(virny_config)

    # Create a config for Alpine Meadow
    config = Config(debug=True)
    config.timeout_seconds = max_time_budget
    config.evaluation_workers_num = num_workers
    config.configurations_per_arm_num = num_pp_candidates
    config.enable_cross_validation = True
    config.log_trace = True
    config.enable_feature_engineering = False
    config.enable_meta_learning = False
    config.fe_hyperparams["verbose"] = True

    # Read data
    test_set_fraction = DATASET_CONFIG[dataset_name]['test_set_fraction']
    data_loader = DATASET_CONFIG[dataset_name]['data_loader'](**DATASET_CONFIG[dataset_name]['data_loader_kwargs'])
    X_train_val, X_test, y_train_val, y_test = train_test_split(data_loader.X_data, data_loader.y_data,
                                                                test_size=test_set_fraction,
                                                                random_state=experiment_seed)
    train_df = X_train_val
    train_df[data_loader.target] = y_train_val
    test_df = X_test
    test_df[data_loader.target] = y_test

    task_type = TaskKeyword.Value('CLASSIFICATION')
    metric = PerformanceMetric.Value('F1_MACRO')
    target_columns = [data_loader.target]
    task = Task([task_type], [metric], target_columns, dataset=train_df)
    test_dataset = task.dataset.from_data_frame(test_df)
    true_results = test_df[target_columns].values
    print("Data is prepared")

    # Define an optimizer
    optimizer = Optimizer(task, config=config)

    # Run an experimental pipeline
    start = time.time()
    for result in optimizer.optimize(pipelines_num_limit=max_total_pipelines_num):
        pipeline = result.pipeline

        time_ = time.time() - start
        predicted_results = pipeline.test([test_dataset]).outputs
        predicted_results = predicted_results.astype(true_results.dtype)
        score = get_score(metric, true_results, predicted_results)
        print('Time: {}, Score: {}'.format(time_, score), flush=True)

    optimization_time = time.time() - start
    print("Optimization finished")

    # Compute virny metrics
    pipeline_overall_metrics_df = compute_metrics_with_virny(pipeline=pipeline,
                                                             virny_config=virny_config,
                                                             task=task,
                                                             data_loader=data_loader,
                                                             X_train_val=X_train_val,
                                                             X_test=X_test,
                                                             y_train_val=y_train_val,
                                                             y_test=y_test)
    total_execution_time = time.time() - start
    print("Virny metrics are computed")

    # Save results in DB
    save_virny_metrics_in_db(pipeline=pipeline,
                             model_metrics_df=pipeline_overall_metrics_df,
                             exp_config_name=exp_config_name,
                             virny_config=virny_config,
                             session_uuid=session_uuid,
                             run_num=run_num,
                             experiment_seed=experiment_seed,
                             optimization_time=optimization_time,
                             total_execution_time=total_execution_time)
    print(f"Virny metrics are saved in DB. Session UUID: {session_uuid}")


if __name__ == "__main__":
    # Example execution command:
    # python3 -m tools.benchmark.exps.run_exp test_exp diabetes 1 3 60 none
    # OR
    # python3 -m tools.benchmark.exps.run_exp test_exp diabetes 1 3 none 10
    input_max_time_budget = sys.argv[5]
    input_max_total_pipelines_num = sys.argv[6]
    run_exp(exp_config_name = sys.argv[1],
            dataset_name = sys.argv[2],
            run_num = int(sys.argv[3]),
            num_workers = int(sys.argv[4]),
            max_time_budget = None if input_max_time_budget.lower() == 'none' else int(input_max_time_budget),
            max_total_pipelines_num = None if input_max_total_pipelines_num.lower() == 'none' else int(input_max_total_pipelines_num))
