import sys
import warnings
from pathlib import Path

# Suppress all warnings
warnings.filterwarnings("ignore")

# Define a correct root path
sys.path.append(str(Path(f"{__file__}").parent.parent.parent))

import os
import sys
import time
import copy
import uuid
import shutil
import pathlib
import pandas as pd
import autosklearn.classification
import autosklearn.pipeline.components.classification

from munch import DefaultMunch
from pprint import pprint
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split
from virny.custom_classes.base_dataset import BaseFlowDataset
from virny.user_interfaces.multiple_models_api import compute_metrics_with_config
from virny.utils.common_helpers import validate_config

from examples.exps.core_db_client import CoreDBClient
from examples.exps.datasets_config import DATASET_CONFIG
from examples.exps.lgbm_classifier import LGBMClassifier
from examples.exps.logistic_regression import LogisticRegression
from examples.exps.xgboost_classifier import XGBoostClassifier


# Global constants
INIT_RANDOM_STATE = 100
SYSTEM_NAME = "autosklearn"


class VirnyInprocessingWrapper:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def __copy__(self):
        new_pipeline = copy.copy(self.pipeline)
        return VirnyInprocessingWrapper(pipeline=new_pipeline)

    def __deepcopy__(self, memo):
        new_pipeline = copy.deepcopy(self.pipeline)
        return VirnyInprocessingWrapper(pipeline=new_pipeline)

    def get_params(self):
        return self.pipeline.get_params()

    def set_params(self, **params):
        pass

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        self.pipeline.refit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)


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


def compute_metrics_with_virny(pipeline, virny_config, data_loader, X_train_val, X_test, y_train_val, y_test):
    dataset_sensitive_attrs = [k for k in virny_config.sensitive_attributes_dct.keys() if '&' not in k]
    pipeline_name = f"{SYSTEM_NAME}_best_pipeline"
    models_dct = {pipeline_name: VirnyInprocessingWrapper(pipeline)}
    main_base_flow_dataset = create_base_flow_dataset(data_loader=data_loader,
                                                      dataset_sensitive_attrs=dataset_sensitive_attrs,
                                                      X_train_val_wo_sensitive_attrs=X_train_val,
                                                      X_test_wo_sensitive_attrs=X_test,
                                                      y_train_val=y_train_val,
                                                      y_test=y_test,
                                                      numerical_columns_wo_sensitive_attrs=data_loader.numerical_columns,
                                                      categorical_columns_wo_sensitive_attrs=data_loader.categorical_columns)
    test_metrics_dct, models_fitted_bootstraps_dct = compute_metrics_with_config(dataset=main_base_flow_dataset,
                                                                                 config=virny_config,
                                                                                 models_config=models_dct,
                                                                                 notebook_logs_stdout=False,
                                                                                 with_predict_proba=False,
                                                                                 return_fitted_bootstrap=True,
                                                                                 verbose=0)
    return test_metrics_dct[pipeline_name], models_fitted_bootstraps_dct[pipeline_name]


def save_virny_metrics_in_db(pipeline, model_metrics_df: pd.DataFrame, exp_config_name: str, virny_config,
                             session_uuid: str, run_num: int, experiment_seed: int, optimization_time: float,
                             total_execution_time: float):
    secrets_path = pathlib.Path(__file__).parent.joinpath('secrets.env')
    db_client = CoreDBClient(secrets_path)
    db_client.connect()

    # Create a dict with custom fields to store in DB
    custom_tbl_fields_dct = dict()
    custom_tbl_fields_dct['session_uuid'] = session_uuid
    custom_tbl_fields_dct['system_name'] = SYSTEM_NAME
    custom_tbl_fields_dct['logical_pipeline_name'] = f"{SYSTEM_NAME}_ensemble"
    custom_tbl_fields_dct['ensemble_config'] = str(pipeline.show_models())
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


def run_exp(exp_config_name: str, dataset_name: str, run_num: int, tmp_folder_prefix: str, num_workers: int,
            memory_limit_per_worker: int = 3072, max_time_budget: int = None, max_total_pipelines_num: int = None):
    if max_total_pipelines_num is None and max_time_budget is None:
        raise ValueError("max_total_pipelines_num and max_time_budget cannot be both None")

    print("============= Input arguments =============")
    print("exp_config_name:", exp_config_name)
    print("dataset_name:", dataset_name)
    print("run_num:", run_num)
    print("tmp_folder_prefix:", tmp_folder_prefix)
    print("num_workers:", num_workers)
    print("memory_limit_per_worker:", memory_limit_per_worker)
    print("max_time_budget:", max_time_budget)
    print("max_total_pipelines_num:", max_total_pipelines_num, '\n')

    # Define configs
    session_uuid = str(uuid.uuid1())
    experiment_seed = INIT_RANDOM_STATE + run_num
    virny_config = DefaultMunch.fromDict(DATASET_CONFIG[dataset_name]["virny_config"])
    virny_config.dataset_name = dataset_name
    virny_config.random_state = experiment_seed  # Set random state for the metric computation with Virny
    validate_config(virny_config)

    # Read data
    test_set_fraction = DATASET_CONFIG[dataset_name]['test_set_fraction']
    data_loader = DATASET_CONFIG[dataset_name]['data_loader'](**DATASET_CONFIG[dataset_name]['data_loader_kwargs'])
    X_train_val, X_test, y_train_val, y_test = train_test_split(data_loader.X_data, data_loader.y_data,
                                                                test_size=test_set_fraction,
                                                                random_state=experiment_seed)
    print("Data is prepared")

    # Register custom classifiers
    autosklearn.pipeline.components.classification.add_classifier(LGBMClassifier)
    autosklearn.pipeline.components.classification.add_classifier(LogisticRegression)
    autosklearn.pipeline.components.classification.add_classifier(XGBoostClassifier)

    # Run optimization
    start = time.time()
    tmp_folder = os.path.join(tmp_folder_prefix, SYSTEM_NAME, f"workers_{num_workers}", dataset_name, str(experiment_seed))
    automl = autosklearn.classification.AutoSklearnClassifier(
        max_total_pipelines_num=max_total_pipelines_num,
        time_left_for_this_task=max_time_budget,
        tmp_folder=tmp_folder,
        delete_tmp_folder_after_terminate=False,
        n_jobs=num_workers,
        memory_limit=memory_limit_per_worker,
        seed=experiment_seed,
        ensemble_kwargs={"ensemble_size": 5} if virny_config.n_estimators > 1 else {},
        allow_string_features=False,
        include={
            "classifier": [
                "decision_tree",
                "LogisticRegression",
                "random_forest",
                "XGBoostClassifier",
                "LGBMClassifier",
            ],
            "feature_preprocessor": ["no_preprocessing"],
        },
    )
    automl.fit(X_train_val, y_train_val)
    optimization_time = time.time() - start

    # View the models found by auto-sklearn
    print()
    print(automl.leaderboard())

    # Print the final ensemble constructed by auto-sklearn
    print()
    pprint(automl.show_models(), indent=4)

    # Print statistics about the auto-sklearn run such as number of
    # iterations, number of models failed with a timeout.
    print()
    print(automl.sprint_statistics())

    # Compute virny metrics
    pipeline_overall_metrics_df, _ = compute_metrics_with_virny(pipeline=automl,
                                                                virny_config=virny_config,
                                                                data_loader=data_loader,
                                                                X_train_val=X_train_val,
                                                                X_test=X_test,
                                                                y_train_val=y_train_val,
                                                                y_test=y_test)
    total_execution_time = time.time() - start
    print("Virny metrics are computed")

    # Save results in DB
    save_virny_metrics_in_db(pipeline=automl,
                             model_metrics_df=pipeline_overall_metrics_df,
                             exp_config_name=exp_config_name,
                             virny_config=virny_config,
                             session_uuid=session_uuid,
                             run_num=run_num,
                             experiment_seed=experiment_seed,
                             optimization_time=optimization_time,
                             total_execution_time=total_execution_time)
    print(f"Virny metrics are saved in DB. Session UUID: {session_uuid}")

    shutil.rmtree(tmp_folder, ignore_errors=True)


if __name__ == "__main__":
    # Example execution command:
    # python3 -m examples.exps.run_exp test_exp diabetes 1 ./tmp/exp3 3 3072 none 10
    # OR
    # python3 -m examples.exps.run_exp test_exp diabetes 1 ./tmp/exp3 3 3072 60 none
    input_memory_limit_per_worker = sys.argv[6]
    input_max_time_budget = sys.argv[7]
    input_max_total_pipelines_num = sys.argv[8]
    run_exp(exp_config_name = sys.argv[1],
            dataset_name = sys.argv[2],
            run_num = int(sys.argv[3]),
            tmp_folder_prefix = sys.argv[4],
            num_workers = int(sys.argv[5]),
            memory_limit_per_worker = None if input_memory_limit_per_worker.lower() == 'none' else int(input_memory_limit_per_worker),
            max_time_budget = None if input_max_time_budget.lower() == 'none' else int(input_max_time_budget),
            max_total_pipelines_num = None if input_max_total_pipelines_num.lower() == 'none' else int(input_max_total_pipelines_num))
