import sys
import uuid
import time
import copy
import pathlib
import pandas as pd

from datetime import datetime, timezone
from munch import DefaultMunch


from virny.utils.common_helpers import validate_config  # type: ignore
from virny.user_interfaces.multiple_models_api import compute_metrics_with_config  # type: ignore
from virny.custom_classes.base_dataset import BaseFlowDataset  # type: ignore

from exps.core_db_client import CoreDBClient
from exps.datasets_config import DATASET_CONFIG

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from flaml import AutoML

INIT_RANDOM_STATE = 100
SYSTEM_NAME = "flaml"


def calculate_fprd(y_true, y_pred, sensitive_attr, privileged_group=None):
    """
    sensitive_attr: Series with sensitive attribute values
    privileged_group: Optional specific privileged group value to compare against

    fprd: Maximum difference from privileged group
    fprs: Dictionary of FPR per subgroup
    """
    subgroups = sensitive_attr.unique()
    fprs = {}

    for subgroup in subgroups:
        mask = (sensitive_attr == subgroup).values
        if mask.sum() == 0:
            continue

        y_true_subgroup = y_true[mask]
        y_pred_subgroup = y_pred[mask]

        try:
            tn, fp, fn, tp = confusion_matrix(
                y_true_subgroup, y_pred_subgroup, labels=[0, 1]
            ).ravel()
        except ValueError:
            continue

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fprs[subgroup] = fpr

    if len(fprs) >= 2:
        if privileged_group is not None and privileged_group in fprs:
            fprd = max(
                abs(fprs[group] - fprs[privileged_group])
                for group in fprs
                if group != privileged_group
            )
        else:
            fprd = max(fprs.values()) - min(fprs.values())
    else:
        fprd = 0.0

    return fprd, fprs


class FairnessMetric:
    def __init__(
        self, sensitive_attrs_train, sensitive_attr_name="Gender", privileged_group=None
    ):
        """
        sensitive_attrs_train: DataFrame
        sensitive_attr_name: Name of the sensitive attribute column
        privileged_group: Value of the privileged group
        """
        self.sensitive_attrs_train = sensitive_attrs_train
        self.sensitive_attr_name = sensitive_attr_name
        self.privileged_group = privileged_group

    def custom_metric_FPRD(
        self,
        X_val,
        y_val,
        estimator,
        labels,
        X_train,
        y_train,
        weight_val=None,
        weight_train=None,
        config=None,
        groups_val=None,
        groups_train=None,
    ):
        try:
            y_pred = estimator.predict(X_val)

            if hasattr(X_val, "index"):
                sensitive_val = self.sensitive_attrs_train.loc[
                    X_val.index, self.sensitive_attr_name
                ]
            else:
                sensitive_val = self.sensitive_attrs_train.iloc[range(len(X_val))][
                    self.sensitive_attr_name
                ]

            fprd, fprs = calculate_fprd(
                y_val, y_pred, sensitive_val, self.privileged_group
            )

            metric_value = -fprd

        except Exception as e:
            print(f"Error calculating FPRD: {e}")
            metric_value = 0.0
            fprd = 0.0

        return metric_value, {"FPRD": abs(fprd)}


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
        return self

    def predict(self, X):
        return self.pipeline.predict(X)


def compute_metrics_with_virny(
    pipeline, virny_config, data_loader, X_train_val, X_test, y_train_val, y_test
):
    dataset_sensitive_attrs = [
        k for k in virny_config.sensitive_attributes_dct.keys() if "&" not in k
    ]
    pipeline_name = f"{SYSTEM_NAME}_best_pipeline"
    models_dct = {pipeline_name: VirnyInprocessingWrapper(pipeline)}
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
    test_metrics_dct = compute_metrics_with_config(
        dataset=main_base_flow_dataset,
        config=virny_config,
        models_config=models_dct,
        notebook_logs_stdout=False,
        with_predict_proba=False,
        verbose=0,
    )
    return test_metrics_dct[pipeline_name]


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


def save_virny_metrics_in_db(
    pipeline,
    model_metrics_df: pd.DataFrame,
    exp_config_name: str,
    virny_config,
    session_uuid: str,
    run_num: int,
    experiment_seed: int,
    optimization_time: float,
    total_execution_time: float,
):
    secrets_path = pathlib.Path(__file__).parent.joinpath("secrets.env")
    db_client = CoreDBClient(secrets_path)
    db_client.connect()

    custom_tbl_fields_dct = dict()
    custom_tbl_fields_dct["session_uuid"] = session_uuid
    custom_tbl_fields_dct["system_name"] = SYSTEM_NAME
    custom_tbl_fields_dct["logical_pipeline_name"] = f"{SYSTEM_NAME}_ensemble"
    custom_tbl_fields_dct["dataset_split_seed"] = experiment_seed
    custom_tbl_fields_dct["model_init_seed"] = experiment_seed
    custom_tbl_fields_dct["experiment_seed"] = experiment_seed
    custom_tbl_fields_dct["exp_config_name"] = exp_config_name
    custom_tbl_fields_dct["run_num"] = run_num
    custom_tbl_fields_dct["optimization_time"] = optimization_time
    custom_tbl_fields_dct["total_execution_time"] = total_execution_time

    model_metrics_df["Dataset_Name"] = virny_config.dataset_name
    model_metrics_df["Num_Estimators"] = virny_config.n_estimators

    model_metrics_df["Tag"] = "OK"
    model_metrics_df["Record_Create_Date_Time"] = datetime.now(timezone.utc)

    for column, value in custom_tbl_fields_dct.items():
        model_metrics_df[column] = value

    subgroup_names = [
        col for col in model_metrics_df.columns if "_priv" in col or "_dis" in col
    ] + ["overall"]
    melted_model_metrics_df = model_metrics_df.melt(
        id_vars=[col for col in model_metrics_df.columns if col not in subgroup_names],
        value_vars=subgroup_names,
        var_name="Subgroup",
        value_name="Metric_Value",
    )

    melted_model_metrics_df.columns = melted_model_metrics_df.columns.str.lower()
    db_client.execute_write_query(
        records=melted_model_metrics_df.to_dict("records"),
        collection_name="all_experiment_metrics",
    )
    db_client.close()


def fute_metrics_with_virny(
    pipeline, virny_config, data_loader, X_train_val, X_test, y_train_val, y_test
):
    dataset_sensitive_attrs = [
        k for k in virny_config.sensitive_attributes_dct.keys() if "&" not in k
    ]
    pipeline_name = f"{SYSTEM_NAME}_best_pipeline"
    models_dct = {pipeline_name: VirnyInprocessingWrapper(pipeline)}
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
    test_metrics_dct = compute_metrics_with_config(
        dataset=main_base_flow_dataset,
        config=virny_config,
        models_config=models_dct,
        notebook_logs_stdout=False,
        with_predict_proba=False,
        verbose=0,
    )
    return test_metrics_dct[pipeline_name]


def run_exp(
    exp_config_name: str,
    dataset_name: str,
    run_num: int,
    tmp_folder_prefix: str,
    num_workers: int,
    memory_limit_per_worker: int = 3072,
    max_time_budget: int = None,
    max_total_pipelines_num: int = None,
):
    if max_total_pipelines_num is None and max_time_budget is None:
        raise ValueError(
            "max_total_pipelines_num and max_time_budget cannot be both None"
        )

    print("============= Input arguments =============")
    print("exp_config_name:", exp_config_name)
    print("dataset_name:", dataset_name)
    print("run_num:", run_num)
    print("tmp_folder_prefix:", tmp_folder_prefix)
    print("num_workers:", num_workers)
    print("memory_limit_per_worker:", memory_limit_per_worker)
    print("max_time_budget:", max_time_budget)
    print("max_total_pipelines_num:", max_total_pipelines_num, "\n")

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
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        data_loader.X_data,
        data_loader.y_data,
        test_size=test_set_fraction,
        random_state=experiment_seed,
    )
    print("Data is prepared")

    sensitive_attributes_dct = virny_config.sensitive_attributes_dct
    base_sensitive_attrs = {
        k: v for k, v in sensitive_attributes_dct.items() if "&" not in k
    }

    sensitive_attr_name = list(base_sensitive_attrs.keys())[0]
    disadvantaged_group = base_sensitive_attrs[sensitive_attr_name]

    unique_values = data_loader.full_df[sensitive_attr_name].unique()

    if len(unique_values) == 2:
        privileged_group = [val for val in unique_values if val != disadvantaged_group][
            0
        ]
    else:
        privileged_group = None

    print(f"Sensitive attribute: {sensitive_attr_name}")
    print(f"Disadvantaged group: {disadvantaged_group}")
    print(f"Privileged group: {privileged_group}")
    print(f"All unique values: {unique_values}")
    print()

    sensitive_attrs_df = data_loader.full_df[[sensitive_attr_name]]
    sensitive_attrs_train_val = sensitive_attrs_df.loc[X_train_val.index]

    X_train_val_reset = X_train_val.reset_index(drop=True)
    y_train_val_reset = y_train_val.reset_index(drop=True)
    sensitive_attrs_train_val_reset = sensitive_attrs_train_val.reset_index(drop=True)

    print(f"Training set shape: {X_train_val_reset.shape}")
    print(f"Sensitive attribute distribution in training set:")
    print(sensitive_attrs_train_val_reset[sensitive_attr_name].value_counts())
    print()

    custom_metric = FairnessMetric(
        sensitive_attrs_train_val_reset, sensitive_attr_name, privileged_group
    )

    automl = AutoML()
    start = time.time()
    automl.fit(
        X_train_val_reset,
        y_train_val_reset,
        task="classification",
        time_budget=max_time_budget,
        n_jobs=num_workers,
        metric=custom_metric.custom_metric_FPRD,
        log_file_name="flaml_logs.json",
        verbose=3,
    )
    optimization_time = time.time() - start

    print(f"\nOptimization completed in {optimization_time:.2f} seconds")
    print(f"Best estimator: {automl.best_estimator}")
    print(f"Best config: {automl.best_config}")
    print(f"Best FPRD: {-automl.best_loss:.6f}\n")

    pipeline_overall_metrics_df = compute_metrics_with_virny(
        pipeline=automl,
        virny_config=virny_config,
        data_loader=data_loader,
        X_train_val=X_train_val,
        X_test=X_test,
        y_train_val=y_train_val,
        y_test=y_test,
    )
    total_execution_time = time.time() - start
    print("Virny metrics are computed")

    # Save results in DB
    save_virny_metrics_in_db(
        pipeline=automl,
        model_metrics_df=pipeline_overall_metrics_df,
        exp_config_name=exp_config_name,
        virny_config=virny_config,
        session_uuid=session_uuid,
        run_num=run_num,
        experiment_seed=experiment_seed,
        optimization_time=optimization_time,
        total_execution_time=total_execution_time,
    )
    print(f"Virny metrics are saved in DB. Session UUID: {session_uuid}")


if __name__ == "__main__":
    # Example execution command:
    # python3 flaml-test.py flaml_test2 folk_emp 1 ./tmp/exp2 4 none 10 none
    # OR
    # python3 flaml-test.py flaml_test2 folk_emp 1 ./tmp/exp2 4 3072 60 none
    input_memory_limit_per_worker = sys.argv[6]
    input_max_time_budget = sys.argv[7]
    input_max_total_pipelines_num = sys.argv[8]
    run_exp(
        exp_config_name=sys.argv[1],
        dataset_name=sys.argv[2],
        run_num=int(sys.argv[3]),
        tmp_folder_prefix=sys.argv[4],
        num_workers=int(sys.argv[5]),
        memory_limit_per_worker=(
            None
            if input_memory_limit_per_worker.lower() == "none"
            else int(input_memory_limit_per_worker)
        ),
        max_time_budget=(
            None
            if input_max_time_budget.lower() == "none"
            else int(input_max_time_budget)
        ),
        max_total_pipelines_num=(
            None
            if input_max_total_pipelines_num.lower() == "none"
            else int(input_max_total_pipelines_num)
        ),
    )
