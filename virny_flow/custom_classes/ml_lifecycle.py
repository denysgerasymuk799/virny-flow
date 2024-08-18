import uuid
import shutil
import pathlib
import pandas as pd

from datetime import datetime, timezone
from sklearn.model_selection import train_test_split
from virny.utils.custom_initializers import create_config_obj

from virny_flow.configs.null_imputers_config import NULL_IMPUTERS_CONFIG
from virny_flow.configs.constants import MODEL_HYPER_PARAMS_COLLECTION_NAME, NUM_FOLDS_FOR_TUNING, ErrorRepairMethod
from virny_flow.utils.custom_logger import get_logger
from virny_flow.utils.model_tuning_utils import tune_ML_models
from virny_flow.utils.common_helpers import generate_guid, create_base_flow_dataset
from virny_flow.custom_classes.database_client import DatabaseClient
from virny_flow.custom_classes.s3_client import S3Client
from virny_flow.validation import is_in_enum


class MLLifecycle:
    """
    Class encapsulates all required ML lifecycle steps to run different experiments
    """
    def __init__(self, exp_config_name: str, dataset_name: str, secrets_path: str,
                 dataset_config: dict, model_params_for_tuning: dict):
        """
        Constructor defining default variables
        """
        self.exp_config_name = exp_config_name
        self.dataset_name = dataset_name
        self.model_params_for_tuning = model_params_for_tuning

        self.num_folds_for_tuning = NUM_FOLDS_FOR_TUNING
        self.test_set_fraction = dataset_config[dataset_name]['test_set_fraction']
        self.virny_config = create_config_obj(dataset_config[dataset_name]['virny_config_path'])
        self.dataset_sensitive_attrs = [col for col in self.virny_config.sensitive_attributes_dct.keys() if '&' not in col]
        self.init_data_loader = dataset_config[dataset_name]['data_loader'](**dataset_config[dataset_name]['data_loader_kwargs'])

        self._logger = get_logger(logger_name='ml_lifecycle')
        self._db = DatabaseClient(secrets_path)
        self._s3_client = S3Client(secrets_path)
        # Create a unique uuid per session to manipulate in the database
        # by all experimental results generated in this session
        self._session_uuid = str(uuid.uuid1())
        print('Session UUID for all results of experiments in the current session:', self._session_uuid)

    def _split_dataset(self, data_loader, experiment_seed: int):
        if self.test_set_fraction < 0.0 or self.test_set_fraction > 1.0:
            raise ValueError("test_set_fraction must be a float in the [0.0-1.0] range")

        X_train_val, X_test, y_train_val, y_test = train_test_split(data_loader.X_data, data_loader.y_data,
                                                                    test_size=self.test_set_fraction,
                                                                    random_state=experiment_seed)
        return X_train_val, X_test, y_train_val, y_test

    def _remove_sensitive_attrs(self, X_train_val: pd.DataFrame, X_tests_lst: list, data_loader):
        X_train_val_wo_sensitive_attrs = X_train_val.drop(self.dataset_sensitive_attrs, axis=1)
        X_tests_wo_sensitive_attrs_lst = list(map(
            lambda X_test: X_test.drop(self.dataset_sensitive_attrs, axis=1),
            X_tests_lst
        ))
        numerical_columns_wo_sensitive_attrs = [col for col in data_loader.numerical_columns if col not in self.dataset_sensitive_attrs]
        categorical_columns_wo_sensitive_attrs = [col for col in data_loader.categorical_columns if col not in self.dataset_sensitive_attrs]

        return (X_train_val_wo_sensitive_attrs, X_tests_wo_sensitive_attrs_lst,
                numerical_columns_wo_sensitive_attrs, categorical_columns_wo_sensitive_attrs)

    def _prepare_baseline_dataset(self, data_loader, experiment_seed: int):
        # Split the dataset
        X_train_val, X_test, y_train_val, y_test = self._split_dataset(data_loader, experiment_seed)

        # Remove sensitive attributes from train and test sets with nulls to avoid their usage during model training
        (X_train_val_wo_sensitive_attrs,
         X_tests_wo_sensitive_attrs_lst,
         numerical_columns_wo_sensitive_attrs,
         categorical_columns_wo_sensitive_attrs) = self._remove_sensitive_attrs(X_train_val=X_train_val,
                                                                                X_tests_lst=[X_test],
                                                                                data_loader=data_loader)
        X_test_wo_sensitive_attrs = X_tests_wo_sensitive_attrs_lst[0]

        # Create a base flow dataset for Virny to compute metrics
        base_flow_dataset = create_base_flow_dataset(data_loader=data_loader,
                                                     dataset_sensitive_attrs=self.dataset_sensitive_attrs,
                                                     X_train_val_wo_sensitive_attrs=X_train_val_wo_sensitive_attrs,
                                                     X_test_wo_sensitive_attrs=X_test_wo_sensitive_attrs,
                                                     y_train_val=y_train_val,
                                                     y_test=y_test,
                                                     numerical_columns_wo_sensitive_attrs=numerical_columns_wo_sensitive_attrs,
                                                     categorical_columns_wo_sensitive_attrs=categorical_columns_wo_sensitive_attrs)

        return base_flow_dataset

    def _impute_nulls(self, X_train_with_nulls, X_tests_with_nulls_lst, null_imputer_name, evaluation_scenario,
                      experiment_seed, numerical_columns, categorical_columns, tune_imputers):
        if not is_in_enum(null_imputer_name, ErrorRepairMethod) or null_imputer_name not in NULL_IMPUTERS_CONFIG.keys():
            raise ValueError(f'{null_imputer_name} null imputer is not implemented')

        # Use a method, kwargs, and hyperparams from NULL_IMPUTERS_CONFIG
        hyperparams = None
        imputation_method = NULL_IMPUTERS_CONFIG[null_imputer_name]["method"]
        imputation_kwargs = NULL_IMPUTERS_CONFIG[null_imputer_name]["kwargs"]
        imputation_kwargs.update({'experiment_seed': experiment_seed})
        imputation_kwargs.update({'dataset_name': self.dataset_name})

        train_set_cols_with_nulls = X_train_with_nulls.columns[X_train_with_nulls.isna().any()].tolist()
        train_numerical_null_columns = list(set(train_set_cols_with_nulls).intersection(numerical_columns))
        train_categorical_null_columns = list(set(train_set_cols_with_nulls).intersection(categorical_columns))

        imputation_start_time = datetime.now()
        if null_imputer_name == ErrorRepairMethod.datawig.value:
            if evaluation_scenario is None:
                output_path = (pathlib.Path(__file__).parent.parent.parent
                               .joinpath('intermediate_state')
                               .joinpath(self.exp_config_name)
                               .joinpath('null_imputation_stage')
                               .joinpath(null_imputer_name)
                               .joinpath(self.dataset_name))
            else:
                output_path = (pathlib.Path(__file__).parent.parent.parent
                               .joinpath('results')
                               .joinpath('intermediate_state')
                               .joinpath(null_imputer_name)
                               .joinpath(self.dataset_name)
                               .joinpath(evaluation_scenario)
                               .joinpath(str(experiment_seed)))
            X_train_imputed, X_tests_imputed_lst, null_imputer_params_dct, null_imputer = (
                imputation_method(X_train_with_nulls=X_train_with_nulls,
                                  X_tests_with_nulls_lst=X_tests_with_nulls_lst,
                                  numeric_columns_with_nulls=train_numerical_null_columns,
                                  categorical_columns_with_nulls=train_categorical_null_columns,
                                  all_numeric_columns=numerical_columns,
                                  all_categorical_columns=categorical_columns,
                                  hyperparams=hyperparams,
                                  output_path=output_path,
                                  **imputation_kwargs))
            # Remove all files created by datawig to save storage space
            shutil.rmtree(output_path)

        elif null_imputer_name == ErrorRepairMethod.automl.value:
            if evaluation_scenario is None:
                output_path = (pathlib.Path(__file__).parent.parent.parent
                               .joinpath('intermediate_state')
                               .joinpath(self.exp_config_name)
                               .joinpath('null_imputation_stage')
                               .joinpath(null_imputer_name)
                               .joinpath(self.dataset_name))
            else:
                output_path = (pathlib.Path(__file__).parent.parent.parent
                               .joinpath('results')
                               .joinpath('intermediate_state')
                               .joinpath(null_imputer_name)
                               .joinpath(self.dataset_name)
                               .joinpath(evaluation_scenario)
                               .joinpath(str(experiment_seed)))

            imputation_kwargs.update({'directory': output_path})
            X_train_imputed, X_tests_imputed_lst, null_imputer_params_dct, null_imputer = (
                imputation_method(X_train_with_nulls=X_train_with_nulls,
                                  X_tests_with_nulls_lst=X_tests_with_nulls_lst,
                                  numeric_columns_with_nulls=train_numerical_null_columns,
                                  categorical_columns_with_nulls=train_categorical_null_columns,
                                  hyperparams=hyperparams,
                                  **imputation_kwargs))
            # Remove all files created by automl to save storage space
            shutil.rmtree(output_path)

        else:
            X_train_imputed, X_tests_imputed_lst, null_imputer_params_dct, null_imputer = (
                imputation_method(X_train_with_nulls=X_train_with_nulls,
                                  X_tests_with_nulls_lst=X_tests_with_nulls_lst,
                                  numeric_columns_with_nulls=train_numerical_null_columns,
                                  categorical_columns_with_nulls=train_categorical_null_columns,
                                  hyperparams=hyperparams,
                                  **imputation_kwargs))

        imputation_end_time = datetime.now()
        imputation_runtime = (imputation_end_time - imputation_start_time).total_seconds() / 60.0
        self._logger.info('Nulls are successfully imputed')

        return X_train_imputed, X_tests_imputed_lst, null_imputer_params_dct, null_imputer, imputation_runtime

    def _tune_ML_models(self, model_names, base_flow_dataset, evaluation_scenario,
                        experiment_seed, null_imputer_name, fairness_intervention_name):
        # Get hyper-parameters for tuning. Each time reinitialize an init model and its hyper-params for tuning.
        all_models_params_for_tuning = self.model_params_for_tuning
        models_params_for_tuning = {model_name: all_models_params_for_tuning[model_name] for model_name in model_names}

        # Tune models and create a models config for metrics computation
        tuned_params_df, models_config = tune_ML_models(models_params_for_tuning=models_params_for_tuning,
                                                        base_flow_dataset=base_flow_dataset,
                                                        dataset_name=self.virny_config.dataset_name,
                                                        n_folds=self.num_folds_for_tuning)

        # Save tunes parameters in database
        date_time_str = datetime.now(timezone.utc)
        tuned_params_df['Model_Best_Params'] = tuned_params_df['Model_Best_Params']
        tuned_params_df['Model_Tuning_Guid'] = tuned_params_df['Model_Name'].apply(
            lambda model_name: generate_guid(ordered_hierarchy_lst=[self.exp_config_name, evaluation_scenario, experiment_seed,
                                                                    self.dataset_name, null_imputer_name, fairness_intervention_name, model_name])
        )
        self._db.write_pandas_df_into_db(collection_name=MODEL_HYPER_PARAMS_COLLECTION_NAME,
                                         df=tuned_params_df,
                                         custom_tbl_fields_dct={
                                             'exp_pipeline_guid': generate_guid(ordered_hierarchy_lst=[self.exp_config_name, evaluation_scenario, experiment_seed, self.dataset_name, null_imputer_name, fairness_intervention_name]),
                                             'session_uuid': self._session_uuid,
                                             'exp_config_name': self.exp_config_name,
                                             'evaluation_scenario': evaluation_scenario,
                                             'experiment_seed': experiment_seed,
                                             'null_imputer_name': null_imputer_name,
                                             'fairness_intervention_name': fairness_intervention_name,
                                             'record_create_date_time': date_time_str,
                                         })
        self._logger.info("Models are tuned and their hyper-params are saved into the database")

        return models_config
