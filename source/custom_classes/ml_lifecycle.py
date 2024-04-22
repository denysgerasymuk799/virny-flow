import os
import uuid
import pathlib
import pandas as pd

from datetime import datetime, timezone
from sklearn.metrics import mean_squared_error, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from virny.utils.custom_initializers import create_config_obj

from configs.models_config_for_tuning import get_models_params_for_tuning
from configs.null_imputers_config import NULL_IMPUTERS_CONFIG, NULL_IMPUTERS_HYPERPARAMS
from configs.constants import (MODEL_HYPER_PARAMS_COLLECTION_NAME, IMPUTATION_PERFORMANCE_METRICS_COLLECTION_NAME,
                               NUM_FOLDS_FOR_TUNING, ErrorRepairMethod, ErrorInjectionStrategy)
from configs.datasets_config import DATASET_CONFIG
from configs.scenarios_config import ERROR_INJECTION_SCENARIOS_CONFIG
from source.utils.custom_logger import get_logger
from source.utils.dataframe_utils import calculate_kl_divergence
from source.utils.model_tuning_utils import tune_ML_models
from source.utils.common_helpers import (generate_guid, create_base_flow_dataset, get_injection_scenarios)
from source.custom_classes.database_client import DatabaseClient
from source.error_injectors.nulls_injector import NullsInjector
from source.validation import is_in_enum


class MLLifecycle:
    """
    Class encapsulates all required ML lifecycle steps to run different experiments
    """
    def __init__(self, dataset_name: str, null_imputers: list, model_names: list):
        """
        Constructor defining default variables
        """
        self.null_imputers = null_imputers
        self.model_names = model_names
        self.dataset_name = dataset_name

        self.num_folds_for_tuning = NUM_FOLDS_FOR_TUNING
        self.test_set_fraction = DATASET_CONFIG[dataset_name]['test_set_fraction']
        self.virny_config = create_config_obj(DATASET_CONFIG[dataset_name]['virny_config_path'])
        self.dataset_sensitive_attrs = [col for col in self.virny_config.sensitive_attributes_dct.keys() if '&' not in col]
        self.init_data_loader = DATASET_CONFIG[dataset_name]['data_loader'](**DATASET_CONFIG[dataset_name]['data_loader_kwargs'])

        self._logger = get_logger()
        self._db = DatabaseClient()
        # Create a unique uuid per session to manipulate in the database
        # by all experimental results generated in this session
        self._session_uuid = str(uuid.uuid1())
        print('Session UUID for all results of experiments in the current benchmark session:', self._session_uuid)

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

    def _tune_ML_models(self, model_names, base_flow_dataset, experiment_seed,
                        evaluation_scenario, null_imputer_name):
        # Get hyper-parameters for tuning. Each time reinitialize an init model and its hyper-params for tuning.
        all_models_params_for_tuning = get_models_params_for_tuning(experiment_seed)
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
            lambda model_name: generate_guid(ordered_hierarchy_lst=[self.dataset_name, null_imputer_name,
                                                                    evaluation_scenario, experiment_seed, model_name])
        )
        self._db.write_pandas_df_into_db(collection_name=MODEL_HYPER_PARAMS_COLLECTION_NAME,
                                         df=tuned_params_df,
                                         custom_tbl_fields_dct={
                                             'exp_pipeline_guid': generate_guid(ordered_hierarchy_lst=[self.dataset_name, null_imputer_name, evaluation_scenario, experiment_seed]),
                                             'session_uuid': self._session_uuid,
                                             'null_imputer_name': null_imputer_name,
                                             'evaluation_scenario': evaluation_scenario,
                                             'experiment_seed': experiment_seed,
                                             'record_create_date_time': date_time_str,
                                         })
        self._logger.info("Models are tuned and their hyper-params are saved into a database")

        return models_config

    def _inject_nulls_into_one_set(self, df: pd.DataFrame, injection_scenario: str, experiment_seed: int):
        injection_strategy, error_rate_str = injection_scenario[:-1], injection_scenario[-1]
        error_rate_idx = int(error_rate_str) - 1
        for scenario_for_dataset in ERROR_INJECTION_SCENARIOS_CONFIG[self.dataset_name][injection_strategy]:
            error_rate = scenario_for_dataset['setting']['error_rates'][error_rate_idx]
            condition = None if injection_strategy == ErrorInjectionStrategy.mcar.value else scenario_for_dataset['setting']['condition']
            nulls_injector = NullsInjector(seed=experiment_seed,
                                           strategy=injection_strategy,
                                           columns_with_nulls=scenario_for_dataset['missing_features'],
                                           null_percentage=error_rate,
                                           condition=condition)
            df = nulls_injector.fit_transform(df)

        return df

    def _inject_nulls(self, X_train_val: pd.DataFrame, X_test: pd.DataFrame, evaluation_scenario: str, experiment_seed: int):
        train_injection_scenario, test_injection_scenarios_lst = get_injection_scenarios(evaluation_scenario)

        X_train_val_with_nulls = self._inject_nulls_into_one_set(df=X_train_val,
                                                                 injection_scenario=train_injection_scenario,
                                                                 experiment_seed=experiment_seed)
        X_tests_with_nulls_lst = list(map(
            lambda test_injection_strategy: self._inject_nulls_into_one_set(df=X_test,
                                                                            injection_scenario=test_injection_strategy,
                                                                            experiment_seed=experiment_seed),
            test_injection_scenarios_lst
        ))
        self._logger.info('Nulls are successfully injected')

        return X_train_val_with_nulls, X_tests_with_nulls_lst

    def _impute_nulls(self, X_train_with_nulls, X_tests_with_nulls_lst, null_imputer_name, evaluation_scenario,
                      experiment_seed, numerical_columns, categorical_columns, tune_imputers):
        if not is_in_enum(null_imputer_name, ErrorRepairMethod) or null_imputer_name not in NULL_IMPUTERS_CONFIG.keys():
            raise ValueError(f'{null_imputer_name} null imputer is not implemented')

        if tune_imputers:
            hyperparams = None
        else:
            train_injection_strategy, _ = get_injection_scenarios(evaluation_scenario)
            hyperparams = NULL_IMPUTERS_HYPERPARAMS.get(null_imputer_name, {}).get(self.dataset_name, {}).get(train_injection_strategy, {})

        # Use a method, kwargs, and hyperparams from NULL_IMPUTERS_CONFIG
        imputation_method = NULL_IMPUTERS_CONFIG[null_imputer_name]["method"]
        imputation_kwargs = NULL_IMPUTERS_CONFIG[null_imputer_name]["kwargs"]
        imputation_kwargs.update({'experiment_seed': experiment_seed})

        # TODO: Save a result imputed dataset in imputed_data_dict for each imputation technique
        train_set_cols_with_nulls = X_train_with_nulls.columns[X_train_with_nulls.isna().any()].tolist()
        train_numerical_null_columns = list(set(train_set_cols_with_nulls).intersection(numerical_columns))
        train_categorical_null_columns = list(set(train_set_cols_with_nulls).intersection(categorical_columns))

        imputation_start_time = datetime.now()
        if null_imputer_name == ErrorRepairMethod.datawig.value:
            output_path = (pathlib.Path(__file__).parent.parent.parent.joinpath('results')
                           .joinpath(null_imputer_name)
                           .joinpath(self.dataset_name)
                           .joinpath(evaluation_scenario)
                           .joinpath(str(experiment_seed)))
            X_train_imputed, X_tests_imputed_lst, null_imputer_params_dct = (
                imputation_method(X_train_with_nulls=X_train_with_nulls,
                                  X_tests_with_nulls_lst=X_tests_with_nulls_lst,
                                  numeric_columns_with_nulls=train_numerical_null_columns,
                                  categorical_columns_with_nulls=train_categorical_null_columns,
                                  all_numeric_columns=numerical_columns,
                                  all_categorical_columns=categorical_columns,
                                  hyperparams=hyperparams,
                                  output_path=output_path,
                                  **imputation_kwargs))

        elif null_imputer_name == ErrorRepairMethod.automl.value:
            output_path = (pathlib.Path(__file__).parent.parent.parent.joinpath('results')
                           .joinpath(null_imputer_name)
                           .joinpath(self.dataset_name)
                           .joinpath(evaluation_scenario)
                           .joinpath(str(experiment_seed)))
            imputation_kwargs.update({'directory': output_path})
            X_train_imputed, X_tests_imputed_lst, null_imputer_params_dct = (
                imputation_method(X_train_with_nulls=X_train_with_nulls,
                                  X_tests_with_nulls_lst=X_tests_with_nulls_lst,
                                  numeric_columns_with_nulls=train_numerical_null_columns,
                                  categorical_columns_with_nulls=train_categorical_null_columns,
                                  hyperparams=hyperparams,
                                  **imputation_kwargs))

        else:
            X_train_imputed, X_tests_imputed_lst, null_imputer_params_dct = (
                imputation_method(X_train_with_nulls=X_train_with_nulls,
                                  X_tests_with_nulls_lst=X_tests_with_nulls_lst,
                                  numeric_columns_with_nulls=train_numerical_null_columns,
                                  categorical_columns_with_nulls=train_categorical_null_columns,
                                  hyperparams=hyperparams,
                                  **imputation_kwargs))

        imputation_end_time = datetime.now()
        imputation_runtime = (imputation_end_time - imputation_start_time).total_seconds() / 60.0
        self._logger.info('Nulls are successfully imputed')

        return X_train_imputed, X_tests_imputed_lst, null_imputer_params_dct, imputation_runtime

    def _evaluate_imputation(self, real, imputed, corrupted, numerical_columns, null_imputer_name, null_imputer_params_dct):
        columns_with_nulls = corrupted.columns[corrupted.isna().any()].tolist()
        metrics_df = pd.DataFrame(columns=('Dataset_Name', 'Null_Imputer_Name', 'Null_Imputer_Params',
                                           'Column_Type', 'Column_With_Nulls', 'KL_Divergence_Pred',
                                           'KL_Divergence_Total', 'RMSE', 'Precision', 'Recall', 'F1_Score'))
        for column_idx, column_name in enumerate(columns_with_nulls):
            column_type = 'numerical' if column_name in numerical_columns else 'categorical'

            indexes = corrupted[column_name].isna()
            true = real.loc[indexes, column_name]
            pred = imputed.loc[indexes, column_name]

            # Column type agnostic metrics
            kl_divergence_pred = calculate_kl_divergence(true, pred)
            print('Predictive KL divergence for {}: {:.2f}'.format(column_name, kl_divergence_pred))
            kl_divergence_total = calculate_kl_divergence(real[column_name], imputed[column_name])
            print('Total KL divergence for {}: {:.2f}'.format(column_name, kl_divergence_total))

            rmse = None
            precision = None
            recall = None
            f1 = None
            if column_type == 'numerical':
                null_imputer_params = null_imputer_params_dct[column_name] if null_imputer_params_dct is not None else None
                rmse = mean_squared_error(true, pred, squared=False)
                print('RMSE for {}: {:.2f}'.format(column_name, rmse))
                print()
            else:
                null_imputer_params = null_imputer_params_dct[column_name] if null_imputer_params_dct is not None else None
                precision, recall, f1, _ = precision_recall_fscore_support(true, pred, average="micro")
                print('Precision for {}: {:.2f}'.format(column_name, precision))
                print('Recall for {}: {:.2f}'.format(column_name, recall))
                print('F1 score for {}: {:.2f}'.format(column_name, f1))
                print()

            # Save imputation performance metric of the imputer in a dataframe
            metrics_df.loc[column_idx] = [self.dataset_name, null_imputer_name, null_imputer_params,
                                          column_type, column_name, kl_divergence_pred, kl_divergence_total,
                                          rmse, precision, recall, f1]

        return metrics_df

    def _save_imputation_metrics_to_db(self, train_imputation_metrics_df: pd.DataFrame, test_imputation_metrics_dfs_lst: list,
                                       imputation_runtime: float, null_imputer_name: str, evaluation_scenario: str, experiment_seed: int):
        train_imputation_metrics_df['Imputation_Guid'] = train_imputation_metrics_df['Column_With_Nulls'].apply(
            lambda column_with_nulls: generate_guid(ordered_hierarchy_lst=[self.dataset_name, null_imputer_name,
                                                                           evaluation_scenario, experiment_seed,
                                                                           'X_train_val', column_with_nulls])
        )
        self._db.write_pandas_df_into_db(collection_name=IMPUTATION_PERFORMANCE_METRICS_COLLECTION_NAME,
                                         df=train_imputation_metrics_df,
                                         custom_tbl_fields_dct={
                                             'exp_pipeline_guid': generate_guid(ordered_hierarchy_lst=[self.dataset_name, null_imputer_name, evaluation_scenario, experiment_seed]),
                                             'session_uuid': self._session_uuid,
                                             'evaluation_scenario': evaluation_scenario,
                                             'experiment_seed': experiment_seed,
                                             'dataset_part': 'X_train_val',
                                             'runtime_in_mins': imputation_runtime,
                                             'record_create_date_time': datetime.now(timezone.utc),
                                         })

        # Save imputation results into a database for each test set from the evaluation scenario
        _, test_injection_scenarios_lst = get_injection_scenarios(evaluation_scenario)
        test_record_create_date_time = datetime.now(timezone.utc)
        for test_set_idx, test_imputation_metrics_df in enumerate(test_imputation_metrics_dfs_lst):
            test_injection_scenario = test_injection_scenarios_lst[test_set_idx]
            test_imputation_metrics_df['Imputation_Guid'] = test_imputation_metrics_df['Column_With_Nulls'].apply(
                lambda column_with_nulls: generate_guid(ordered_hierarchy_lst=[self.dataset_name, null_imputer_name,
                                                                               evaluation_scenario, experiment_seed,
                                                                               f'X_test_{test_injection_scenario}',
                                                                               column_with_nulls])
            )
            self._db.write_pandas_df_into_db(collection_name=IMPUTATION_PERFORMANCE_METRICS_COLLECTION_NAME,
                                             df=test_imputation_metrics_df,
                                             custom_tbl_fields_dct={
                                                 'exp_pipeline_guid': generate_guid(ordered_hierarchy_lst=[self.dataset_name, null_imputer_name, evaluation_scenario, experiment_seed]),
                                                 'session_uuid': self._session_uuid,
                                                 'evaluation_scenario': evaluation_scenario,
                                                 'experiment_seed': experiment_seed,
                                                 'dataset_part': f'X_test_{test_injection_scenario}',
                                                 'runtime_in_mins': imputation_runtime,
                                                 'record_create_date_time': test_record_create_date_time,
                                            })

        self._logger.info("Performance metrics and tuned parameters of the null imputer are saved into a database")

    def _save_imputed_datasets_to_fs(self, X_train_val: pd.DataFrame, X_tests_lst: pd.DataFrame,
                                     null_imputer_name: str, evaluation_scenario: str, experiment_seed: int):
        save_sets_dir_path = (pathlib.Path(__file__).parent.parent.parent
                              .joinpath('results')
                              .joinpath(self.dataset_name)
                              .joinpath(null_imputer_name)
                              .joinpath(evaluation_scenario)
                              .joinpath(str(experiment_seed)))
        os.makedirs(save_sets_dir_path, exist_ok=True)

        train_set_filename = f'imputed_{self.dataset_name}_{null_imputer_name}_{evaluation_scenario}_{experiment_seed}_X_train_val.csv'
        X_train_val.to_csv(os.path.join(save_sets_dir_path, train_set_filename),
                           sep=",",
                           columns=X_train_val.columns,
                           index=True)

        # Save each imputed test set in a local filesystem
        _, test_injection_scenarios_lst = get_injection_scenarios(evaluation_scenario)
        for test_set_idx, X_test in enumerate(X_tests_lst):
            test_injection_scenario = test_injection_scenarios_lst[test_set_idx]
            test_set_filename = f'imputed_{self.dataset_name}_{null_imputer_name}_{evaluation_scenario}_{experiment_seed}_X_test_{test_injection_scenario}.csv'
            X_test.to_csv(os.path.join(save_sets_dir_path, test_set_filename),
                          sep=",",
                          columns=X_test.columns,
                          index=True)

        self._logger.info("Imputed train and test sets are saved locally")

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
