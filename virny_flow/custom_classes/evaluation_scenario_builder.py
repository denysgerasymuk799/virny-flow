import copy
import numpy as np
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from virny.custom_classes import BaseFlowDataset
from virny.user_interfaces import compute_metrics_with_fitted_bootstrap

from virny_flow.utils.custom_logger import get_logger
from virny_flow.custom_classes.ml_lifecycle import MLLifecycle
from virny_flow.error_injectors.nulls_injector import NullsInjector
from virny_flow.utils.pipeline_utils import get_dis_group_condition
from virny_flow.configs.constants import (ErrorType, STAGE_SEPARATOR, S3Folder, FairnessIntervention,
                                          NO_FAIRNESS_INTERVENTION)


class EvaluationScenarioBuilder(MLLifecycle):
    def __init__(self, exp_config, dataset_config: dict):
        """
        Constructor defining default variables
        """
        super().__init__(exp_config_name=exp_config.exp_config_name,
                         dataset_name=exp_config.dataset,
                         secrets_path=exp_config.secrets_path,
                         dataset_config=dataset_config,
                         model_params_for_tuning=None)

        self.exp_config = exp_config
        self._logger = get_logger(logger_name='evaluation_scenario_builder')

    def _validate_evaluation_scenario(self, evaluation_scenario: list):
        required_params = {'columns_to_inject', 'error_type', 'condition', 'error_rate'}
        for injection_scenario in evaluation_scenario:
            if not required_params.issubset(set(injection_scenario.keys())):
                raise ValueError(f'Not all parameters are defined in the input evaluation scenario. '
                                 f'The required parameters are {required_params}.')
            if not isinstance(injection_scenario['columns_to_inject'], str):
                raise ValueError('The columns_to_inject parameter should be string.')
            if not isinstance(injection_scenario['condition'], str):
                raise ValueError('The condition parameter must be string.')
            if not (isinstance(injection_scenario['error_rate'], float) and 0.0 < injection_scenario['error_rate'] < 1.0):
                raise ValueError('The error_rate parameter must be float from the (0.0-1.0) range.')
            if not (isinstance(injection_scenario['error_type'], str) and ErrorType.has_value(injection_scenario['error_type'])):
                raise ValueError('The error_type parameter must be a value from the ErrorType enum.')

    def test_evaluation_scenario(self, evaluation_scenario_name: str, evaluation_scenario: list, experiment_seed: int):
        self._db.connect()

        # Read pipeline names from the database
        pipeline_names = self._db.read_pipeline_names(exp_config_name=self.exp_config_name)
        # Test each ml pipeline using the defined evaluation scenario
        for pipeline_name in pipeline_names:
            null_imputer_name, fairness_intervention_name, model_name = pipeline_name.split(STAGE_SEPARATOR)
            self._test_ml_pipeline(evaluation_scenario_name=evaluation_scenario_name,
                                   evaluation_scenario=evaluation_scenario,
                                   null_imputer_name=null_imputer_name,
                                   fairness_intervention_name=fairness_intervention_name,
                                   model_name=model_name,
                                   experiment_seed=experiment_seed)

        self._db.close()
        self._logger.info(f"Evaluation scenario {evaluation_scenario_name} was successfully executed!")

    def _test_ml_pipeline(self, evaluation_scenario_name: str, evaluation_scenario: list, null_imputer_name: str,
                          fairness_intervention_name: str, model_name: str, experiment_seed: int):
        # Init pipeline components
        pipeline_name = f'{null_imputer_name}&{fairness_intervention_name}&{model_name}'
        data_loader = copy.deepcopy(self.init_data_loader)
        (null_imputer, column_transformer, fair_preprocessor, fitted_bootstrap,
         train_numerical_null_columns, train_categorical_null_columns, preprocessed_train_set_columns) =\
            self._read_pipeline_components(null_imputer_name=null_imputer_name,
                                           fairness_intervention_name=fairness_intervention_name,
                                           model_name=model_name)

        # Split the dataset
        _, X_test_with_nulls, _, y_test = self._split_dataset(data_loader, experiment_seed)

        # Implement an evaluation scenario
        X_test_with_target_errors = self.implement_evaluation_scenario(df=X_test_with_nulls,
                                                                       evaluation_scenario=evaluation_scenario,
                                                                       experiment_seed=experiment_seed)

        # Remove sensitive attributes from train and test sets with nulls to avoid their usage during imputation
        _, X_test_with_target_errors_wo_sensitive_attrs_lst, _, _ = self._remove_sensitive_attrs(X_train_val=pd.DataFrame(),
                                                                                                 X_tests_lst=[X_test_with_target_errors],
                                                                                                 data_loader=data_loader)
        X_test_with_target_errors_wo_sensitive_attrs = X_test_with_target_errors_wo_sensitive_attrs_lst[0]

        # Impute nulls
        X_test = self._impute_nulls_in_test_set(null_imputer_name=null_imputer_name,
                                                null_imputer=null_imputer,
                                                X_test=X_test_with_target_errors_wo_sensitive_attrs,
                                                numerical_null_columns=train_numerical_null_columns,
                                                categorical_null_columns=train_categorical_null_columns)

        # Apply column transformer
        X_test = column_transformer.transform(X_test)

        # Reorder and add missing columns to the new data
        for col in preprocessed_train_set_columns:
            if col not in X_test.columns:
                X_test[col] = 0  # or some default missing value for the column type

        X_test = X_test[preprocessed_train_set_columns]

        # Apply pre-processing fairness-enhancing intervention
        if fairness_intervention_name != NO_FAIRNESS_INTERVENTION:
            # Add a binary sensitive attribute for intervention to X_test
            input_sensitive_attrs_for_intervention = self.exp_config.sensitive_attrs_for_intervention
            binary_sensitive_attr_for_intervention = '&'.join(input_sensitive_attrs_for_intervention) + '_binary'
            X_test = self._add_sensitive_attr_for_intervention(X_test=X_test,
                                                               sensitive_attr_for_intervention=binary_sensitive_attr_for_intervention,
                                                               data_loader=data_loader,
                                                               experiment_seed=experiment_seed)

        if fairness_intervention_name in (FairnessIntervention.DIR.value, FairnessIntervention.LFR.value):
            X_test = self._apply_fair_preprocessor(X_test=X_test,
                                                   y_test=y_test,
                                                   target_column=data_loader.target,
                                                   fairness_intervention_name=fairness_intervention_name,
                                                   fair_preprocessor=fair_preprocessor,
                                                   sensitive_attr_for_intervention=binary_sensitive_attr_for_intervention)

        # Compute metrics with Virny using the fitted bootstrap
        self.virny_config.random_state = experiment_seed  # Set random state for the metric computation with Virny
        test_base_flow_dataset = BaseFlowDataset(init_sensitive_attrs_df=data_loader.full_df[self.dataset_sensitive_attrs],
                                                 X_train_val=pd.DataFrame(),
                                                 X_test=X_test,
                                                 y_train_val=pd.DataFrame(),
                                                 y_test=y_test,
                                                 target=data_loader.target,
                                                 numerical_columns=data_loader.numerical_columns,
                                                 categorical_columns=data_loader.categorical_columns)
        metrics_df = compute_metrics_with_fitted_bootstrap(fitted_bootstrap=fitted_bootstrap,
                                                           test_base_flow_dataset=test_base_flow_dataset,
                                                           config=self.virny_config,
                                                           with_predict_proba=True)

        # Store resulted metrics in S3
        target_dir_path = f'{S3Folder.experiments.value}/{self.exp_config_name}/{S3Folder.evaluation_scenarios.value}/{evaluation_scenario_name}'
        metrics_filename = f'{pipeline_name}_metrics.csv'
        self._s3_client.write_csv(metrics_df, f'{target_dir_path}/{metrics_filename}', index=True)

    def _read_pipeline_components(self, null_imputer_name: str, fairness_intervention_name: str, model_name: str):
        pipeline_name = f'{null_imputer_name}&{fairness_intervention_name}&{model_name}'
        source_dir_path = f'{S3Folder.experiments.value}/{self.exp_config_name}/{S3Folder.artifacts.value}/{pipeline_name}'
        null_imputer = self._s3_client.read_pickle(f'{source_dir_path}/null_imputer.pkl')
        column_transformer = self._s3_client.read_pickle(f'{source_dir_path}/column_transformer.pkl')
        fair_preprocessor = self._s3_client.read_pickle(f'{source_dir_path}/fair_preprocessor.pkl')\
            if fairness_intervention_name in (FairnessIntervention.DIR.value, FairnessIntervention.LFR.value) else None
        fitted_model_bootstrap = self._s3_client.read_pickle(f'{source_dir_path}/fitted_model_bootstrap.pkl')
        train_numerical_null_columns = self._s3_client.read_pickle(f'{source_dir_path}/train_numerical_null_columns.pkl')
        train_categorical_null_columns = self._s3_client.read_pickle(f'{source_dir_path}/train_categorical_null_columns.pkl')
        preprocessed_train_set_columns = self._s3_client.read_pickle(f'{source_dir_path}/preprocessed_train_set_columns.pkl')

        return (null_imputer, column_transformer, fair_preprocessor, fitted_model_bootstrap,
                train_numerical_null_columns, train_categorical_null_columns, preprocessed_train_set_columns)

    def implement_evaluation_scenario(self, df: pd.DataFrame, evaluation_scenario: list, experiment_seed: int):
        # Validate user input
        self._validate_evaluation_scenario(evaluation_scenario)

        # Parse the input evaluation scenario
        parsed_evaluation_scenario = dict()
        for injection_scenario in evaluation_scenario:
            # Parse user inputs
            parsed_injection_scenario = {
                'columns_to_inject': injection_scenario['columns_to_inject'].split(','),
                'condition': injection_scenario['condition'],
                'error_rate': injection_scenario['error_rate'],
            }
            parsed_evaluation_scenario.setdefault(injection_scenario['error_type'], []).append(parsed_injection_scenario)

        # Inject nulls based on the evaluation scenario
        if ErrorType.missing_value.value in parsed_evaluation_scenario.keys():
            nulls_injector = NullsInjector(seed=experiment_seed)
            for injection_scenario in parsed_evaluation_scenario[ErrorType.missing_value.value]:
                df = nulls_injector.fit_transform(df=df,
                                                  columns_with_nulls=injection_scenario['columns_to_inject'],
                                                  null_percentage=injection_scenario['error_rate'],
                                                  condition=injection_scenario['condition'])
        return df

    def _impute_nulls_in_test_set(self, null_imputer_name: str, null_imputer, X_test: pd.DataFrame,
                                  numerical_null_columns: list, categorical_null_columns: list):
        if null_imputer.num_imputer is not None:
            X_test[numerical_null_columns] = null_imputer.num_imputer.transform(X_test[numerical_null_columns])
        if null_imputer.cat_imputer is not None:
            X_test[categorical_null_columns] = null_imputer.cat_imputer.transform(X_test[categorical_null_columns])

        return X_test

    def _add_sensitive_attr_for_intervention(self, X_test: pd.DataFrame, sensitive_attr_for_intervention: str,
                                             data_loader, experiment_seed: int):
        # Create train and test sensitive attr dfs
        _, test_sensitive_attrs_df, _, _ = self._split_dataset(data_loader, experiment_seed)
        test_sensitive_attrs_df = test_sensitive_attrs_df[self.exp_config.sensitive_attrs_for_intervention]

        # Add a new binary column to the test set
        test_dis_group_mask = get_dis_group_condition(test_sensitive_attrs_df,
                                                      attrs=self.exp_config.sensitive_attrs_for_intervention,
                                                      dis_values=[self.virny_config.sensitive_attributes_dct[attr]
                                                                  for attr in self.exp_config.sensitive_attrs_for_intervention])
        X_test[sensitive_attr_for_intervention] = None  # Create a new column for the fairness intervention
        X_test.loc[test_dis_group_mask, sensitive_attr_for_intervention] = 0
        X_test.loc[~test_dis_group_mask, sensitive_attr_for_intervention] = 1

        return X_test

    def _apply_fair_preprocessor(self, X_test: pd.DataFrame, y_test: pd.DataFrame, target_column: str, fair_preprocessor,
                                 fairness_intervention_name: str, sensitive_attr_for_intervention: str):
        test_df = X_test
        test_df[target_column] = y_test
        test_binary_dataset = BinaryLabelDataset(df=test_df,
                                                 label_names=[target_column],
                                                 protected_attribute_names=[sensitive_attr_for_intervention],
                                                 favorable_label=1,
                                                 unfavorable_label=0)

        # Set labels (aka y_test) to zeros since we do not know labels during inference
        test_binary_dataset.labels = np.zeros(shape=np.shape(test_binary_dataset.labels))

        if fairness_intervention_name == FairnessIntervention.DIR.value:
            test_repaired_df , _ = fair_preprocessor.fit_transform(test_binary_dataset).convert_to_dataframe()
        else:
            test_repaired_df , _ = fair_preprocessor.transform(test_binary_dataset).convert_to_dataframe()

        test_repaired_df.index = test_repaired_df.index.astype(dtype='int64')
        X_test = test_repaired_df.drop([target_column, sensitive_attr_for_intervention], axis=1)

        return X_test
