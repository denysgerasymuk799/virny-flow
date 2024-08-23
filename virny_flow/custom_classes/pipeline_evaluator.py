import copy
import pandas as pd

from sklearn.model_selection import train_test_split
from virny.user_interfaces.multiple_models_with_db_writer_api import compute_metrics_with_db_writer

from virny_flow.custom_classes.ml_lifecycle import MLLifecycle
from virny_flow.utils.pipeline_utils import get_dis_group_condition
from virny_flow.preprocessing import preprocess_base_flow_dataset
from virny_flow.fairness_interventions.preprocessors import remove_disparate_impact, apply_learning_fair_representations
from virny_flow.fairness_interventions.postprocessors import get_eq_odds_postprocessor, get_reject_option_classification_postprocessor
from virny_flow.fairness_interventions.inprocessors import get_adversarial_debiasing_wrapper_config, get_exponentiated_gradient_reduction_wrapper
from virny_flow.utils.common_helpers import generate_guid, create_base_flow_dataset
from virny_flow.configs.constants import (EXP_COLLECTION_NAME, ErrorRepairMethod, STAGE_SEPARATOR,
                                          NO_FAIRNESS_INTERVENTION, FairnessIntervention, S3Folder)


class PipelineEvaluator(MLLifecycle):
    """
    Class encapsulates all experimental pipelines
    """
    def __init__(self, exp_config, dataset_config, fairness_intervention_config, models_config):
        """
        Constructor defining default variables
        """
        super().__init__(exp_config_name=exp_config.exp_config_name,
                         dataset_name=exp_config.dataset,
                         secrets_path=exp_config.secrets_path,
                         dataset_config=dataset_config,
                         model_params_for_tuning=models_config)

        self.exp_config = exp_config
        self.fairness_intervention_config = fairness_intervention_config
        self.task_name = None

    def execute_task(self, task_name: str, seed: int):
        self._db.connect()

        self.task_name = task_name
        stage_num = task_name.count(STAGE_SEPARATOR) + 1
        if stage_num == 1:
            null_imputer_name = task_name
            self.run_null_imputation_stage(init_data_loader=self.init_data_loader,
                                           experiment_seed=seed,
                                           null_imputer_name=null_imputer_name,
                                           tune_imputers=True,
                                           save_imputed_datasets=True)
            execution_status = True

        elif stage_num == 2:
            null_imputer_name, fairness_intervention_name = task_name.split(STAGE_SEPARATOR)
            self.run_fairness_intervention_stage(init_data_loader=self.init_data_loader,
                                                 experiment_seed=seed,
                                                 null_imputer_name=null_imputer_name,
                                                 fairness_intervention_name=fairness_intervention_name,
                                                 tune_fairness_interventions=True,
                                                 save_preprocessed_datasets=True)
            execution_status = True

        elif stage_num == 3:
            null_imputer_name, fairness_intervention_name, model_name = task_name.split(STAGE_SEPARATOR)
            self.run_model_evaluation_stage(init_data_loader=self.init_data_loader,
                                            experiment_seed=seed,
                                            null_imputer_name=null_imputer_name,
                                            fairness_intervention_name=fairness_intervention_name,
                                            model_name=model_name)
            execution_status = True

        else:
            self._logger.info(f"Task {task_name} has an incorrect format")
            execution_status = False

        self._db.close()
        self._logger.info(f"Task {task_name} was executed!")

        return execution_status

    def run_null_imputation_stage(self, init_data_loader, experiment_seed: int, null_imputer_name: str,
                                  tune_imputers: bool, save_imputed_datasets: bool):
        if null_imputer_name in (ErrorRepairMethod.boost_clean.value, ErrorRepairMethod.cp_clean.value):
            raise ValueError(f'To work with {ErrorRepairMethod.boost_clean.value} or {ErrorRepairMethod.cp_clean.value}, '
                             f'use scripts/evaluate_models.py')

        # Split the dataset
        data_loader = copy.deepcopy(init_data_loader)
        X_train_val_with_nulls, X_test_with_nulls, y_train_val, y_test = self._split_dataset(data_loader, experiment_seed)

        # Remove sensitive attributes from train and test sets with nulls to avoid their usage during imputation
        (X_train_val_with_nulls_wo_sensitive_attrs,
         X_tests_with_nulls_wo_sensitive_attrs_lst,
         numerical_columns_wo_sensitive_attrs,
         categorical_columns_wo_sensitive_attrs) = self._remove_sensitive_attrs(X_train_val=X_train_val_with_nulls,
                                                                                X_tests_lst=[X_test_with_nulls],
                                                                                data_loader=data_loader)

        # Impute nulls
        (X_train_val_imputed_wo_sensitive_attrs, X_tests_imputed_wo_sensitive_attrs_lst,
         null_imputer_params_dct, null_imputer,
         imputation_runtime) = self._impute_nulls(X_train_with_nulls=X_train_val_with_nulls_wo_sensitive_attrs,
                                                  X_tests_with_nulls_lst=X_tests_with_nulls_wo_sensitive_attrs_lst,
                                                  null_imputer_name=null_imputer_name,
                                                  evaluation_scenario=None,
                                                  experiment_seed=experiment_seed,
                                                  categorical_columns=categorical_columns_wo_sensitive_attrs,
                                                  numerical_columns=numerical_columns_wo_sensitive_attrs,
                                                  tune_imputers=tune_imputers)

        if save_imputed_datasets:
            X_test_imputed_wo_sensitive_attrs = X_tests_imputed_wo_sensitive_attrs_lst[0]
            self._save_preprocessed_datasets_to_s3(X_train_val=X_train_val_imputed_wo_sensitive_attrs,
                                                   X_test=X_test_imputed_wo_sensitive_attrs,
                                                   stage_name="null_imputation",
                                                   preprocessor_name=null_imputer_name)
            self._save_artifacts_to_s3(preprocessor_name='null_imputer',
                                       preprocessor=null_imputer,
                                       preprocessor_params_dct=null_imputer_params_dct)
            # Save numerical and categorical columns of the training set in pickle files
            train_set_cols_with_nulls = X_train_val_with_nulls_wo_sensitive_attrs.columns[X_train_val_with_nulls_wo_sensitive_attrs.isna().any()].tolist()
            train_numerical_null_columns = list(set(train_set_cols_with_nulls).intersection(numerical_columns_wo_sensitive_attrs))
            train_categorical_null_columns = list(set(train_set_cols_with_nulls).intersection(categorical_columns_wo_sensitive_attrs))

            self._save_custom_object_to_s3(obj=train_numerical_null_columns, obj_name='train_numerical_null_columns')
            self._save_custom_object_to_s3(obj=train_categorical_null_columns, obj_name='train_categorical_null_columns')

    def run_fairness_intervention_stage(self, init_data_loader, experiment_seed: int, null_imputer_name: str,
                                        fairness_intervention_name: str, tune_fairness_interventions: bool,
                                        save_preprocessed_datasets: bool):
        data_loader = copy.deepcopy(init_data_loader)
        main_base_flow_dataset = self._load_preprocessed_train_test_sets(data_loader=data_loader,
                                                                         stage_name="null_imputation",
                                                                         preprocessor_name=null_imputer_name,
                                                                         experiment_seed=experiment_seed)
        # Preprocess the dataset using the pre-defined preprocessor
        preprocessed_base_flow_dataset, column_transformer = preprocess_base_flow_dataset(main_base_flow_dataset)

        if fairness_intervention_name != NO_FAIRNESS_INTERVENTION:
            # Create a binary column for the fairness intervention
            input_sensitive_attrs_for_intervention = self.exp_config.sensitive_attrs_for_intervention
            binary_sensitive_attr_for_intervention = '&'.join(input_sensitive_attrs_for_intervention) + '_binary'

            # Create train and test sensitive attr dfs
            train_sensitive_attrs_df, test_sensitive_attrs_df, _, _ = self._split_dataset(data_loader, experiment_seed)
            train_sensitive_attrs_df = train_sensitive_attrs_df[input_sensitive_attrs_for_intervention]
            test_sensitive_attrs_df = test_sensitive_attrs_df[input_sensitive_attrs_for_intervention]

            # Add a new binary column to the training set
            train_dis_group_mask = get_dis_group_condition(train_sensitive_attrs_df,
                                                           attrs=input_sensitive_attrs_for_intervention,
                                                           dis_values=[self.virny_config.sensitive_attributes_dct[attr]
                                                                       for attr in input_sensitive_attrs_for_intervention])
            preprocessed_base_flow_dataset.X_train_val[binary_sensitive_attr_for_intervention] = None  # Create a new column for the fairness intervention
            preprocessed_base_flow_dataset.X_train_val.loc[train_dis_group_mask, binary_sensitive_attr_for_intervention] = 0
            preprocessed_base_flow_dataset.X_train_val.loc[~train_dis_group_mask, binary_sensitive_attr_for_intervention] = 1

            # Add a new binary column to the test set
            test_dis_group_mask = get_dis_group_condition(test_sensitive_attrs_df,
                                                          attrs=input_sensitive_attrs_for_intervention,
                                                          dis_values=[self.virny_config.sensitive_attributes_dct[attr]
                                                                      for attr in input_sensitive_attrs_for_intervention])
            preprocessed_base_flow_dataset.X_test[binary_sensitive_attr_for_intervention] = None  # Create a new column for the fairness intervention
            preprocessed_base_flow_dataset.X_test.loc[test_dis_group_mask, binary_sensitive_attr_for_intervention] = 0
            preprocessed_base_flow_dataset.X_test.loc[~test_dis_group_mask, binary_sensitive_attr_for_intervention] = 1

        # Apply fairness-enhancing preprocessors
        if fairness_intervention_name in (FairnessIntervention.DIR.value, FairnessIntervention.LFR.value):
            if fairness_intervention_name == FairnessIntervention.DIR.value:
                preprocessed_base_flow_dataset, fair_preprocessor =\
                    remove_disparate_impact(preprocessed_base_flow_dataset,
                                            repair_level=self.fairness_intervention_config[FairnessIntervention.DIR.value]["repair_level"],
                                            sensitive_attribute=binary_sensitive_attr_for_intervention)
            elif fairness_intervention_name == FairnessIntervention.LFR.value:
                preprocessed_base_flow_dataset, fair_preprocessor =\
                    apply_learning_fair_representations(preprocessed_base_flow_dataset,
                                                        intervention_options=self.fairness_intervention_config[FairnessIntervention.LFR.value],
                                                        sensitive_attribute=binary_sensitive_attr_for_intervention)

            fair_preprocessor_params_dct = self.fairness_intervention_config[fairness_intervention_name]
            self._save_artifacts_to_s3(preprocessor_name='fair_preprocessor',
                                       preprocessor=fair_preprocessor,
                                       preprocessor_params_dct=fair_preprocessor_params_dct)

        if save_preprocessed_datasets:
            column_transformer_params = {"string_params": str(column_transformer.get_params())}
            self._save_artifacts_to_s3(preprocessor_name='column_transformer',
                                       preprocessor=column_transformer,
                                       preprocessor_params_dct=column_transformer_params)
            self._save_preprocessed_datasets_to_s3(X_train_val=preprocessed_base_flow_dataset.X_train_val,
                                                   X_test=preprocessed_base_flow_dataset.X_test,
                                                   stage_name="fairness_intervention",
                                                   preprocessor_name=fairness_intervention_name)

    def run_model_evaluation_stage(self, init_data_loader, experiment_seed: int, null_imputer_name: str,
                                   fairness_intervention_name: str, model_name: str):
        data_loader = copy.deepcopy(init_data_loader)

        evaluation_scenario = None
        custom_table_fields_dct = dict()
        custom_table_fields_dct['session_uuid'] = self._session_uuid
        custom_table_fields_dct['dataset_split_seed'] = experiment_seed
        custom_table_fields_dct['model_init_seed'] = experiment_seed
        custom_table_fields_dct['experiment_seed'] = experiment_seed
        custom_table_fields_dct['exp_config_name'] = self.exp_config_name
        custom_table_fields_dct['evaluation_scenario'] = evaluation_scenario
        custom_table_fields_dct['null_imputer_name'] = null_imputer_name
        custom_table_fields_dct['fairness_intervention_name'] = fairness_intervention_name

        # Create exp_pipeline_guid to define a row level of granularity.
        # concat(exp_pipeline_guid, model_name, subgroup, metric) can be used to check duplicates of results
        # for the same experimental pipeline.
        custom_table_fields_dct['exp_pipeline_guid'] = (
            generate_guid(ordered_hierarchy_lst=[self.exp_config_name, evaluation_scenario, experiment_seed,
                                                 self.dataset_name, null_imputer_name, fairness_intervention_name]))

        # Load fairness-enhanced datasets
        main_base_flow_dataset = self._load_preprocessed_train_test_sets(data_loader=data_loader,
                                                                         stage_name="fairness_intervention",
                                                                         preprocessor_name=fairness_intervention_name,
                                                                         experiment_seed=experiment_seed)
        self._save_custom_object_to_s3(obj=main_base_flow_dataset.X_train_val.columns,
                                       obj_name='preprocessed_train_set_columns')
        # Tune ML models
        models_config = self._tune_ML_models(model_names=[model_name],
                                             base_flow_dataset=main_base_flow_dataset,
                                             evaluation_scenario=evaluation_scenario,
                                             experiment_seed=experiment_seed,
                                             null_imputer_name=null_imputer_name,
                                             fairness_intervention_name=fairness_intervention_name)

        # Prepare fairness in-processor or post-processor if needed
        sensitive_attribute = '&'.join(self.exp_config.sensitive_attrs_for_intervention) + '_binary'
        privileged_groups = [{sensitive_attribute: 1}]
        unprivileged_groups = [{sensitive_attribute: 0}]
        if fairness_intervention_name == FairnessIntervention.EOP.value:
            postprocessor = get_eq_odds_postprocessor(privileged_groups=privileged_groups,
                                                      unprivileged_groups=unprivileged_groups,
                                                      seed=experiment_seed)
            self.virny_config['postprocessing_sensitive_attribute'] = sensitive_attribute
        elif fairness_intervention_name == FairnessIntervention.ROC.value:
            postprocessor_configs = self.fairness_intervention_config[FairnessIntervention.ROC.value]
            postprocessor = get_reject_option_classification_postprocessor(privileged_groups=privileged_groups,
                                                                           unprivileged_groups=unprivileged_groups,
                                                                           postprocessor_configs=postprocessor_configs)
            self.virny_config['postprocessing_sensitive_attribute'] = sensitive_attribute
        else:
            postprocessor = None
            if fairness_intervention_name == FairnessIntervention.EGR.value:
                models_config = get_exponentiated_gradient_reduction_wrapper(inprocessor_configs=self.fairness_intervention_config[FairnessIntervention.EGR.value],
                                                                             sensitive_attr_for_intervention=sensitive_attribute)
            elif fairness_intervention_name == FairnessIntervention.AD.value:
                models_config = get_adversarial_debiasing_wrapper_config(privileged_groups=privileged_groups,
                                                                         unprivileged_groups=unprivileged_groups,
                                                                         inprocessor_configs=self.fairness_intervention_config[FairnessIntervention.AD.value],
                                                                         sensitive_attr_for_intervention=sensitive_attribute)

        # Compute metrics for tuned models
        self.virny_config.random_state = experiment_seed  # Set random state for the metric computation with Virny
        _, models_fitted_bootstraps_dct = compute_metrics_with_db_writer(dataset=main_base_flow_dataset,
                                                                         config=self.virny_config,
                                                                         models_config=models_config,
                                                                         custom_tbl_fields_dct=custom_table_fields_dct,
                                                                         db_writer_func=self._db.get_db_writer(collection_name=EXP_COLLECTION_NAME),
                                                                         notebook_logs_stdout=False,
                                                                         postprocessor=postprocessor,
                                                                         return_fitted_bootstrap=True,
                                                                         verbose=0)
        print(f'Metric computation for {null_imputer_name}&{fairness_intervention_name}&{model_name} was finished\n', flush=True)

        # Save the fitted bootstrap in S3
        self._save_virny_bootstrap_to_s3(bootstrap=models_fitted_bootstraps_dct[model_name])

    def _save_preprocessed_datasets_to_s3(self, X_train_val: pd.DataFrame, X_test: pd.DataFrame,
                                          stage_name: str, preprocessor_name: str):
        save_sets_dir_path = f'{S3Folder.experiments.value}/{self.exp_config_name}/{S3Folder.intermediate_state.value}/{stage_name}_stage/{self.dataset_name}/{preprocessor_name}'

        # Write X_train_val to S3 as a CSV
        train_set_filename = f'preprocessed_{self.exp_config_name}_{self.dataset_name}_{preprocessor_name}_X_train_val.csv'
        self._s3_client.write_csv(X_train_val, f'{save_sets_dir_path}/{train_set_filename}', index=True)

        # Save X_test set in S3
        test_set_filename = f'preprocessed_{self.exp_config_name}_{self.dataset_name}_{preprocessor_name}_X_test.csv'
        self._s3_client.write_csv(X_test, f'{save_sets_dir_path}/{test_set_filename}', index=True)

        self._logger.info(f"Train and test sets preprocessed by {preprocessor_name} were saved in S3")

    def _save_artifacts_to_s3(self, preprocessor_name: str, preprocessor, preprocessor_params_dct: dict):
        target_dir_path = f'{S3Folder.experiments.value}/{self.exp_config_name}/{S3Folder.artifacts.value}'
        pipeline_names_with_prefix = self._db.read_pipeline_names_by_prefix(exp_config_name=self.exp_config_name,
                                                                            prefix=self.task_name)

        # Write artifacts to each related pipeline in S3
        for pipeline_name in pipeline_names_with_prefix:
            # Write preprocessor object to S3
            preprocessor_filename = f'{preprocessor_name}.pkl'
            self._s3_client.write_pickle(obj=preprocessor, key=f'{target_dir_path}/{pipeline_name}/{preprocessor_filename}')

            # Write preprocessor params to S3
            preprocessor_params_filename = f'{preprocessor_name}_params.json'
            self._s3_client.write_json(data=preprocessor_params_dct, key=f'{target_dir_path}/{pipeline_name}/{preprocessor_params_filename}')

        self._logger.info(f"Artifacts for {preprocessor_name} were saved in S3")

    def _save_custom_object_to_s3(self, obj_name: str, obj):
        target_dir_path = f'{S3Folder.experiments.value}/{self.exp_config_name}/{S3Folder.artifacts.value}'
        pipeline_names_with_prefix = self._db.read_pipeline_names_by_prefix(exp_config_name=self.exp_config_name,
                                                                            prefix=self.task_name)
        # Write artifacts to each related pipeline in S3
        for pipeline_name in pipeline_names_with_prefix:
            # Write preprocessor object to S3
            preprocessor_filename = f'{obj_name}.pkl'
            self._s3_client.write_pickle(obj=obj, key=f'{target_dir_path}/{pipeline_name}/{preprocessor_filename}')

    def _save_virny_bootstrap_to_s3(self, bootstrap: list):
        target_dir_path = f'{S3Folder.experiments.value}/{self.exp_config_name}/{S3Folder.artifacts.value}'
        pipeline_names_with_prefix = self._db.read_pipeline_names_by_prefix(exp_config_name=self.exp_config_name,
                                                                            prefix=self.task_name)

        # Write artifacts to each related pipeline in S3
        for pipeline_name in pipeline_names_with_prefix:
            # Write model to S3
            bootstrap_filename = 'fitted_model_bootstrap.pkl'
            self._s3_client.write_pickle(obj=bootstrap,
                                         key=f'{target_dir_path}/{pipeline_name}/{bootstrap_filename}')
            # Write model params to S3
            model_params_filename = 'model_params.json'
            self._s3_client.write_json(data=bootstrap[0]["model_obj"].get_params(),
                                       key=f'{target_dir_path}/{pipeline_name}/{model_params_filename}')

        self._logger.info("Artifacts for the model bootstrap were saved in S3")

    def _load_preprocessed_train_test_sets(self, data_loader, stage_name: str, preprocessor_name: str, experiment_seed: int):
        if self.test_set_fraction < 0.0 or self.test_set_fraction > 1.0:
            raise ValueError("test_set_fraction must be a float in the [0.0-1.0] range")

        # Split the dataset
        y_train_val, y_test = train_test_split(data_loader.y_data,
                                               test_size=self.test_set_fraction,
                                               random_state=experiment_seed)

        # Create a base flow dataset for Virny to compute metrics
        numerical_columns_wo_sensitive_attrs = [col for col in data_loader.numerical_columns if col not in self.dataset_sensitive_attrs]
        categorical_columns_wo_sensitive_attrs = [col for col in data_loader.categorical_columns if col not in self.dataset_sensitive_attrs]

        # Read X_train_val set from S3 as CSV
        save_sets_dir_path = f'{S3Folder.experiments.value}/{self.exp_config_name}/{S3Folder.intermediate_state.value}/{stage_name}_stage/{self.dataset_name}/{preprocessor_name}'
        train_set_filename = f'preprocessed_{self.exp_config_name}_{self.dataset_name}_{preprocessor_name}_X_train_val.csv'
        X_train_val_imputed_wo_sensitive_attrs = self._s3_client.read_csv(key=f'{save_sets_dir_path}/{train_set_filename}',
                                                                          index=True)
        if stage_name == 'null_imputation':
            X_train_val_imputed_wo_sensitive_attrs[categorical_columns_wo_sensitive_attrs] = (
                X_train_val_imputed_wo_sensitive_attrs[categorical_columns_wo_sensitive_attrs].astype(str))

        # Subset y_train_val to align with X_train_val_imputed_wo_sensitive_attrs
        if preprocessor_name == ErrorRepairMethod.deletion.value:
            y_train_val = y_train_val.loc[X_train_val_imputed_wo_sensitive_attrs.index]

        # Read X_test set from S3 as CSV
        test_set_filename = f'preprocessed_{self.exp_config_name}_{self.dataset_name}_{preprocessor_name}_X_test.csv'
        X_test_imputed_wo_sensitive_attrs = self._s3_client.read_csv(key=f'{save_sets_dir_path}/{test_set_filename}',
                                                                     index=True)
        if stage_name == 'null_imputation':
            X_test_imputed_wo_sensitive_attrs[categorical_columns_wo_sensitive_attrs] = (
                X_test_imputed_wo_sensitive_attrs[categorical_columns_wo_sensitive_attrs].astype(str))

        # Create base flow datasets for Virny to compute metrics
        main_base_flow_dataset = create_base_flow_dataset(data_loader=copy.deepcopy(data_loader),
                                                          dataset_sensitive_attrs=self.dataset_sensitive_attrs,
                                                          X_train_val_wo_sensitive_attrs=X_train_val_imputed_wo_sensitive_attrs,
                                                          X_test_wo_sensitive_attrs=X_test_imputed_wo_sensitive_attrs,
                                                          y_train_val=y_train_val,
                                                          y_test=copy.deepcopy(y_test),
                                                          numerical_columns_wo_sensitive_attrs=numerical_columns_wo_sensitive_attrs,
                                                          categorical_columns_wo_sensitive_attrs=categorical_columns_wo_sensitive_attrs)

        return main_base_flow_dataset
