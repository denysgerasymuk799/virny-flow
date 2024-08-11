import copy
import tqdm

from pprint import pprint
from virny.user_interfaces.multiple_models_with_multiple_test_sets_api import compute_metrics_with_multiple_test_sets

from virny_flow.utils.common_helpers import generate_guid
from virny_flow.custom_classes.ml_lifecycle import MLLifecycle
from virny_flow.utils.dataframe_utils import preprocess_mult_base_flow_datasets
from virny_flow.configs.constants import EXP_COLLECTION_NAME, EXPERIMENT_RUN_SEEDS, ErrorRepairMethod, STAGE_SEPARATOR


class PipelineEvaluator(MLLifecycle):
    """
    Class encapsulates all experimental pipelines
    """
    def __init__(self, exp_config, dataset_config, models_config):
        """
        Constructor defining default variables
        """
        super().__init__(exp_config_name=exp_config.exp_config_name,
                         dataset_name=exp_config.dataset,
                         secrets_path=exp_config.secrets_path,
                         dataset_config=dataset_config,
                         model_params_for_tuning=models_config)

    def execute_task(self, task_name: str, seed: int):
        self._db.connect()

        num_stages = task_name.count(STAGE_SEPARATOR) + 1
        if num_stages == 1:
            null_imputer_name = task_name
            self._run_null_imputation_stage(init_data_loader=self.init_data_loader,
                                            experiment_seed=seed,
                                            null_imputer_name=null_imputer_name,
                                            tune_imputers=True,
                                            save_imputed_datasets=True)
            execution_status = True
        else:
            self._logger.info(f"Task {task_name} has an incorrect format")
            execution_status = False

        self._db.close()
        self._logger.info(f"Task {task_name} was executed!")

        return execution_status

    def _run_null_imputation_stage(self, init_data_loader, experiment_seed, null_imputer_name,
                                   tune_imputers, save_imputed_datasets):
        if null_imputer_name in (ErrorRepairMethod.boost_clean.value, ErrorRepairMethod.cp_clean.value):
            raise ValueError(f'To work with {ErrorRepairMethod.boost_clean.value} or {ErrorRepairMethod.cp_clean.value}, '
                             f'use scripts/evaluate_models.py')

        data_loader = copy.deepcopy(init_data_loader)
        self.impute_nulls(data_loader=data_loader,
                          null_imputer_name=null_imputer_name,
                          experiment_seed=experiment_seed,
                          tune_imputers=tune_imputers,
                          save_imputed_datasets=save_imputed_datasets)

    def impute_nulls(self, data_loader, null_imputer_name: str, experiment_seed: int,
                     tune_imputers: bool = True, save_imputed_datasets: bool = False):
        # Split the dataset
        X_train_val_with_nulls, X_test_with_nulls, y_train_val, y_test = self._split_dataset(data_loader, experiment_seed)

        # Remove sensitive attributes from train and test sets with nulls to avoid their usage during imputation
        (X_train_val_with_nulls_wo_sensitive_attrs,
         X_tests_with_nulls_wo_sensitive_attrs_lst,
         numerical_columns_wo_sensitive_attrs,
         categorical_columns_wo_sensitive_attrs) = self._remove_sensitive_attrs(X_train_val=X_train_val_with_nulls,
                                                                                X_tests_lst=[X_test_with_nulls],
                                                                                data_loader=data_loader)

        # Impute nulls
        (X_train_val_imputed_wo_sensitive_attrs, X_tests_imputed_wo_sensitive_attrs_lst, null_imputer_params_dct,
         imputation_runtime) = self._impute_nulls(X_train_with_nulls=X_train_val_with_nulls_wo_sensitive_attrs,
                                                  X_tests_with_nulls_lst=X_tests_with_nulls_wo_sensitive_attrs_lst,
                                                  null_imputer_name=null_imputer_name,
                                                  evaluation_scenario=None,
                                                  experiment_seed=experiment_seed,
                                                  categorical_columns=categorical_columns_wo_sensitive_attrs,
                                                  numerical_columns=numerical_columns_wo_sensitive_attrs,
                                                  tune_imputers=tune_imputers)
        print('X_tests_imputed_wo_sensitive_attrs_lst[0].columns -- ', X_tests_imputed_wo_sensitive_attrs_lst[0].columns)

        if save_imputed_datasets:
            self._save_imputed_datasets_to_s3(X_train_val=X_train_val_imputed_wo_sensitive_attrs,
                                              X_tests_lst=X_tests_imputed_wo_sensitive_attrs_lst,
                                              null_imputer_name=null_imputer_name)

    def _run_exp_iter_for_standard_imputation(self, data_loader, experiment_seed: int, evaluation_scenario: str,
                                              null_imputer_name: str, model_names: list, tune_imputers: bool,
                                              ml_impute: bool, save_imputed_datasets: bool, custom_table_fields_dct: dict):
        if ml_impute:
            main_base_flow_dataset, extra_base_flow_datasets = self.inject_and_impute_nulls(data_loader=data_loader,
                                                                                            null_imputer_name=null_imputer_name,
                                                                                            evaluation_scenario=evaluation_scenario,
                                                                                            tune_imputers=tune_imputers,
                                                                                            experiment_seed=experiment_seed,
                                                                                            save_imputed_datasets=save_imputed_datasets)
        else:
            main_base_flow_dataset, extra_base_flow_datasets = self.load_imputed_train_test_sets(data_loader=data_loader,
                                                                                                 null_imputer_name=null_imputer_name,
                                                                                                 evaluation_scenario=evaluation_scenario,
                                                                                                 experiment_seed=experiment_seed)

        # Preprocess the dataset using the defined preprocessor
        main_base_flow_dataset, extra_test_sets = preprocess_mult_base_flow_datasets(main_base_flow_dataset, extra_base_flow_datasets)

        # Tune ML models
        models_config = self._tune_ML_models(model_names=model_names,
                                             base_flow_dataset=main_base_flow_dataset,
                                             experiment_seed=experiment_seed,
                                             evaluation_scenario=evaluation_scenario,
                                             null_imputer_name=null_imputer_name)

        # Compute metrics for tuned models
        self.virny_config.random_state = experiment_seed  # Set random state for the metric computation with Virny
        compute_metrics_with_multiple_test_sets(dataset=main_base_flow_dataset,
                                                extra_test_sets_lst=extra_test_sets,
                                                config=self.virny_config,
                                                models_config=models_config,
                                                custom_tbl_fields_dct=custom_table_fields_dct,
                                                db_writer_func=self._db.get_db_writer(collection_name=EXP_COLLECTION_NAME),
                                                notebook_logs_stdout=False,
                                                verbose=0)

    def _run_exp_iter(self, init_data_loader, run_num, evaluation_scenario, null_imputer_name,
                      model_names, tune_imputers, ml_impute, save_imputed_datasets):
        data_loader = copy.deepcopy(init_data_loader)

        custom_table_fields_dct = dict()
        experiment_seed = EXPERIMENT_RUN_SEEDS[run_num - 1]
        custom_table_fields_dct['session_uuid'] = self._session_uuid
        custom_table_fields_dct['null_imputer_name'] = null_imputer_name
        custom_table_fields_dct['evaluation_scenario'] = evaluation_scenario
        custom_table_fields_dct['experiment_iteration'] = f'exp_iter_{run_num}'
        custom_table_fields_dct['dataset_split_seed'] = experiment_seed
        custom_table_fields_dct['model_init_seed'] = experiment_seed

        # Create exp_pipeline_guid to define a row level of granularity.
        # concat(exp_pipeline_guid, model_name, subgroup, metric) can be used to check duplicates of results
        # for the same experimental pipeline.
        custom_table_fields_dct['exp_pipeline_guid'] = (
            generate_guid(ordered_hierarchy_lst=[self.dataset_name, null_imputer_name, evaluation_scenario, experiment_seed]))

        self._logger.info("Start an experiment iteration for the following custom params:")
        pprint(custom_table_fields_dct)
        print('\n', flush=True)

        if null_imputer_name in (ErrorRepairMethod.boost_clean.value, ErrorRepairMethod.cp_clean.value):
            self._run_exp_iter_for_joint_cleaning_and_training(data_loader=data_loader,
                                                               experiment_seed=experiment_seed,
                                                               evaluation_scenario=evaluation_scenario,
                                                               null_imputer_name=null_imputer_name,
                                                               model_names=model_names,
                                                               tune_imputers=tune_imputers,
                                                               custom_table_fields_dct=custom_table_fields_dct)
        else:
            self._run_exp_iter_for_standard_imputation(data_loader=data_loader,
                                                       experiment_seed=experiment_seed,
                                                       evaluation_scenario=evaluation_scenario,
                                                       null_imputer_name=null_imputer_name,
                                                       model_names=model_names,
                                                       tune_imputers=tune_imputers,
                                                       ml_impute=ml_impute,
                                                       save_imputed_datasets=save_imputed_datasets,
                                                       custom_table_fields_dct=custom_table_fields_dct)

    def run_experiment(self, run_nums: list, evaluation_scenarios: list, model_names: list,
                       tune_imputers: bool, ml_impute: bool, save_imputed_datasets: bool):
        self._db.connect()

        total_iterations = len(self.null_imputers) * len(evaluation_scenarios) * len(run_nums)
        with tqdm.tqdm(total=total_iterations, desc="Experiment Progress") as pbar:
            for null_imputer_idx, null_imputer_name in enumerate(self.null_imputers):
                for evaluation_scenario_idx, evaluation_scenario in enumerate(evaluation_scenarios):
                    for run_idx, run_num in enumerate(run_nums):
                        self._logger.info(f"{'=' * 30} NEW EXPERIMENT RUN {'=' * 30}")
                        print('Configs for a new experiment run:')
                        print(
                            f"Null imputer: {null_imputer_name} ({null_imputer_idx + 1} out of {len(self.null_imputers)})\n"
                            f"Evaluation scenario: {evaluation_scenario} ({evaluation_scenario_idx + 1} out of {len(evaluation_scenarios)})\n"
                            f"Run num: {run_num} ({run_idx + 1} out of {len(run_nums)})\n"
                        )
                        self._run_exp_iter(init_data_loader=self.init_data_loader,
                                           run_num=run_num,
                                           evaluation_scenario=evaluation_scenario,
                                           null_imputer_name=null_imputer_name,
                                           model_names=model_names,
                                           tune_imputers=tune_imputers,
                                           ml_impute=ml_impute,
                                           save_imputed_datasets=save_imputed_datasets)
                        pbar.update(1)
                        print('\n\n\n\n', flush=True)

        self._db.close()
        self._logger.info("Experimental results were successfully saved!")
