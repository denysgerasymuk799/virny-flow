import copy
import time

from munch import DefaultMunch
from openbox.utils.constants import SUCCESS
from openbox.utils.history import Observation
from openbox.utils.util_funcs import parse_result
from virny.user_interfaces.multiple_models_with_db_writer_api import compute_metrics_with_db_writer

from .ml_lifecycle import MLLifecycle
from virny_flow.core.utils.pipeline_utils import get_dis_group_condition, nested_dict_from_flat
from virny_flow.core.utils.common_helpers import generate_guid, create_base_flow_dataset
from virny_flow.core.preprocessing import preprocess_base_flow_dataset
from virny_flow.core.fairness_interventions.preprocessors import remove_disparate_impact, apply_learning_fair_representations
from virny_flow.core.fairness_interventions.postprocessors import get_eq_odds_postprocessor, get_reject_option_classification_postprocessor
from virny_flow.core.fairness_interventions.inprocessors import get_adversarial_debiasing_wrapper_config, get_exponentiated_gradient_reduction_wrapper
from virny_flow.configs.structs import Task
from virny_flow.configs.constants import (ErrorRepairMethod, STAGE_SEPARATOR, ALL_EXPERIMENT_METRICS_TABLE,
                                          NO_FAIRNESS_INTERVENTION, FairnessIntervention)
from virny_flow.task_manager.domain_logic.bayesian_optimization import get_objective_losses


class PipelineEvaluator(MLLifecycle):
    """
    Class encapsulates all experimental pipelines
    """
    def __init__(self, exp_config: DefaultMunch, dataset_config: dict, null_imputation_config: dict,
                 fairness_intervention_config: dict, models_config: dict):
        """
        Constructor defining default variables
        """
        super().__init__(exp_config_name=exp_config.exp_config_name,
                         dataset_name=exp_config.dataset,
                         secrets_path=exp_config.secrets_path,
                         dataset_config=dataset_config,
                         models_config=models_config)

        self.exp_config = exp_config
        self.null_imputation_config = null_imputation_config
        self.fairness_intervention_config = fairness_intervention_config

    def execute_task(self, task: Task, seed: int):
        self._db.connect()

        # Parse an input task
        null_imputer_name, fairness_intervention_name, model_name = task.physical_pipeline.logical_pipeline_name.split(STAGE_SEPARATOR)
        null_imputer_params = nested_dict_from_flat(task.physical_pipeline.null_imputer_params)
        fairness_intervention_params = nested_dict_from_flat(task.physical_pipeline.fairness_intervention_params)
        model_params = nested_dict_from_flat(task.physical_pipeline.model_params)

        # Perform null imputation
        main_base_flow_dataset = self.run_null_imputation_stage(init_data_loader=self.init_data_loader,
                                                                experiment_seed=seed,
                                                                null_imputer_name=null_imputer_name,
                                                                null_imputer_params=null_imputer_params)
        # Apply pre-processing fairness intervention if defined
        preprocessed_base_flow_dataset = self.run_fairness_intervention_stage(main_base_flow_dataset=main_base_flow_dataset,
                                                                              init_data_loader=self.init_data_loader,
                                                                              experiment_seed=seed,
                                                                              fairness_intervention_name=fairness_intervention_name,
                                                                              fairness_intervention_params=fairness_intervention_params)
        # Evaluate the model
        start_time = time.time()
        multiple_models_metrics_dct = self.run_model_evaluation_stage(main_base_flow_dataset=preprocessed_base_flow_dataset,
                                                                      experiment_seed=seed,
                                                                      null_imputer_name=null_imputer_name,
                                                                      fairness_intervention_name=fairness_intervention_name,
                                                                      model_name=model_name,
                                                                      model_params=model_params)
        elapsed_time = time.time() - start_time

        # Create an observation for MO-BO based on the computed metrics
        result_losses = get_objective_losses(metrics_dct=multiple_models_metrics_dct,
                                             objectives=task.objectives,
                                             model_name=model_name,
                                             sensitive_attributes_dct=self.virny_config.sensitive_attributes_dct)
        print("result_losses:", result_losses)
        objectives, constraints, extra_info = parse_result(result_losses)
        print("objectives:", objectives)
        observation = Observation(
            config=task.physical_pipeline.suggestion,
            objectives=objectives,
            constraints=constraints,
            trial_state=SUCCESS,
            elapsed_time=elapsed_time,
            extra_info=extra_info,
        )

        self._db.close()
        self._logger.info(f"Task with UUID {task.task_uuid} was executed!")

        return observation

    def run_null_imputation_stage(self, init_data_loader, experiment_seed: int,
                                  null_imputer_name: str, null_imputer_params: dict):
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
        (X_train_val_imputed_wo_sensitive_attrs,
         X_tests_imputed_wo_sensitive_attrs_lst,
         null_imputer_params_dct,
         imputation_runtime) = self._impute_nulls(X_train_with_nulls=X_train_val_with_nulls_wo_sensitive_attrs,
                                                  X_tests_with_nulls_lst=X_tests_with_nulls_wo_sensitive_attrs_lst,
                                                  null_imputer_name=null_imputer_name,
                                                  null_imputer_params=null_imputer_params,
                                                  experiment_seed=experiment_seed,
                                                  categorical_columns=categorical_columns_wo_sensitive_attrs,
                                                  numerical_columns=numerical_columns_wo_sensitive_attrs)

        # Create base flow datasets for Virny to compute metrics
        X_test_imputed_wo_sensitive_attrs = X_tests_imputed_wo_sensitive_attrs_lst[0]
        main_base_flow_dataset = create_base_flow_dataset(data_loader=data_loader,
                                                          dataset_sensitive_attrs=self.dataset_sensitive_attrs,
                                                          X_train_val_wo_sensitive_attrs=X_train_val_imputed_wo_sensitive_attrs,
                                                          X_test_wo_sensitive_attrs=X_test_imputed_wo_sensitive_attrs,
                                                          y_train_val=y_train_val,
                                                          y_test=copy.deepcopy(y_test),
                                                          numerical_columns_wo_sensitive_attrs=numerical_columns_wo_sensitive_attrs,
                                                          categorical_columns_wo_sensitive_attrs=categorical_columns_wo_sensitive_attrs)

        return main_base_flow_dataset

    def run_fairness_intervention_stage(self, main_base_flow_dataset, init_data_loader, experiment_seed: int,
                                        fairness_intervention_name: str, fairness_intervention_params: dict):
        data_loader = copy.deepcopy(init_data_loader)

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
                                            repair_level=fairness_intervention_params["repair_level"],
                                            sensitive_attribute=binary_sensitive_attr_for_intervention)
            elif fairness_intervention_name == FairnessIntervention.LFR.value:
                preprocessed_base_flow_dataset, fair_preprocessor =\
                    apply_learning_fair_representations(preprocessed_base_flow_dataset,
                                                        intervention_options=fairness_intervention_params,
                                                        sensitive_attribute=binary_sensitive_attr_for_intervention)

        return preprocessed_base_flow_dataset

    def run_model_evaluation_stage(self, main_base_flow_dataset, experiment_seed: int, null_imputer_name: str,
                                   fairness_intervention_name: str, model_name: str, model_params: dict):
        custom_table_fields_dct = dict()
        custom_table_fields_dct['session_uuid'] = self._session_uuid
        custom_table_fields_dct['dataset_split_seed'] = experiment_seed
        custom_table_fields_dct['model_init_seed'] = experiment_seed
        custom_table_fields_dct['experiment_seed'] = experiment_seed
        custom_table_fields_dct['exp_config_name'] = self.exp_config_name
        custom_table_fields_dct['null_imputer_name'] = null_imputer_name
        custom_table_fields_dct['fairness_intervention_name'] = fairness_intervention_name

        # Create exp_pipeline_guid to define a row level of granularity.
        # concat(exp_pipeline_guid, model_name, subgroup, metric) can be used to check duplicates of results
        # for the same experimental pipeline.
        custom_table_fields_dct['exp_pipeline_guid'] = (
            generate_guid(ordered_hierarchy_lst=[self.exp_config_name, experiment_seed, self.dataset_name,
                                                 null_imputer_name, fairness_intervention_name]))

        # Tune ML models
        all_model_params = {**model_params, **self.models_config[model_name]['default_kwargs']}
        models_dct = {
            model_name: self.models_config[model_name]['model'](**all_model_params)
        }

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
                models_dct = get_exponentiated_gradient_reduction_wrapper(inprocessor_configs=self.fairness_intervention_config[FairnessIntervention.EGR.value],
                                                                          sensitive_attr_for_intervention=sensitive_attribute)
            elif fairness_intervention_name == FairnessIntervention.AD.value:
                models_dct = get_adversarial_debiasing_wrapper_config(privileged_groups=privileged_groups,
                                                                      unprivileged_groups=unprivileged_groups,
                                                                      inprocessor_configs=self.fairness_intervention_config[FairnessIntervention.AD.value],
                                                                      sensitive_attr_for_intervention=sensitive_attribute)

        # Compute metrics for tuned models
        self.virny_config.random_state = experiment_seed  # Set random state for the metric computation with Virny
        multiple_models_metrics_dct = compute_metrics_with_db_writer(dataset=main_base_flow_dataset,
                                                                     config=self.virny_config,
                                                                     models_config=models_dct,
                                                                     custom_tbl_fields_dct=custom_table_fields_dct,
                                                                     db_writer_func=self._db.get_db_writer(collection_name=ALL_EXPERIMENT_METRICS_TABLE),
                                                                     notebook_logs_stdout=False,
                                                                     postprocessor=postprocessor,
                                                                     verbose=0)
        print(f'Metric computation for {null_imputer_name}&{fairness_intervention_name}&{model_name} was finished\n', flush=True)

        return multiple_models_metrics_dct
