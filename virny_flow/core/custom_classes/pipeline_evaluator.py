import copy
import time
import pandas as pd

from datetime import datetime, timezone
from munch import DefaultMunch
from openbox.utils.constants import SUCCESS
from openbox.utils.history import Observation
from openbox.utils.util_funcs import parse_result
from sklearn.model_selection import train_test_split
from virny.user_interfaces import compute_metrics_with_fitted_bootstrap, compute_metrics_with_config

from .ml_lifecycle import MLLifecycle
from virny_flow.core.utils.common_helpers import create_base_flow_dataset
from virny_flow.core.utils.pipeline_utils import get_dis_group_condition, nested_dict_from_flat
from virny_flow.core.preprocessing import preprocess_base_flow_dataset
from virny_flow.core.fairness_interventions.preprocessors import remove_disparate_impact, apply_learning_fair_representations
from virny_flow.core.fairness_interventions.postprocessors import get_eq_odds_postprocessor, get_reject_option_classification_postprocessor
from virny_flow.core.fairness_interventions.inprocessors import get_adversarial_debiasing_wrapper_config, get_exponentiated_gradient_reduction_wrapper
from virny_flow.core.custom_classes.core_db_client import run_transaction_with_retry, commit_with_retry
from virny_flow.configs.structs import Task, PhysicalPipeline
from virny_flow.configs.constants import (ErrorRepairMethod, STAGE_SEPARATOR, NO_FAIRNESS_INTERVENTION, ALL_EXPERIMENT_METRICS_TABLE,
                                          FairnessIntervention, LOGICAL_PIPELINE_SCORES_TABLE)
from virny_flow.task_manager.domain_logic.bayesian_optimization import get_objective_losses


def get_best_compound_pp_improvement(session, db, logical_pipeline_uuid: str, exp_config_name: str, use_init_best_score: bool):
    # Read best_compound_pp_improvement for lp uuid and pipeline_quality_mean for each objective
    logical_pipeline_record = db.read_one_query(collection_name=LOGICAL_PIPELINE_SCORES_TABLE,
                                                session=session,
                                                query={"logical_pipeline_uuid": logical_pipeline_uuid,
                                                       "exp_config_name": exp_config_name})
    # Since the first batch for a new logical pipeline does not have pipeline_quality_mean, we need to use another table
    # to save best_compound_pp_improvement and use it for pruning unpromising pipelines in batch 1
    best_compound_pp_improvement = logical_pipeline_record.get('init_best_compound_pp_improvement', 0) if use_init_best_score \
                                        else logical_pipeline_record['best_compound_pp_improvement']

    return best_compound_pp_improvement


def adaptive_execution_transaction(session, db, task: Task, exp_config_name: str,
                                   cur_test_compound_pp_improvement: float, use_init_best_score: bool):
    # Read best_compound_pp_improvement for lp uuid and pipeline_quality_mean for each objective
    best_compound_pp_improvement = get_best_compound_pp_improvement(session=session,
                                                                    db=db,
                                                                    logical_pipeline_uuid=task.physical_pipeline.logical_pipeline_uuid,
                                                                    exp_config_name=exp_config_name,
                                                                    use_init_best_score=use_init_best_score)

    if cur_test_compound_pp_improvement > best_compound_pp_improvement:
        best_compound_pp_improvement = cur_test_compound_pp_improvement

        # Update best_compound_pp_improvement in DB in best_physical_pipeline_observations
        if use_init_best_score:
            update_val_dct = {"init_best_compound_pp_improvement": best_compound_pp_improvement,
                              "init_best_physical_pipeline_uuid": task.physical_pipeline.physical_pipeline_uuid}
        else:
            update_val_dct = {"best_compound_pp_improvement": best_compound_pp_improvement,
                              "best_physical_pipeline_uuid": task.physical_pipeline.physical_pipeline_uuid}

        db.update_query(collection_name=LOGICAL_PIPELINE_SCORES_TABLE,
                        session=session,
                        condition={"exp_config_name": exp_config_name,
                                   "logical_pipeline_uuid": task.physical_pipeline.logical_pipeline_uuid},
                        update_val_dct=update_val_dct)


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
        self.training_set_fractions_for_halting = exp_config.training_set_fractions_for_halting

    def execute_task(self, task: Task, seed: int):
        self._db.connect()

        pipeline_quality_mean = task.pipeline_quality_mean
        use_init_best_score = True if set(pipeline_quality_mean.values()) == {0.0} else False
        print("\n\nuse_init_best_score:", use_init_best_score)
        print("pipeline_quality_mean:", pipeline_quality_mean)

        # Start a client session
        session = self._db.client.start_session()
        adaptive_pipeline_generator = self.execute_adaptive_pipeline(physical_pipeline=task.physical_pipeline,
                                                                     objectives=task.objectives,
                                                                     pipeline_quality_mean=pipeline_quality_mean,
                                                                     exp_config_name=self.exp_config_name,
                                                                     use_init_best_score=use_init_best_score,
                                                                     seed=seed)
        for cur_test_compound_pp_improvement, observation, test_multiple_models_metrics_df in adaptive_pipeline_generator:
            try:
                session.start_transaction()
                run_transaction_with_retry(adaptive_execution_transaction,
                                           session,
                                           db=self._db,
                                           task=task,
                                           exp_config_name=self.exp_config_name,
                                           cur_test_compound_pp_improvement=cur_test_compound_pp_improvement,
                                           use_init_best_score=use_init_best_score)
                commit_with_retry(session)
            except Exception as e:
                print(f"Transaction aborted: {e}")
                session.abort_transaction()

        # Write virny metrics from the latest halting round into database
        null_imputer_name, fairness_intervention_name, model_name = task.physical_pipeline.logical_pipeline_name.split(STAGE_SEPARATOR)
        custom_tbl_fields_dct = dict()
        custom_tbl_fields_dct['session_uuid'] = self._session_uuid
        custom_tbl_fields_dct['task_uuid'] = task.task_uuid
        custom_tbl_fields_dct['physical_pipeline_uuid'] = task.physical_pipeline.physical_pipeline_uuid
        custom_tbl_fields_dct['logical_pipeline_name'] = task.physical_pipeline.logical_pipeline_name
        custom_tbl_fields_dct['dataset_split_seed'] = seed
        custom_tbl_fields_dct['model_init_seed'] = seed
        custom_tbl_fields_dct['experiment_seed'] = seed
        custom_tbl_fields_dct['exp_config_name'] = self.exp_config_name
        custom_tbl_fields_dct['null_imputer_name'] = null_imputer_name
        custom_tbl_fields_dct['fairness_intervention_name'] = fairness_intervention_name

        self.save_virny_metrics_in_db(model_metrics_df=test_multiple_models_metrics_df,
                                      custom_tbl_fields_dct=custom_tbl_fields_dct)

        # End the session
        session.end_session()

        self._logger.info(f"Task with UUID {task.task_uuid} and logical pipeline {task.physical_pipeline.logical_pipeline_name} was executed!")
        self._db.close()

        return observation

    def execute_adaptive_pipeline(self, physical_pipeline: PhysicalPipeline, objectives: list, pipeline_quality_mean: dict,
                                  exp_config_name: str, use_init_best_score: bool, seed: int):
        data_loader = copy.deepcopy(self.init_data_loader)
        X_train_val, X_test, y_train_val, y_test = self._split_dataset(data_loader, seed)

        test_compound_pp_improvement = None
        observation = None
        test_multiple_models_metrics_df = None
        for training_set_fraction in self.training_set_fractions_for_halting:
            print('training_set_fraction:', training_set_fraction)

            # Split a dataset based on the current fraction
            X_train, _, y_train, _ = (X_train_val, None, y_train_val, None if training_set_fraction == 1.0
                                        else train_test_split(X_train_val, y_train_val,
                                                              train_size=training_set_fraction, stratify=y_train_val))
            # Execute a pipeline for the current incremental training set
            (train_objectives, test_objectives,
             observation, test_multiple_models_metrics_df) = self.execute_pipeline(X_train=X_train,
                                                                                   X_test=X_test,
                                                                                   y_train=y_train,
                                                                                   y_test=y_test,
                                                                                   physical_pipeline=physical_pipeline,
                                                                                   objectives=objectives,
                                                                                   seed=seed)
            # Create train_compound_pp_improvement and test_compound_pp_improvement based on the test_losses
            train_compound_pp_improvement = 0.0
            test_compound_pp_improvement = 0.0
            for idx, objective in enumerate(objectives):
                train_compound_pp_improvement += train_objectives[idx] - pipeline_quality_mean[objective["name"]]
                test_compound_pp_improvement += test_objectives[idx] - pipeline_quality_mean[objective["name"]]

            best_compound_pp_improvement = get_best_compound_pp_improvement(session=None,
                                                                            db=self._db,
                                                                            logical_pipeline_uuid=physical_pipeline.logical_pipeline_uuid,
                                                                            exp_config_name=exp_config_name,
                                                                            use_init_best_score=use_init_best_score)
            print("train_compound_pp_improvement:", train_compound_pp_improvement)
            print("test_compound_pp_improvement:", test_compound_pp_improvement)
            print("best_compound_pp_improvement:", best_compound_pp_improvement)
            if test_compound_pp_improvement > best_compound_pp_improvement:
                best_compound_pp_improvement = test_compound_pp_improvement

            yield test_compound_pp_improvement, observation, test_multiple_models_metrics_df

            if train_compound_pp_improvement < best_compound_pp_improvement:
                return test_compound_pp_improvement, observation, test_multiple_models_metrics_df

        return test_compound_pp_improvement, observation, test_multiple_models_metrics_df

    def execute_pipeline(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame,
                         physical_pipeline: PhysicalPipeline, objectives: list, seed: int):
        # Parse an input physical pipeline
        null_imputer_name, fairness_intervention_name, model_name = physical_pipeline.logical_pipeline_name.split(STAGE_SEPARATOR)
        null_imputer_params = nested_dict_from_flat(physical_pipeline.null_imputer_params)
        fairness_intervention_params = nested_dict_from_flat(physical_pipeline.fairness_intervention_params)
        model_params = nested_dict_from_flat(physical_pipeline.model_params)

        start_time = time.time()

        # Perform null imputation
        main_base_flow_dataset = self.run_null_imputation_stage(X_train_val_with_nulls=X_train,
                                                                X_test_with_nulls=X_test,
                                                                y_train_val=y_train,
                                                                y_test=y_test,
                                                                experiment_seed=seed,
                                                                null_imputer_name=null_imputer_name,
                                                                null_imputer_params=null_imputer_params)
        # Apply a pre-processing fairness intervention if defined
        preprocessed_base_flow_dataset = self.run_fairness_intervention_stage(main_base_flow_dataset=main_base_flow_dataset,
                                                                              init_data_loader=self.init_data_loader,
                                                                              experiment_seed=seed,
                                                                              fairness_intervention_name=fairness_intervention_name,
                                                                              fairness_intervention_params=fairness_intervention_params)
        # Evaluate a model
        (test_multiple_models_metrics_dct,
         train_multiple_models_metrics_dct) = self.run_model_evaluation_stage(main_base_flow_dataset=preprocessed_base_flow_dataset,
                                                                              experiment_seed=seed,
                                                                              null_imputer_name=null_imputer_name,
                                                                              fairness_intervention_name=fairness_intervention_name,
                                                                              model_name=model_name,
                                                                              model_params=model_params)
        elapsed_time = time.time() - start_time

        # Create an observation for MO optimisation based on the computed metrics
        train_objectives = get_objective_losses(metrics_dct=train_multiple_models_metrics_dct,
                                                objectives=objectives,
                                                model_name=model_name,
                                                sensitive_attributes_dct=self.virny_config.sensitive_attributes_dct)
        print("train_objectives:", train_objectives)

        test_objectives = get_objective_losses(metrics_dct=test_multiple_models_metrics_dct,
                                               objectives=objectives,
                                               model_name=model_name,
                                               sensitive_attributes_dct=self.virny_config.sensitive_attributes_dct)
        print("test_objectives:", test_objectives)
        train_reversed_objectives = train_objectives.pop("reversed_objectives")
        test_reversed_objectives = test_objectives.pop("reversed_objectives")

        objective_values, constraints, extra_info = parse_result(copy.copy(test_objectives))
        observation = Observation(
            config=physical_pipeline.suggestion,
            objectives=objective_values,
            constraints=constraints,
            trial_state=SUCCESS,
            elapsed_time=elapsed_time,
            extra_info=extra_info,
        )
        observation.extra_info["reversed_objectives"] = test_reversed_objectives
        observation.extra_info["exp_config_objectives"] = objectives

        return train_reversed_objectives, test_reversed_objectives, observation, test_multiple_models_metrics_dct[model_name]

    def run_null_imputation_stage(self, X_train_val_with_nulls, X_test_with_nulls, y_train_val, y_test,
                                  experiment_seed: int, null_imputer_name: str, null_imputer_params: dict):
        if null_imputer_name in (ErrorRepairMethod.boost_clean.value, ErrorRepairMethod.cp_clean.value):
            raise ValueError(f'To work with {ErrorRepairMethod.boost_clean.value} or {ErrorRepairMethod.cp_clean.value}, '
                             f'use scripts/evaluate_models.py')

        # Remove sensitive attributes from train and test sets with nulls to avoid their usage during imputation
        (X_train_val_with_nulls_wo_sensitive_attrs,
         X_tests_with_nulls_wo_sensitive_attrs_lst,
         numerical_columns_wo_sensitive_attrs,
         categorical_columns_wo_sensitive_attrs) = self._remove_sensitive_attrs(X_train_val=X_train_val_with_nulls,
                                                                                X_tests_lst=[X_test_with_nulls],
                                                                                data_loader=self.init_data_loader)

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
        main_base_flow_dataset = create_base_flow_dataset(data_loader=self.init_data_loader,
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

        # Compute metrics for tuned models based on the test set
        self.virny_config.random_state = experiment_seed  # Set random state for the metric computation with Virny
        (test_multiple_models_metrics_dct,
         models_fitted_bootstraps_dct) = compute_metrics_with_config(dataset=main_base_flow_dataset,
                                                                     config=self.virny_config,
                                                                     models_config=models_dct,
                                                                     notebook_logs_stdout=False,
                                                                     postprocessor=postprocessor,
                                                                     return_fitted_bootstrap=True,
                                                                     verbose=0)

        # Compute metrics based on the training set to check model fit
        train_base_flow_dataset = copy.deepcopy(main_base_flow_dataset)
        train_base_flow_dataset.X_train_val = pd.DataFrame()
        train_base_flow_dataset.X_test = main_base_flow_dataset.X_train_val
        train_base_flow_dataset.y_train_val = pd.DataFrame()
        train_base_flow_dataset.y_test = main_base_flow_dataset.y_train_val
        train_metrics_df = compute_metrics_with_fitted_bootstrap(fitted_bootstrap=models_fitted_bootstraps_dct[model_name],
                                                                 test_base_flow_dataset=train_base_flow_dataset,
                                                                 config=self.virny_config,
                                                                 with_predict_proba=True)
        train_metrics_df['Model_Name'] = model_name
        train_multiple_models_metrics_dct = {model_name: train_metrics_df}
        print(f'Metric computation for {null_imputer_name}&{fairness_intervention_name}&{model_name} was finished\n', flush=True)

        return test_multiple_models_metrics_dct, train_multiple_models_metrics_dct

    def save_virny_metrics_in_db(self, model_metrics_df: pd.DataFrame, custom_tbl_fields_dct: dict):
        # Concatenate current run metrics with previous results and
        # create melted_model_metrics_df to save it in a database
        model_metrics_df['Dataset_Name'] = self.virny_config.dataset_name
        model_metrics_df['Num_Estimators'] = self.virny_config.n_estimators

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
        self._db.execute_write_query(records=melted_model_metrics_df.to_dict('records'),
                                     collection_name=ALL_EXPERIMENT_METRICS_TABLE)
