"""Base class for Evaluation methods."""

from abc import ABC, abstractmethod
import traceback
import time

import numpy as np

from alpine_meadow.common import TaskKeyword, PerformanceMetric, ValidationMethod
from alpine_meadow.core.meta_learning.run import StepRun, PipelineRun
from alpine_meadow.utils.performance import time_calls, timer
from alpine_meadow.utils import AMException
from .utils import compute_classification_metrics, compute_regression_metrics


class Evaluation(ABC):
    """
    The base class for evaluation method
    """

    def __init__(self):
        self._initialized = False
        self._config = None
        self._optimizer = None
        self._dataset = None
        self._train_dataset = None
        self._validation_dataset = None
        self._validation_target = None
        self._validation_method = None
        self._splits = []

    @time_calls
    def init(self, optimizer, dataset):
        """
        Initialize the train/validation split and cross-validation splits (if enabled).
        :param optimizer:
        :param dataset:
        :return:
        """

        if self._initialized:
            return

        self._config = optimizer.config
        self._optimizer = optimizer
        self._dataset = dataset

        # get labels for stratified sampling
        df = dataset.to_data_frame()
        task = optimizer.task
        target_columns = task.target_columns
        if task.type == TaskKeyword.CLASSIFICATION:
            labels = df[target_columns[0]].values
        else:
            labels = np.ones(len(df))

        # get train and validation dataset
        if task.train_dataset is not None:
            if task.validation_dataset is None:
                raise AMException("Validation dataset cannot be None")
            self._train_dataset = task.train_dataset
            self._validation_dataset = task.validation_dataset
            self._validation_method = ValidationMethod()
            self._validation_method.user_provided.SetInParent()
        else:
            while True:
                from sklearn.model_selection import train_test_split
                try:
                    train_df, test_df = train_test_split(df, train_size=self._config.train_data_size,
                                                         stratify=labels, random_state=self._config.seed)
                except BaseException:  # pylint: disable=broad-except
                    train_df, test_df = train_test_split(df, train_size=self._config.train_data_size,
                                                         random_state=self._config.seed)

                # create train dataset
                self._train_dataset = task.dataset.from_data_frame(train_df)
                self._train_dataset.tags.append('train')

                # create validation dataset
                self._validation_dataset = task.dataset.from_data_frame(test_df)
                self._validation_dataset.tags.append('validation')

                if df.empty:
                    break
                if self._train_dataset.num_instances > 0 and self._validation_dataset.num_instances > 0:
                    break
            task.train_dataset = self._train_dataset
            task.validation_dataset = self._validation_dataset
            self._validation_method = ValidationMethod()
            self._validation_method.holdout.train_proportion = self._config.train_data_size
            self._validation_method.holdout.test_proportion = 1 - self._config.train_data_size
        self._validation_target = self._validation_dataset.to_data_frame()[target_columns]

        # get splits for cross validation
        if self.is_cross_validation_enabled():
            self._config.logger.info('Cross validation enabled')
            # get splits
            if self._config.cross_validation_strategy == 'kfold':
                from sklearn.model_selection import KFold

                cv = KFold(n_splits=self._config.cross_validation_k_folds_num,
                           shuffle=True, random_state=self._config.seed)
            else:
                raise RuntimeError(f'Invalid cross validation strategy: {self._config.cross_validation_strategy}!')

            self._splits = self._optimizer.task.create_dataset_splits(cv)
            self._validation_method = ValidationMethod()
            self._validation_method.cross_validation.num_folds = self._config.cross_validation_k_folds_num

        # initialized
        self._initialized = True

    @time_calls
    def train_pipeline(self, pipeline_executor, train_dataset, is_cross_validation=False):
        """
        Train the pipeline over the dataset.
        :param pipeline_executor:
        :param train_dataset:
        :param is_cross_validation:
        :return:
        """

        new_pipeline_executor = self._optimizer.backend.get_pipeline_executor(
            pipeline_executor.pipeline, pipeline_executor.metrics)

        # train
        train_start = time.perf_counter()
        kwargs = {}
        train_response = new_pipeline_executor.train(
            [train_dataset],
            time_limit=self._optimizer.state.get_remaining_time(),
            **kwargs)
        train_end = time.perf_counter()

        # add pipeline run
        if is_cross_validation:
            tags = ['cross-validation']
        else:
            tags = []
        self._on_response(train_response, 'TRAIN', tags, train_start, train_end,
                          pipeline_executor.pipeline,
                          getattr(pipeline_executor, 'trained_dataset_ids', []),
                          None)

        return new_pipeline_executor

    @time_calls
    def test_pipeline(self, pipeline_executor, test_datasets, is_cross_validation=False):
        """
        Test the pipeline over the dataset.
        :param pipeline_executor:
        :param test_datasets:
        :param is_cross_validation:
        :return:
        """

        task = self._optimizer.task
        kwargs = {}
        if len(task.metrics) == 1 and 'ROC_AUC' in PerformanceMetric.Name(task.metrics[0]):
            kwargs['produce_proba'] = True

        # test
        for test_dataset, test_target in test_datasets:
            score_start = time.perf_counter()
            score_response = pipeline_executor.score(
                [test_dataset], [test_target],
                time_limit=self._optimizer.state.get_remaining_time(),
                **self._optimizer.task.scoring_kwargs,
                **kwargs)
            score_end = time.perf_counter()

            # add pipeline run
            if is_cross_validation:
                tags = ['cross-validation']
            else:
                tags = []
            self._on_response(score_response, 'TEST', tags, score_start, score_end,
                              pipeline_executor.pipeline,
                              getattr(pipeline_executor, 'trained_dataset_ids', []),
                              test_dataset.id)

        return pipeline_executor

    def is_cross_validation_enabled(self):
        if not self._config.enable_cross_validation:
            return False

        # if instances num is big, we don't do cross validation
        # instances_num = self.dataset.num_instances
        # if instances_num > self.config.cross_validation_instances_num_threshold:
        #     return False

        return True

    @time_calls
    def evaluate_pipeline_with_cross_validation(self, pipeline):
        """
        Evaluate a pipeline with cross-validation.
        :param pipeline:
        :return:
        """

        task = self._optimizer.task

        # cross validation
        pipeline_executor = None
        for train_dataset, test_dataset in self._splits:
            new_pipeline_executor = self._optimizer.backend.get_pipeline_executor(pipeline, task.metrics)

            new_pipeline_executor = self.train_pipeline(new_pipeline_executor, train_dataset,
                                                        is_cross_validation=True)
            new_pipeline_executor = self.test_pipeline(new_pipeline_executor, [test_dataset],
                                                       is_cross_validation=True)

            if pipeline_executor is None:
                pipeline_executor = new_pipeline_executor
            else:
                pipeline_executor.merge_scores(new_pipeline_executor)

        # update scores and validation method
        pipeline_executor.pipeline.metrics.score = pipeline_executor.get_score(to_utility=True)
        pipeline_executor.pipeline.metrics.cross_validation_score = pipeline_executor.get_score(to_utility=True)
        pipeline_executor.validation_method = self._validation_method

        return pipeline_executor

    @abstractmethod
    def validate_pipeline(self, pipeline):
        pass

    def compute_all_metrics(self, pipeline_executor):
        """
        Compute all metrics per the task type for the given pipeline executor
        """

        task = self._optimizer.task
        if self.is_cross_validation_enabled():
            test_datasets = map(lambda x: x[1][0], self._splits)
        else:
            test_datasets = [self._validation_dataset]

        pipeline_executor.clear_scores()
        for test_dataset in test_datasets:
            y_true = test_dataset.to_data_frame()[task.target_columns]
            y_pred = pipeline_executor.test([test_dataset]).outputs
            y_pred_proba = None
            try:
                y_pred_proba = pipeline_executor.test_proba([test_dataset]).outputs
            except:  # pylint: disable=bare-except # noqa: E722
                traceback.print_exc()

            if task.type == TaskKeyword.CLASSIFICATION:
                classes = pipeline_executor.model.primitive.classes_
                metrics = compute_classification_metrics(
                    y_true=y_true, y_pred=y_pred, y_pred_proba=y_pred_proba, classes=classes,
                    **task.scoring_kwargs)
            else:
                assert task.type == TaskKeyword.REGRESSION
                metrics = compute_regression_metrics(y_true=y_true, y_pred=y_pred,
                                                     **task.scoring_kwargs)

            for metric, score in metrics.items():
                pipeline_executor.add_score(metric, score)

    def _on_response(self, response, context, tags, start_time, end_time, pipeline,
                     trained_splits, test_split):
        """
        Add a pipeline run for the train/test response
        :param response:
        :param context:
        :param tags:
        :param start_time:
        :param end_time:
        :param pipeline:
        :param trained_splits:
        :param test_split:
        :return:
        """

        # update pipeline history
        for metric_name, metric_value in response.metrics.items():
            if metric_name == 'step_runs':
                continue

            # update metrics in profiling
            if metric_name.endswith('time'):
                timer(metric_name)._update(metric_value)  # pylint: disable=protected-access
            else:
                if metric_name.endswith('train_count') or metric_name.endswith('inference_count'):
                    pass
                else:
                    raise AMException(f"Unknown metric: {metric_name}")

        if not self._config.log_trace:
            return

        step_runs = []
        for step_run_raw in response.metrics.get('step_runs', []):
            step_run = StepRun(
                index=step_run_raw['index'],
                primitive=step_run_raw['primitive'],
                fit_time=step_run_raw['fit_time'],
                produce_time=step_run_raw['produce_time']
            )
            step_runs.append(step_run)
        pipeline_run = PipelineRun(
            pipeline=pipeline,
            context=context,
            trained_splits=trained_splits,
            test_split=test_split,
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            tags=tags,
            step_runs=step_runs)
        self._optimizer.pipeline_history.add_pipeline_run(pipeline_run)
