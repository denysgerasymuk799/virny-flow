"""Adaptive pipeline selection."""

import numpy as np

from alpine_meadow.common import Dataset
from alpine_meadow.utils.performance import time_calls, timer
from alpine_meadow.utils import AMException
from .base import Evaluation


class AdaptiveEvaluation(Evaluation):
    """
    The evaluation method for Adaptive Pipeline Selection, which trains and tests the pipeline on increasingly-large
    samples, and if the train error of a pipeline is beyond the best validation so far, the pipeline will be pruned
    """

    def __init__(self):
        self._initialized = False
        self.sub_train_datasets = None
        self.sub_train_datasets_sizes = None
        self.sub_train_targets = None
        self.best_validation_error = None
        super().__init__()

    def init(self, optimizer, dataset):
        if self._initialized:
            return

        super().init(optimizer, dataset)

        # sub epochs
        sub_epoch_num = max(1,
                            self._train_dataset.num_bytes // self._config.aps_minimum_slice_size)
        self._config.logger.debug(f'[AdaptiveEvaluation] Number of sub epochs: {sub_epoch_num}')

        task = self._optimizer.task
        target_columns = task.target_columns

        # get datasets
        sub_train_datasets, sub_train_datasets_sizes = \
            self.get_sub_datasets(self._train_dataset, sub_epoch_num, return_sizes=True)

        epoch_size = 1
        self.sub_train_datasets = []
        self.sub_train_datasets_sizes = []
        # we do 1, 2, 4, ..., epoch_num
        while epoch_size < sub_epoch_num:
            self.sub_train_datasets.append(sub_train_datasets[epoch_size - 1])
            self.sub_train_datasets_sizes.append(sub_train_datasets_sizes[epoch_size - 1])
            epoch_size *= 2
        self.sub_train_datasets.append(sub_train_datasets[-1])
        self.sub_train_datasets_sizes.append(sub_train_datasets_sizes[-1])

        # get targets
        self.sub_train_targets = []
        for sub_train_dataset in self.sub_train_datasets:
            sub_train_target = sub_train_dataset.to_data_frame()[target_columns]
            self.sub_train_targets.append(sub_train_target)

        self.best_validation_error = np.inf

    @time_calls
    def validate_pipeline(self, pipeline):
        import time

        task = self._optimizer.task
        pipeline_executor = self._optimizer.backend.get_pipeline_executor(pipeline, task.metrics)
        pipeline_history = pipeline.pipeline_arm.pipeline_history

        pruned = False
        start_time = time.perf_counter()
        train_errors = []
        validation_errors = []
        for sub_epoch_index, sub_train_dataset in enumerate(self.sub_train_datasets):
            self._optimizer.metrics['total_pipelines_num'] += 1
            sub_train_target = self.sub_train_targets[sub_epoch_index]

            # train
            train_start = time.perf_counter()
            pipeline_executor = self.train_pipeline(pipeline_executor, sub_train_dataset)
            train_time = time.perf_counter() - train_start

            # validation
            validation_start = time.perf_counter()
            pipeline_executor = self.test_pipeline(
                pipeline_executor,
                [(sub_train_dataset, sub_train_target), (self._validation_dataset, self._validation_target)])
            validation_time = time.perf_counter() - validation_start

            # get scores
            _, validation_score = pipeline_executor.get_scores(to_utility=True)[-2:]
            train_error, validation_error = pipeline_executor.get_scores(to_error=True)[-2:]
            train_errors.append(train_error)
            validation_errors.append(validation_error)

            # remove train scores and set validation method
            for metric in pipeline_executor.metrics:
                pipeline_executor.scored_dataset_ids = pipeline_executor.scored_dataset_ids[-1:]
                pipeline_executor.set_scores([pipeline_executor.get_scores(metric=metric)[-1]], metric=metric)
            pipeline_executor.validation_method = self._validation_method
            # logger.debug('Pipeline: {}, Phase: {}, Error: {}'.format(pipeline.id, sub_epoch_index, train_error))
            # logger.debug('Best validation error: {}'.format(self.best_validation_error))

            # update best validation error
            if validation_error < self.best_validation_error:
                self.best_validation_error = validation_error

                if not self.is_cross_validation_enabled():
                    yield pipeline_executor

            # pruning
            if self._config.enable_aps_pruning:
                if train_error > self.best_validation_error:
                    pruned = True
                    self._config.logger.debug(f'Pipeline {pipeline.id}, Phase {sub_epoch_index}, pruned')

            # save metrics
            self._optimizer.state.add_score(validation_score)
            self._optimizer.state.add_time(train_time + validation_time)
            pipeline.metrics.score = validation_score
            pipeline.metrics.time = train_time + validation_time
            pipeline_history.update_run_history(pipeline.id)

            self._optimizer.metrics[f'phase_{sub_epoch_index}_pipelines_num'] += 1
            timer(f'pipeline.phase_{sub_epoch_index}_time')._update(  # pylint: disable=protected-access
                time.perf_counter() - start_time)

            # if pruned we stop
            if pruned:
                break

            # if done we stop
            if self._optimizer.state.done:
                break

        if self._optimizer.state.done:
            return

        # curve fitting
        if self._config.enable_aps_curve_fitting:
            from .curve_fitting import CurveModel

            model = CurveModel(len(validation_errors) - 1, logger=self._config.logger)

            predicted_error = model.predict(train_errors[:-1])
            self._config.logger.debug(f'Train error {train_errors[-1]}, predicted {predicted_error}')

            predicted_error = model.predict(validation_errors[:-1])
            self._config.logger.debug(f'Validation error {validation_errors[-1]}, predicted {predicted_error}')

        pipeline.evaluated = True
        self._optimizer.metrics['validated_pipelines_num'] += 1
        if pruned:
            self._optimizer.metrics['pruned_pipelines_num'] += 1
        else:
            self._optimizer.metrics['un_pruned_pipelines_num'] += 1

            # do cross validation in batch mode
            if self.is_cross_validation_enabled():
                pipeline_executor = self.evaluate_pipeline_with_cross_validation(pipeline)

            yield pipeline_executor

    def get_sub_datasets(self, dataset: Dataset, sub_datasets_num, return_sizes=False):
        """
        Split the dataset into sub_datasets_num equal-sized pieces, and return first $i$ pieces for each sub dataset
        """

        import math

        instances_num = dataset.num_instances
        data_frame = dataset.to_data_frame()
        interval_size = int(math.floor(instances_num / sub_datasets_num))
        if interval_size <= 0:
            raise AMException(f"interval size must be larger than 0: {interval_size}")
        # if the interval is too big, we will create smaller intervals
        # interval_size = min(interval_size, 10000)

        sub_datasets = []
        sub_sizes = []
        for start in range(0, instances_num, interval_size):
            if start + interval_size < instances_num:
                sub_dataset = dataset.from_data_frame(data_frame[:start + interval_size])
                sub_size = start + interval_size
            else:
                sub_dataset = dataset
                sub_size = instances_num
            sub_datasets.append(sub_dataset)
            sub_sizes.append(sub_size)

        if return_sizes:
            return sub_datasets, sub_sizes
        return sub_datasets
