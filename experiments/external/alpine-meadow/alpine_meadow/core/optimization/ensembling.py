# pylint: disable=no-else-return, consider-using-with
"""Ensembling."""

import time
import concurrent.futures

import numpy as np
import pandas as pd

from alpine_meadow.common import TaskKeyword
from alpine_meadow.backend.base import TrainResult, ScoreResult, TestResult
from alpine_meadow.backend.default.pipeline_executor import DefaultPipelineExecutor
from alpine_meadow.utils import AMException


class EnsembleVotingPipeline(DefaultPipelineExecutor):
    """
    Pipeline executor for majority voting based ensemble.
    """

    def __init__(self, task, pipelines):
        super().__init__(pipelines[0].pipeline, task.metrics)

        self._pipelines = pipelines

    def train(self, datasets, **kwargs) -> TrainResult:
        raise RuntimeError("No need to train ensemble!")

    def test(self, datasets, **kwargs) -> TestResult:
        columns = None
        predictions = []
        for pipeline in self._pipelines:
            output = pipeline.test(datasets, **kwargs).outputs
            columns = list(output.columns)
            predictions.append(output.values)
        predictions = np.asarray(predictions)
        if len(predictions.shape) == 3 and predictions.shape[-1] == 1:
            predictions = predictions.reshape(predictions.shape[:-1])
        predictions = predictions.T

        # encode
        from sklearn.preprocessing import LabelEncoder

        label_encoder = LabelEncoder()
        label_encoder.fit(predictions.flatten())

        # vote
        values = np.apply_along_axis(lambda x: np.argmax(np.bincount(label_encoder.transform(x))), axis=1,
                                     arr=predictions)

        # decode
        values = label_encoder.inverse_transform(values)

        return TestResult(outputs=pd.DataFrame(columns=columns, data=values), metrics=None)


class EnsembleStackingPipeline(DefaultPipelineExecutor):
    """
    Pipeline executor for stacking based ensemble.
    """

    def __init__(self, task, pipelines, stacking_data, num_models):
        super().__init__(pipelines[0].pipeline, task.metrics)

        self._pipelines = pipelines

        # train stacking model
        X, y = stacking_data  # pylint: disable=invalid-name
        from xgboost import XGBRegressor
        self._stacking_model = XGBRegressor()
        self._stacking_model.fit(X, y)

        if len(pipelines) % num_models != 0:
            raise AMException(f"The number of pipelines should be a"
                              f"multiple of the number of models: {len(pipelines)}, {num_models}")
        self._num_models = num_models
        self._num_pipelines_per_model = len(pipelines) // num_models

    def train(self, datasets, **kwargs) -> TrainResult:
        raise RuntimeError("No need to train ensemble!")

    def test(self, datasets, **kwargs) -> TestResult:
        # get predictions from all models
        columns = None
        predictions = None
        sub_predictions = None
        for index, pipeline in enumerate(self._pipelines):
            output = pipeline.test(datasets, **kwargs).outputs
            if columns is None:
                columns = list(output.columns)
            output = output.values.flatten()

            if predictions is None:
                predictions = np.zeros((len(output), self._num_models))
            if sub_predictions is None:
                sub_predictions = np.zeros((len(output), self._num_pipelines_per_model))

            # write predictions
            pipeline_index = index % self._num_pipelines_per_model
            sub_predictions[:, pipeline_index] = output
            if pipeline_index == self._num_pipelines_per_model - 1:
                model_index = index // self._num_pipelines_per_model
                predictions[:, model_index] = sub_predictions.mean(1)

        # run the stacking model
        values = self._stacking_model.predict(predictions)

        return TestResult(outputs=pd.DataFrame(columns=columns, data=values), metrics=None)


class EnsemblePipelineExecutor(DefaultPipelineExecutor):
    """
    Ensemble pipeline executor for building the ensemble of pipelines.
    """

    def __init__(self, task, pipelines):
        super().__init__(pipelines[0], task.metrics)

        self._task = task
        self._pipelines = pipelines

        # for ensemble, we use voting for classification and stacking for regression
        self._ensemble_method = {
            TaskKeyword.Value('CLASSIFICATION'): 'VOTING',
            TaskKeyword.Value('REGRESSION'): 'STACKING'
        }[task.type]
        self._ensemble = None

    def train(self, datasets, **kwargs) -> TrainResult:
        self._ensemble = self._train(datasets)

    def test(self, datasets, **kwargs) -> TestResult:
        return self._ensemble.test(datasets, **kwargs)

    def test_proba(self, datasets, **kwargs) -> TestResult:
        return self.test(datasets, **kwargs)

    def score(self, datasets, targets, **kwargs) -> ScoreResult:
        """
        Compute the scores based on the inputs and metrics
        :param datasets:
        :param targets:
        :param kwargs:
        :return:
        """

        from alpine_meadow.common.metric import get_score

        test_result = self.test(datasets, **kwargs)
        outputs = test_result.outputs
        metrics = test_result.metrics
        truth = targets[0]
        scores = []
        for metric in self.metrics:
            score = get_score(metric, truth.values, outputs.values, **self._task.scoring_kwargs)
            self.add_score(metric, score)
            scores.append(score)
        return ScoreResult(scores=scores, metrics=metrics)

    def _train(self, datasets):
        """
        Train the ensemble.
        :param datasets:
        :return:
        """

        # variables
        task = self._task
        pipelines = self._pipelines
        metrics = self.metrics
        ensemble_method = self._ensemble_method
        if len(datasets) != 1:
            raise AMException(f"The number of datasets must be 1: {len(datasets)}")
        dataset = datasets[0]
        backend = concurrent.futures.ThreadPoolExecutor(max_workers=len(pipelines))

        def evaluate_pipeline(pipeline, train_dataset, test_dataset=None):
            try:
                # init
                pipeline_executor = DefaultPipelineExecutor(pipeline, metrics)

                # train
                pipeline_executor.train([train_dataset])

                # test
                if test_dataset is not None:
                    predictions = pipeline_executor.test([test_dataset]).outputs
                else:
                    predictions = None

                return pipeline_executor, predictions
            except Exception as e:
                task.config.logger.debug(msg='', exc_info=True)
                raise e

        if ensemble_method == 'VOTING':
            # train pipelines
            train_test_futures = []
            for pipeline in pipelines:
                train_test_futures.append(backend.submit(evaluate_pipeline, pipeline, dataset))
            completed_futures, uncompleted_futures = concurrent.futures.wait(train_test_futures)
            if uncompleted_futures:
                raise RuntimeError('Ensemble train fails!')

            pipeline_executors = []
            for future in completed_futures:
                pipeline_executor, _ = future.result()
                pipeline_executors.append(pipeline_executor)

            return EnsembleVotingPipeline(task, pipeline_executors)

        else:
            if ensemble_method != 'STACKING':
                raise AMException(f"Unknown ensemble method: {ensemble_method}")

            # get splits
            from sklearn.model_selection import KFold

            cv = KFold(n_splits=task.config.ensembling_folds_num,
                       shuffle=True, random_state=2020)
            df = dataset.to_data_frame()
            index_splits = list(cv.split(df, None))
            dataset_splits = self._task.create_dataset_splits(cv, df=df)

            # train/test on splits
            train_test_futures = []
            for pipeline in self._pipelines:
                for train_dataset, test_dataset in dataset_splits:
                    train_test_futures.append(backend.submit(
                        evaluate_pipeline, pipeline, train_dataset, test_dataset[0]))
            completed_futures, uncompleted_futures = concurrent.futures.wait(train_test_futures)
            if uncompleted_futures:
                raise RuntimeError('Ensemble train fails!')

            # prepare stacking data
            stacking_train_data = np.zeros((dataset.num_instances, len(pipelines)))
            for index, future in enumerate(train_test_futures):
                pipeline_executor, predictions = future.result()
                predictions = predictions.values.flatten()
                test_index = index_splits[index % len(index_splits)][1]
                pipeline_index = index // len(index_splits)
                stacking_train_data[test_index, pipeline_index] = predictions
            stacking_labels = dataset.to_data_frame()[task.target_columns]
            stacking_data = (stacking_train_data, stacking_labels)

            pipeline_executors = []
            for future in completed_futures:
                pipeline_executor, _ = future.result()
                pipeline_executors.append(pipeline_executor)

            return EnsembleStackingPipeline(task, pipeline_executors, stacking_data, len(pipelines))


def ensembling(optimizer):
    """
    Apply ensembling
    :param optimizer:
    :param dataset:
    :return:
    """

    # only apply ensembling for classification and regression problems
    if optimizer.task.type not in [TaskKeyword.Value('CLASSIFICATION'),
                                   TaskKeyword.Value('REGRESSION')]:
        return None

    start = time.perf_counter()

    # get best k pipelines
    best_k_pipelines_with_scores = optimizer.pipeline_history.get_best_k_pipelines_with_scores(
        optimizer.config.ensembling_input_pipelines_num)
    optimizer.logger.info(f'Ensembled pipeline scores: {list(map(lambda x: x[0], best_k_pipelines_with_scores))}')
    best_k_pipelines = list(map(lambda x: x[1], best_k_pipelines_with_scores))

    # ensembling
    task = optimizer.task
    ensemble_pipeline_executor = EnsemblePipelineExecutor(optimizer.task, best_k_pipelines)
    ensemble_pipeline_executor.train([task.train_dataset])
    validation_target = task.validation_dataset.to_data_frame()[task.target_columns]
    ensemble_pipeline_executor.score([task.validation_dataset], [validation_target],
                                     **optimizer.task.scoring_kwargs)
    optimizer.logger.info(f'Ensembling time: {time.perf_counter() - start}')

    return ensemble_pipeline_executor
