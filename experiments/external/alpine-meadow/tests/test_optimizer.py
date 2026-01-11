# pylint: disable=missing-docstring
"""Tests for optimizers."""
import tempfile

import pytest

from alpine_meadow.common import Task, TaskKeyword, PerformanceMetric, Config
from alpine_meadow.core import Optimizer
from alpine_meadow.common.proto import pipeline_pb2 as base
from alpine_meadow.core.evaluation.utils import CLASSIFICATION_METRICS, REGRESSION_METRICS


def test_classification(df_185):
    # make task
    df1 = df_185.head(10000)
    df2 = df_185.tail(10000)
    metrics = [PerformanceMetric.Value('F1_MACRO'), PerformanceMetric.Value('ACCURACY'),
               PerformanceMetric.Value('PRECISION')]
    target_column = "Hall_of_Fame"
    task = Task([TaskKeyword.Value('CLASSIFICATION')], metrics, [target_column], df1,
                pos_label='0')
    print('Naive score', task.compute_naive_score())

    # run optimizer
    config = Config(timeout_seconds=15)
    optimizer = Optimizer(task, config=config)
    for result in optimizer.optimize():
        print(f'Time: {result.elapsed_time}, Score: {result.score}')
        for metric in metrics:
            result.pipeline.get_scores(metric)

    # run optimizer again (online/streaming optimizer)
    task.dataset = df2
    for result in optimizer.optimize():
        print(f'Time: {result.elapsed_time}, Score: {result.score}')
        for metric in metrics:
            result.pipeline.get_scores(metric)

    # dump trace
    with tempfile.NamedTemporaryFile() as f:
        optimizer.dump(f.name)


def test_regression(df_196):
    # make task
    df = df_196
    metric = PerformanceMetric.Value('MEAN_SQUARED_ERROR')
    target_column = "class"
    task = Task([TaskKeyword.Value('REGRESSION')], [metric], [target_column], df)
    print('Naive score', task.compute_naive_score())

    # run optimizer
    config = Config(timeout_seconds=15)
    optimizer = Optimizer(task, config=config)
    for result in optimizer.optimize():
        print(f'Time: {result.elapsed_time}, Score: {result.score}')


@pytest.mark.skip()
def test_fe(df_185):
    # make task
    df = df_185
    metric = PerformanceMetric.Value('F1_MACRO')
    target_column = "Hall_of_Fame"
    task = Task([TaskKeyword.Value('CLASSIFICATION')], [metric], [target_column], df)

    # run optimizer
    config = Config(enable_feature_engineering=True)
    optimizer = Optimizer(task, config=config)
    for result in optimizer.optimize():
        print(f'Time: {result.elapsed_time}, Score: {result.score}')


def test_limiting_random_forest(df_185):
    # make task
    df = df_185
    metric = PerformanceMetric.Value('F1')
    target_column = "Hall_of_Fame"
    class_weights = {"0": 1, "1": 10, "2": 2}
    task = Task([TaskKeyword.Value('CLASSIFICATION')], [metric], [target_column], df,
                class_weights=class_weights)

    # run optimizer
    config = Config(including_primitives=[base.Primitive.RandomForestClassifier],
                    timeout_seconds=5)
    optimizer = Optimizer(task, config=config)
    for result in optimizer.optimize():
        print(f'Time: {result.elapsed_time}, Score: {result.score}')
        model_name = base.Primitive.Name.Name(result.pipeline.pipeline.steps[-1].primitive.name)
        assert model_name == 'RandomForestClassifier'


def test_user_provided_train_validation(df_185):
    # make task
    df = df_185
    train_df = df[:800]
    test_df = df[800:]
    metric = PerformanceMetric.Value('F1_MACRO')
    target_column = "Hall_of_Fame"
    task = Task([TaskKeyword.Value('CLASSIFICATION')], [metric], [target_column],
                train_dataset=train_df, validation_dataset=test_df)

    # run optimizer
    config = Config(timeout_seconds=5)
    optimizer = Optimizer(task, config=config)
    for result in optimizer.optimize():
        assert result.pipeline.validation_method.HasField('user_provided')
        print(f'Time: {result.elapsed_time}, Score: {result.score}')


def test_predict_proba_and_thresholding(df_185):
    # make task
    df = df_185
    metric = PerformanceMetric.Value('F1_MACRO')
    target_column = "Hall_of_Fame"
    task = Task([TaskKeyword.Value('CLASSIFICATION')], [metric], [target_column], df)

    # run optimizer
    config = Config(predict_proba=True, timeout_seconds=5,
                    including_primitives=[base.Primitive.ThresholdingPrimitive])
    optimizer = Optimizer(task, config=config)
    result = None
    for result in optimizer.optimize():
        print(f'Time: {result.elapsed_time}, Score: {result.score}')

    # compute prob
    prob = result.pipeline.test_proba([task.dataset]).outputs
    assert prob.shape == (1073, 3)


def test_one_row_dataset(df_185):
    # make task
    df = df_185
    train_df = df.sample(n=1)
    test_df = df.sample(n=1)
    metric = PerformanceMetric.Value('F1_MACRO')
    target_column = "Hall_of_Fame"
    task = Task([TaskKeyword.Value('CLASSIFICATION')], [metric], [target_column],
                train_dataset=train_df, validation_dataset=test_df)

    # run optimizer
    config = Config(timeout_seconds=5)
    optimizer = Optimizer(task, config=config)
    for result in optimizer.optimize():
        print(f'Time: {result.elapsed_time}, Score: {result.score}')


def test_compute_all_metrics_classification(df_185):
    # make task
    df = df_185
    metric = PerformanceMetric.Value('F1_MACRO')
    target_column = "Hall_of_Fame"
    task = Task([TaskKeyword.Value('CLASSIFICATION')], [metric], [target_column], df)

    # run optimizer
    for enable_cross_validation in [True, False]:
        config = Config(timeout_seconds=15, compute_all_metrics=True,
                        enable_cross_validation=enable_cross_validation)
        optimizer = Optimizer(task, config=config)
        for result in optimizer.optimize():
            for metric in CLASSIFICATION_METRICS:
                if 'ROC_AUC' not in PerformanceMetric.Name(metric):
                    score = result.pipeline.get_score(metric)
                    assert score is not None


def test_compute_all_metrics_regression(df_196):
    # make task
    df = df_196
    metric = PerformanceMetric.Value('MEAN_SQUARED_ERROR')
    target_column = "class"
    task = Task([TaskKeyword.Value('REGRESSION')], [metric], [target_column], df)

    # run optimizer
    for enable_cross_validation in [True, False]:
        config = Config(timeout_seconds=15, compute_all_metrics=True,
                        enable_cross_validation=enable_cross_validation)
        optimizer = Optimizer(task, config=config)
        for result in optimizer.optimize():
            for metric in REGRESSION_METRICS:
                score = result.pipeline.get_score(metric)
                assert score is not None
