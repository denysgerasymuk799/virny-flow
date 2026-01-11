# pylint: disable=missing-docstring
"""Tests for evaluation methods."""

import random

from alpine_meadow.common import Task, TaskKeyword, PerformanceMetric, Config
from alpine_meadow.core import Optimizer
from alpine_meadow.core.evaluation import NaiveEvaluation, AdaptiveEvaluation


def check_pipeline(pipeline_executor, config, metrics):
    for metric in metrics:
        scores = pipeline_executor.get_scores(metric=metric)
        if config.enable_cross_validation:
            assert len(scores) == config.cross_validation_k_folds_num
            assert pipeline_executor.validation_method.HasField('cross_validation')
            cross_validation = pipeline_executor.validation_method.cross_validation
            assert cross_validation.num_folds == config.cross_validation_k_folds_num
        else:
            assert len(scores) == 1
            assert pipeline_executor.validation_method.HasField('holdout')
            holdout = pipeline_executor.validation_method.holdout
            assert abs(holdout.train_proportion - config.train_data_size) < 1e-4
            assert abs(holdout.test_proportion - (1 - config.train_data_size)) < 1e-4


def test_naive(df_185):
    # make task
    df = df_185
    metrics = [PerformanceMetric.Value('F1_MACRO'), PerformanceMetric.Value('ACCURACY')]
    target_column = "Hall_of_Fame"

    # make evaluation method
    for enable_cross_validation in [True, False]:
        task = Task([TaskKeyword.Value('CLASSIFICATION')], metrics, [target_column], df)
        config = Config(evaluation_method='naive', enable_cross_validation=enable_cross_validation)
        optimizer = Optimizer(task, config=config)
        evaluation = optimizer.evaluation_method
        assert isinstance(evaluation, NaiveEvaluation)

        # evaluate
        pipeline_arm = random.choice(optimizer.search_space.pipeline_arms)
        pipeline = pipeline_arm.get_next_pipelines(
            use_bayesian_optimization=optimizer.config.enable_bayesian_optimization,
            pipelines_num=optimizer.config.configurations_per_arm_num)[0]
        pipeline_executor = next(evaluation.validate_pipeline(pipeline))

        # check pipeline executor
        check_pipeline(pipeline_executor, config, metrics)


def test_adaptive(df_185):
    # make task
    df = df_185
    metrics = [PerformanceMetric.Value('F1_MACRO'), PerformanceMetric.Value('ACCURACY')]
    target_column = "Hall_of_Fame"

    for enable_cross_validation in [True, False]:
        task = Task([TaskKeyword.Value('CLASSIFICATION')], metrics, [target_column], df)
        config = Config(evaluation_method='adaptive', enable_aps_pruning=True,
                        enable_cross_validation=enable_cross_validation)
        optimizer = Optimizer(task, config=config)
        evaluation = optimizer.evaluation_method
        assert isinstance(evaluation, AdaptiveEvaluation)

        # evaluate
        pipeline_arm = random.choice(optimizer.search_space.pipeline_arms)
        pipeline = pipeline_arm.get_next_pipelines(
            use_bayesian_optimization=optimizer.config.enable_bayesian_optimization,
            pipelines_num=optimizer.config.configurations_per_arm_num)[0]

        # check pipeline executor
        for pipeline_executor in evaluation.validate_pipeline(pipeline):
            check_pipeline(pipeline_executor, config, metrics)


def test_adaptive_curve_fitting(df_185):
    # make task
    df = df_185
    metrics = [PerformanceMetric.Value('F1_MACRO'), PerformanceMetric.Value('ACCURACY')]
    target_column = "Hall_of_Fame"
    task = Task([TaskKeyword.Value('CLASSIFICATION')], metrics, [target_column], df)

    config = Config(evaluation_method='adaptive', enable_cross_validation=False,
                    enable_aps_pruning=False, enable_aps_curve_fitting=True)
    optimizer = Optimizer(task, config=config)
    evaluation = optimizer.evaluation_method
    assert isinstance(evaluation, AdaptiveEvaluation)

    # evaluate
    pipeline_arm = random.choice(optimizer.search_space.pipeline_arms)
    pipeline = pipeline_arm.get_next_pipelines(
        use_bayesian_optimization=optimizer.config.enable_bayesian_optimization,
        pipelines_num=optimizer.config.configurations_per_arm_num)[0]

    # check pipeline executor
    for pipeline_executor in evaluation.validate_pipeline(pipeline):
        for metric in metrics:
            scores = pipeline_executor.get_scores(metric=metric)
            assert scores
