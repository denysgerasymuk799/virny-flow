# pylint: disable=missing-docstring
"""Tests for cost model."""

from alpine_meadow.common import Task, TaskKeyword, PerformanceMetric, Config
from alpine_meadow.core import Optimizer


def test_cost_model(df_185):
    # make task
    df = df_185
    metrics = [PerformanceMetric.Value('F1_MACRO'), PerformanceMetric.Value('ACCURACY')]
    target_column = "Hall_of_Fame"
    task = Task([TaskKeyword.Value('CLASSIFICATION')], metrics, [target_column], df)

    # run optimizer
    config = Config(timeout_seconds=15)
    optimizer = Optimizer(task, config=config)
    for result in optimizer.optimize():
        print(f'Time: {result.elapsed_time}, Score: {result.score}')

    # check cost model
    cost_model = optimizer.cost_model
    pipeline_arms = optimizer.search_space.pipeline_arms
    current_quality_num = 0
    external_quality_num = 0
    time_num = 0
    for pipeline_arm in pipeline_arms:
        pipeline_arm.compute_metrics()
        score = cost_model.compute_score(pipeline_arm)
        assert score is not None
        current_quality_mean, current_quality_std = cost_model.quality_model.estimate_current_quality(pipeline_arm)
        if current_quality_mean is not None and current_quality_std is not None:
            current_quality_num += 1
        external_quality_mean, external_quality_std = cost_model.quality_model.get_external_quality(pipeline_arm)
        if external_quality_mean is not None and external_quality_std is not None:
            external_quality_num += 1
        time_mean = cost_model.time_model.estimate_time(pipeline_arm)
        if time_mean is not None:
            time_num += 1
    assert current_quality_num > 0
    assert external_quality_num > 100
    assert time_num > 0
    assert current_quality_num == time_num
